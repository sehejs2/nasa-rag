"""Embed data/processed/chunks.jsonl and upsert into the pgvector store.

Idempotent: a chunk is only (re-)embedded if it's new, its text changed since
the last ingest (content_hash mismatch), or a previous run left it without an
embedding (e.g. it was interrupted or failed after retries). Unchanged chunks
are skipped with no OpenAI call and no DB write. Safe to interrupt and re-run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import psycopg

from app.ingestion.db import get_connection, init_schema
from app.ingestion.embedder import Embedder

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MANIFEST_PATH = DATA_DIR / "raw" / "manifest.json"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"

COST_PER_MILLION_TOKENS_USD = 0.02

UPSERT_CHUNK_SQL = """
INSERT INTO chunks (chunk_id, doc_id, text, token_count, section, content_hash, embedding, created_at)
VALUES (%(chunk_id)s, %(doc_id)s, %(text)s, %(token_count)s, %(section)s, %(content_hash)s, NULL, now())
ON CONFLICT (chunk_id) DO UPDATE SET
    doc_id = EXCLUDED.doc_id,
    text = EXCLUDED.text,
    token_count = EXCLUDED.token_count,
    section = EXCLUDED.section,
    content_hash = EXCLUDED.content_hash,
    embedding = CASE
        WHEN chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash THEN NULL
        ELSE chunks.embedding
    END
"""

UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (doc_id, title, source_url, source_family, fetched_at)
VALUES (%(id)s, %(title)s, %(source_url)s, %(source_family)s, %(fetched_at)s)
ON CONFLICT (doc_id) DO UPDATE SET
    title = EXCLUDED.title,
    source_url = EXCLUDED.source_url,
    source_family = EXCLUDED.source_family,
    fetched_at = EXCLUDED.fetched_at
"""


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_documents(path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_existing_state(conn: psycopg.Connection) -> dict[str, tuple[str, bool]]:
    """chunk_id -> (stored content_hash, embedding_is_missing)."""
    rows = conn.execute("SELECT chunk_id, content_hash, embedding IS NULL FROM chunks").fetchall()
    return {chunk_id: (stored_hash, embedding_missing) for chunk_id, stored_hash, embedding_missing in rows}


def determine_pending(
    chunks: list[dict[str, Any]], existing: dict[str, tuple[str, bool]]
) -> list[dict[str, Any]]:
    """Return chunks that are new, changed, or missing an embedding from a prior run.

    Each returned chunk is annotated with `_content_hash`. A chunk already in
    `existing` with a matching content_hash and a non-null embedding is left out
    entirely - it triggers no DB write and no embedding call.
    """
    pending = []
    for chunk in chunks:
        new_hash = content_hash(chunk["text"])
        stored = existing.get(chunk["chunk_id"])
        needs_embedding = stored is None or stored[0] != new_hash or stored[1]
        if needs_embedding:
            pending.append({**chunk, "_content_hash": new_hash})
    return pending


def upsert_documents(conn: psycopg.Connection, documents: list[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(UPSERT_DOCUMENT_SQL, documents)


def upsert_pending_chunk_rows(conn: psycopg.Connection, pending: list[dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            UPSERT_CHUNK_SQL,
            [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "text": c["text"],
                    "token_count": c["token_count"],
                    "section": c["metadata"].get("section_heading"),
                    "content_hash": c["_content_hash"],
                }
                for c in pending
            ],
        )


def write_embeddings(conn: psycopg.Connection, embeddings: dict[str, list[float]]) -> None:
    """Write embeddings (plain list[float], as returned by Embedder) into the vector column.

    Cast to ::vector explicitly - psycopg has no dumper registered for plain
    Python lists, so an uncast parameter would be sent as a double precision[]
    array. Assignment happens to tolerate that via an implicit cast, but the
    equivalent `<=>` comparison in app/ingestion/search.py does not, so we cast
    explicitly here too rather than rely on that asymmetry.
    """
    if not embeddings:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %(embedding)s::vector WHERE chunk_id = %(chunk_id)s",
            [{"chunk_id": cid, "embedding": vec} for cid, vec in embeddings.items()],
        )


def print_plan(total_chunks: int, pending: list[dict[str, Any]]) -> None:
    total_tokens = sum(c["token_count"] for c in pending)
    estimated_cost = total_tokens / 1_000_000 * COST_PER_MILLION_TOKENS_USD
    print(f"Chunks total: {total_chunks}")
    print(f"Chunks needing embedding: {len(pending)}")
    print(f"Estimated tokens to embed: {total_tokens}")
    print(f"Estimated cost: ${estimated_cost:.4f} (text-embedding-3-small @ $0.02 / 1M tokens)")


def main() -> None:
    documents = load_documents()
    chunks = load_chunks()

    conn = get_connection()
    init_schema(conn)

    upsert_documents(conn, documents)
    existing = fetch_existing_state(conn)
    pending = determine_pending(chunks, existing)

    print_plan(len(chunks), pending)

    if not pending:
        print(f"\nEmbedded: 0, skipped: {len(chunks)}, failed: 0")
        conn.close()
        return

    upsert_pending_chunk_rows(conn, pending)

    def on_batch_done(index: int, total: int, size: int, success: bool) -> None:
        status = "ok" if success else "FAILED"
        print(f"[batch {index}/{total}] {size} chunks -> {status}", file=sys.stderr)

    embedder = Embedder()
    embeddings, failed_ids = asyncio.run(
        embedder.embed_all(
            [(c["chunk_id"], c["text"]) for c in pending],
            on_batch_done=on_batch_done,
        )
    )

    write_embeddings(conn, embeddings)
    conn.close()

    skipped = len(chunks) - len(pending)
    print(f"\nEmbedded: {len(embeddings)}, skipped: {skipped}, failed: {len(failed_ids)}")
    if failed_ids:
        print("Failed chunk_ids (will retry on next `make ingest` run):")
        for chunk_id in failed_ids:
            print(f"  - {chunk_id}")


if __name__ == "__main__":
    main()
