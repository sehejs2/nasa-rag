"""Integration test for the pgvector similarity search query.

Regression coverage for a real bug: query_embedding is a plain Python
list[float] (exactly what Embedder.embed_all returns), and psycopg has no
dumper registered for plain lists - an uncast parameter is sent as a
`double precision[]` array, and `vector <=> double precision[]` doesn't
exist as an operator. This raised psycopg.errors.UndefinedFunction in
scripts/search_smoke.py until the query cast the parameter to ::vector.
"""

from __future__ import annotations

import pytest

from app.ingestion.ingest import (
    determine_pending,
    fetch_existing_state,
    upsert_documents,
    upsert_pending_chunk_rows,
    write_embeddings,
)
from app.ingestion.search import similarity_search

pytestmark = pytest.mark.integration


def _document(doc_id: str) -> dict:
    return {
        "id": doc_id,
        "title": "Search Test Document",
        "source_url": "https://example.com/search",
        "source_family": "nasa_general",
        "fetched_at": "2026-01-01T00:00:00+00:00",
    }


def _chunk(chunk_id: str, text: str, doc_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "token_count": len(text.split()),
        "metadata": {"section_heading": "Intro"},
    }


def test_similarity_search_accepts_plain_list_query_embedding(db_conn):
    doc_id = "doc-search"
    upsert_documents(db_conn, [_document(doc_id)])

    chunks = [
        _chunk("chunk-aligned", "aligned match text", doc_id),
        _chunk("chunk-orthogonal", "orthogonal match text", doc_id),
    ]
    existing = fetch_existing_state(db_conn)
    pending = determine_pending(chunks, existing)
    upsert_pending_chunk_rows(db_conn, pending)

    aligned_vec = [1.0] + [0.0] * 1535
    orthogonal_vec = [0.0, 1.0] + [0.0] * 1534
    write_embeddings(db_conn, {"chunk-aligned": aligned_vec, "chunk-orthogonal": orthogonal_vec})

    # Plain Python list, unwrapped - matches what Embedder.embed_all returns.
    query_embedding = [1.0] + [0.0] * 1535

    # Other integration tests share this session-scoped test database, so filter
    # down to this test's own document rather than assuming an empty table.
    all_rows = similarity_search(db_conn, query_embedding, top_k=50)
    rows = [row for row in all_rows if row[2] == "Search Test Document"]

    assert len(rows) == 2
    texts_by_rank = [row[0] for row in rows]
    similarities_by_rank = [row[3] for row in rows]
    assert texts_by_rank[0] == "aligned match text"
    assert similarities_by_rank[0] == pytest.approx(1.0, abs=1e-6)
    assert similarities_by_rank[1] == pytest.approx(0.0, abs=1e-6)
