"""DB-touching tests. Require a reachable Postgres (see conftest.py); auto-skipped otherwise."""

from __future__ import annotations

import pytest

from app.ingestion.ingest import (
    determine_pending,
    fetch_existing_state,
    upsert_documents,
    upsert_pending_chunk_rows,
)

pytestmark = pytest.mark.integration


def _document(doc_id: str) -> dict:
    return {
        "id": doc_id,
        "title": "Test Document",
        "source_url": "https://example.com/test",
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


def test_upsert_produces_no_duplicates_when_run_twice(db_conn):
    doc = _document("doc-dup")
    chunk = _chunk("chunk-dup", "hello world", "doc-dup")

    for _ in range(2):
        upsert_documents(db_conn, [doc])
        existing = fetch_existing_state(db_conn)
        pending = determine_pending([chunk], existing)
        upsert_pending_chunk_rows(db_conn, pending)

    doc_count = db_conn.execute("SELECT count(*) FROM documents WHERE doc_id = %s", (doc["id"],)).fetchone()[0]
    chunk_count = db_conn.execute(
        "SELECT count(*) FROM chunks WHERE chunk_id = %s", (chunk["chunk_id"],)
    ).fetchone()[0]

    assert doc_count == 1
    assert chunk_count == 1


def test_unchanged_chunk_skipped_on_second_pass(db_conn):
    doc = _document("doc-skip")
    chunk = _chunk("chunk-skip", "hello world", "doc-skip")
    upsert_documents(db_conn, [doc])

    existing = fetch_existing_state(db_conn)
    first_pass_pending = determine_pending([chunk], existing)
    assert len(first_pass_pending) == 1
    upsert_pending_chunk_rows(db_conn, first_pass_pending)

    # Simulate a successful embedding write, as ingest.main() would do.
    db_conn.execute(
        "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
        ([0.1] * 1536, chunk["chunk_id"]),
    )

    existing = fetch_existing_state(db_conn)
    second_pass_pending = determine_pending([chunk], existing)

    assert second_pass_pending == []


def test_changed_content_gets_reembedded_and_replaces_old_row(db_conn):
    doc = _document("doc-change")
    original = _chunk("chunk-change", "original text", "doc-change")
    upsert_documents(db_conn, [doc])

    existing = fetch_existing_state(db_conn)
    pending = determine_pending([original], existing)
    upsert_pending_chunk_rows(db_conn, pending)
    db_conn.execute(
        "UPDATE chunks SET embedding = %s WHERE chunk_id = %s",
        ([0.1] * 1536, original["chunk_id"]),
    )

    updated = _chunk("chunk-change", "completely different text", "doc-change")
    existing = fetch_existing_state(db_conn)
    pending = determine_pending([updated], existing)

    assert [c["chunk_id"] for c in pending] == ["chunk-change"]
    upsert_pending_chunk_rows(db_conn, pending)

    row = db_conn.execute(
        "SELECT text, embedding FROM chunks WHERE chunk_id = %s", (updated["chunk_id"],)
    ).fetchone()
    assert row[0] == "completely different text"
    assert row[1] is None  # stale embedding invalidated, awaiting re-embedding
