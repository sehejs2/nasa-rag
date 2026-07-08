"""Pure-logic tests for the content_hash-based skip decision. No DB, no network."""

from app.ingestion.ingest import content_hash, determine_pending


def _chunk(chunk_id: str, text: str, doc_id: str = "doc-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "token_count": len(text.split()),
        "metadata": {"section_heading": None},
    }


def test_new_chunk_is_pending():
    chunks = [_chunk("c1", "hello world")]
    pending = determine_pending(chunks, existing={})
    assert [c["chunk_id"] for c in pending] == ["c1"]


def test_unchanged_chunk_produces_zero_pending():
    text = "hello world"
    chunks = [_chunk("c1", text)]
    existing = {"c1": (content_hash(text), False)}

    pending = determine_pending(chunks, existing)

    assert pending == []


def test_changed_content_is_pending():
    chunks = [_chunk("c1", "brand new text")]
    existing = {"c1": (content_hash("stale old text"), False)}

    pending = determine_pending(chunks, existing)

    assert [c["chunk_id"] for c in pending] == ["c1"]
    assert pending[0]["_content_hash"] == content_hash("brand new text")


def test_missing_embedding_is_retried_even_if_content_unchanged():
    text = "hello world"
    chunks = [_chunk("c1", text)]
    existing = {"c1": (content_hash(text), True)}  # embedding_missing=True

    pending = determine_pending(chunks, existing)

    assert [c["chunk_id"] for c in pending] == ["c1"]


def test_mixed_batch_only_returns_the_chunks_that_need_work():
    chunks = [
        _chunk("new", "new chunk"),
        _chunk("unchanged", "same as before"),
        _chunk("changed", "updated text"),
    ]
    existing = {
        "unchanged": (content_hash("same as before"), False),
        "changed": (content_hash("old text"), False),
    }

    pending = determine_pending(chunks, existing)

    assert {c["chunk_id"] for c in pending} == {"new", "changed"}
