"""Unit tests for the two-stage retrieve() orchestration. No network, no DB.

DB access is faked at the connection level (a stub with .execute().fetchall()
returning canned rows) rather than mocking fetch_candidates itself, so the row
-> RetrievedChunk mapping is exercised too.
"""

from __future__ import annotations

import pytest

from app.config import settings
from app.retrieval import retriever
from app.retrieval.reranker import NoopReranker


class _FakeEmbedder:
    def __init__(self, vector: list[float]):
        self._vector = vector

    async def embed_all(self, id_text_pairs):
        return {chunk_id: self._vector for chunk_id, _ in id_text_pairs}, []


class _ReverseReranker:
    """Test double that reverses vector order, to prove stage 2 actually runs."""

    async def rerank(self, query, candidates):
        reversed_candidates = list(reversed(candidates))
        return [
            c.model_copy(update={"rerank_score": float(len(candidates) - i)})
            for i, c in enumerate(reversed_candidates)
        ]


def _row(chunk_id: str, similarity: float) -> tuple:
    return (chunk_id, "doc-1", f"text for {chunk_id}", "Title", "https://example.com/doc", "nasa_general", None, similarity)


class _FakeCursorResult:
    def __init__(self, rows: list[tuple]):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple]):
        self._rows = rows

    def execute(self, sql, params):
        return _FakeCursorResult(self._rows)

    def close(self):
        pass


def test_fetch_candidates_maps_db_rows_to_retrieved_chunks():
    fake_conn = _FakeConn([_row("c1", 0.87)])

    candidates = retriever.fetch_candidates(fake_conn, [0.1] * 1536, pool_size=20)

    assert len(candidates) == 1
    chunk = candidates[0]
    assert chunk.chunk_id == "c1"
    assert chunk.title == "Title"
    assert chunk.vector_score == pytest.approx(0.87)
    assert chunk.rerank_score == pytest.approx(0.87)
    assert chunk.rank == 0


async def test_two_stage_flow_reranks_and_respects_top_k(monkeypatch):
    rows = [_row(f"c{i}", 1.0 - i * 0.01) for i in range(20)]
    fake_conn = _FakeConn(rows)
    monkeypatch.setattr(retriever, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retriever, "init_schema", lambda conn: None)

    result = await retriever.retrieve(
        "some query",
        top_k=5,
        embedder=_FakeEmbedder([0.1] * 1536),
        reranker=_ReverseReranker(),
    )

    assert len(result.chunks) == 5
    # _ReverseReranker reversed vector order, so the top result is the last
    # vector-order candidate (c19), proving stage 2 actually reordered stage 1's output.
    assert result.chunks[0].chunk_id == "c19"
    assert [c.rank for c in result.chunks] == [1, 2, 3, 4, 5]
    assert result.reranker_used == "_ReverseReranker"
    assert result.timings.total_ms >= 0


async def test_two_stage_flow_with_noop_reranker_preserves_vector_order(monkeypatch):
    rows = [_row(f"c{i}", 1.0 - i * 0.01) for i in range(20)]
    fake_conn = _FakeConn(rows)
    monkeypatch.setattr(retriever, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retriever, "init_schema", lambda conn: None)

    result = await retriever.retrieve(
        "some query",
        top_k=3,
        embedder=_FakeEmbedder([0.1] * 1536),
        reranker=NoopReranker(),
    )

    assert [c.chunk_id for c in result.chunks] == ["c0", "c1", "c2"]


async def test_default_top_k_comes_from_settings(monkeypatch):
    rows = [_row(f"c{i}", 1.0 - i * 0.01) for i in range(20)]
    fake_conn = _FakeConn(rows)
    monkeypatch.setattr(retriever, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(retriever, "init_schema", lambda conn: None)

    result = await retriever.retrieve(
        "some query",
        embedder=_FakeEmbedder([0.1] * 1536),
        reranker=NoopReranker(),
    )

    assert len(result.chunks) == settings.RETRIEVAL_TOP_K
