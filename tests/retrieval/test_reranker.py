"""Reranker tests. All OpenAI calls are faked - never hit the network."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import MAX_RERANK_ATTEMPTS, LLMReranker, NoopReranker


class _FakeCompletionsAPI:
    def __init__(self, create_fn):
        self._create_fn = create_fn

    async def create(self, **kwargs):
        return await self._create_fn(**kwargs)


class _FakeChat:
    def __init__(self, create_fn):
        self.completions = _FakeCompletionsAPI(create_fn)


class _FakeClient:
    def __init__(self, create_fn):
        self.chat = _FakeChat(create_fn)


def _fake_llm_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _chunk(chunk_id: str, vector_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=f"text for {chunk_id}",
        title="Title",
        source_url="https://example.com/doc",
        source_family="nasa_general",
        section=None,
        vector_score=vector_score,
        rerank_score=vector_score,
        rank=0,
    )


async def test_well_formed_scores_reorder_candidates():
    candidates = [_chunk("c1", 0.9), _chunk("c2", 0.8), _chunk("c3", 0.7)]

    async def create_fn(**kwargs):
        return _fake_llm_response(json.dumps({"scores": [2, 9, 5]}))

    result = await LLMReranker(client=_FakeClient(create_fn)).rerank("query", candidates)

    assert [c.chunk_id for c in result] == ["c2", "c3", "c1"]
    assert [c.rerank_score for c in result] == [9.0, 5.0, 2.0]


async def test_malformed_json_then_retry_succeeds():
    attempts = {"count": 0}
    candidates = [_chunk("c1", 0.9), _chunk("c2", 0.8)]

    async def create_fn(**kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _fake_llm_response("not valid json at all")
        return _fake_llm_response(json.dumps({"scores": [1, 9]}))

    result = await LLMReranker(client=_FakeClient(create_fn)).rerank("query", candidates)

    assert attempts["count"] == 2
    assert [c.chunk_id for c in result] == ["c2", "c1"]


async def test_malformed_json_exhausts_retries_falls_back_to_vector_order():
    attempts = {"count": 0}
    candidates = [_chunk("c1", 0.9), _chunk("c2", 0.8)]

    async def create_fn(**kwargs):
        attempts["count"] += 1
        return _fake_llm_response("still not json")

    result = await LLMReranker(client=_FakeClient(create_fn)).rerank("query", candidates)

    assert attempts["count"] == MAX_RERANK_ATTEMPTS
    assert [c.chunk_id for c in result] == ["c1", "c2"]
    assert [c.rerank_score for c in result] == [0.9, 0.8]


async def test_wrong_length_score_list_is_treated_as_malformed():
    candidates = [_chunk("c1", 0.9), _chunk("c2", 0.8), _chunk("c3", 0.7)]

    async def create_fn(**kwargs):
        return _fake_llm_response(json.dumps({"scores": [1, 2]}))  # only 2 for 3 candidates

    result = await LLMReranker(client=_FakeClient(create_fn)).rerank("query", candidates)

    assert [c.chunk_id for c in result] == ["c1", "c2", "c3"]  # fell back to vector order


async def test_empty_candidates_makes_no_api_call():
    called = {"count": 0}

    async def create_fn(**kwargs):
        called["count"] += 1
        return _fake_llm_response(json.dumps({"scores": []}))

    result = await LLMReranker(client=_FakeClient(create_fn)).rerank("query", [])

    assert result == []
    assert called["count"] == 0


async def test_noop_reranker_preserves_vector_order():
    candidates = [_chunk("c1", 0.9), _chunk("c2", 0.8), _chunk("c3", 0.7)]

    result = await NoopReranker().rerank("query", candidates)

    assert [c.chunk_id for c in result] == ["c1", "c2", "c3"]
    assert [c.rerank_score for c in result] == [0.9, 0.8, 0.7]
