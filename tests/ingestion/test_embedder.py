"""Embedder tests. All OpenAI calls are faked - never hit the network."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
from openai import RateLimitError

from app.ingestion.embedder import BATCH_SIZE, MAX_ATTEMPTS, Embedder


class _FakeEmbeddingsAPI:
    def __init__(self, create_fn):
        self._create_fn = create_fn

    async def create(self, model, input):  # noqa: A002 - matches OpenAI SDK's parameter name
        return await self._create_fn(model, input)


class _FakeClient:
    def __init__(self, create_fn):
        self.embeddings = _FakeEmbeddingsAPI(create_fn)


def _fake_response(texts: list[str]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))]) for t in texts])


def _rate_limit_error(retry_after: str = "0") -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
    return RateLimitError("rate limited", response=response, body=None)


async def test_batching_preserves_order_and_ids():
    calls: list[list[str]] = []

    async def create_fn(model, input):
        calls.append(list(input))
        return _fake_response(input)

    embedder = Embedder(client=_FakeClient(create_fn))
    pairs = [(f"id-{i}", f"text number {i}") for i in range(BATCH_SIZE * 2 + 30)]

    embeddings, failed = await embedder.embed_all(pairs)

    assert failed == []
    assert len(embeddings) == len(pairs)
    for chunk_id, text in pairs:
        assert embeddings[chunk_id] == [float(len(text))]

    batch_sizes = sorted(len(c) for c in calls)
    assert batch_sizes == [30, BATCH_SIZE, BATCH_SIZE]


async def test_retry_then_success():
    attempts = {"count": 0}

    async def create_fn(model, input):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _rate_limit_error()
        return _fake_response(input)

    embedder = Embedder(client=_FakeClient(create_fn))
    embeddings, failed = await embedder.embed_all([("a", "hello")])

    assert failed == []
    assert embeddings["a"] == [float(len("hello"))]
    assert attempts["count"] == 3


async def test_retry_exhausts_and_reports_failure_without_raising():
    attempts = {"count": 0}

    async def create_fn(model, input):
        attempts["count"] += 1
        raise _rate_limit_error()

    embedder = Embedder(client=_FakeClient(create_fn))
    embeddings, failed = await embedder.embed_all([("a", "hello")])

    assert embeddings == {}
    assert failed == ["a"]
    assert attempts["count"] == MAX_ATTEMPTS


async def test_one_batch_failing_does_not_drop_others():
    # BATCH_SIZE good pairs fill exactly one batch; the bad pair lands in its own
    # second batch, so we can assert one batch's failure doesn't affect the other.
    async def create_fn(model, input):
        if any("bad" in t for t in input):
            raise _rate_limit_error()
        return _fake_response(input)

    good_pairs = [(f"good-{i}", f"good text {i}") for i in range(BATCH_SIZE)]
    bad_pairs = [("bad-1", "bad text")]

    embedder = Embedder(client=_FakeClient(create_fn))
    embeddings, failed = await embedder.embed_all(good_pairs + bad_pairs)

    assert failed == ["bad-1"]
    assert set(embeddings.keys()) == {chunk_id for chunk_id, _ in good_pairs}
