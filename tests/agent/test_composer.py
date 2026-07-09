"""Unit tests for citation validation and composition prompt assembly.

All OpenAI calls are faked - never hit the network.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.agent.composer import build_composition_messages, compose_answer, validate_citations
from app.agent.models import AgentTrace, TokenUsage, ToolCallRecord
from app.agent.sources import Source
from app.retrieval.models import RetrievedChunk


def _source(number: int, kind: str = "chunk", ref_id: str = "c1") -> Source:
    return Source(number=number, kind=kind, title=f"Title {number}", url=None, detail="detail", ref_id=ref_id)


def _trace(**overrides) -> AgentTrace:
    defaults = dict(
        query="q",
        route="direct",
        tool_calls=[],
        retrieved_chunks=[],
        draft_answer="",
        iterations_used=1,
        stopped_reason="model_finished",
        total_latency_ms=1.0,
        token_usage=TokenUsage(),
    )
    defaults.update(overrides)
    return AgentTrace(**defaults)


def test_validate_citations_splits_valid_and_invalid():
    text = "Some claim [1]. Another [2][3]. A bad one [7]."
    sources = [_source(1), _source(2), _source(3), _source(4), _source(5)]

    summary = validate_citations(text, sources)

    assert summary.cited_sources == [1, 2, 3]
    assert summary.invalid_citations == [7]


def test_validate_citations_no_markers():
    summary = validate_citations("No citations here.", [_source(1)])

    assert summary.cited_sources == []
    assert summary.invalid_citations == []


def test_validate_citations_deduplicates_repeated_markers():
    summary = validate_citations("[1] and again [1] and [1]", [_source(1)])

    assert summary.cited_sources == [1]


def test_composition_prompt_includes_all_numbered_sources():
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="doc-1",
        text="Chunk body text.",
        title="Chunk Title",
        source_url="https://example.com/doc",
        source_family="nasa_general",
        section="Intro",
        vector_score=0.9,
        rerank_score=8.0,
        rank=1,
    )
    tool_record = ToolCallRecord(
        iteration=1,
        tool_name="iss_now",
        arguments={},
        result_ok=True,
        result_summary="summary",
        latency_ms=1.0,
        call_id="call-1",
        result_data={"latitude": 1.0, "longitude": 2.0},
    )
    trace = _trace(retrieved_chunks=[chunk], tool_calls=[tool_record])
    sources = [
        Source(number=1, kind="chunk", title="Chunk Title", url="https://example.com/doc", detail="Intro", ref_id="c1"),
        Source(number=2, kind="tool", title="iss_now", url=None, detail="iss_now()", ref_id="call-1"),
    ]

    messages = build_composition_messages("What is going on?", trace, sources)

    content = messages[0]["content"]
    assert "What is going on?" in content
    assert "[1] Chunk Title" in content
    assert "Chunk body text." in content
    assert "[2] iss_now" in content
    assert '"latitude": 1.0' in content


def test_composition_prompt_handles_no_sources():
    trace = _trace()
    messages = build_composition_messages("q", trace, [])

    assert "no sources" in messages[0]["content"].lower()


class _FakeStream:
    def __init__(self, chunks: list[str]):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for text in self._chunks:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    async def close(self):
        self.closed = True


class _FakeCompletionsAPI:
    def __init__(self, stream: _FakeStream):
        self._stream = stream

    async def create(self, **kwargs):
        assert kwargs["stream"] is True
        return self._stream


class _FakeChat:
    def __init__(self, stream: _FakeStream):
        self.completions = _FakeCompletionsAPI(stream)


class _FakeClient:
    def __init__(self, stream: _FakeStream):
        self.chat = _FakeChat(stream)


async def test_compose_answer_yields_deltas_and_closes_stream():
    stream = _FakeStream(["Hello ", "world."])
    client = _FakeClient(stream)
    trace = _trace()

    deltas = [d async for d in compose_answer("q", trace, [], client=client)]

    assert deltas == ["Hello ", "world."]
    assert stream.closed is True
