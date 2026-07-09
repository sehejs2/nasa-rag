"""Unit tests for the /chat SSE event stream. OpenAI and the agent loop are fully faked."""

from __future__ import annotations

import json

from app.agent import chat_stream
from app.agent.models import AgentTrace, TokenUsage
from app.agent.sources import Source


def _trace(**overrides) -> AgentTrace:
    defaults = dict(
        query="q",
        route="direct",
        tool_calls=[],
        retrieved_chunks=[],
        draft_answer="ignored - superseded by composition",
        iterations_used=1,
        stopped_reason="model_finished",
        total_latency_ms=42.0,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    defaults.update(overrides)
    return AgentTrace(**defaults)


def _parse_events(raw_events: list[str]) -> list[tuple[str, object]]:
    parsed = []
    for raw in raw_events:
        lines = raw.strip("\n").split("\n")
        event_name = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        parsed.append((event_name, data))
    return parsed


async def test_event_order_and_payloads_happy_path(monkeypatch):
    trace = _trace(route="direct")

    async def fake_run_agent(query, mark_draft_superseded=False):
        assert mark_draft_superseded is True
        return trace

    async def fake_compose_answer(query, trace_arg, sources):
        for piece in ["Hello ", "world [1]."]:
            yield piece

    fake_source = Source(number=1, kind="chunk", title="T", url="https://x", detail="d", ref_id="c1")

    monkeypatch.setattr(chat_stream, "run_agent", fake_run_agent)
    monkeypatch.setattr(chat_stream, "compose_answer", fake_compose_answer)
    monkeypatch.setattr(chat_stream, "build_sources", lambda t: [fake_source])

    raw_events = [c async for c in chat_stream.chat_event_stream("q")]
    events = _parse_events(raw_events)

    assert [name for name, _ in events] == ["meta", "sources", "delta", "delta", "done"]

    meta = events[0][1]
    assert meta == {"route": "direct", "tools": [], "iterations": 1}

    sources_payload = events[1][1]
    assert sources_payload == [fake_source.model_dump()]

    deltas = [data["text"] for name, data in events if name == "delta"]
    assert deltas == ["Hello ", "world [1]."]

    done = events[-1][1]
    assert done["cited_sources"] == [1]
    assert done["invalid_citations"] == []
    assert done["token_usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert done["total_latency_ms"] == 42.0


async def test_meta_tools_reflect_unique_tool_names_sorted(monkeypatch):
    from app.agent.models import ToolCallRecord

    trace = _trace(
        route="both",
        tool_calls=[
            ToolCallRecord(
                iteration=1, tool_name="iss_now", arguments={}, result_ok=True,
                result_summary="", latency_ms=1.0, call_id="c1",
            ),
            ToolCallRecord(
                iteration=1, tool_name="apod", arguments={}, result_ok=True,
                result_summary="", latency_ms=1.0, call_id="c2",
            ),
        ],
    )

    async def fake_run_agent(query, mark_draft_superseded=False):
        return trace

    async def fake_compose_answer(query, trace_arg, sources):
        return
        yield  # pragma: no cover - makes this an async generator with no items

    monkeypatch.setattr(chat_stream, "run_agent", fake_run_agent)
    monkeypatch.setattr(chat_stream, "compose_answer", fake_compose_answer)
    monkeypatch.setattr(chat_stream, "build_sources", lambda t: [])

    raw_events = [c async for c in chat_stream.chat_event_stream("q")]
    events = _parse_events(raw_events)

    meta = events[0][1]
    assert meta["tools"] == ["apod", "iss_now"]


async def test_agent_loop_failure_produces_error_before_any_delta(monkeypatch):
    async def fake_run_agent(query, mark_draft_superseded=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_stream, "run_agent", fake_run_agent)

    raw_events = [c async for c in chat_stream.chat_event_stream("q")]
    events = _parse_events(raw_events)

    assert len(events) == 1
    assert events[0][0] == "error"
    assert "boom" in events[0][1]["message"]


async def test_composition_failure_mid_stream_produces_error_after_deltas(monkeypatch):
    trace = _trace()

    async def fake_run_agent(query, mark_draft_superseded=False):
        return trace

    async def fake_compose_answer(query, trace_arg, sources):
        yield "partial "
        raise RuntimeError("stream broke")

    monkeypatch.setattr(chat_stream, "run_agent", fake_run_agent)
    monkeypatch.setattr(chat_stream, "compose_answer", fake_compose_answer)
    monkeypatch.setattr(chat_stream, "build_sources", lambda t: [])

    raw_events = [c async for c in chat_stream.chat_event_stream("q")]
    events = _parse_events(raw_events)

    assert [name for name, _ in events] == ["meta", "sources", "delta", "error"]
    assert "stream broke" in events[-1][1]["message"]


def test_format_sse_event_shape():
    formatted = chat_stream.format_sse_event("meta", {"a": 1})

    assert formatted == 'event: meta\ndata: {"a": 1}\n\n'
