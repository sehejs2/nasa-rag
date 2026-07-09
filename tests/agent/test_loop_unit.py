"""Unit tests for agent loop mechanics. OpenAI and tool execution are fully faked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent import loop
from app.tools.base import ToolResult


def _tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(content: str | None = None, tool_calls=None, usage=(10, 5, 15)) -> SimpleNamespace:
    prompt, completion, total = usage
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total),
    )


class _FakeCompletionsAPI:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        # `messages` is a mutable list the loop keeps appending to across
        # iterations - snapshot it so later mutations don't retroactively
        # change what an earlier call appears to have been sent.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.calls.append(snapshot)
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses: list):
        self.completions = _FakeCompletionsAPI(responses)


class _FakeClient:
    def __init__(self, responses: list):
        self.chat = _FakeChat(responses)


async def test_tool_call_then_content_drives_correct_flow(monkeypatch):
    executed = []

    async def fake_execute(name, args):
        executed.append((name, args))
        return ToolResult(tool_name=name, ok=True, data={"foo": "bar"}, latency_ms=1.0, called_with=args)

    monkeypatch.setattr(loop, "execute", fake_execute)

    tool_call = _tool_call("call_1", "iss_now", {})
    responses = [
        _response(tool_calls=[tool_call]),
        _response(content="Final answer here."),
    ]
    client = _FakeClient(responses)

    trace = await loop.run_agent("where is the iss", client=client)

    assert executed == [("iss_now", {})]
    assert trace.draft_answer == "Final answer here."
    assert trace.iterations_used == 2
    assert trace.stopped_reason == "model_finished"
    assert trace.route == "tools"
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "iss_now"
    assert trace.tool_calls[0].result_ok is True
    assert trace.token_usage.total_tokens == 30  # two responses of 15 each

    second_call_messages = client.chat.completions.calls[1]["messages"]
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1]["role"] == "user"
    assert second_call_messages[2]["role"] == "assistant"
    assert second_call_messages[2]["tool_calls"][0]["id"] == "call_1"
    assert second_call_messages[3]["role"] == "tool"
    assert second_call_messages[3]["tool_call_id"] == "call_1"


async def test_parallel_tool_calls_all_execute(monkeypatch):
    executed = []

    async def fake_execute(name, args):
        executed.append(name)
        return ToolResult(tool_name=name, ok=True, data={}, latency_ms=1.0, called_with=args)

    monkeypatch.setattr(loop, "execute", fake_execute)

    tool_calls = [_tool_call("c1", "iss_now", {}), _tool_call("c2", "apod", {})]
    responses = [_response(tool_calls=tool_calls), _response(content="done")]
    client = _FakeClient(responses)

    trace = await loop.run_agent("q", client=client)

    assert set(executed) == {"iss_now", "apod"}
    assert len(trace.tool_calls) == 2
    assert trace.route == "tools"

    tool_messages = [m for m in client.chat.completions.calls[1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_messages} == {"c1", "c2"}


async def test_failed_tool_result_serialized_back_not_raised(monkeypatch):
    async def fake_execute(name, args):
        return ToolResult(tool_name=name, ok=False, error="boom", latency_ms=1.0, called_with=args)

    monkeypatch.setattr(loop, "execute", fake_execute)

    tool_call = _tool_call("c1", "iss_now", {})
    responses = [_response(tool_calls=[tool_call]), _response(content="handled gracefully")]
    client = _FakeClient(responses)

    trace = await loop.run_agent("q", client=client)

    assert trace.tool_calls[0].result_ok is False
    assert "boom" in trace.tool_calls[0].result_summary
    assert trace.draft_answer == "handled gracefully"

    tool_message = client.chat.completions.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    payload = json.loads(tool_message["content"])
    assert payload["ok"] is False
    assert payload["error"] == "boom"


async def test_search_documents_result_populates_retrieved_chunks(monkeypatch):
    chunk_payload = {
        "chunks": [
            {
                "chunk_id": "abc123",
                "doc_id": "doc-1",
                "title": "Some Title",
                "source_url": "https://example.com/doc",
                "source_family": "nasa_general",
                "section": "Intro",
                "text": "Some retrieved text.",
                "similarity": 8.5,
            }
        ]
    }

    async def fake_execute(name, args):
        return ToolResult(tool_name=name, ok=True, data=chunk_payload, latency_ms=1.0, called_with=args)

    monkeypatch.setattr(loop, "execute", fake_execute)

    tool_call = _tool_call("c1", "search_documents", {"query": "some query"})
    responses = [_response(tool_calls=[tool_call]), _response(content="answer")]
    client = _FakeClient(responses)

    trace = await loop.run_agent("q", client=client)

    assert trace.route == "retrieval"
    assert len(trace.retrieved_chunks) == 1
    chunk = trace.retrieved_chunks[0]
    assert chunk.chunk_id == "abc123"
    assert chunk.source_url == "https://example.com/doc"
    assert chunk.vector_score == 8.5
    assert chunk.rerank_score == 8.5
    assert chunk.rank == 1


async def test_max_iterations_guard_forces_final_answer(monkeypatch):
    async def fake_execute(name, args):
        return ToolResult(tool_name=name, ok=True, data={}, latency_ms=1.0, called_with=args)

    monkeypatch.setattr(loop, "execute", fake_execute)

    tool_call = _tool_call("c1", "iss_now", {})
    # Every iteration returns another tool call, never a final content-only message.
    responses = [_response(tool_calls=[tool_call]) for _ in range(4)] + [
        _response(content="forced final answer")
    ]
    client = _FakeClient(responses)

    trace = await loop.run_agent("q", max_iterations=4, client=client)

    assert trace.stopped_reason == "max_iterations"
    assert trace.iterations_used == 4
    assert trace.draft_answer == "forced final answer"

    final_call_kwargs = client.chat.completions.calls[-1]
    assert "tools" not in final_call_kwargs
    assert "tool_choice" not in final_call_kwargs


@pytest.mark.parametrize(
    ("tool_names", "expected_route"),
    [
        (set(), "direct"),
        ({"search_documents"}, "retrieval"),
        ({"iss_now"}, "tools"),
        ({"search_documents", "apod"}, "both"),
    ],
)
def test_derive_route(tool_names, expected_route):
    assert loop._derive_route(tool_names) == expected_route
