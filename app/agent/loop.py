"""Hand-rolled agent loop.

gpt-4o-mini routes between document retrieval, live NASA tools, both, or
neither via native OpenAI function calling over the unified tool registry
(search_documents included) - there is no separate hand-written classifier.
Tool failures (ToolResult(ok=False)) are serialized back to the model as data;
the loop itself never raises on a tool failure. Answer composition with
citations and streaming is Phase 6 - this produces a trace and a plain draft
answer only.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from openai import AsyncOpenAI

from app.agent.models import AgentTrace, TokenUsage, ToolCallRecord, accumulate_usage
from app.agent.prompts import SYSTEM_PROMPT
from app.config import settings
from app.retrieval.models import RetrievedChunk
from app.tools.base import ToolResult
from app.tools.registry import execute, get_openai_tools

MODEL = "gpt-4o-mini"
# See app/tools/base.py and app/retrieval/reranker.py for why this is explicit:
# the SDK's own default (600s, 2 internal retries) can turn one stalled call
# into an effectively indefinite hang.
OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0

MAX_MESSAGE_FIELD_CHARS = 1000
RESULT_SUMMARY_CHARS = 200

SEARCH_DOCUMENTS_TOOL = "search_documents"
LIVE_TOOL_NAMES = {"apod", "iss_now", "mars_rover_photos", "jwst_images"}


def _truncate_for_message(value: Any, limit: int = MAX_MESSAGE_FIELD_CHARS) -> Any:
    """Generic safety net: shrink any oversized string in a tool payload before
    it's JSON-serialized into a message sent back to the model. Individual
    tools may already truncate their own large fields (e.g. search_documents'
    chunk text); this catches anything that doesn't.
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...(truncated)"
    if isinstance(value, dict):
        return {k: _truncate_for_message(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_for_message(v, limit) for v in value]
    return value


def _derive_route(tool_names: set[str]) -> str:
    has_retrieval = SEARCH_DOCUMENTS_TOOL in tool_names
    has_live_tool = bool(tool_names & LIVE_TOOL_NAMES)
    if has_retrieval and has_live_tool:
        return "both"
    if has_retrieval:
        return "retrieval"
    if has_live_tool:
        return "tools"
    return "direct"


def _extract_retrieved_chunks(result: ToolResult) -> list[RetrievedChunk]:
    """Rebuild full RetrievedChunk objects from a successful search_documents result.

    search_documents' ToolResult.data already carries every citation-relevant
    field except a split vector/rerank score (it exposes one merged
    `similarity`); that single value is reused for both here since the
    distinction doesn't matter for citation rendering.
    """
    chunks = []
    for i, item in enumerate((result.data or {}).get("chunks", []), start=1):
        chunks.append(
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                title=item["title"],
                source_url=item["source_url"],
                source_family=item["source_family"],
                section=item.get("section"),
                vector_score=item["similarity"],
                rerank_score=item["similarity"],
                rank=i,
            )
        )
    return chunks


async def _execute_tool_call(tool_call: Any, iteration: int) -> tuple[ToolCallRecord, ToolResult]:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    result = await execute(name, args)

    if result.ok:
        summary = json.dumps(result.data)[:RESULT_SUMMARY_CHARS]
    else:
        summary = (result.error or "")[:RESULT_SUMMARY_CHARS]

    record = ToolCallRecord(
        iteration=iteration,
        tool_name=name,
        arguments=args,
        result_ok=result.ok,
        result_summary=summary,
        latency_ms=result.latency_ms,
        call_id=tool_call.id,
        result_data=result.data if result.ok else None,
    )
    return record, result


def _build_trace(
    *,
    query: str,
    tool_names_called: set[str],
    tool_calls_seen: list[ToolCallRecord],
    retrieved_chunks: list[RetrievedChunk],
    draft_answer: str,
    iterations_used: int,
    stopped_reason: str,
    start: float,
    token_usage: TokenUsage,
    draft_answer_superseded: bool,
) -> AgentTrace:
    return AgentTrace(
        query=query,
        route=_derive_route(tool_names_called),
        tool_calls=tool_calls_seen,
        retrieved_chunks=retrieved_chunks,
        draft_answer=draft_answer,
        iterations_used=iterations_used,
        stopped_reason=stopped_reason,
        total_latency_ms=(time.perf_counter() - start) * 1000,
        token_usage=token_usage,
        draft_answer_superseded=draft_answer_superseded,
    )


async def run_agent(
    query: str,
    max_iterations: int = 4,
    *,
    client: AsyncOpenAI | None = None,
    mark_draft_superseded: bool = False,
) -> AgentTrace:
    """Run the agent loop for tool orchestration and a draft answer.

    mark_draft_superseded=True is for the Phase 6 /chat path: the loop still
    runs exactly the same way and draft_answer is still captured the same way
    (nothing is skipped), but the trace is flagged so callers know
    draft_answer isn't meant to be shown directly - /chat recomposes a cited
    answer from the trace's sources instead. /agent/debug and make ask don't
    pass this, so their behavior is unchanged.
    """
    client = client or AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    tool_calls_seen: list[ToolCallRecord] = []
    retrieved_chunks: list[RetrievedChunk] = []
    tool_names_called: set[str] = set()
    token_usage = TokenUsage()
    iterations_used = 0
    start = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=get_openai_tools(),
            tool_choice="auto",
        )
        accumulate_usage(token_usage, response.usage)

        message = response.choices[0].message
        assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_message)

        if not message.tool_calls:
            return _build_trace(
                query=query,
                tool_names_called=tool_names_called,
                tool_calls_seen=tool_calls_seen,
                retrieved_chunks=retrieved_chunks,
                draft_answer=message.content or "",
                iterations_used=iterations_used,
                stopped_reason="model_finished",
                start=start,
                token_usage=token_usage,
                draft_answer_superseded=mark_draft_superseded,
            )

        results = await asyncio.gather(
            *(_execute_tool_call(tc, iteration) for tc in message.tool_calls)
        )

        for tool_call, (record, result) in zip(message.tool_calls, results, strict=True):
            tool_calls_seen.append(record)
            tool_names_called.add(record.tool_name)
            if record.tool_name == SEARCH_DOCUMENTS_TOOL and result.ok:
                retrieved_chunks.extend(_extract_retrieved_chunks(result))

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(_truncate_for_message(result.model_dump())),
                }
            )

    # Hit max_iterations without a final answer: force one last no-tools completion.
    final_response = await client.chat.completions.create(model=MODEL, messages=messages)
    accumulate_usage(token_usage, final_response.usage)

    return _build_trace(
        query=query,
        tool_names_called=tool_names_called,
        tool_calls_seen=tool_calls_seen,
        retrieved_chunks=retrieved_chunks,
        draft_answer=final_response.choices[0].message.content or "",
        iterations_used=iterations_used,
        stopped_reason="max_iterations",
        start=start,
        token_usage=token_usage,
        draft_answer_superseded=mark_draft_superseded,
    )
