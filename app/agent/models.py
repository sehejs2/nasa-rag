"""Trace models for the hand-rolled agent loop."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.retrieval.models import RetrievedChunk

Route = Literal["retrieval", "tools", "both", "direct"]
StoppedReason = Literal["model_finished", "max_iterations"]


class ToolCallRecord(BaseModel):
    iteration: int
    tool_name: str
    arguments: dict
    result_ok: bool
    result_summary: str
    latency_ms: float
    call_id: str = ""
    # Full, untruncated ToolResult.data (None for failed calls) - result_summary
    # above is truncated for human-readable display; Phase 6 source assembly
    # (app/agent/sources.py) needs the full payload to extract e.g. image URLs.
    result_data: dict | None = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def accumulate_usage(token_usage: TokenUsage, usage) -> None:
    """Add an OpenAI response's `.usage` onto a running TokenUsage total.

    Shared by the agent loop, the composer, and the Phase 7 eval harness's
    judge calls, so every code path that spends tokens accumulates the same way.
    """
    if usage is None:
        return
    token_usage.prompt_tokens += usage.prompt_tokens or 0
    token_usage.completion_tokens += usage.completion_tokens or 0
    token_usage.total_tokens += usage.total_tokens or 0


class AgentTrace(BaseModel):
    query: str
    route: Route
    tool_calls: list[ToolCallRecord]
    retrieved_chunks: list[RetrievedChunk]
    draft_answer: str
    iterations_used: int
    stopped_reason: StoppedReason
    total_latency_ms: float
    token_usage: TokenUsage
    # True when the caller (the Phase 6 /chat path) intends to recompose a
    # cited answer from the trace's sources rather than show draft_answer
    # directly. draft_answer is always populated the same way either way -
    # this is purely an informational marker for consumers.
    draft_answer_superseded: bool = False
