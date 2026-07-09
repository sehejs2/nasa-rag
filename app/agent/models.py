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


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
