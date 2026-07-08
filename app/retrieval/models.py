"""Shared data contracts for the retrieval module.

RetrievedChunk is the contract the agent (Phase 5) and citation rendering
(Phase 6) will consume - keep it stable and free of retrieval-internal detail.
"""

from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    title: str
    source_url: str
    source_family: str
    section: str | None
    vector_score: float
    rerank_score: float
    rank: int


class RetrievalTimings(BaseModel):
    embed_ms: float
    vector_search_ms: float
    rerank_ms: float
    total_ms: float


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    timings: RetrievalTimings
    reranker_used: str
