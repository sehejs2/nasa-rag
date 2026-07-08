"""Stage 2 of retrieval: rerank the vector-recall candidate pool.

A cross-encoder reranker could be swapped in behind the same Reranker
interface and compared against LLMReranker via the Phase 7 eval harness once
there's a labeled set to score against.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

from openai import AsyncOpenAI

from app.config import settings
from app.retrieval.models import RetrievedChunk

RERANK_MODEL = "gpt-4o-mini"
MAX_PASSAGE_CHARS_FOR_PROMPT = 1000
MAX_RERANK_ATTEMPTS = 2

# The SDK's own default timeout is 600s (read/write/pool) with 2 internal
# retries on top - a single stalled request can silently eat 30+ minutes.
# We do our own retry loop below, so disable the SDK's and cap each request
# at a much shorter, explicit timeout instead.
OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Return candidates reordered by relevance, each with rerank_score set."""
        ...


class NoopReranker:
    """Preserves vector order. Used as the LLMReranker fallback and directly in tests."""

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        return [c.model_copy(update={"rerank_score": c.vector_score}) for c in candidates]


def _build_prompt(query: str, candidates: list[RetrievedChunk]) -> str:
    passages = "\n\n".join(
        f"[{i}] {c.text[:MAX_PASSAGE_CHARS_FOR_PROMPT]}" for i, c in enumerate(candidates, start=1)
    )
    return (
        f'Query: "{query}"\n\n'
        f"Passages:\n{passages}\n\n"
        f"Score how well each numbered passage answers the query, from 0 (irrelevant) "
        f"to 10 (directly answers it). Respond with strict JSON only, no prose or "
        f'markdown fencing: {{"scores": [<{len(candidates)} numbers, one per passage, '
        f"in passage order>]}}"
    )


def _parse_scores(content: str, expected_count: int) -> list[float] | None:
    try:
        data = json.loads(content)
        scores = data["scores"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(scores, list) or len(scores) != expected_count:
        return None
    try:
        return [float(s) for s in scores]
    except (TypeError, ValueError):
        return None


class LLMReranker:
    """One gpt-4o-mini call scoring all candidates 0-10 as strict JSON.

    Malformed output gets one retry; if that also fails, falls back to vector
    order (NoopReranker) rather than raising - a debug/demo endpoint shouldn't
    500 because the reranker returned bad JSON.
    """

    def __init__(self, client: AsyncOpenAI | None = None):
        self._client = client or AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        self._fallback = NoopReranker()

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        prompt = _build_prompt(query, candidates)
        scores: list[float] | None = None

        for attempt in range(1, MAX_RERANK_ATTEMPTS + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=RERANK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
            except Exception:
                logger.exception("LLM rerank call failed (attempt %d)", attempt)
                continue

            scores = _parse_scores(content, len(candidates))
            if scores is not None:
                break
            logger.warning("LLM rerank returned malformed JSON (attempt %d): %r", attempt, content)

        if scores is None:
            logger.warning("LLM reranker exhausted retries; falling back to vector order")
            return await self._fallback.rerank(query, candidates)

        scored = [
            c.model_copy(update={"rerank_score": score})
            for c, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda c: (c.rerank_score, c.vector_score), reverse=True)
        return scored
