"""Async, batched embedding client for OpenAI's text-embedding-3-small."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100
MAX_CONCURRENT_BATCHES = 4
MAX_ATTEMPTS = 5

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_FALLBACK_WAIT = wait_exponential(multiplier=1, min=1, max=30)


def _wait_respecting_retry_after(retry_state):
    """Honor a Retry-After header if the provider sent one; else exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return _FALLBACK_WAIT(retry_state)


class Embedder:
    """Batches embedding requests with bounded concurrency and retry/backoff."""

    def __init__(self, client: AsyncOpenAI | None = None):
        self._client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=_wait_respecting_retry_after,
        stop=stop_after_attempt(MAX_ATTEMPTS),
        reraise=True,
    )
    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=MODEL, input=texts)
        return [item.embedding for item in response.data]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self._semaphore:
            return await self._embed_batch_with_retry(texts)

    async def embed_all(
        self,
        id_text_pairs: list[tuple[str, str]],
        on_batch_done: Callable[[int, int, int, bool], None] | None = None,
    ) -> tuple[dict[str, list[float]], list[str]]:
        """Embed (id, text) pairs in batches of BATCH_SIZE, up to MAX_CONCURRENT_BATCHES at once.

        Returns (chunk_id -> embedding for every chunk that succeeded, chunk_ids that
        failed after exhausting retries). A batch failure never drops other batches.
        `on_batch_done(batch_index, total_batches, batch_size, success)` fires as each
        batch finishes, in completion order, for progress reporting.
        """
        batches = [
            id_text_pairs[i : i + BATCH_SIZE] for i in range(0, len(id_text_pairs), BATCH_SIZE)
        ]
        total_batches = len(batches)

        async def run_batch(
            batch: list[tuple[str, str]], index: int
        ) -> tuple[dict[str, list[float]], list[str]]:
            ids = [chunk_id for chunk_id, _ in batch]
            texts = [text for _, text in batch]
            try:
                vectors = await self.embed_batch(texts)
            except Exception:
                logger.exception("Batch of %d chunks failed after retries", len(ids))
                if on_batch_done:
                    on_batch_done(index, total_batches, len(ids), False)
                return {}, ids
            if on_batch_done:
                on_batch_done(index, total_batches, len(ids), True)
            return dict(zip(ids, vectors, strict=True)), []

        results = await asyncio.gather(
            *(run_batch(batch, i) for i, batch in enumerate(batches, start=1))
        )

        embeddings: dict[str, list[float]] = {}
        failed_ids: list[str] = []
        for batch_embeddings, batch_failed in results:
            embeddings.update(batch_embeddings)
            failed_ids.extend(batch_failed)
        return embeddings, failed_ids
