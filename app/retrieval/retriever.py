"""Two-stage retrieval: vector recall (stage 1) + reranking (stage 2).

Stage 1 embeds the query and pulls a candidate pool from pgvector by cosine
similarity. Stage 2 hands that pool to a Reranker (LLMReranker by default) to
reorder for precision, then the result is truncated to top_k.
"""

from __future__ import annotations

import time

import psycopg

from app.config import settings
from app.ingestion.db import get_connection, init_schema
from app.ingestion.embedder import Embedder
from app.retrieval.models import RetrievalResult, RetrievalTimings, RetrievedChunk
from app.retrieval.reranker import LLMReranker, NoopReranker, Reranker

# Candidate embedding is a plain Python list[float] (see app/ingestion/search.py
# for the <=> vs double precision[] bug this cast avoids).
CANDIDATE_SQL = """
SELECT c.chunk_id, c.doc_id, c.text, d.title, d.source_url, d.source_family, c.section,
       1 - (c.embedding <=> %(query_embedding)s::vector) AS similarity
FROM chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(query_embedding)s::vector
LIMIT %(limit)s
"""


async def embed_query(query: str, embedder: Embedder) -> list[float]:
    embeddings, failed = await embedder.embed_all([("query", query)])
    if failed:
        raise RuntimeError("Failed to embed query for retrieval.")
    return embeddings["query"]


def fetch_candidates(
    conn: psycopg.Connection, query_embedding: list[float], pool_size: int
) -> list[RetrievedChunk]:
    """Stage 1: top `pool_size` candidates by cosine similarity, as RetrievedChunk stubs.

    rerank_score is seeded with vector_score and rank with 0; both are overwritten
    once stage 2 (reranking) and top_k truncation run.
    """
    rows = conn.execute(
        CANDIDATE_SQL, {"query_embedding": query_embedding, "limit": pool_size}
    ).fetchall()
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            title=title,
            source_url=source_url,
            source_family=source_family,
            section=section,
            vector_score=similarity,
            rerank_score=similarity,
            rank=0,
        )
        for chunk_id, doc_id, text, title, source_url, source_family, section, similarity in rows
    ]


async def retrieve(
    query: str,
    top_k: int | None = None,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    resolved_top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    pool_size = settings.RETRIEVAL_CANDIDATE_POOL_SIZE
    embedder = embedder or Embedder()
    if reranker is None:
        reranker = LLMReranker() if settings.RETRIEVAL_RERANK_ENABLED else NoopReranker()

    total_start = time.perf_counter()

    embed_start = time.perf_counter()
    query_embedding = await embed_query(query, embedder)
    embed_ms = (time.perf_counter() - embed_start) * 1000

    vector_start = time.perf_counter()
    conn = get_connection()
    try:
        init_schema(conn)
        candidates = fetch_candidates(conn, query_embedding, pool_size)
    finally:
        conn.close()
    vector_search_ms = (time.perf_counter() - vector_start) * 1000

    rerank_start = time.perf_counter()
    reranked = await reranker.rerank(query, candidates)
    rerank_ms = (time.perf_counter() - rerank_start) * 1000

    top = [
        chunk.model_copy(update={"rank": i})
        for i, chunk in enumerate(reranked[:resolved_top_k], start=1)
    ]

    total_ms = (time.perf_counter() - total_start) * 1000

    return RetrievalResult(
        chunks=top,
        timings=RetrievalTimings(
            embed_ms=embed_ms,
            vector_search_ms=vector_search_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
        ),
        reranker_used=type(reranker).__name__,
    )
