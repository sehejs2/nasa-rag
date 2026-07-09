"""Doc-level retrieval metrics. Pure functions, no I/O.

Retrieval happens at chunk granularity, but eval cases label relevance at the
document level (relevant_doc_ids), since that's what a human reviewer can
verify by reading a doc. `chunks_to_doc_ids` bridges the two: it maps ranked
chunks to a deduplicated, rank-ordered list of doc_ids.

precision@k divides by the fixed k (the conventional IR definition) - an
under-full top-k (fewer than k chunks retrieved) still divides by k, since an
empty slot is a missed opportunity, not a null result. recall@k and mrr divide
by/search within actual content, so they need explicit handling for the
zero-relevant-docs edge case (recall is undefined with no relevant docs and
returns None; mrr and precision are well-defined with a 0.0 value in every
edge case since they never divide by len(relevant)).
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_K = 5


def chunks_to_doc_ids(chunks: Iterable) -> list[str]:
    """Ranked chunks (objects or dicts with .doc_id/["doc_id"]) -> deduplicated,
    rank-ordered doc_ids (first occurrence wins).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        doc_id = chunk["doc_id"] if isinstance(chunk, dict) else chunk.doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def precision_at_k(
    retrieved_doc_ids: list[str], relevant_doc_ids: Iterable[str], k: int = DEFAULT_K
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(
    retrieved_doc_ids: list[str], relevant_doc_ids: Iterable[str], k: int = DEFAULT_K
) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_doc_ids)
    if not relevant:
        return None  # undefined: nothing was relevant, so nothing could be recalled
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved_doc_ids: list[str], relevant_doc_ids: Iterable[str], k: int = DEFAULT_K) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_doc_ids)
    for position, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0
