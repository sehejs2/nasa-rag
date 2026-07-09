"""Unit tests for doc-level retrieval metrics against hand-computed values."""

from __future__ import annotations

import pytest

from app.evals.retrieval_metrics import chunks_to_doc_ids, mrr, precision_at_k, recall_at_k


def test_precision_at_k_hand_computed():
    retrieved = ["d1", "d2", "d3", "d4", "d5", "d6"]
    relevant = {"d2", "d5", "d9"}  # d9 is never retrieved

    # top-5 = [d1..d5]; hits = {d2, d5} = 2; precision divides by fixed k=5.
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)


def test_recall_at_k_hand_computed():
    retrieved = ["d1", "d2", "d3", "d4", "d5", "d6"]
    relevant = {"d2", "d5", "d9"}

    # 2 of 3 relevant docs found within top-5.
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_mrr_hand_computed():
    retrieved = ["d1", "d2", "d3", "d4", "d5", "d6"]
    relevant = {"d2", "d5", "d9"}

    # first relevant hit (d2) is at position 2 -> reciprocal rank 1/2.
    assert mrr(retrieved, relevant, k=5) == pytest.approx(0.5)


def test_perfect_single_hit():
    assert precision_at_k(["d1"], {"d1"}, k=5) == pytest.approx(1 / 5)
    assert recall_at_k(["d1"], {"d1"}, k=5) == pytest.approx(1.0)
    assert mrr(["d1"], {"d1"}, k=5) == pytest.approx(1.0)


def test_no_retrieved_docs():
    assert precision_at_k([], {"d1"}, k=5) == 0.0
    assert recall_at_k([], {"d1"}, k=5) == 0.0
    assert mrr([], {"d1"}, k=5) == 0.0


def test_zero_relevant_docs():
    retrieved = ["d1", "d2", "d3"]

    assert precision_at_k(retrieved, [], k=5) == 0.0
    assert recall_at_k(retrieved, [], k=5) is None  # undefined: nothing was relevant
    assert mrr(retrieved, [], k=5) == 0.0


def test_no_retrieved_and_no_relevant():
    assert precision_at_k([], [], k=5) == 0.0
    assert recall_at_k([], [], k=5) is None
    assert mrr([], [], k=5) == 0.0


def test_mrr_beyond_k_not_counted():
    # relevant doc is at position 6, but k=5 - should not be found.
    retrieved = ["d1", "d2", "d3", "d4", "d5", "d6"]
    assert mrr(retrieved, {"d6"}, k=5) == 0.0
    assert mrr(retrieved, {"d6"}, k=6) == pytest.approx(1 / 6)


@pytest.mark.parametrize("fn", [precision_at_k, recall_at_k, mrr])
def test_invalid_k_raises(fn):
    with pytest.raises(ValueError):
        fn(["d1"], {"d1"}, k=0)


def test_chunks_to_doc_ids_dedup_preserves_rank_order():
    chunks = [
        {"doc_id": "d1"},
        {"doc_id": "d2"},
        {"doc_id": "d1"},  # duplicate, should be dropped (first occurrence wins)
        {"doc_id": "d3"},
    ]

    assert chunks_to_doc_ids(chunks) == ["d1", "d2", "d3"]


def test_chunks_to_doc_ids_supports_attribute_access():
    from types import SimpleNamespace

    chunks = [SimpleNamespace(doc_id="d1"), SimpleNamespace(doc_id="d2")]

    assert chunks_to_doc_ids(chunks) == ["d1", "d2"]


def test_chunks_to_doc_ids_empty():
    assert chunks_to_doc_ids([]) == []
