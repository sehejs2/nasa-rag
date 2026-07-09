"""Routing accuracy scoring: does the agent's actual route match expected_route?

Pure functions, no I/O - mirrors app/evals/retrieval_metrics.py.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple


class RoutingObservation(NamedTuple):
    category: str
    expected_route: str
    actual_route: str


def route_correct(expected_route: str, actual_route: str) -> bool:
    return expected_route == actual_route


def routing_accuracy(observations: list[RoutingObservation]) -> dict:
    """Overall + per-category routing accuracy.

    Returns {"overall": float | None, "by_category": {category: float},
    "counts": {category: int}}. overall/by_category entries are None only
    when there are zero observations to average (empty input).
    """
    if not observations:
        return {"overall": None, "by_category": {}, "counts": {}}

    correct_by_category: dict[str, int] = defaultdict(int)
    total_by_category: dict[str, int] = defaultdict(int)

    for obs in observations:
        is_correct = route_correct(obs.expected_route, obs.actual_route)
        correct_by_category[obs.category] += int(is_correct)
        total_by_category[obs.category] += 1

    total_correct = sum(correct_by_category.values())
    overall = total_correct / len(observations)
    by_category = {
        category: correct_by_category[category] / total_by_category[category]
        for category in total_by_category
    }
    counts = dict(total_by_category)

    return {"overall": overall, "by_category": by_category, "counts": counts}
