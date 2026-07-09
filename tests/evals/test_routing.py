"""Unit tests for routing accuracy scoring."""

from __future__ import annotations

import pytest

from app.evals.routing import RoutingObservation, route_correct, routing_accuracy


def test_route_correct():
    assert route_correct("retrieval", "retrieval") is True
    assert route_correct("retrieval", "both") is False


def test_routing_accuracy_overall_and_per_category():
    observations = [
        RoutingObservation("retrieval", "retrieval", "retrieval"),
        RoutingObservation("retrieval", "retrieval", "both"),  # wrong
        RoutingObservation("tools", "tools", "tools"),
        RoutingObservation("tools", "tools", "tools"),
    ]

    result = routing_accuracy(observations)

    assert result["overall"] == pytest.approx(3 / 4)
    assert result["by_category"]["retrieval"] == pytest.approx(1 / 2)
    assert result["by_category"]["tools"] == pytest.approx(1.0)
    assert result["counts"] == {"retrieval": 2, "tools": 2}


def test_routing_accuracy_empty():
    result = routing_accuracy([])

    assert result == {"overall": None, "by_category": {}, "counts": {}}
