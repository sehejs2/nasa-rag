"""Routing accuracy test: run each labeled case through the real agent loop.

Marked both `integration` (needs the local DB) and `llm` (makes real gpt-4o-mini
calls) - skipped if either OPENAI_API_KEY or Postgres is unavailable. Live NASA
tool executions are stubbed (see conftest.py) since we're testing routing
decisions, not NASA API uptime; search_documents hits the real local DB and
makes real embedding/rerank calls.

Failures are reported, never papered over: if a case's route doesn't exactly
match (e.g. model chose "both" where "retrieval" was expected), that's a
reported failure, not a softened assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from app.agent.loop import run_agent
from app.config import settings

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "routing_cases.json"

pytestmark = [pytest.mark.integration, pytest.mark.llm]


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.OperationalError:
        return False


@pytest.mark.timeout(300)
async def test_routing_accuracy_against_labeled_cases():
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set; skipping routing accuracy test.")
    if not _db_reachable():
        pytest.skip("Postgres is not reachable; skipping routing accuracy test.")

    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        trace = await run_agent(case["query"])
        called_tools = {tc.tool_name for tc in trace.tool_calls}
        route_ok = trace.route == case["expected_route"]
        tools_ok = set(case["expected_tools"]).issubset(called_tools)
        ok = route_ok and tools_ok
        rows.append(
            {
                "query": case["query"],
                "expected_route": case["expected_route"],
                "actual_route": trace.route,
                "expected_tools": case["expected_tools"],
                "called_tools": sorted(called_tools),
                "ok": ok,
            }
        )

    passed = sum(row["ok"] for row in rows)

    print(f"\nRouting accuracy: {passed}/{len(cases)}\n")
    print(f"{'RESULT':<6}{'expected':<12}{'actual':<12}query")
    for row in rows:
        status = "PASS" if row["ok"] else "FAIL"
        print(f"{status:<6}{row['expected_route']:<12}{row['actual_route']:<12}{row['query'][:60]}")
        if not row["ok"]:
            print(f"       expected_tools={row['expected_tools']} called_tools={row['called_tools']}")

    failures = [row for row in rows if not row["ok"]]
    assert not failures, f"{len(failures)}/{len(cases)} routing case(s) failed: {[row['query'] for row in failures]}"
