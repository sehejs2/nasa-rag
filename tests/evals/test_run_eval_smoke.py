"""End-to-end smoke test for the eval harness CLI: real DB, real LLM calls
(only 3 cases, kept cheap). Skips if OPENAI_API_KEY or the DB is unavailable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from app.config import settings

pytestmark = [pytest.mark.integration, pytest.mark.llm]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.OperationalError:
        return False


@pytest.mark.timeout(180)
def test_run_eval_limit_3_end_to_end():
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set; skipping eval harness smoke test.")
    if not _db_reachable():
        pytest.skip("Postgres is not reachable; skipping eval harness smoke test.")

    existing_files = set(RESULTS_DIR.glob("*_results.json")) if RESULTS_DIR.exists() else set()

    result = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--limit", "3"],
        capture_output=True,
        text=True,
        timeout=170,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "EVAL REPORT" in result.stdout

    new_files = (set(RESULTS_DIR.glob("*_results.json")) if RESULTS_DIR.exists() else set()) - existing_files
    assert len(new_files) == 1, f"expected exactly one new results file, got {new_files}"

    payload = json.loads(new_files.pop().read_text(encoding="utf-8"))
    assert payload["summary"]["total_cases"] == 3
    assert len(payload["cases"]) == 3
