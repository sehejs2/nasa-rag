"""Integration test: does real two-stage retrieval find the right document for grounded queries?

Requires the real corpus to already be embedded (see `make ingest`, Phase 2)
and a live OPENAI_API_KEY with quota - skips (rather than fails) if either is
unavailable, so `make test` stays green without live dependencies.

Cheap by design: 8 queries x (1 embedding call + 1 batched rerank call).
"""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from app.config import settings
from app.retrieval.retriever import retrieve

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "retrieval_sanity.json"
MIN_PASSING = 7
TOP_K = 5

pytestmark = pytest.mark.integration


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.DATABASE_URL, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except psycopg.OperationalError:
        return False


async def test_retrieval_sanity_set_finds_expected_document_in_top_k():
    if not settings.OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set; skipping retrieval sanity test.")
    if not _db_reachable():
        pytest.skip("Postgres is not reachable; skipping retrieval sanity test.")

    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    misses = []
    for case in cases:
        result = await retrieve(case["query"], top_k=TOP_K)
        doc_ids = [chunk.doc_id for chunk in result.chunks]
        if case["expected_doc_id"] not in doc_ids:
            misses.append(case["query"])

    passed = len(cases) - len(misses)
    assert passed >= MIN_PASSING, (
        f"Only {passed}/{len(cases)} queries found their expected doc in the top {TOP_K}. "
        f"Misses: {misses}"
    )
