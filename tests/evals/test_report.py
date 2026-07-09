"""Unit tests for report building/rendering, including empty and partial results."""

from __future__ import annotations

from app.evals.faithfulness import FaithfulnessResult
from app.evals.key_facts import KeyFactsResult
from app.evals.report import build_summary, format_console_report, worst_cases, write_results_artifact
from app.evals.runner import CaseResult, RetrievalMetrics


def test_build_summary_empty_results():
    summary = build_summary([])

    assert summary["total_cases"] == 0
    assert summary["n_errored"] == 0
    assert summary["routing"]["overall"] is None
    assert summary["retrieval"]["n_scored"] == 0
    assert summary["retrieval"]["mean_precision_at_k"] is None
    assert summary["retrieval_rerank_off"] is None
    assert summary["faithfulness"]["mean_faithfulness"] is None
    assert summary["key_facts"]["mean_answer_completeness"] is None
    assert summary["estimated_cost_usd"] == 0.0


def test_format_console_report_empty_results_does_not_raise():
    summary = build_summary([])
    report = format_console_report([], summary)

    assert "EVAL REPORT" in report
    assert "Total cases: 0" in report


def test_build_summary_partial_results_mixes_errored_and_scored():
    errored = CaseResult(case_id="c1", query="q1", category="retrieval", expected_route="retrieval", error="boom")
    scored = CaseResult(
        case_id="c2",
        query="q2",
        category="retrieval",
        expected_route="retrieval",
        actual_route="retrieval",
        route_correct=True,
        retrieval=RetrievalMetrics(precision_at_k=0.4, recall_at_k=0.6, mrr=0.5, retrieved_doc_ids=["d1"]),
        faithfulness=FaithfulnessResult(faithfulness=0.8, supported_count=4, unsupported_count=1),
        key_facts_result=KeyFactsResult(answer_completeness=0.5),
    )

    summary = build_summary([errored, scored])

    assert summary["total_cases"] == 2
    assert summary["n_errored"] == 1
    assert summary["errored_case_ids"] == ["c1"]
    assert summary["routing"]["overall"] == 1.0  # only c2 has an actual_route
    assert summary["retrieval"]["n_scored"] == 1
    assert summary["retrieval"]["mean_precision_at_k"] == 0.4
    assert summary["faithfulness"]["mean_faithfulness"] == 0.8
    assert summary["key_facts"]["mean_answer_completeness"] == 0.5


def test_build_summary_rerank_off_only_present_when_used():
    scored = CaseResult(
        case_id="c1",
        query="q1",
        category="retrieval",
        expected_route="retrieval",
        actual_route="retrieval",
        retrieval=RetrievalMetrics(precision_at_k=0.4, recall_at_k=0.6, mrr=0.5, retrieved_doc_ids=["d1"]),
        retrieval_rerank_off=RetrievalMetrics(precision_at_k=0.2, recall_at_k=0.3, mrr=0.25, retrieved_doc_ids=["d2"]),
    )

    summary = build_summary([scored])

    assert summary["retrieval_rerank_off"] is not None
    assert summary["retrieval_rerank_off"]["mean_precision_at_k"] == 0.2

    report = format_console_report([scored], summary)
    assert "rerank OFF" in report


def test_faithfulness_zero_claims_counted_separately_from_scored():
    zero_claims = CaseResult(
        case_id="c1",
        query="q1",
        category="direct",
        expected_route="direct",
        actual_route="direct",
        faithfulness=FaithfulnessResult(claims=[], faithfulness=None),
    )

    summary = build_summary([zero_claims])

    assert summary["faithfulness"]["n_zero_claims"] == 1
    assert summary["faithfulness"]["n_scored"] == 0
    assert summary["faithfulness"]["mean_faithfulness"] is None


def test_worst_cases_sorts_errors_first_and_respects_n():
    results = [
        CaseResult(case_id=f"c{i}", query=f"q{i}", category="retrieval", expected_route="retrieval", route_correct=True)
        for i in range(10)
    ]
    results.append(
        CaseResult(case_id="errored", query="qe", category="retrieval", expected_route="retrieval", error="boom")
    )

    worst = worst_cases(results, n=5)

    assert len(worst) == 5
    assert worst[0].case_id == "errored"


def test_write_results_artifact_creates_unique_timestamped_file(tmp_path):
    results = [
        CaseResult(case_id="c1", query="q1", category="direct", expected_route="direct", actual_route="direct")
    ]

    path1 = write_results_artifact(results, rerank="on", retrieval_k=5, cases_path="eval/cases.json", output_dir=tmp_path)
    assert path1.exists()
    assert path1.name.endswith("_results.json")

    import json

    payload = json.loads(path1.read_text())
    assert payload["run_config"]["rerank"] == "on"
    assert payload["summary"]["total_cases"] == 1
    assert len(payload["cases"]) == 1
