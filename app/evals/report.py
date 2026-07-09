"""Console report + JSON artifact rendering for eval runs. Pure/IO-light: the
only I/O is writing the results file and (best-effort) reading the git commit hash.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.evals.runner import CaseResult
from app.evals.routing import RoutingObservation, routing_accuracy

# Approximate gpt-4o-mini pricing (USD per 1M tokens) - a rough cost estimate
# only, not billing-accurate. Update if OpenAI's published pricing changes.
COST_PER_1M_INPUT_TOKENS_USD = 0.15
COST_PER_1M_OUTPUT_TOKENS_USD = 0.60

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval" / "results"
WORST_N = 5


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * COST_PER_1M_INPUT_TOKENS_USD
        + completion_tokens / 1_000_000 * COST_PER_1M_OUTPUT_TOKENS_USD
    )


def build_summary(results: list[CaseResult]) -> dict:
    total = len(results)
    errored = [r for r in results if r.error]
    scored = [r for r in results if not r.error]

    routing_obs = [
        RoutingObservation(category=r.category, expected_route=r.expected_route, actual_route=r.actual_route)
        for r in scored
        if r.actual_route is not None
    ]
    routing = routing_accuracy(routing_obs)

    retrieval_cases = [r for r in scored if r.retrieval is not None]
    retrieval_summary = {
        "n_scored": len(retrieval_cases),
        "mean_precision_at_k": _mean([r.retrieval.precision_at_k for r in retrieval_cases]),
        "mean_recall_at_k": _mean([r.retrieval.recall_at_k for r in retrieval_cases]),
        "mean_mrr": _mean([r.retrieval.mrr for r in retrieval_cases]),
    }

    rerank_off_cases = [r for r in scored if r.retrieval_rerank_off is not None]
    retrieval_rerank_off_summary = None
    if rerank_off_cases:
        retrieval_rerank_off_summary = {
            "n_scored": len(rerank_off_cases),
            "mean_precision_at_k": _mean([r.retrieval_rerank_off.precision_at_k for r in rerank_off_cases]),
            "mean_recall_at_k": _mean([r.retrieval_rerank_off.recall_at_k for r in rerank_off_cases]),
            "mean_mrr": _mean([r.retrieval_rerank_off.mrr for r in rerank_off_cases]),
        }

    faithfulness_cases = [r for r in scored if r.faithfulness is not None]
    faithfulness_judge_errors = [r for r in faithfulness_cases if r.faithfulness.judge_error]
    faithfulness_scored = [r for r in faithfulness_cases if r.faithfulness.faithfulness is not None]
    faithfulness_zero_claims = [
        r
        for r in faithfulness_cases
        if not r.faithfulness.judge_error and r.faithfulness.faithfulness is None
    ]
    faithfulness_summary = {
        "n_scored": len(faithfulness_scored),
        "n_zero_claims": len(faithfulness_zero_claims),
        "n_judge_errors": len(faithfulness_judge_errors),
        "mean_faithfulness": _mean([r.faithfulness.faithfulness for r in faithfulness_scored]),
        "supported_total": sum(r.faithfulness.supported_count for r in faithfulness_cases),
        "unsupported_total": sum(r.faithfulness.unsupported_count for r in faithfulness_cases),
        "contradicted_total": sum(r.faithfulness.contradicted_count for r in faithfulness_cases),
    }

    key_facts_cases = [r for r in scored if r.key_facts_result is not None]
    key_facts_judge_errors = [r for r in key_facts_cases if r.key_facts_result.judge_error]
    key_facts_scored = [r for r in key_facts_cases if r.key_facts_result.answer_completeness is not None]
    key_facts_summary = {
        "n_scored": len(key_facts_scored),
        "n_judge_errors": len(key_facts_judge_errors),
        "mean_answer_completeness": _mean([r.key_facts_result.answer_completeness for r in key_facts_scored]),
    }

    token_usage = {
        "prompt_tokens": sum(r.token_usage.prompt_tokens for r in results),
        "completion_tokens": sum(r.token_usage.completion_tokens for r in results),
        "total_tokens": sum(r.token_usage.total_tokens for r in results),
    }

    return {
        "total_cases": total,
        "n_errored": len(errored),
        "errored_case_ids": [r.case_id for r in errored],
        "routing": routing,
        "retrieval": retrieval_summary,
        "retrieval_rerank_off": retrieval_rerank_off_summary,
        "faithfulness": faithfulness_summary,
        "key_facts": key_facts_summary,
        "token_usage": token_usage,
        "estimated_cost_usd": estimate_cost_usd(token_usage["prompt_tokens"], token_usage["completion_tokens"]),
    }


def _case_score(result: CaseResult) -> float:
    """A single [0,1] composite used only to rank the "worst cases" section.
    Errored cases sort first (score -1); cases with nothing scored are neutral (0.5).
    """
    if result.error:
        return -1.0
    components: list[float] = []
    if result.route_correct is not None:
        components.append(1.0 if result.route_correct else 0.0)
    if result.faithfulness and result.faithfulness.faithfulness is not None:
        components.append(result.faithfulness.faithfulness)
    if result.key_facts_result and result.key_facts_result.answer_completeness is not None:
        components.append(result.key_facts_result.answer_completeness)
    if result.retrieval:
        retrieval_components = [
            v
            for v in (result.retrieval.precision_at_k, result.retrieval.recall_at_k, result.retrieval.mrr)
            if v is not None
        ]
        if retrieval_components:
            components.append(statistics.fmean(retrieval_components))
    return statistics.fmean(components) if components else 0.5


def _what_failed(result: CaseResult) -> str:
    if result.error:
        return f"error: {result.error[:100]}"
    reasons = []
    if result.route_correct is False:
        reasons.append(f"route {result.actual_route!r} != expected {result.expected_route!r}")
    faithfulness = result.faithfulness
    if faithfulness and faithfulness.faithfulness is not None and faithfulness.faithfulness < 1.0:
        reasons.append(f"faithfulness {faithfulness.faithfulness:.2f}")
    key_facts = result.key_facts_result
    if key_facts and key_facts.answer_completeness is not None and key_facts.answer_completeness < 1.0:
        reasons.append(f"completeness {key_facts.answer_completeness:.2f}")
    if result.retrieval and (result.retrieval.precision_at_k or 0) < 0.2:
        reasons.append(f"precision@k {result.retrieval.precision_at_k:.2f}")
    return "; ".join(reasons) if reasons else "no issues found"


def worst_cases(results: list[CaseResult], n: int = WORST_N) -> list[CaseResult]:
    return sorted(results, key=_case_score)[:n]


def format_console_report(results: list[CaseResult], summary: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("EVAL REPORT")
    lines.append("=" * 70)
    lines.append(f"Total cases: {summary['total_cases']}  (errored: {summary['n_errored']})")
    lines.append("")

    routing = summary["routing"]
    if routing["overall"] is not None:
        lines.append(f"Routing accuracy: {routing['overall']:.1%} overall")
        for category in sorted(routing["by_category"]):
            lines.append(
                f"  {category:<14} {routing['by_category'][category]:.1%}  (n={routing['counts'][category]})"
            )
    else:
        lines.append("Routing accuracy: n/a (no scored cases)")
    lines.append("")

    retrieval = summary["retrieval"]
    lines.append(f"Retrieval (n={retrieval['n_scored']}):")
    lines.append(f"  precision@k: {_fmt(retrieval['mean_precision_at_k'])}")
    lines.append(f"  recall@k:    {_fmt(retrieval['mean_recall_at_k'])}")
    lines.append(f"  MRR:         {_fmt(retrieval['mean_mrr'])}")

    if summary["retrieval_rerank_off"]:
        rerank_off = summary["retrieval_rerank_off"]
        lines.append("")
        lines.append(f"Retrieval, rerank OFF (n={rerank_off['n_scored']}) - compare against rerank ON above:")
        lines.append(f"  precision@k: {_fmt(rerank_off['mean_precision_at_k'])}")
        lines.append(f"  recall@k:    {_fmt(rerank_off['mean_recall_at_k'])}")
        lines.append(f"  MRR:         {_fmt(rerank_off['mean_mrr'])}")
    lines.append("")

    faithfulness = summary["faithfulness"]
    lines.append(
        f"Faithfulness: mean={_fmt(faithfulness['mean_faithfulness'])} "
        f"(n={faithfulness['n_scored']}, zero_claims={faithfulness['n_zero_claims']}, "
        f"judge_errors={faithfulness['n_judge_errors']})"
    )
    lines.append(
        f"  claims: supported={faithfulness['supported_total']} "
        f"unsupported={faithfulness['unsupported_total']} "
        f"contradicted={faithfulness['contradicted_total']}"
    )
    lines.append("")

    key_facts = summary["key_facts"]
    lines.append(
        f"Answer completeness: mean={_fmt(key_facts['mean_answer_completeness'])} "
        f"(n={key_facts['n_scored']}, judge_errors={key_facts['n_judge_errors']})"
    )
    lines.append("")

    tokens = summary["token_usage"]
    lines.append(
        f"Tokens: {tokens['total_tokens']:,} total "
        f"({tokens['prompt_tokens']:,} prompt / {tokens['completion_tokens']:,} completion)"
    )
    lines.append(f"Estimated cost: ${summary['estimated_cost_usd']:.4f} (gpt-4o-mini rates, approximate)")

    if summary["errored_case_ids"]:
        lines.append("")
        lines.append(f"Errored cases ({len(summary['errored_case_ids'])}): {', '.join(summary['errored_case_ids'])}")

    lines.append("")
    lines.append("-" * 70)
    lines.append(f"Worst {WORST_N} cases:")
    lines.append("-" * 70)
    for result in worst_cases(results):
        lines.append(f"  [{_case_score(result):.2f}] {result.case_id}: {result.query[:70]}")
        lines.append(f"      {_what_failed(result)}")

    return "\n".join(lines)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - the commit hash is best-effort metadata
        return None


def write_results_artifact(
    results: list[CaseResult],
    *,
    rerank: str,
    retrieval_k: int,
    cases_path: str,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}_results.json"

    payload = {
        "run_config": {
            "git_commit": _git_commit_hash(),
            "rerank": rerank,
            "retrieval_k": retrieval_k,
            "cases_path": str(cases_path),
            "agent_model": "gpt-4o-mini",
            "judge_model": settings.JUDGE_MODEL,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "summary": build_summary(results),
        "cases": [json.loads(r.model_dump_json()) for r in results],
    }

    output_path.write_text(json.dumps(payload, indent=2))
    return output_path
