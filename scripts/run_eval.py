"""Eval harness CLI. Usage: make eval [args="--limit 3 --rerank both"]

Runs every case in eval/cases.json through the full agent + composition
pipeline, scores routing accuracy, doc-level retrieval metrics, LLM-judged
faithfulness, and key-facts completeness, then prints a console report and
writes a full-detail JSON artifact to eval/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from app.evals.report import build_summary, format_console_report, write_results_artifact
from app.evals.runner import DEFAULT_CASES_PATH, CaseResult, load_cases, run_eval


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NASA RAG eval harness.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to the eval cases JSON file.")
    parser.add_argument(
        "--rerank",
        choices=["on", "off", "both"],
        default="on",
        help="Reranker setting for retrieval-scored cases. 'both' runs those cases twice for a side-by-side comparison.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases (for smoke runs).")
    parser.add_argument("--concurrency", type=int, default=4, help="Max cases running concurrently.")
    return parser.parse_args(argv)


def _on_progress(completed: int, total: int, result: CaseResult, phase: str) -> None:
    status = "ERROR" if result.error else "ok"
    label = "rerank-off retrieval pass" if phase == "rerank_off" else result.category
    print(
        f"[{phase}: {completed}/{total}] {result.case_id} ({label}) -> {status} "
        f"({result.latency_ms:.0f}ms)",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    print(
        f"Running {len(cases)} case(s) from {args.cases} "
        f"(rerank={args.rerank}, concurrency={args.concurrency})...",
        file=sys.stderr,
    )

    start = time.perf_counter()
    results = asyncio.run(
        run_eval(cases, rerank=args.rerank, concurrency=args.concurrency, on_progress=_on_progress)
    )
    wall_time_seconds = time.perf_counter() - start

    summary = build_summary(results)
    print()
    print(format_console_report(results, summary))
    print()
    print(f"Wall time: {wall_time_seconds:.1f}s")

    output_path = write_results_artifact(results, rerank=args.rerank, retrieval_k=5, cases_path=args.cases)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
