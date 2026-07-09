"""Eval case runner: executes the full agent + composition pipeline per case and scores it.

Composition uses a non-streaming call (compose_answer_sync) rather than the
real /chat streaming path - we need the final text and token usage in one
shot for scoring, not incremental UI deltas, and the task explicitly allows
this. Individual case failures (network errors, judge_error) are caught and
recorded on that case's CaseResult.error rather than aborting the whole run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.agent.composer import build_composition_messages
from app.agent.loop import run_agent
from app.agent.models import AgentTrace, TokenUsage, accumulate_usage
from app.agent.sources import build_sources
from app.config import settings
from app.evals.faithfulness import FaithfulnessResult, score_faithfulness
from app.evals.key_facts import KeyFactsResult, score_key_facts
from app.evals.retrieval_metrics import DEFAULT_K, chunks_to_doc_ids, mrr, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)

COMPOSER_MODEL = "gpt-4o-mini"
OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0

DEFAULT_CASES_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "cases.json"


class EvalCase(BaseModel):
    id: str
    query: str
    category: str
    expected_route: str
    relevant_doc_ids: list[str] = []
    key_facts: list[str] = []
    notes: str = ""


class RetrievalMetrics(BaseModel):
    precision_at_k: float | None
    recall_at_k: float | None
    mrr: float | None
    retrieved_doc_ids: list[str]


class CaseResult(BaseModel):
    case_id: str
    query: str
    category: str
    expected_route: str
    actual_route: str | None = None
    route_correct: bool | None = None
    answer: str = ""
    retrieval: RetrievalMetrics | None = None
    # Only populated under --rerank both: the same retrieval metrics re-run
    # with reranking disabled, for a side-by-side comparison.
    retrieval_rerank_off: RetrievalMetrics | None = None
    faithfulness: FaithfulnessResult | None = None
    key_facts_result: KeyFactsResult | None = None
    token_usage: TokenUsage = TokenUsage()
    latency_ms: float = 0.0
    error: str | None = None


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data]


async def compose_answer_sync(
    query: str, trace: AgentTrace, sources: list, *, client: AsyncOpenAI | None = None
) -> tuple[str, TokenUsage]:
    """Non-streaming variant of app.agent.composer.compose_answer, for eval scoring."""
    client = client or AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY, timeout=OPENAI_REQUEST_TIMEOUT_SECONDS, max_retries=0
    )
    messages = build_composition_messages(query, trace, sources)
    response = await client.chat.completions.create(model=COMPOSER_MODEL, messages=messages)
    usage = TokenUsage()
    accumulate_usage(usage, response.usage)
    return response.choices[0].message.content or "", usage


def _score_retrieval(case: EvalCase, trace: AgentTrace, k: int) -> RetrievalMetrics | None:
    if not case.relevant_doc_ids:
        return None
    retrieved_doc_ids = chunks_to_doc_ids(trace.retrieved_chunks)
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved_doc_ids, case.relevant_doc_ids, k=k),
        recall_at_k=recall_at_k(retrieved_doc_ids, case.relevant_doc_ids, k=k),
        mrr=mrr(retrieved_doc_ids, case.relevant_doc_ids, k=k),
        retrieved_doc_ids=retrieved_doc_ids,
    )


async def run_single_case(
    case: EvalCase, *, semaphore: asyncio.Semaphore, retrieval_k: int = DEFAULT_K
) -> CaseResult:
    start = time.perf_counter()
    async with semaphore:
        try:
            trace = await run_agent(case.query)
            sources = build_sources(trace)
            answer, compose_usage = await compose_answer_sync(case.query, trace, sources)

            faithfulness_result = await score_faithfulness(answer, trace)
            key_facts_result = await score_key_facts(answer, case.key_facts)

            combined_usage = TokenUsage()
            accumulate_usage(combined_usage, trace.token_usage)
            accumulate_usage(combined_usage, compose_usage)
            accumulate_usage(combined_usage, faithfulness_result.token_usage)
            accumulate_usage(combined_usage, key_facts_result.token_usage)

            return CaseResult(
                case_id=case.id,
                query=case.query,
                category=case.category,
                expected_route=case.expected_route,
                actual_route=trace.route,
                route_correct=(trace.route == case.expected_route),
                answer=answer,
                retrieval=_score_retrieval(case, trace, retrieval_k),
                faithfulness=faithfulness_result,
                key_facts_result=key_facts_result,
                token_usage=combined_usage,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 - a case failure must not abort the run
            logger.exception("Case %s failed", case.id)
            return CaseResult(
                case_id=case.id,
                query=case.query,
                category=case.category,
                expected_route=case.expected_route,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=str(exc),
            )


async def _run_retrieval_only(
    case: EvalCase, *, semaphore: asyncio.Semaphore, retrieval_k: int
) -> RetrievalMetrics | None:
    async with semaphore:
        try:
            trace = await run_agent(case.query)
            return _score_retrieval(case, trace, retrieval_k)
        except Exception:  # noqa: BLE001 - rerank-off comparison pass must not abort the run
            logger.exception("Rerank-off retrieval pass failed for %s", case.id)
            return None


async def run_eval(
    cases: list[EvalCase],
    *,
    rerank: str = "on",
    concurrency: int = 4,
    retrieval_k: int = DEFAULT_K,
    on_progress: Callable[[int, int, CaseResult, str], None] | None = None,
) -> list[CaseResult]:
    """Run every case through the full pipeline and score it.

    rerank: "on" runs with RETRIEVAL_RERANK_ENABLED=True, "off" with False,
    "both" runs the full pipeline once with reranking on, then re-runs just
    the retrieval-scored cases (non-empty relevant_doc_ids) a second time
    with reranking off, filling in retrieval_rerank_off for comparison.

    on_progress(completed, total, result, phase) fires per completed case;
    phase is "primary" for the full pipeline pass and "rerank_off" for the
    (--rerank both only) retrieval-only comparison pass.
    """
    if rerank not in ("on", "off", "both"):
        raise ValueError(f"rerank must be 'on', 'off', or 'both', got {rerank!r}")

    semaphore = asyncio.Semaphore(concurrency)
    total = len(cases)
    results: list[CaseResult | None] = [None] * total
    completed = 0

    original_rerank_setting = settings.RETRIEVAL_RERANK_ENABLED
    try:
        settings.RETRIEVAL_RERANK_ENABLED = rerank != "off"

        async def run_and_report(index: int, case: EvalCase) -> None:
            nonlocal completed
            result = await run_single_case(case, semaphore=semaphore, retrieval_k=retrieval_k)
            completed += 1
            if on_progress:
                on_progress(completed, total, result, "primary")
            results[index] = result

        await asyncio.gather(*(run_and_report(i, c) for i, c in enumerate(cases)))

        if rerank == "both":
            settings.RETRIEVAL_RERANK_ENABLED = False
            retrieval_cases = [(i, c) for i, c in enumerate(cases) if c.relevant_doc_ids]
            second_pass_completed = 0
            second_pass_total = len(retrieval_cases)

            async def run_second_pass(index: int, case: EvalCase) -> None:
                nonlocal second_pass_completed
                metrics = await _run_retrieval_only(case, semaphore=semaphore, retrieval_k=retrieval_k)
                results[index].retrieval_rerank_off = metrics  # type: ignore[union-attr]
                second_pass_completed += 1
                if on_progress:
                    on_progress(second_pass_completed, second_pass_total, results[index], "rerank_off")  # type: ignore[arg-type]

            await asyncio.gather(*(run_second_pass(i, c) for i, c in retrieval_cases))
    finally:
        settings.RETRIEVAL_RERANK_ENABLED = original_rerank_setting

    return results  # type: ignore[return-value]
