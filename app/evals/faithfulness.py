"""LLM-as-judge faithfulness scoring for the Phase 7 eval harness.

Two-step judging: (1) extract the final answer's atomic factual claims, (2)
verify each claim against the ACTUAL sources the agent used in that run
(chunks + tool results from the trace - the same content the composer saw,
via app.agent.composer.render_sources_block). faithfulness = supported /
total claims; answers with zero claims (e.g. a correct decline-to-answer)
score None rather than 0 or 1, and should be reported/averaged separately.

Known limitation: the default judge is gpt-4o-mini, the same model that
composes the answers being judged. Same-model-family judging tends to be
lenient - a model is more likely to rate its own ambiguous phrasing as
"supported" than an independent, stronger judge would. JUDGE_MODEL
(app/config.py) lets a different judge model be swapped in without touching
this module's logic, for exactly that reason.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.agent.composer import render_sources_block
from app.agent.models import AgentTrace, TokenUsage, accumulate_usage
from app.agent.sources import build_sources
from app.config import settings

JUDGE_TIMEOUT_SECONDS = 30.0
MAX_JUDGE_ATTEMPTS = 2

logger = logging.getLogger(__name__)

ClaimVerdict = Literal["supported", "unsupported", "contradicted"]
_VALID_VERDICTS = {"supported", "unsupported", "contradicted"}

EXTRACT_CLAIMS_PROMPT = """Given the following answer text, list its atomic factual claims - \
each a single, standalone factual assertion someone could check independently. If the answer \
makes no factual claims at all (e.g. it declines to answer, or is pure smalltalk), return an \
empty list.

Answer:
\"\"\"{answer}\"\"\"

Respond with strict JSON only: {{"claims": ["claim 1", "claim 2", ...]}}"""

VERIFY_CLAIMS_PROMPT = """Given the following claims and the actual sources available to the \
system that made them, classify each claim as exactly one of:
- "supported": the sources directly support the claim
- "unsupported": the sources neither support nor contradict it (not addressed)
- "contradicted": the sources directly contradict it

Sources:
{sources_text}

Claims (in order):
{claims_list}

Respond with strict JSON only, with exactly {n} entries in the same order as the claims: \
{{"verdicts": ["supported"|"unsupported"|"contradicted", ...]}}"""


class ClaimJudgement(BaseModel):
    claim: str
    verdict: ClaimVerdict


class FaithfulnessResult(BaseModel):
    claims: list[ClaimJudgement] = []
    faithfulness: float | None = None
    supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    judge_error: bool = False
    error: str | None = None
    token_usage: TokenUsage = TokenUsage()


async def _call_judge_json(client: AsyncOpenAI, prompt: str, token_usage: TokenUsage) -> dict:
    response = await client.chat.completions.create(
        model=settings.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    accumulate_usage(token_usage, response.usage)
    content = response.choices[0].message.content or ""
    return json.loads(content)


async def _extract_claims(client: AsyncOpenAI, answer: str, token_usage: TokenUsage) -> list[str]:
    prompt = EXTRACT_CLAIMS_PROMPT.format(answer=answer)
    last_error: Exception | None = None
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            data = await _call_judge_json(client, prompt, token_usage)
            claims = data["claims"]
            if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
                raise ValueError(f"claims must be a list of strings, got {claims!r}")
            return claims
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning("Claim extraction malformed (attempt %d): %s", attempt, exc)
    raise ValueError(f"Claim extraction failed after {MAX_JUDGE_ATTEMPTS} attempts: {last_error}")


async def _verify_claims(
    client: AsyncOpenAI, claims: list[str], sources_text: str, token_usage: TokenUsage
) -> list[str]:
    claims_list = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    prompt = VERIFY_CLAIMS_PROMPT.format(sources_text=sources_text, claims_list=claims_list, n=len(claims))
    last_error: Exception | None = None
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            data = await _call_judge_json(client, prompt, token_usage)
            verdicts = data["verdicts"]
            if not isinstance(verdicts, list) or len(verdicts) != len(claims):
                raise ValueError(f"expected {len(claims)} verdicts, got {verdicts!r}")
            if not all(v in _VALID_VERDICTS for v in verdicts):
                raise ValueError(f"invalid verdict value(s) in {verdicts!r}")
            return verdicts
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning("Claim verification malformed (attempt %d): %s", attempt, exc)
    raise ValueError(f"Claim verification failed after {MAX_JUDGE_ATTEMPTS} attempts: {last_error}")


async def score_faithfulness(
    answer: str, trace: AgentTrace, *, client: AsyncOpenAI | None = None
) -> FaithfulnessResult:
    client = client or AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY, timeout=JUDGE_TIMEOUT_SECONDS, max_retries=0
    )
    token_usage = TokenUsage()

    try:
        claims = await _extract_claims(client, answer, token_usage)
    except ValueError as exc:
        return FaithfulnessResult(judge_error=True, error=str(exc), token_usage=token_usage)

    if not claims:
        return FaithfulnessResult(claims=[], faithfulness=None, token_usage=token_usage)

    sources_text = render_sources_block(build_sources(trace), trace)

    try:
        verdicts = await _verify_claims(client, claims, sources_text, token_usage)
    except ValueError as exc:
        return FaithfulnessResult(judge_error=True, error=str(exc), token_usage=token_usage)

    judgements = [ClaimJudgement(claim=c, verdict=v) for c, v in zip(claims, verdicts, strict=True)]
    supported = sum(1 for j in judgements if j.verdict == "supported")
    unsupported = sum(1 for j in judgements if j.verdict == "unsupported")
    contradicted = sum(1 for j in judgements if j.verdict == "contradicted")

    return FaithfulnessResult(
        claims=judgements,
        faithfulness=supported / len(judgements),
        supported_count=supported,
        unsupported_count=unsupported,
        contradicted_count=contradicted,
        token_usage=token_usage,
    )
