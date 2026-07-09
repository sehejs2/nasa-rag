"""Key-facts completeness scoring: does the answer contain each labeled key fact?

Semantic containment, judged by LLM (paraphrasing counts; exact wording does
not need to match). answer_completeness = facts_present / facts_total. Cases
with zero key_facts score None (nothing to check). Same JUDGE_MODEL and
malformed-JSON-retry conventions as app/evals/faithfulness.py.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.agent.models import TokenUsage, accumulate_usage
from app.config import settings

JUDGE_TIMEOUT_SECONDS = 30.0
MAX_JUDGE_ATTEMPTS = 2

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Given the following answer text and a list of key facts it should ideally \
contain, judge for each key fact whether the answer contains it. Judge by meaning, not exact \
wording - paraphrasing counts as present.

Answer:
\"\"\"{answer}\"\"\"

Key facts (in order):
{facts_list}

Respond with strict JSON only, with exactly {n} boolean entries in the same order as the key \
facts: {{"present": [true|false, ...]}}"""


class KeyFactJudgement(BaseModel):
    key_fact: str
    present: bool


class KeyFactsResult(BaseModel):
    judgements: list[KeyFactJudgement] = []
    answer_completeness: float | None = None
    judge_error: bool = False
    error: str | None = None
    token_usage: TokenUsage = TokenUsage()


async def score_key_facts(
    answer: str, key_facts: list[str], *, client: AsyncOpenAI | None = None
) -> KeyFactsResult:
    if not key_facts:
        return KeyFactsResult(judgements=[], answer_completeness=None)

    client = client or AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY, timeout=JUDGE_TIMEOUT_SECONDS, max_retries=0
    )
    facts_list = "\n".join(f"{i}. {fact}" for i, fact in enumerate(key_facts, start=1))
    prompt = PROMPT_TEMPLATE.format(answer=answer, facts_list=facts_list, n=len(key_facts))

    token_usage = TokenUsage()
    last_error: Exception | None = None
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            response = await client.chat.completions.create(
                model=settings.JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            accumulate_usage(token_usage, response.usage)
            content = response.choices[0].message.content or ""
            data = json.loads(content)
            present = data["present"]
            if not isinstance(present, list) or len(present) != len(key_facts):
                raise ValueError(f"expected {len(key_facts)} booleans, got {present!r}")
            if not all(isinstance(p, bool) for p in present):
                raise ValueError(f"non-boolean entries in {present!r}")

            judgements = [
                KeyFactJudgement(key_fact=fact, present=is_present)
                for fact, is_present in zip(key_facts, present, strict=True)
            ]
            return KeyFactsResult(
                judgements=judgements,
                answer_completeness=sum(present) / len(present),
                token_usage=token_usage,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning("Key-facts judging malformed (attempt %d): %s", attempt, exc)

    return KeyFactsResult(
        judge_error=True,
        error=f"Key-facts judging failed after {MAX_JUDGE_ATTEMPTS} attempts: {last_error}",
        token_usage=token_usage,
    )
