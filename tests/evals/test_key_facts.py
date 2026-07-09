"""Unit tests for key-facts completeness scoring. All OpenAI calls are faked."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.evals.key_facts import score_key_facts


class _FakeCompletionsAPI:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[str] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs["messages"][0]["content"])
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses: list):
        self.completions = _FakeCompletionsAPI(responses)


class _FakeClient:
    def __init__(self, responses: list):
        self.chat = _FakeChat(responses)


def _response(content: str, usage=(10, 5, 15)) -> SimpleNamespace:
    prompt, completion, total = usage
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total),
    )


async def test_zero_key_facts_scores_none_with_no_api_call():
    client = _FakeClient([])

    result = await score_key_facts("Any answer.", [], client=client)

    assert result.answer_completeness is None
    assert result.judgements == []
    assert len(client.chat.completions.calls) == 0


async def test_well_formed_partial_completeness():
    responses = [_response(json.dumps({"present": [True, False, True]}))]
    client = _FakeClient(responses)

    result = await score_key_facts("An answer.", ["fact one", "fact two", "fact three"], client=client)

    assert result.judge_error is False
    assert result.answer_completeness == 2 / 3
    assert [j.present for j in result.judgements] == [True, False, True]
    assert result.token_usage.total_tokens == 15


async def test_malformed_then_retry_succeeds():
    responses = [_response("not json"), _response(json.dumps({"present": [True]}))]
    client = _FakeClient(responses)

    result = await score_key_facts("An answer.", ["fact one"], client=client)

    assert result.judge_error is False
    assert result.answer_completeness == 1.0


async def test_wrong_length_is_judge_error_after_retries():
    responses = [
        _response(json.dumps({"present": [True, False]})),  # 2, need 3
        _response(json.dumps({"present": [True]})),  # still wrong
    ]
    client = _FakeClient(responses)

    result = await score_key_facts("An answer.", ["a", "b", "c"], client=client)

    assert result.judge_error is True
    assert result.answer_completeness is None
    assert result.error is not None


async def test_non_boolean_entries_treated_as_malformed():
    responses = [
        _response(json.dumps({"present": ["yes"]})),
        _response(json.dumps({"present": [True]})),
    ]
    client = _FakeClient(responses)

    result = await score_key_facts("An answer.", ["a"], client=client)

    assert result.judge_error is False
    assert result.answer_completeness == 1.0
