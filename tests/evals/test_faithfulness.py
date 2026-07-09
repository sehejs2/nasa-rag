"""Unit tests for LLM-as-judge faithfulness scoring. All OpenAI calls are faked."""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.agent.models import AgentTrace, TokenUsage
from app.evals.faithfulness import score_faithfulness
from app.retrieval.models import RetrievedChunk


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


def _trace(**overrides) -> AgentTrace:
    defaults = dict(
        query="q",
        route="retrieval",
        tool_calls=[],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="c1",
                doc_id="doc-1",
                text="The sky is blue.",
                title="T",
                source_url="https://example.com",
                source_family="nasa_general",
                section=None,
                vector_score=0.9,
                rerank_score=8.0,
                rank=1,
            )
        ],
        draft_answer="",
        iterations_used=1,
        stopped_reason="model_finished",
        total_latency_ms=1.0,
        token_usage=TokenUsage(),
    )
    defaults.update(overrides)
    return AgentTrace(**defaults)


async def test_well_formed_extraction_and_verification():
    responses = [
        _response(json.dumps({"claims": ["The sky is blue.", "The grass is purple."]})),
        _response(json.dumps({"verdicts": ["supported", "contradicted"]})),
    ]
    client = _FakeClient(responses)

    result = await score_faithfulness("The sky is blue. The grass is purple.", _trace(), client=client)

    assert result.judge_error is False
    assert result.faithfulness == 0.5
    assert result.supported_count == 1
    assert result.contradicted_count == 1
    assert result.unsupported_count == 0
    assert result.token_usage.total_tokens == 30  # two calls of 15 each


async def test_zero_claims_scores_none():
    responses = [_response(json.dumps({"claims": []}))]
    client = _FakeClient(responses)

    result = await score_faithfulness("I don't have grounds to answer that.", _trace(), client=client)

    assert result.faithfulness is None
    assert result.claims == []
    assert result.judge_error is False
    assert len(client.chat.completions.calls) == 1  # verification never called


async def test_malformed_extraction_then_retry_succeeds():
    responses = [
        _response("not valid json"),
        _response(json.dumps({"claims": ["A claim."]})),
        _response(json.dumps({"verdicts": ["supported"]})),
    ]
    client = _FakeClient(responses)

    result = await score_faithfulness("A claim.", _trace(), client=client)

    assert result.judge_error is False
    assert result.faithfulness == 1.0
    assert len(client.chat.completions.calls) == 3


async def test_extraction_malformed_twice_is_judge_error():
    responses = [_response("not json"), _response("still not json")]
    client = _FakeClient(responses)

    result = await score_faithfulness("Some answer.", _trace(), client=client)

    assert result.judge_error is True
    assert result.faithfulness is None
    assert result.error is not None


async def test_verification_wrong_length_is_treated_as_malformed():
    responses = [
        _response(json.dumps({"claims": ["Claim one.", "Claim two."]})),
        _response(json.dumps({"verdicts": ["supported"]})),  # only 1, need 2
        _response(json.dumps({"verdicts": ["supported"]})),  # retry also wrong
    ]
    client = _FakeClient(responses)

    result = await score_faithfulness("Claim one. Claim two.", _trace(), client=client)

    assert result.judge_error is True


async def test_verification_invalid_verdict_value_is_treated_as_malformed():
    responses = [
        _response(json.dumps({"claims": ["Claim one."]})),
        _response(json.dumps({"verdicts": ["maybe"]})),  # not a valid verdict
        _response(json.dumps({"verdicts": ["supported"]})),  # retry succeeds
    ]
    client = _FakeClient(responses)

    result = await score_faithfulness("Claim one.", _trace(), client=client)

    assert result.judge_error is False
    assert result.faithfulness == 1.0
