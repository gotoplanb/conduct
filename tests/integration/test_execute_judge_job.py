"""Integration tests for the Phase-1 single-model pointwise LLM-as-judge
(worker.executor.execute_judge_job). See GitHub issue #17.

A judge job carries a `target_job_id`; the executor resolves the target's
prompt+response, scores it against the judge's rubric (here supplied via the
job's system_prompt override), parses a {score, rationale} verdict, stores it,
and — only when `apply_to_target` is set — appends the score to the target via
the normal quality-score lane.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from models.job import Job
from models.types import JobStatus, Sensitivity
from providers.base import BaseProvider, ProviderError, ProviderResponse
from providers.registry import ProviderRegistry
from routing.engine import RoutingDecision
from worker.executor import (
    _parse_judge_verdict,
    execute_judge_job,
    is_judge_job,
)


class _StubJudgeProvider(BaseProvider):
    name = "ollama"

    def __init__(self, text: str, *, raise_error: bool = False) -> None:
        self._text = text
        self._raise = raise_error
        self.last_call: dict = {}

    async def complete(
        self, prompt="", model="", system_prompt="", max_tokens=1000,
        temperature=None, seed=None, **kwargs,
    ) -> ProviderResponse:
        if self._raise:
            raise ProviderError("boom")
        self.last_call = {
            "prompt": prompt, "system_prompt": system_prompt,
            "temperature": temperature, "seed": seed,
        }
        return ProviderResponse(
            response=self._text, tokens_in=10, tokens_out=5,
            cost_usd=Decimal("0"), latency_ms=1, model_used=model, provider=self.name,
        )


def _decision(sensitivity: Sensitivity = Sensitivity.INTERNAL) -> RoutingDecision:
    return RoutingDecision(
        model="gemma4:e4b",
        provider="ollama",
        fallback_model=None,
        fallback_provider=None,
        effective_sensitivity=sensitivity,
        max_tokens=400,
        reason="test",
        temperature=0.0,
        deterministic_seed=True,
    )


async def _seed_target(
    db, client_id, *, sensitivity="internal", response="The capital is Paris."
) -> Job:
    job = Job(
        client_app_id=client_id, task_type="qa", sensitivity=sensitivity,
        prompt="What is the capital of France?", response=response,
        status=JobStatus.COMPLETE.value,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _seed_judge(db, client_id, *, target_id, apply_to_target=False, mode="pointwise") -> Job:
    job = Job(
        client_app_id=client_id, task_type="judge", sensitivity="internal",
        prompt="(judge)", system_prompt="Score the response 1-5 against the prompt.",
        status=JobStatus.PENDING.value,
        inputs={"mode": mode, "target_job_id": str(target_id), "apply_to_target": apply_to_target},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _run(judge, decision, provider, client, session) -> Job:
    reg = ProviderRegistry()
    reg.register(provider)
    return await execute_judge_job(
        job=judge, decision=decision, client=client, providers=reg, session=session
    )


# --- pure helpers ---------------------------------------------------------


def test_is_judge_job() -> None:
    assert is_judge_job({"target_job_id": "abc"}) is True
    assert is_judge_job({"mode": "pointwise"}) is False
    assert is_judge_job({}) is False
    assert is_judge_job(None) is False


def test_parse_judge_verdict_plain() -> None:
    score, rationale = _parse_judge_verdict('{"score": 4, "rationale": "solid"}')
    assert score == 4
    assert rationale == "solid"


def test_parse_judge_verdict_tolerates_fences_and_prose() -> None:
    # The exact failure mode we hit on gemma4:12b — fenced JSON with preamble.
    text = 'Here is my verdict:\n```json\n{"score": 5, "rationale": "great"}\n```'
    score, rationale = _parse_judge_verdict(text)
    assert score == 5
    assert rationale == "great"


def test_parse_judge_verdict_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        _parse_judge_verdict('{"score": 9}')


def test_parse_judge_verdict_rejects_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        _parse_judge_verdict("I think it was pretty good, maybe a 4.")


# --- execution ------------------------------------------------------------


async def test_pointwise_judge_scores_and_applies_to_target(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, target_id=target.id, apply_to_target=True)
    provider = _StubJudgeProvider('{"score": 4, "rationale": "accurate and on-topic"}')

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    meta = (out.job_metadata or {}).get("judge") or {}
    assert meta["score"] == 4
    assert meta["target_kind"] == "job"
    assert meta["applied_to_target"] is True

    # The score landed on the TARGET via the judge lane.
    await db_session.refresh(target)
    scores = (target.job_metadata or {}).get("quality_scores") or []
    assert len(scores) == 1
    assert scores[0]["score"] == 4
    assert scores[0]["via"] == "judge"
    assert scores[0]["reviewer"] == "gemma4:e4b"

    # Deterministic profile reached the provider (temp 0 + a derived seed).
    assert provider.last_call["temperature"] == 0.0
    assert isinstance(provider.last_call["seed"], int)


async def test_verdict_only_does_not_touch_target(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, target_id=target.id, apply_to_target=False)
    provider = _StubJudgeProvider('{"score": 3, "rationale": "partial"}')

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    assert (out.job_metadata or {})["judge"]["score"] == 3
    await db_session.refresh(target)
    assert (target.job_metadata or {}).get("quality_scores", []) == []


async def test_unparseable_verdict_fails_loudly(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id)
    judge = await _seed_judge(db_session, client.id, target_id=target.id, apply_to_target=True)
    provider = _StubJudgeProvider("It was fine, I'd say a four out of five.")

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.FAILED.value
    assert "could not parse judge verdict" in out.error
    # A failed judge must not score the target.
    await db_session.refresh(target)
    assert (target.job_metadata or {}).get("quality_scores", []) == []


async def test_unknown_target_fails(db_session: AsyncSession, seeded_client) -> None:
    client = seeded_client[0]
    judge = await _seed_judge(
        db_session, client.id, target_id="00000000-0000-0000-0000-000000000000"
    )
    provider = _StubJudgeProvider('{"score": 5}')  # never called

    out = await _run(judge, _decision(), provider, client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "not found" in out.error
    assert provider.last_call == {}  # provider never invoked


async def test_sensitivity_guard_blocks_looser_judge(
    db_session: AsyncSession, seeded_client
) -> None:
    """A confidential target must not be judged by a job that resolved to a
    looser (public) sensitivity — that could route its content to a cloud
    model. Fail before calling any provider."""
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, sensitivity="confidential")
    judge = await _seed_judge(db_session, client.id, target_id=target.id, apply_to_target=True)
    provider = _StubJudgeProvider('{"score": 5}')

    out = await _run(judge, _decision(sensitivity=Sensitivity.PUBLIC), provider, client, db_session)

    assert out.status == JobStatus.FAILED.value
    assert "confidential" in out.error
    assert provider.last_call == {}  # never reached the model
    await db_session.refresh(target)
    assert (target.job_metadata or {}).get("quality_scores", []) == []
