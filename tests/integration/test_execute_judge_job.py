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
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from models.job import Job
from models.types import JobStatus, Sensitivity
from providers.base import BaseProvider, ProviderError, ProviderResponse
from providers.registry import ProviderRegistry
from routing.engine import RoutingDecision
from worker.executor import (
    _aggregate_panel,
    _build_panel,
    _model_family,
    _parse_judge_verdict,
    _parse_pairwise_verdict,
    _reconcile_pairwise,
    execute_judge_job,
    is_judge_job,
)


class _StubJudgeProvider(BaseProvider):
    name = "ollama"

    def __init__(
        self, text: str = "", *, texts: list[str] | None = None, raise_error: bool = False
    ) -> None:
        # `texts` returns one response per call in order (for the two
        # order-swapped pairwise calls); `text` repeats a single response.
        self._texts = list(texts) if texts is not None else None
        self._text = text
        self._raise = raise_error
        self.last_call: dict = {}
        self.call_count = 0

    async def complete(
        self, prompt="", model="", system_prompt="", max_tokens=1000,
        temperature=None, seed=None, **kwargs,
    ) -> ProviderResponse:
        if self._raise:
            raise ProviderError("boom")
        self.call_count += 1
        self.last_call = {
            "prompt": prompt, "system_prompt": system_prompt,
            "temperature": temperature, "seed": seed,
        }
        out = self._texts.pop(0) if self._texts is not None else self._text
        return ProviderResponse(
            response=out, tokens_in=10, tokens_out=5,
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
    db, client_id, *, sensitivity="internal", response="The capital is Paris.", model_used=""
) -> Job:
    job = Job(
        client_app_id=client_id, task_type="qa", sensitivity=sensitivity,
        prompt="What is the capital of France?", response=response, model_used=model_used,
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


# --- pairwise (Phase 2) ---------------------------------------------------


def test_reconcile_pairwise_consistent_winner() -> None:
    a, b = uuid4(), uuid4()
    # call1 (A=a, B=b) → "A"=a; call2 (A=b, B=a) → "B"=a : both name a → consistent
    chosen, rejected, tie, consistent = _reconcile_pairwise("A", "B", a, b)
    assert chosen == a and rejected == b
    assert tie is False and consistent is True


def test_reconcile_pairwise_position_bias_is_tie() -> None:
    a, b = uuid4(), uuid4()
    # winner "A" both calls → call1=a, call2=b : disagree → tie
    chosen, rejected, tie, consistent = _reconcile_pairwise("A", "A", a, b)
    assert chosen is None and rejected is None
    assert tie is True and consistent is False


def test_reconcile_pairwise_explicit_tie() -> None:
    a, b = uuid4(), uuid4()
    chosen, _r, tie, consistent = _reconcile_pairwise("TIE", "B", a, b)
    assert chosen is None and tie is True and consistent is False


def test_parse_pairwise_verdict() -> None:
    w, r = _parse_pairwise_verdict('{"winner": "B", "rationale": "B is sharper"}')
    assert w == "B" and r == "B is sharper"
    with pytest.raises(ValueError, match="not in A/B/tie"):
        _parse_pairwise_verdict('{"winner": "C"}')


async def _seed_pairwise_judge(
    db, client_id, *, target_id, against_id, apply_to_target=False
) -> Job:
    job = Job(
        client_app_id=client_id, task_type="judge", sensitivity="internal",
        prompt="(judge)", system_prompt="Compare the two responses against the prompt.",
        status=JobStatus.PENDING.value,
        inputs={
            "mode": "pairwise", "target_job_id": str(target_id),
            "against_job_id": str(against_id), "apply_to_target": apply_to_target,
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_pairwise_consistent_winner_records_preference(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    a = await _seed_target(db_session, client.id, response="Paris is the capital.")
    b = await _seed_target(db_session, client.id, response="Maybe London?")
    judge = await _seed_pairwise_judge(
        db_session, client.id, target_id=a.id, against_id=b.id, apply_to_target=True
    )
    # call1 (A=a) → "A"=a ; call2 (A=b) → "B"=a : a wins consistently.
    provider = _StubJudgeProvider(texts=[
        '{"winner": "A", "rationale": "A is correct"}',
        '{"winner": "B", "rationale": "A is still correct"}',
    ])

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    assert provider.call_count == 2  # order-swapped: two calls
    meta = out.job_metadata["judge"]
    assert meta["mode"] == "pairwise"
    assert meta["chosen_job_id"] == str(a.id)
    assert meta["rejected_job_id"] == str(b.id)
    assert meta["tie"] is False
    assert meta["position_consistent"] is True
    assert meta["applied_to_target"] is True

    # Preference recorded on BOTH participants (not in the 1-5 quality lane).
    await db_session.refresh(a)
    await db_session.refresh(b)
    av = (a.job_metadata or {}).get("pairwise_verdicts") or []
    bv = (b.job_metadata or {}).get("pairwise_verdicts") or []
    assert len(av) == 1 and av[0]["outcome"] == "win" and av[0]["against_job_id"] == str(b.id)
    assert len(bv) == 1 and bv[0]["outcome"] == "loss"
    assert (a.job_metadata or {}).get("quality_scores", []) == []  # pointwise lane untouched


async def test_pairwise_position_bias_yields_tie(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    a = await _seed_target(db_session, client.id)
    b = await _seed_target(db_session, client.id, response="other")
    judge = await _seed_pairwise_judge(
        db_session, client.id, target_id=a.id, against_id=b.id, apply_to_target=True
    )
    # Always picks "A" → favors whichever is shown first → disagreement → tie.
    provider = _StubJudgeProvider(texts=[
        '{"winner": "A", "rationale": "first one"}',
        '{"winner": "A", "rationale": "first one"}',
    ])

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    meta = out.job_metadata["judge"]
    assert meta["tie"] is True
    assert meta["position_consistent"] is False
    assert meta["chosen_job_id"] is None
    assert meta["applied_to_target"] is False
    await db_session.refresh(a)
    assert (a.job_metadata or {}).get("pairwise_verdicts", []) == []


async def test_pairwise_requires_against_job_id(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    a = await _seed_target(db_session, client.id)
    judge = Job(
        client_app_id=client.id, task_type="judge", sensitivity="internal",
        prompt="(judge)", system_prompt="x", status=JobStatus.PENDING.value,
        inputs={"mode": "pairwise", "target_job_id": str(a.id)},  # no against_job_id
    )
    db_session.add(judge)
    await db_session.commit()
    await db_session.refresh(judge)
    provider = _StubJudgeProvider('{"winner": "A"}')

    out = await _run(judge, _decision(), provider, client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "against_job_id" in out.error
    assert provider.call_count == 0


# --- panel / jury (Phase 3) -----------------------------------------------


def test_model_family() -> None:
    assert _model_family("gemma4:e4b") == "gemma"
    assert _model_family("llama3.2:3b") == "llama"
    assert _model_family("qwen3.5:9b") == "qwen"
    assert _model_family("mistral-small3.2") == "mistral"
    assert _model_family("claude-sonnet-4-6") == "claude"
    assert _model_family("us.anthropic.claude-haiku-4-5-20251001-v1:0") == "claude"


def test_aggregate_panel_median_and_disagreement() -> None:
    agg = _aggregate_panel([
        {"model": "a", "score": 4}, {"model": "b", "score": 2}, {"model": "c", "score": 5},
    ])
    assert agg["score"] == 4  # median(4, 2, 5)
    assert agg["n"] == 3 and agg["spread"] == 3 and agg["disagreement"] is True

    tight = _aggregate_panel([{"model": "a", "score": 4}, {"model": "b", "score": 4}])
    assert tight["score"] == 4 and tight["disagreement"] is False


def test_build_panel_excludes_self_preference_and_sensitivity() -> None:
    # producer is gemma4:12b → the gemma juror is dropped (self-preference);
    # the cloud juror is dropped because the target is confidential.
    eligible, excluded = _build_panel(
        ["gemma4:e4b", "llama3.2:3b", "claude-haiku-4-5"],
        "gemma4:12b", Sensitivity.CONFIDENTIAL, allow_cloud_for_internal=False,
    )
    assert eligible == ["llama3.2:3b"]
    assert "gemma4:e4b" in excluded
    assert "claude-haiku-4-5" in excluded


async def _seed_panel_judge(db, client_id, *, target_id, panel, apply_to_target=False) -> Job:
    job = Job(
        client_app_id=client_id, task_type="judge", sensitivity="internal",
        prompt="(judge)", system_prompt="Score the response 1-5 against the prompt.",
        status=JobStatus.PENDING.value,
        inputs={
            "mode": "panel", "target_job_id": str(target_id),
            "panel": panel, "apply_to_target": apply_to_target,
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_panel_aggregates_and_excludes_producer_family(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    # Target was produced by gemma → gemma jurors excluded (self-preference).
    target = await _seed_target(db_session, client.id, model_used="gemma4:e4b")
    judge = await _seed_panel_judge(
        db_session, client.id, target_id=target.id,
        panel=["gemma4:e4b", "llama3.2:3b", "qwen3.5:4b"], apply_to_target=True,
    )
    # eligible = [llama, qwen] in order → scores 4 then 2 → median 3, spread 2.
    provider = _StubJudgeProvider(texts=[
        '{"score": 4, "rationale": "good"}',
        '{"score": 2, "rationale": "meh"}',
    ])

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    assert provider.call_count == 2  # the gemma candidate was never called
    meta = out.job_metadata["judge"]
    assert meta["mode"] == "panel"
    assert meta["n"] == 2
    assert meta["score"] == 3
    assert meta["disagreement"] is True  # spread 2
    assert "gemma4:e4b" in meta["excluded"]
    assert {p["model"] for p in meta["panelists"]} == {"llama3.2:3b", "qwen3.5:4b"}
    assert meta["applied_to_target"] is True

    # Median applied to the target under a distinct via tag.
    await db_session.refresh(target)
    qs = (target.job_metadata or {}).get("quality_scores") or []
    assert len(qs) == 1 and qs[0]["score"] == 3 and qs[0]["via"] == "judge-panel"


async def test_panel_skips_failed_juror(db_session: AsyncSession, seeded_client) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, model_used="claude-sonnet-4-6")
    judge = await _seed_panel_judge(
        db_session, client.id, target_id=target.id,
        panel=["llama3.2:3b", "qwen3.5:4b", "mistral-small3.2"],
    )
    # middle juror returns garbage → skipped; aggregate over the survivors.
    provider = _StubJudgeProvider(texts=[
        '{"score": 5, "rationale": "a"}',
        "not json at all",
        '{"score": 3, "rationale": "c"}',
    ])

    out = await _run(judge, _decision(), provider, client, db_session)

    assert out.status == JobStatus.COMPLETE.value
    meta = out.job_metadata["judge"]
    assert meta["n"] == 2  # one of three failed
    assert len(meta["failures"]) == 1
    assert meta["score"] == 4  # median(5, 3)


async def test_panel_empty_after_exclusion_fails(
    db_session: AsyncSession, seeded_client
) -> None:
    client = seeded_client[0]
    target = await _seed_target(db_session, client.id, model_used="gemma4:e4b")
    judge = await _seed_panel_judge(
        db_session, client.id, target_id=target.id,
        panel=["gemma4:e4b", "gemma4:12b"],  # all same family as producer
    )
    provider = _StubJudgeProvider('{"score": 5}')

    out = await _run(judge, _decision(), provider, client, db_session)
    assert out.status == JobStatus.FAILED.value
    assert "no eligible panelists" in out.error
    assert provider.call_count == 0
