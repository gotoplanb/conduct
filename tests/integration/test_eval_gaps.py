"""Coverage gaps in eval/scoring.py and eval/datasets.py: validation raises,
eval-token failure modes, shadow-side pairwise records, and the export
filters/limits that the main dataset tests don't reach.

Unscoped queries (no task_type) run against the shared dev database, so every
test here scopes by the per-test client_app_id or a unique task_type to keep
pre-existing rows out of the assertions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from eval.datasets import _system_for_job, iter_preferences, iter_sft
from eval.scoring import (
    EvalTokenError,
    apply_pairwise_preference,
    apply_score,
    mint_eval_token,
    redeem_eval_token,
    score_state,
)
from models.job import Job
from models.prompt import PromptVersion
from models.shadow import JobShadow
from models.types import JobStatus


@pytest.fixture
def task_type() -> str:
    return f"gap-{uuid4().hex[:8]}"


def _qs(*scores, via="judge"):
    return {"quality_scores": [
        {"score": s, "via": via, "reviewer": "m", "note": "", "at": "x"} for s in scores
    ]}


async def _job(db, client_id, *, task_type, prompt="Q?", response="A", model="m",
               metadata=None) -> Job:
    job = Job(
        client_app_id=client_id, task_type=task_type, sensitivity="internal",
        prompt=prompt, response=response, model_used=model,
        status=JobStatus.COMPLETE.value, job_metadata=metadata or {},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _shadow(db, parent_id, *, model="s", response="B", metadata=None) -> JobShadow:
    sh = JobShadow(
        parent_job_id=parent_id, model=model, provider="ollama", response=response,
        status=JobStatus.COMPLETE.value, shadow_metadata=metadata or {},
    )
    db.add(sh)
    await db.commit()
    await db.refresh(sh)
    return sh


# --- scoring: validation + score_state --------------------------------------


async def test_apply_score_dimension_out_of_range(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="out of range 1-5"):
        await apply_score(db_session, uuid4(), scores={"correctness": 9})


async def test_apply_score_overall_out_of_range(db_session: AsyncSession) -> None:
    with pytest.raises(ValueError, match="out of range 1-5"):
        await apply_score(db_session, uuid4(), score=0)


def test_score_state_skips_non_numeric_dimension_values() -> None:
    state = score_state([
        {"score": 4, "scores": {"format": "not-a-number", "craft": 5}},
        {"score": 2, "scores": {"format": 3, "craft": None}},
    ])
    assert state["count"] == 2
    assert state["dimensions"] == {
        "format": {"count": 1, "avg": 3.0},
        "craft": {"count": 1, "avg": 5.0},
    }


def test_score_state_dimension_with_only_bad_values_is_dropped() -> None:
    # Regression: float() must be attempted before the defaultdict key exists,
    # or an all-bad dimension leaves an empty list and a ZeroDivisionError.
    state = score_state([{"score": 4, "scores": {"format": "n/a"}}])
    assert state["count"] == 1
    assert state["dimensions"] == {}


# --- scoring: pairwise on shadows / misses ----------------------------------


async def test_pairwise_preference_lands_on_shadow(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    parent = await _job(db_session, seeded_client[0].id, task_type=task_type)
    shadow = await _shadow(db_session, parent.id)
    other = await _job(db_session, seeded_client[0].id, task_type=task_type)

    ok = await apply_pairwise_preference(
        db_session, shadow.id, against_job_id=other.id,
        outcome="win", judge_job_id=parent.id,
    )
    assert ok is True
    await db_session.refresh(shadow)
    verdicts = shadow.shadow_metadata["pairwise_verdicts"]
    assert verdicts[-1]["outcome"] == "win"
    assert verdicts[-1]["against_job_id"] == str(other.id)


async def test_pairwise_preference_unknown_target_returns_false(
    db_session: AsyncSession,
) -> None:
    ok = await apply_pairwise_preference(
        db_session, uuid4(), against_job_id=uuid4(), outcome="win", judge_job_id=uuid4(),
    )
    assert ok is False


# --- scoring: eval-token failure modes --------------------------------------


async def test_redeem_without_issued_token_is_404(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    job = await _job(db_session, seeded_client[0].id, task_type=task_type)
    with pytest.raises(EvalTokenError) as exc:
        await redeem_eval_token(db_session, job, raw_token="cdt_ev_x", score=4, note=None)
    assert exc.value.status == 404


async def test_redeem_expired_token_is_401(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    job = await _job(db_session, seeded_client[0].id, task_type=task_type)
    raw, _expires = await mint_eval_token(db_session, job)
    job.eval_token_expires_at = datetime.now(UTC) - timedelta(days=1)
    await db_session.commit()
    with pytest.raises(EvalTokenError) as exc:
        await redeem_eval_token(db_session, job, raw_token=raw, score=4, note=None)
    assert exc.value.status == 401
    assert "expired" in exc.value.message


# --- datasets: system-prompt resolution -------------------------------------


async def test_system_for_job_none_job_is_empty(db_session: AsyncSession) -> None:
    # A shadow whose parent vanished still exports (with an empty system).
    assert await _system_for_job(db_session, None, {}) == ""


# --- datasets: SFT filters + limits -----------------------------------------


async def test_sft_client_scope_without_task_type(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    await _job(db_session, cid, task_type=task_type, metadata=_qs(5))
    rows = await iter_sft(db_session, min_score=4, client_app_id=cid)
    assert len(rows) == 1
    assert rows[0]["meta"]["task_type"] == task_type


async def test_sft_prompt_version_filter_excludes_mismatch(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    await _job(db_session, cid, task_type=task_type, metadata=_qs(5))  # pv None
    rows = await iter_sft(
        db_session, task_type=task_type, min_score=4, prompt_version=424242
    )
    assert rows == []


async def test_sft_limit_stops_job_scan(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    await _job(db_session, cid, task_type=task_type, prompt="a", metadata=_qs(5))
    await _job(db_session, cid, task_type=task_type, prompt="b", metadata=_qs(5))
    rows = await iter_sft(db_session, task_type=task_type, min_score=4, limit=1)
    assert len(rows) == 1


async def test_sft_shadow_scan_respects_limit_already_met(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    # Parent below min_score so the limit is filled inside the shadow loop:
    # the first qualifying shadow lands, the second is cut off by the limit.
    cid = seeded_client[0].id
    parent = await _job(db_session, cid, task_type=task_type, metadata=_qs(2))
    await _shadow(db_session, parent.id, response="great-1", metadata=_qs(5))
    await _shadow(db_session, parent.id, response="great-2", metadata=_qs(5))
    rows = await iter_sft(
        db_session, task_type=task_type, min_score=4, include_shadows=True, limit=1
    )
    assert len(rows) == 1
    assert rows[0]["meta"]["source"] == "shadow"


async def test_sft_prompt_version_content_cached_across_jobs(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    pv = PromptVersion(task_type=task_type, client_id=None, content="You are a tutor.")
    db_session.add(pv)
    await db_session.commit()
    await db_session.refresh(pv)
    md = {**_qs(5), "prompt": {"source": "library", "version_id": pv.id}}
    await _job(db_session, cid, task_type=task_type, prompt="a", metadata=md)
    await _job(db_session, cid, task_type=task_type, prompt="b", metadata=md)

    rows = await iter_sft(db_session, task_type=task_type, min_score=4)
    assert len(rows) == 2
    # Second row resolves the same version via the cache — identical system.
    assert {r["system"] for r in rows} == {"You are a tutor."}


async def test_sft_shadow_prompt_version_filter(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    # Parent below min_score so only the shadow could qualify; its parent has
    # no prompt_version -> the pv filter must drop it.
    parent = await _job(db_session, cid, task_type=task_type, metadata=_qs(2))
    await _shadow(db_session, parent.id, metadata=_qs(5))
    rows = await iter_sft(
        db_session, task_type=task_type, min_score=4,
        include_shadows=True, prompt_version=424242,
    )
    assert rows == []


# --- datasets: pairwise preferences -----------------------------------------


async def test_pairwise_limit_and_client_scope(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    loser_a = await _job(db_session, cid, task_type=task_type, response="meh-a")
    loser_b = await _job(db_session, cid, task_type=task_type, response="meh-b")
    await _job(
        db_session, cid, task_type=task_type, response="best",
        metadata={"pairwise_verdicts": [
            {"against_job_id": str(loser_a.id), "outcome": "win", "judge_job_id": "j"},
            {"against_job_id": str(loser_b.id), "outcome": "win", "judge_job_id": "j"},
        ]},
    )
    rows = await iter_preferences(
        db_session, method="pairwise", limit=1, client_app_id=cid
    )
    assert len(rows) == 1
    assert rows[0]["chosen"] == "best"


async def test_pairwise_skips_winner_without_response(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    loser = await _job(db_session, cid, task_type=task_type)
    await _job(
        db_session, cid, task_type=task_type, response="",
        metadata={"pairwise_verdicts": [
            {"against_job_id": str(loser.id), "outcome": "win", "judge_job_id": "j"},
        ]},
    )
    rows = await iter_preferences(db_session, task_type=task_type, method="pairwise")
    assert rows == []


async def test_pairwise_winner_prompt_version_filter(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    loser = await _job(db_session, cid, task_type=task_type)
    await _job(
        db_session, cid, task_type=task_type, response="win",
        metadata={"pairwise_verdicts": [
            {"against_job_id": str(loser.id), "outcome": "win", "judge_job_id": "j"},
        ]},
    )
    rows = await iter_preferences(
        db_session, task_type=task_type, method="pairwise", prompt_version=424242
    )
    assert rows == []


async def test_pairwise_skips_missing_loser(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    await _job(
        db_session, cid, task_type=task_type, response="win",
        metadata={"pairwise_verdicts": [
            {"against_job_id": str(uuid4()), "outcome": "win", "judge_job_id": "j"},
        ]},
    )
    rows = await iter_preferences(db_session, task_type=task_type, method="pairwise")
    assert rows == []


# --- datasets: score-differential preferences -------------------------------


async def test_score_prefs_prompt_version_filter(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    parent = await _job(db_session, cid, task_type=task_type, metadata=_qs(5))
    await _shadow(db_session, parent.id, metadata=_qs(1))
    rows = await iter_preferences(
        db_session, task_type=task_type, method="score", min_gap=2, prompt_version=424242
    )
    assert rows == []


async def test_score_prefs_needs_two_candidates(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    await _job(db_session, cid, task_type=task_type, metadata=_qs(5))  # no shadows
    rows = await iter_preferences(db_session, task_type=task_type, method="score", min_gap=2)
    assert rows == []


async def test_score_prefs_limit(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    cid = seeded_client[0].id
    for _ in range(2):
        parent = await _job(db_session, cid, task_type=task_type, metadata=_qs(5))
        await _shadow(db_session, parent.id, metadata=_qs(1))
    rows = await iter_preferences(
        db_session, task_type=task_type, method="score", min_gap=2, limit=1
    )
    assert len(rows) == 1


async def test_score_prefs_skip_unusable_candidates(
    db_session: AsyncSession, seeded_client, task_type
) -> None:
    # Parent scored but has no response -> excluded as a candidate; the pair
    # comes from the two scored shadows; an unscored shadow is skipped. No
    # task_type arg (client-scoped) to also cover the unfiltered scan path.
    cid = seeded_client[0].id
    parent = await _job(
        db_session, cid, task_type=task_type, response="", metadata=_qs(3)
    )
    await _shadow(db_session, parent.id, model="hi", response="good", metadata=_qs(5))
    await _shadow(db_session, parent.id, model="lo", response="bad", metadata=_qs(1))
    await _shadow(db_session, parent.id, model="unscored", response="x", metadata={})
    rows = await iter_preferences(
        db_session, method="score", min_gap=2, client_app_id=cid
    )
    assert len(rows) == 1
    assert rows[0]["chosen"] == "good"
    assert rows[0]["rejected"] == "bad"
    assert rows[0]["meta"]["chosen_model"] == "hi"
