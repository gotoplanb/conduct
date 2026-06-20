"""Tests for the training-dataset export (eval/datasets.py, routes/datasets.py).
See GitHub issue #16."""

from __future__ import annotations

import json

from sqlalchemy.ext.asyncio import AsyncSession

from eval.datasets import iter_preferences, iter_sft
from models.job import Job
from models.prompt import PromptVersion
from models.shadow import JobShadow
from models.types import JobStatus


def _qs(*scores, via="judge"):
    return {"quality_scores": [
        {"score": s, "via": via, "reviewer": "m", "note": "", "at": "x"} for s in scores
    ]}


def _qs_dims(dims, via="judge"):
    overall = round(sum(dims.values()) / len(dims))
    return {"quality_scores": [
        {"score": overall, "scores": dims, "via": via, "reviewer": "m", "at": "x"}
    ]}


async def _job(db, client_id, *, task_type="qa", prompt="Q?", response="A", model="m",
               system_prompt="", metadata=None, sensitivity="internal") -> Job:
    job = Job(
        client_app_id=client_id, task_type=task_type, sensitivity=sensitivity,
        prompt=prompt, response=response, system_prompt=system_prompt, model_used=model,
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


# --- SFT -------------------------------------------------------------------


async def test_sft_keeps_high_scored_only(db_session: AsyncSession, seeded_client) -> None:
    cid = seeded_client[0].id
    good = await _job(db_session, cid, prompt="good?", response="great", metadata=_qs(5, 4))
    await _job(db_session, cid, prompt="bad?", response="weak", metadata=_qs(2))  # below min_score
    await _job(db_session, cid, prompt="unscored?", response="x", metadata={})    # no scores

    rows = await iter_sft(db_session, task_type="qa", min_score=4)
    assert len(rows) == 1
    r = rows[0]
    assert r["prompt"] == "good?" and r["completion"] == "great"
    assert r["meta"]["score"] == 4.5 and r["meta"]["n_scores"] == 2
    assert r["meta"]["source"] == "job" and r["meta"]["id"] == str(good.id)


async def test_sft_label_dim_filters_and_surfaces_dimensions(
    db_session: AsyncSession, seeded_client
) -> None:
    cid = seeded_client[0].id
    # high overall (4.3) but low correctness (2) → excluded when label_dim=correctness
    await _job(db_session, cid, prompt="a", response="x",
               metadata=_qs_dims({"correctness": 2, "format": 5, "craft": 6 - 1}))
    # high correctness (5) → kept
    await _job(db_session, cid, prompt="b", response="y",
               metadata=_qs_dims({"correctness": 5, "format": 3, "craft": 4}))

    rows = await iter_sft(db_session, task_type="qa", min_score=4, label_dim="correctness")
    assert len(rows) == 1
    assert rows[0]["prompt"] == "b"
    assert rows[0]["meta"]["score"] == 5.0  # the correctness dimension
    assert rows[0]["meta"]["dimensions"] == {"correctness": 5.0, "format": 3.0, "craft": 4.0}


async def test_sft_via_filter(db_session: AsyncSession, seeded_client) -> None:
    cid = seeded_client[0].id
    # high human score, low judge score → excluded when filtering via=judge
    md = {"quality_scores": [
        {"score": 5, "via": "admin"}, {"score": 2, "via": "judge"},
    ]}
    await _job(db_session, cid, metadata=md)
    assert await iter_sft(db_session, task_type="qa", min_score=4, via="judge") == []
    assert len(await iter_sft(db_session, task_type="qa", min_score=4, via="admin")) == 1


async def test_sft_reconstructs_system_from_prompt_version(
    db_session: AsyncSession, seeded_client
) -> None:
    cid = seeded_client[0].id
    pv = PromptVersion(task_type="qa", client_id=None, content="You are a tutor.")
    db_session.add(pv)
    await db_session.commit()
    await db_session.refresh(pv)
    # library-sourced job: no system_prompt override, but a version_id in metadata
    md = {**_qs(5), "prompt": {"source": "library", "version_id": pv.id}}
    await _job(db_session, cid, system_prompt="", metadata=md)

    rows = await iter_sft(db_session, task_type="qa", min_score=4)
    assert rows[0]["system"] == "You are a tutor."
    assert rows[0]["meta"]["prompt_version"] == pv.id


async def test_sft_include_shadows(db_session: AsyncSession, seeded_client) -> None:
    cid = seeded_client[0].id
    parent = await _job(db_session, cid, prompt="P?", response="parent-ans", metadata=_qs(3))
    await _shadow(db_session, parent.id, model="big", response="shadow-ans", metadata=_qs(5))

    without = await iter_sft(db_session, task_type="qa", min_score=4, include_shadows=False)
    assert without == []  # parent scored 3, no shadows pulled
    with_sh = await iter_sft(db_session, task_type="qa", min_score=4, include_shadows=True)
    assert len(with_sh) == 1
    assert with_sh[0]["completion"] == "shadow-ans"
    assert with_sh[0]["meta"]["source"] == "shadow" and with_sh[0]["meta"]["model"] == "big"


# --- preferences: pairwise -------------------------------------------------


async def test_preferences_pairwise(db_session: AsyncSession, seeded_client) -> None:
    cid = seeded_client[0].id
    loser = await _job(db_session, cid, prompt="P?", response="meh", model="weak")
    winner = await _job(
        db_session, cid, prompt="P?", response="excellent", model="strong",
        metadata={"pairwise_verdicts": [
            {"against_job_id": str(loser.id), "outcome": "win", "judge_job_id": "j1"}
        ]},
    )
    # the loser also carries the mirror 'loss' record — which must NOT create a pair
    loser.job_metadata = {"pairwise_verdicts": [
        {"against_job_id": str(winner.id), "outcome": "loss", "judge_job_id": "j1"}
    ]}
    await db_session.commit()

    rows = await iter_preferences(db_session, task_type="qa", method="pairwise")
    assert len(rows) == 1  # exactly one pair, not two
    r = rows[0]
    assert r["chosen"] == "excellent" and r["rejected"] == "meh"
    assert r["meta"]["chosen_model"] == "strong" and r["meta"]["rejected_model"] == "weak"
    assert r["meta"]["method"] == "pairwise"


# --- preferences: score differential ---------------------------------------


async def test_preferences_from_score_differential(
    db_session: AsyncSession, seeded_client
) -> None:
    cid = seeded_client[0].id
    parent = await _job(
        db_session, cid, prompt="P?", response="parent-good", model="p", metadata=_qs(5)
    )
    await _shadow(db_session, parent.id, model="s", response="shadow-bad", metadata=_qs(2))

    rows = await iter_preferences(db_session, task_type="qa", method="score", min_gap=2)
    assert len(rows) == 1
    r = rows[0]
    assert r["chosen"] == "parent-good" and r["rejected"] == "shadow-bad"
    assert r["meta"]["chosen_score"] == 5 and r["meta"]["rejected_score"] == 2
    assert r["meta"]["method"] == "score"


async def test_preferences_score_skips_small_gap(
    db_session: AsyncSession, seeded_client
) -> None:
    cid = seeded_client[0].id
    parent = await _job(db_session, cid, response="a", metadata=_qs(4))
    await _shadow(db_session, parent.id, response="b", metadata=_qs(3))  # gap 1 < min_gap 2
    assert await iter_preferences(db_session, task_type="qa", method="score", min_gap=2) == []


# --- preferences: composite (#31) ------------------------------------------


async def test_preferences_from_composite(db_session: AsyncSession, seeded_client) -> None:
    cid = seeded_client[0].id
    # Two solutions to one spec, ranked by the deterministic composite (#30).
    parent = await _job(
        db_session, cid, prompt="P?", response="good", model="p",
        metadata=_qs_dims({"compile": 5, "golden": 5}),  # composite 5.0
    )
    await _shadow(
        db_session, parent.id, model="s", response="bad",
        metadata=_qs_dims({"compile": 1, "golden": 1}),  # composite 1.0
    )

    rows = await iter_preferences(db_session, task_type="qa", method="composite", min_gap=2)
    assert len(rows) == 1
    r = rows[0]
    assert r["chosen"] == "good" and r["rejected"] == "bad"
    assert r["meta"]["method"] == "composite"
    assert r["meta"]["chosen_score"] == 5.0 and r["meta"]["rejected_score"] == 1.0
    # provenance back to the originating job + shadow
    assert r["meta"]["chosen_id"] == str(parent.id)
    assert r["meta"]["rejected_id"]


async def test_preferences_composite_skips_small_gap(
    db_session: AsyncSession, seeded_client
) -> None:
    cid = seeded_client[0].id
    parent = await _job(
        db_session, cid, response="a", metadata=_qs_dims({"compile": 5, "golden": 5}),  # 5.0
    )
    # composite (3*4+3*5)/6 = 4.5 -> gap 0.5 < min_gap 2
    await _shadow(
        db_session, parent.id, response="b", metadata=_qs_dims({"compile": 4, "golden": 5}),
    )
    assert await iter_preferences(db_session, task_type="qa", method="composite", min_gap=2) == []


# --- route smoke (auth + JSONL) --------------------------------------------


async def test_export_routes_stream_jsonl(
    client, db_session, seeded_client, admin_headers
) -> None:
    cid = seeded_client[0].id
    await _job(db_session, cid, prompt="q", response="good", metadata=_qs(5))

    r = await client.get("/datasets/sft?task_type=qa&min_score=4", headers=admin_headers)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0]["completion"] == "good"

    # admin-gated
    rp = await client.get("/datasets/preferences", headers=admin_headers)
    assert rp.status_code == 200


async def test_export_requires_admin(client, seeded_client) -> None:
    r = await client.get(
        "/datasets/sft", headers={"Authorization": f"Bearer {seeded_client[1]}"}
    )
    assert r.status_code in (401, 403)
