"""Routes under /ui/eval/review — the human scoring page + score POST,
and the shared eval.scoring.apply_score helper it sits on."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from eval.scoring import apply_score, score_state
from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus


@pytest.fixture
def task_type() -> str:
    return f"review-test-{uuid4().hex[:8]}"


async def _seed_job_with_shadow(
    db: AsyncSession, *, client_id, task_type: str
) -> tuple[Job, JobShadow]:
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="internal",
        prompt="Write a bio for Jane.",
        status=JobStatus.COMPLETE.value,
        model_used="llama3.3:70b",
        response="Jane is a realtor.",
        latency_ms=100,
        tokens_in=10,
        tokens_out=20,
        completed_at=datetime.now(UTC),
        job_metadata={},
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    shadow = JobShadow(
        parent_job_id=job.id,
        model="claude-haiku-4-5",
        provider="anthropic",
        status=JobStatus.COMPLETE.value,
        response="Jane sells homes with care.",
    )
    db.add(shadow)
    await db.commit()
    await db.refresh(shadow)
    return job, shadow


# --- helper ---


def test_score_state_empty() -> None:
    assert score_state([]) == {"count": 0, "avg": None}


def test_score_state_averages() -> None:
    st = score_state([{"score": 4}, {"score": 2}, {"score": "bad"}])
    assert st["count"] == 2
    assert st["avg"] == 3.0


async def test_apply_score_to_job(db_session, seeded_client, task_type) -> None:
    job, _ = await _seed_job_with_shadow(
        db_session, client_id=seeded_client[0].id, task_type=task_type
    )
    result = await apply_score(db_session, job.id, score=5, reviewer="ui")
    assert result is not None
    kind, scores = result
    assert kind == "job"
    assert scores[-1]["score"] == 5


async def test_apply_score_unknown_id_returns_none(db_session) -> None:
    assert await apply_score(db_session, uuid4(), score=3) is None


# --- UI routes ---


async def test_review_unauth_redirects(client) -> None:
    r = await client.get("/ui/eval/review", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/ui/login"


async def test_review_page_shows_prompt_and_candidates(
    client, db_session, seeded_client, admin_token, task_type
) -> None:
    await _seed_job_with_shadow(db_session, client_id=seeded_client[0].id, task_type=task_type)
    r = await client.get(
        f"/ui/eval/review?task_type={task_type}",
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "Write a bio for Jane." in r.text
    assert "Jane is a realtor." in r.text  # production response
    assert "Jane sells homes with care." in r.text  # shadow response
    assert "claude-haiku-4-5" in r.text
    assert "production" in r.text


async def test_review_score_records_and_returns_partial(
    client, db_session, seeded_client, admin_token, task_type
) -> None:
    _, shadow = await _seed_job_with_shadow(
        db_session, client_id=seeded_client[0].id, task_type=task_type
    )
    r = await client.post(
        "/ui/eval/review/score",
        data={"target_id": str(shadow.id), "score": "4", "note": "tone is good"},
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 200
    assert "scored 1×" in r.text
    assert "avg 4.0" in r.text


async def test_review_score_out_of_range_is_400(
    client, db_session, seeded_client, admin_token, task_type
) -> None:
    job, _ = await _seed_job_with_shadow(
        db_session, client_id=seeded_client[0].id, task_type=task_type
    )
    r = await client.post(
        "/ui/eval/review/score",
        data={"target_id": str(job.id), "score": "9"},
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 400


async def test_review_score_unknown_id_404(client, admin_token) -> None:
    r = await client.post(
        "/ui/eval/review/score",
        data={"target_id": str(uuid4()), "score": "3"},
        cookies={"conduct_admin": admin_token},
    )
    assert r.status_code == 404
