"""Worker dispatch safety net (#45 follow-up): a dispatch exception must mark
the job FAILED, never leave it silently PENDING.

This is the latent bug a stale-worker dpo_fine_tune job surfaced — the worker
hit PromptNotFoundError, it escaped execute_job, and the Job row stayed
'pending' forever. _run_async now wraps dispatch and fails the job.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import worker.runner as runner
from models.job import Job
from models.types import JobStatus


@pytest.fixture
def worker_sessionmaker(db_conn, monkeypatch):
    maker = async_sessionmaker(
        bind=db_conn, expire_on_commit=False,
        join_transaction_mode="create_savepoint", class_=AsyncSession,
    )
    monkeypatch.setattr(runner, "get_worker_session_maker", lambda: maker)
    return maker


async def _seed(db, client_id, status=JobStatus.PENDING.value) -> Job:
    job = Job(
        client_app_id=client_id, task_type="dpo_fine_tune", sensitivity="internal",
        prompt="", status=status,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_dispatch_exception_marks_job_failed_not_pending(
    db_session: AsyncSession, seeded_client, worker_sessionmaker, monkeypatch
) -> None:
    job = await _seed(db_session, seeded_client[0].id)

    async def boom(_job_id):
        raise RuntimeError("kaboom in dispatch")

    monkeypatch.setattr(runner, "_dispatch_job", boom)

    with pytest.raises(RuntimeError, match="kaboom"):  # re-raised for RQ
        await runner._run_async(job.id)

    await db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value  # not stuck pending
    assert "worker dispatch error" in job.error and "kaboom" in job.error
    assert job.completed_at is not None


async def test_safe_marker_leaves_finished_jobs_alone(
    db_session: AsyncSession, seeded_client, worker_sessionmaker
) -> None:
    # A job that already completed must not be clobbered to failed.
    job = await _seed(db_session, seeded_client[0].id, status=JobStatus.COMPLETE.value)
    await runner._mark_job_failed_safe(job.id, "should be ignored")
    await db_session.refresh(job)
    assert job.status == JobStatus.COMPLETE.value


async def test_safe_marker_never_raises_on_missing_job(worker_sessionmaker) -> None:
    from uuid import uuid4
    # Unknown id → best-effort no-op, no exception.
    await runner._mark_job_failed_safe(uuid4(), "nope")
