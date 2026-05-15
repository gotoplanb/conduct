"""Eval endpoints: side-by-side model comparison + manual quality scoring.

`/eval/compare` aggregates per-model performance across both real jobs and
their shadows — the routing-decision feedback loop. `/eval/review` surfaces
unscored shadow responses for human rating. The score endpoint accepts
either a Job ID or a JobShadow ID (ID-discriminated, single URL).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from eval.rollup import compute_rollup
from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus

router = APIRouter(prefix="/eval", tags=["eval"], dependencies=[Depends(admin_only)])


class ModelEval(BaseModel):
    model: str
    job_count: int
    success_count: int
    failure_count: int
    failure_rate: float
    avg_latency_ms: float | None
    avg_tokens_out: float | None
    cost_per_job_usd: float
    avg_score: float | None = None
    score_count: int = 0


class CompareOut(BaseModel):
    task_type: str
    period_days: int
    models: list[ModelEval]


@router.get("/compare")
async def compare(
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str, Query(min_length=1)],
    days: Annotated[int, Query(ge=1, le=365)] = 30,
) -> CompareOut:
    rows = await compute_rollup(session, task_type=task_type, days=days)
    return CompareOut(
        task_type=task_type,
        period_days=days,
        models=[ModelEval(**row) for row in rows],
    )


class ScoreIn(BaseModel):
    score: int = Field(ge=1, le=5, description="Quality rating 1-5")
    reviewer: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=500)


@router.post("/jobs/{target_id}/score")
async def score_target(
    target_id: UUID,
    body: ScoreIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """Score either a Job or a JobShadow. The URL stays the same; we look
    up by ID in jobs first, then job_shadows. (UUIDs don't collide across
    tables.)"""
    entry = {
        "score": body.score,
        "reviewer": body.reviewer or "",
        "note": body.note or "",
        "at": datetime.now(UTC).isoformat(),
    }

    job = await session.get(Job, target_id)
    if job is not None:
        existing = job.job_metadata.get("quality_scores", []) if job.job_metadata else []
        existing.append(entry)
        job.job_metadata = {**(job.job_metadata or {}), "quality_scores": existing}
        await session.commit()
        return {"kind": "job", "id": str(target_id), "scores": existing}

    shadow = await session.get(JobShadow, target_id)
    if shadow is not None:
        existing = (
            shadow.shadow_metadata.get("quality_scores", []) if shadow.shadow_metadata else []
        )
        existing.append(entry)
        shadow.shadow_metadata = {
            **(shadow.shadow_metadata or {}),
            "quality_scores": existing,
        }
        await session.commit()
        return {"kind": "shadow", "id": str(target_id), "scores": existing}

    raise HTTPException(status.HTTP_404_NOT_FOUND, "no job or shadow with that id")


class ReviewItem(BaseModel):
    parent_job_id: UUID
    shadow_id: UUID
    task_type: str
    model: str
    prompt: str
    response: str
    created_at: datetime


class ReviewOut(BaseModel):
    items: list[ReviewItem]


@router.get("/review")
async def review_queue(
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ReviewOut:
    """Return completed shadow rows that haven't been scored yet, joined with
    their parent for the task_type and prompt content. Oldest-first."""
    stmt = (
        select(JobShadow, Job)
        .join(Job, Job.id == JobShadow.parent_job_id)
        .where(JobShadow.status == JobStatus.COMPLETE.value)
        .order_by(JobShadow.created_at.asc())
    )
    if task_type:
        stmt = stmt.where(Job.task_type == task_type)
    # We over-fetch and filter unscored in Python — `metadata.quality_scores`
    # is a JSON array; expressing "empty or missing" cleanly in SQLAlchemy
    # core would clutter the query. The N here is bounded by `limit * ~3`.
    rows = (await session.execute(stmt.limit(limit * 3))).all()

    items: list[ReviewItem] = []
    for shadow, parent in rows:
        scores = (shadow.shadow_metadata or {}).get("quality_scores", [])
        if scores:
            continue
        items.append(
            ReviewItem(
                parent_job_id=parent.id,
                shadow_id=shadow.id,
                task_type=parent.task_type,
                model=shadow.model,
                prompt=parent.prompt,
                response=shadow.response,
                created_at=shadow.created_at,
            )
        )
        if len(items) >= limit:
            break
    return ReviewOut(items=items)
