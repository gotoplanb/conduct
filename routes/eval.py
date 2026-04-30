"""Eval endpoints: side-by-side model comparison + optional manual quality scoring.

`/eval/compare` powers the routing-decision feedback loop — once 10+ jobs of a
task type have run, the tradeoffs are visible. The score endpoint is the human
override on top of the automatic latency/cost/failure signals.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from models.job import Job
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


class CompareOut(BaseModel):
    task_type: str
    period_days: int
    models: list[ModelEval]


@router.get("/compare", response_model=CompareOut)
async def compare(
    task_type: str = Query(..., min_length=1),
    days: int = Query(default=30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> CompareOut:
    since = datetime.now(UTC) - timedelta(days=days)

    is_complete = case((Job.status == JobStatus.COMPLETE.value, 1), else_=0)
    is_failed = case((Job.status == JobStatus.FAILED.value, 1), else_=0)

    rows = (
        await session.execute(
            select(
                Job.model_used,
                func.count().label("attempts"),
                func.sum(is_complete).label("successes"),
                func.sum(is_failed).label("failures"),
                func.avg(Job.latency_ms).filter(Job.status == JobStatus.COMPLETE.value),
                func.avg(Job.tokens_out).filter(Job.status == JobStatus.COMPLETE.value),
                func.coalesce(
                    func.sum(Job.cost_usd).filter(Job.status == JobStatus.COMPLETE.value),
                    0,
                ),
            )
            .where(
                Job.task_type == task_type,
                Job.created_at >= since,
                Job.model_used != "",
            )
            .group_by(Job.model_used)
            .order_by(func.count().desc())
        )
    ).all()

    models: list[ModelEval] = []
    for model, attempts, successes, failures, avg_latency, avg_tok_out, total_cost in rows:
        attempts_i = int(attempts or 0)
        successes_i = int(successes or 0)
        failures_i = int(failures or 0)
        cost_per_job = float(total_cost or 0) / successes_i if successes_i else 0.0
        models.append(
            ModelEval(
                model=model,
                job_count=attempts_i,
                success_count=successes_i,
                failure_count=failures_i,
                failure_rate=(failures_i / attempts_i) if attempts_i else 0.0,
                avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
                avg_tokens_out=float(avg_tok_out) if avg_tok_out is not None else None,
                cost_per_job_usd=cost_per_job,
            )
        )

    return CompareOut(task_type=task_type, period_days=days, models=models)


class ScoreIn(BaseModel):
    score: int = Field(ge=1, le=5, description="Quality rating 1-5")
    reviewer: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=500)


@router.post("/jobs/{job_id}/score")
async def score_job(
    job_id: UUID,
    body: ScoreIn,
    session: AsyncSession = Depends(get_session),
) -> dict:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    quality = job.job_metadata.get("quality_scores", []) if job.job_metadata else []
    quality.append(
        {
            "score": body.score,
            "reviewer": body.reviewer or "",
            "note": body.note or "",
            "at": datetime.now(UTC).isoformat(),
        }
    )
    job.job_metadata = {**(job.job_metadata or {}), "quality_scores": quality}
    await session.commit()
    return {"job_id": str(job_id), "scores": quality}
