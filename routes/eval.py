"""Eval endpoints: side-by-side model comparison + manual quality scoring.

`/eval/compare` aggregates per-model performance across both real jobs and
their shadows — the routing-decision feedback loop. `/eval/review` surfaces
unscored shadow responses for human rating. The score endpoint accepts
either a Job ID or a JobShadow ID (ID-discriminated, single URL).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
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


def _aggregate_metadata_scores(
    pairs: list[tuple[str, dict | None]],
) -> tuple[dict[str, float], dict[str, int]]:
    """Walk metadata.quality_scores entries, return per-model avg + count."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for model, meta in pairs:
        if not model:
            continue
        scores = (meta or {}).get("quality_scores", [])
        for entry in scores:
            try:
                value = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            sums[model] += value
            counts[model] += 1
    avgs = {m: sums[m] / counts[m] for m in counts}
    return avgs, dict(counts)


@router.get("/compare")
async def compare(
    task_type: str = Query(..., min_length=1),
    days: int = Query(default=30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> CompareOut:
    since = datetime.now(UTC) - timedelta(days=days)

    job_is_complete = case((Job.status == JobStatus.COMPLETE.value, 1), else_=0)
    job_is_failed = case((Job.status == JobStatus.FAILED.value, 1), else_=0)
    shadow_is_complete = case((JobShadow.status == JobStatus.COMPLETE.value, 1), else_=0)
    shadow_is_failed = case((JobShadow.status == JobStatus.FAILED.value, 1), else_=0)

    job_rows = (
        await session.execute(
            select(
                Job.model_used.label("model"),
                func.count().label("attempts"),
                func.sum(job_is_complete).label("successes"),
                func.sum(job_is_failed).label("failures"),
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
        )
    ).all()

    # Shadows: join to parent for the task_type filter.
    shadow_rows = (
        await session.execute(
            select(
                JobShadow.model.label("model"),
                func.count().label("attempts"),
                func.sum(shadow_is_complete).label("successes"),
                func.sum(shadow_is_failed).label("failures"),
                func.avg(JobShadow.latency_ms).filter(JobShadow.status == JobStatus.COMPLETE.value),
                func.avg(JobShadow.tokens_out).filter(JobShadow.status == JobStatus.COMPLETE.value),
                func.coalesce(
                    func.sum(JobShadow.cost_usd).filter(
                        JobShadow.status == JobStatus.COMPLETE.value
                    ),
                    0,
                ),
            )
            .join(Job, Job.id == JobShadow.parent_job_id)
            .where(
                Job.task_type == task_type,
                JobShadow.created_at >= since,
            )
            .group_by(JobShadow.model)
        )
    ).all()

    # Merge job + shadow stats per model.
    rolled: dict[str, dict] = {}
    for source_rows in (job_rows, shadow_rows):
        for row in source_rows:
            model, attempts, successes, failures, avg_latency, avg_tok_out, total_cost = row
            entry = rolled.setdefault(
                model,
                {
                    "attempts": 0,
                    "successes": 0,
                    "failures": 0,
                    "latency_sum": 0.0,
                    "latency_count": 0,
                    "tokens_sum": 0.0,
                    "tokens_count": 0,
                    "cost_total": 0.0,
                },
            )
            attempts_i = int(attempts or 0)
            successes_i = int(successes or 0)
            entry["attempts"] += attempts_i
            entry["successes"] += successes_i
            entry["failures"] += int(failures or 0)
            entry["cost_total"] += float(total_cost or 0)
            if avg_latency is not None and successes_i:
                entry["latency_sum"] += float(avg_latency) * successes_i
                entry["latency_count"] += successes_i
            if avg_tok_out is not None and successes_i:
                entry["tokens_sum"] += float(avg_tok_out) * successes_i
                entry["tokens_count"] += successes_i

    # Score rollup — single pass over both tables' metadata blobs.
    score_pairs: list[tuple[str, dict | None]] = []
    score_pairs.extend(
        (m, meta)
        for m, meta in (
            await session.execute(
                select(Job.model_used, Job.job_metadata).where(
                    Job.task_type == task_type,
                    Job.created_at >= since,
                    Job.model_used != "",
                )
            )
        ).all()
    )
    score_pairs.extend(
        (m, meta)
        for m, meta in (
            await session.execute(
                select(JobShadow.model, JobShadow.shadow_metadata)
                .join(Job, Job.id == JobShadow.parent_job_id)
                .where(Job.task_type == task_type, JobShadow.created_at >= since)
            )
        ).all()
    )
    avg_scores, score_counts = _aggregate_metadata_scores(score_pairs)

    models: list[ModelEval] = []
    for model, e in sorted(rolled.items(), key=lambda kv: -kv[1]["attempts"]):
        successes = e["successes"]
        cost_per_job = (e["cost_total"] / successes) if successes else 0.0
        avg_lat = (e["latency_sum"] / e["latency_count"]) if e["latency_count"] else None
        avg_tok = (e["tokens_sum"] / e["tokens_count"]) if e["tokens_count"] else None
        models.append(
            ModelEval(
                model=model,
                job_count=e["attempts"],
                success_count=successes,
                failure_count=e["failures"],
                failure_rate=(e["failures"] / e["attempts"]) if e["attempts"] else 0.0,
                avg_latency_ms=avg_lat,
                avg_tokens_out=avg_tok,
                cost_per_job_usd=cost_per_job,
                avg_score=avg_scores.get(model),
                score_count=score_counts.get(model, 0),
            )
        )

    return CompareOut(task_type=task_type, period_days=days, models=models)


class ScoreIn(BaseModel):
    score: int = Field(ge=1, le=5, description="Quality rating 1-5")
    reviewer: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=500)


@router.post("/jobs/{target_id}/score")
async def score_target(
    target_id: UUID,
    body: ScoreIn,
    session: AsyncSession = Depends(get_session),
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
    task_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
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
