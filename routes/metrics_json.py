"""JSON metrics aggregator. Distinct from Prometheus scrape — built for dashboards
and ad-hoc queries with filters. Admin-only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus

router = APIRouter(tags=["observability"], dependencies=[Depends(admin_only)])


class ModelStats(BaseModel):
    count: int
    avg_latency_ms: float | None
    total_cost_usd: float


class TaskTypeStats(BaseModel):
    count: int
    preferred_model: str | None


class MetricsOut(BaseModel):
    period_days: int
    total_jobs: int
    jobs_by_status: dict[str, int]
    jobs_by_model: dict[str, ModelStats]
    jobs_by_task_type: dict[str, TaskTypeStats]


@router.get("/metrics", response_model=MetricsOut)
async def metrics(
    days: int = Query(default=30, ge=1, le=365),
    client_app_id: UUID | None = None,
    task_type: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> MetricsOut:
    since = datetime.now(UTC) - timedelta(days=days)

    base_filters = [Job.created_at >= since]
    if client_app_id is not None:
        base_filters.append(Job.client_app_id == client_app_id)
    if task_type is not None:
        base_filters.append(Job.task_type == task_type)

    total = (
        await session.execute(select(func.count()).select_from(Job).where(*base_filters))
    ).scalar_one()

    by_status_rows = (
        await session.execute(
            select(Job.status, func.count()).where(*base_filters).group_by(Job.status)
        )
    ).all()
    jobs_by_status = {status: int(count) for status, count in by_status_rows}

    # Per-model stats: count from all (including failures, attributable since
    # we set model_used eagerly in execute_job), latency/cost from completes only.
    by_model_rows = (
        await session.execute(
            select(
                Job.model_used,
                func.count(),
                func.avg(Job.latency_ms).filter(Job.status == JobStatus.COMPLETE.value),
                func.coalesce(
                    func.sum(Job.cost_usd).filter(Job.status == JobStatus.COMPLETE.value),
                    0,
                ),
            )
            .where(*base_filters, Job.model_used != "")
            .group_by(Job.model_used)
        )
    ).all()
    jobs_by_model: dict[str, ModelStats] = {}
    for model, count, avg_latency, total_cost in by_model_rows:
        jobs_by_model[model] = ModelStats(
            count=int(count),
            avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
            total_cost_usd=float(total_cost or 0),
        )

    # Per-task-type stats with preferred model from RoutingRule.
    by_task_rows = (
        await session.execute(
            select(Job.task_type, func.count()).where(*base_filters).group_by(Job.task_type)
        )
    ).all()
    rules = {r.task_type: r for r in (await session.scalars(select(RoutingRule))).all()}
    jobs_by_task_type: dict[str, TaskTypeStats] = {}
    for tt, count in by_task_rows:
        rule = rules.get(tt)
        jobs_by_task_type[tt] = TaskTypeStats(
            count=int(count),
            preferred_model=rule.preferred_model if rule else None,
        )

    return MetricsOut(
        period_days=days,
        total_jobs=int(total),
        jobs_by_status=jobs_by_status,
        jobs_by_model=jobs_by_model,
        jobs_by_task_type=jobs_by_task_type,
    )
