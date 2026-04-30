from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from rq.exceptions import NoSuchJobError
from rq.job import Job as RQJob
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import current_client
from config.settings import get_settings
from db.session import get_session
from deps import get_provider_registry
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus, Sensitivity
from prompt_loader import PromptNotFoundError
from providers.base import ProviderError
from providers.registry import ProviderRegistry
from rate_limit import rate_limited_client
from routing.engine import RoutingDecision, SensitivityViolation, decide
from worker.executor import execute_job
from worker.queue import DEFAULT_JOB_TIMEOUT_S, get_queue, get_redis
from worker.runner import run_job

log = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobCreateIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_type: str = Field(min_length=1, max_length=100)
    prompt: str
    system_prompt: str = ""
    sensitivity: Sensitivity | None = None
    priority: int = Field(default=5, ge=1, le=10)
    model: str | None = None
    is_async: bool = Field(default=False, alias="async")
    metadata: dict = Field(default_factory=dict)


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: UUID
    status: JobStatus
    task_type: str
    sensitivity: Sensitivity
    response: str | None = None
    model_used: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict = {}

    @classmethod
    def from_job(cls, job: Job) -> JobOut:
        return cls(
            job_id=job.id,
            status=JobStatus(job.status),
            task_type=job.task_type,
            sensitivity=Sensitivity(job.sensitivity),
            response=job.response or None,
            model_used=job.model_used or None,
            tokens_in=job.tokens_in,
            tokens_out=job.tokens_out,
            cost_usd=job.cost_usd,
            latency_ms=job.latency_ms,
            error=job.error or None,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            metadata=job.job_metadata or {},
        )


def _should_enqueue(body: JobCreateIn, decision: RoutingDecision) -> bool:
    """Async path is taken when the client asked for it OR when the target is
    local. The worker is the sole owner of Ollama inference (and model swaps),
    so all local jobs flow through the queue. Cloud calls run sync."""
    if body.is_async:
        return True
    return decision.provider == "ollama"


@router.post(
    "",
    response_model=None,
    responses={
        200: {"model": JobOut, "description": "sync result"},
        202: {"description": "queued for async execution"},
    },
)
async def submit_job(
    body: JobCreateIn,
    client: ClientApp = Depends(rate_limited_client),
    session: AsyncSession = Depends(get_session),
    providers: ProviderRegistry = Depends(get_provider_registry),
) -> JSONResponse | JobOut:
    rule = await session.scalar(select(RoutingRule).where(RoutingRule.task_type == body.task_type))
    requested_sensitivity = body.sensitivity or (
        Sensitivity(rule.sensitivity) if rule else Sensitivity.INTERNAL
    )

    settings = get_settings()
    try:
        decision = decide(
            sensitivity=requested_sensitivity,
            model_requested=body.model,
            allow_cloud_for_internal=client.allow_cloud_for_internal,
            rule=rule,
            default_model=settings.default_model,
            default_sensitive_model=settings.default_sensitive_model,
        )
    except SensitivityViolation as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    if not providers.has(decision.provider) and not _should_enqueue(body, decision):
        # Sync path needs the provider in-process; async path defers the check
        # to the worker (which builds its own registry).
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"provider {decision.provider} not configured (missing API key?)",
        )

    job = Job(
        client_app_id=client.id,
        task_type=body.task_type,
        sensitivity=decision.effective_sensitivity.value,
        priority=body.priority,
        prompt=body.prompt,
        system_prompt=body.system_prompt,
        model_requested=body.model or "",
        status=JobStatus.PENDING.value,
        job_metadata=body.metadata or {},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    if _should_enqueue(body, decision):
        try:
            get_queue().enqueue(
                run_job,
                str(job.id),
                job_id=str(job.id),
                job_timeout=DEFAULT_JOB_TIMEOUT_S,
            )
        except Exception as e:
            # If Redis is down, we still have the row — flip it to failed so the
            # client gets a definite signal rather than a perpetually-pending job.
            log.exception("failed to enqueue job %s", job.id)
            job.status = JobStatus.FAILED.value
            job.error = f"enqueue failed: {e}"
            await session.commit()
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "queue backend unavailable"
            ) from e
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": str(job.id),
                "status": JobStatus.PENDING.value,
                "poll_url": f"/jobs/{job.id}",
            },
        )

    # Sync path (cloud only).
    try:
        await execute_job(
            job=job,
            decision=decision,
            client_name=client.name,
            providers=providers,
            session=session,
        )
    except PromptNotFoundError as e:
        job.status = JobStatus.FAILED.value
        job.error = f"prompt resolution failed: {e}"
        await session.commit()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e)) from e
    except ProviderError as e:
        await session.rollback()
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e

    return JobOut.from_job(job)


@router.get("/{job_id}", response_model=JobOut)
async def get_job(
    job_id: UUID,
    client: ClientApp = Depends(current_client),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    job = await session.get(Job, job_id)
    if job is None or job.client_app_id != client.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return JobOut.from_job(job)


@router.delete("/{job_id}", response_model=JobOut)
async def cancel_job(
    job_id: UUID,
    client: ClientApp = Depends(current_client),
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    job = await session.get(Job, job_id)
    if job is None or job.client_app_id != client.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    if job.status == JobStatus.RUNNING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "cannot cancel a running job")
    if job.status == JobStatus.PENDING.value:
        try:
            rq_job = RQJob.fetch(str(job.id), connection=get_redis())
            rq_job.cancel()
            rq_job.delete()
        except NoSuchJobError:
            # Worker may have already pulled it, or RQ TTL expired the record.
            pass
        except Exception:
            # Don't fail the cancel API on Redis hiccups — DB is source of truth.
            log.exception("rq cancel/delete failed for job %s", job.id)
        job.status = JobStatus.CANCELLED.value
        await session.commit()
        await session.refresh(job)
    return JobOut.from_job(job)
