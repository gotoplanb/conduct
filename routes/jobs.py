from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from rq.exceptions import NoSuchJobError
from rq.job import Job as RQJob
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only, current_client, current_client_or_admin
from config.settings import get_settings
from db.session import get_session
from deps import get_provider_registry
from eval.fanout import FanoutValidationError, run_fanout_secondaries, validate_fanout_targets
from eval.scoring import EvalTokenError, mint_eval_token, redeem_eval_token, score_state
from eval.shadow_runner import enqueue_shadows_for_parent
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus, Sensitivity
from prompt_loader import PromptNotFoundError
from providers.base import ProviderError
from providers.registry import ProviderRegistry, is_cloud
from providers.resident import is_resident
from rate_limit import rate_limited_client
from routing.engine import RoutingDecision, SensitivityViolation, decide
from worker.executor import execute_job
from worker.queue import DEFAULT_JOB_TIMEOUT_S, get_queue, get_redis
from worker.runner import run_job

log = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

_JOB_NOT_FOUND = "job not found"


class JobCreateIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_type: str = Field(min_length=1, max_length=100)
    prompt: str
    system_prompt: str = ""
    sensitivity: Sensitivity | None = None
    priority: int = Field(default=5, ge=1, le=10)
    model: str | None = None
    is_async: bool = Field(default=False, alias="async")
    # Fan-out targets — extra models to run in parallel for real-time eval.
    # Each target must be cloud or in RESIDENT_MODELS (the API can't trigger
    # worker swaps mid-request). Results land in JobShadow rows attached to
    # the primary's Job; the response body is still just the primary's.
    fanout: list[str] = Field(default_factory=list, max_length=10)
    # Per-request override of the rule's eval-shadow sampling: if true, every
    # eligible shadow on the rule fans out for THIS job regardless of `rate`.
    # Handy for "I want the full comparison for this specific input."
    force_shadows: bool = False
    metadata: dict = Field(default_factory=dict)
    # Typed inputs for media tasks. Per-task-type shape (e.g.
    # `{"source_image_url": "/output/abc.png"}` for image→video,
    # `{"source_video_url": "...", "source_audio_url": "..."}` for mux).
    # Text-only tasks leave this empty. The provider decides what's required;
    # missing required inputs surface as task-time errors.
    inputs: dict = Field(default_factory=dict)


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
    eval_url: str | None = None
    # For media tasks: served from the API's /output/ static handler.
    # Text tasks leave this null; clients should check `task_type`'s rule
    # `media_kind` or just look for `media_url` being non-null.
    media_url: str | None = None

    @classmethod
    def from_job(cls, job: Job) -> JobOut:
        base = get_settings().public_base_url.rstrip("/")
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
            eval_url=f"{base}/jobs/{job.id}/eval",
            media_url=job.media_url,
        )


#: Cloud providers Conduct knows how to route to. Each one is gated per-client
#: through the registry; the routing engine consults the resulting set when
#: deciding whether a cloud model is reachable for the requesting client.
_CLOUD_PROVIDERS = ("anthropic", "bedrock")


def _cloud_providers_for(client: ClientApp, providers: ProviderRegistry) -> frozenset[str]:
    return frozenset(
        name for name in _CLOUD_PROVIDERS if providers.has_for_client(client, name)
    )


def _should_enqueue(body: JobCreateIn, decision: RoutingDecision) -> bool:
    """Async path is taken when the client asked for it OR when the target is
    local non-resident. The worker is the sole owner of Ollama inference for
    non-resident models (it does the swaps); resident models can be called
    directly by the API. Cloud calls always run sync. Fan-out forces sync
    so the parallel calls land in one request."""
    if body.fanout:
        return False
    if body.is_async:
        return True
    if decision.provider != "ollama":
        return False
    return not is_resident(decision.model)


def _resolve_decision(
    body: JobCreateIn,
    client: ClientApp,
    rule: RoutingRule | None,
    providers: ProviderRegistry,
) -> RoutingDecision:
    """Run the routing engine, mapping SensitivityViolation to a 400."""
    settings = get_settings()
    requested_sensitivity = body.sensitivity or (
        Sensitivity(rule.sensitivity) if rule else Sensitivity.INTERNAL
    )
    try:
        return decide(
            sensitivity=requested_sensitivity,
            model_requested=body.model,
            allow_cloud_for_internal=client.allow_cloud_for_internal,
            rule=rule,
            default_model=settings.default_model,
            default_sensitive_model=settings.default_sensitive_model,
            available_cloud_providers=_cloud_providers_for(client, providers),
        )
    except SensitivityViolation as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


def _validate_fanout(
    body: JobCreateIn, decision: RoutingDecision, client: ClientApp
) -> None:
    """Every fan-out target (including the primary) must be directly callable
    from the API path — cloud or resident-local. Cloud targets are further
    gated on sensitivity. Raises HTTPException(400) on any violation."""
    if not body.fanout:
        return
    try:
        validate_fanout_targets(body.fanout)
    except FanoutValidationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    if not (is_cloud(decision.model) or is_resident(decision.model)):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"primary model {decision.model!r} is non-resident local — "
            "fanout requires the primary to be cloud or in RESIDENT_MODELS",
        )
    for target in body.fanout:
        if is_cloud(target):
            _check_cloud_target_sensitivity(target, decision, client)


def _check_cloud_target_sensitivity(
    target: str, decision: RoutingDecision, client: ClientApp
) -> None:
    if decision.effective_sensitivity == Sensitivity.CONFIDENTIAL:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"fanout target {target!r} is cloud — disallowed for confidential",
        )
    if (
        decision.effective_sensitivity == Sensitivity.INTERNAL
        and not client.allow_cloud_for_internal
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"fanout target {target!r} is cloud — client lacks allow_cloud_for_internal",
        )


async def _enqueue_for_media_async(job: Job, session: AsyncSession) -> JSONResponse:
    """Media tasks land on the conduct-media RQ queue with the longer
    DEFAULT_MEDIA_JOB_TIMEOUT_S — Wan 2.2 I2V routinely runs 5-30 minutes,
    well past the 10-min default for text jobs. Same shape as the text
    enqueue path otherwise."""
    from worker.queue import (  # noqa: PLC0415
        DEFAULT_MEDIA_JOB_TIMEOUT_S,
        get_media_queue,
    )

    try:
        get_media_queue().enqueue(
            run_job,
            str(job.id),
            job_id=str(job.id),
            job_timeout=DEFAULT_MEDIA_JOB_TIMEOUT_S,
        )
    except Exception as e:
        log.exception("failed to enqueue media job %s", job.id)
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


async def _enqueue_for_async(job: Job, session: AsyncSession) -> JSONResponse:
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


async def _execute_sync(
    *,
    job: Job,
    decision: RoutingDecision,
    body: JobCreateIn,
    client: ClientApp,
    providers: ProviderRegistry,
    session: AsyncSession,
) -> None:
    """Run the primary (and any fan-out secondaries) inline. Translates
    PromptNotFoundError → 500 and ProviderError → 502."""
    import asyncio as _asyncio

    try:
        primary = execute_job(
            job=job,
            decision=decision,
            client=client,
            providers=providers,
            session=session,
        )
        if body.fanout:
            secondaries = run_fanout_secondaries(
                parent=job,
                secondary_models=body.fanout,
                client=client,
                max_tokens=decision.max_tokens,
                providers=providers,
                session=session,
            )
            await _asyncio.gather(primary, secondaries)
        else:
            await primary
    except PromptNotFoundError as e:
        job.status = JobStatus.FAILED.value
        job.error = f"prompt resolution failed: {e}"
        await session.commit()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e)) from e
    except ProviderError as e:
        await session.rollback()
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e)) from e


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
    client: Annotated[ClientApp, Depends(rate_limited_client)],
    session: Annotated[AsyncSession, Depends(get_session)],
    providers: Annotated[ProviderRegistry, Depends(get_provider_registry)],
) -> JSONResponse | JobOut:
    rule = await session.scalar(
        select(RoutingRule).where(
            RoutingRule.task_type == body.task_type,
            RoutingRule.is_archived.is_(False),
        )
    )

    # Media tasks (image/video/audio/mux) bypass the text routing engine
    # entirely — there's no model/provider/sensitivity decision to make on
    # the API side. Just enqueue onto the conduct-media queue and let the
    # worker dispatch via execute_media_job. The worker re-reads the rule
    # to pick the workflow template + provider, so the API doesn't need to.
    if rule is not None and rule.media_kind != "text":
        job = Job(
            client_app_id=client.id,
            task_type=body.task_type,
            sensitivity=Sensitivity(rule.sensitivity).value,
            priority=body.priority,
            prompt=body.prompt,
            system_prompt=body.system_prompt,
            model_requested=body.model or "",
            status=JobStatus.PENDING.value,
            inputs=body.inputs or {},
            job_metadata={**(body.metadata or {})},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return await _enqueue_for_media_async(job, session)

    decision = _resolve_decision(body, client, rule, providers)
    _validate_fanout(body, decision, client)

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
        inputs=body.inputs or {},
        job_metadata={
            **(body.metadata or {}),
            **({"force_shadows": True} if body.force_shadows else {}),
        },
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    if _should_enqueue(body, decision):
        return await _enqueue_for_async(job, session)

    await _execute_sync(
        job=job,
        decision=decision,
        body=body,
        client=client,
        providers=providers,
        session=session,
    )
    # Fan out the rule's eval shadows once the primary lands successfully —
    # mirrors the MCP path (mcp_server.create_job). Without this, sync HTTP
    # jobs silently skipped every shadow in their rule. We don't fan out on
    # failure (no point comparing against a broken primary), and we don't
    # block the response on shadow enqueueing — RQ takes them from here.
    if job.status == JobStatus.COMPLETE.value:
        await enqueue_shadows_for_parent(
            parent_job=job, rule=rule, client=client, session=session
        )
    return JobOut.from_job(job)


class JobListItem(BaseModel):
    job_id: UUID
    task_type: str
    status: JobStatus
    client_app: str
    model_used: str | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    created_at: datetime
    avg_score: float | None = None
    score_count: int = 0


class JobListOut(BaseModel):
    jobs: list[JobListItem]


def _in_score_range(avg: float | None, lo: float | None, hi: float | None) -> bool:
    if avg is None:
        return False
    return (lo is None or avg >= lo) and (hi is None or avg <= hi)


@router.get("", dependencies=[Depends(admin_only)])
async def list_jobs(
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    job_status: Annotated[str | None, Query(alias="status")] = None,
    q: Annotated[str | None, Query()] = None,
    min_score: Annotated[float | None, Query(ge=1, le=5)] = None,
    max_score: Annotated[float | None, Query(ge=1, le=5)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> JobListOut:
    """Admin: list recent jobs across all clients, newest first. With
    min_score/max_score, only jobs whose average quality score falls in range
    are returned (unscored jobs are excluded) — handy for triaging mediocre
    outputs."""
    has_score_filter = min_score is not None or max_score is not None
    # Score is a JSON aggregate, so it's filtered in Python; over-fetch when a
    # score filter is set so the post-filter result can still fill `limit`.
    fetch_cap = 500 if has_score_filter else limit
    stmt = select(Job).order_by(Job.created_at.desc()).limit(fetch_cap)
    if task_type:
        stmt = stmt.where(Job.task_type == task_type)
    if job_status:
        stmt = stmt.where(Job.status == job_status)
    if q:
        stmt = stmt.where(Job.prompt.ilike(f"%{q}%"))
    rows = (await session.scalars(stmt)).all()

    client_ids = {j.client_app_id for j in rows}
    names = (
        {
            c.id: c.name
            for c in (
                await session.scalars(select(ClientApp).where(ClientApp.id.in_(client_ids)))
            ).all()
        }
        if client_ids
        else {}
    )

    out: list[JobListItem] = []
    for j in rows:
        st = score_state((j.job_metadata or {}).get("quality_scores", []))
        if has_score_filter and not _in_score_range(st["avg"], min_score, max_score):
            continue
        out.append(
            JobListItem(
                job_id=j.id,
                task_type=j.task_type,
                status=JobStatus(j.status),
                client_app=names.get(j.client_app_id, "?"),
                model_used=j.model_used or None,
                cost_usd=j.cost_usd,
                latency_ms=j.latency_ms,
                created_at=j.created_at,
                avg_score=st["avg"],
                score_count=st["count"],
            )
        )
        if len(out) >= limit:
            break
    return JobListOut(jobs=out)


@router.get("/{job_id}")
async def get_job(
    job_id: UUID,
    principal: Annotated[ClientApp | None, Depends(current_client_or_admin)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobOut:
    job = await session.get(Job, job_id)
    # Admin (principal is None) sees any job; a client sees only its own.
    if job is None or (principal is not None and job.client_app_id != principal.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, _JOB_NOT_FOUND)
    return JobOut.from_job(job)


class ShadowOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model: str
    provider: str
    status: str
    response: str | None
    error: str | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: Decimal | None
    latency_ms: int | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ShadowsOut(BaseModel):
    parent_job_id: UUID
    shadows: list[ShadowOut]


@router.get("/{job_id}/shadows")
async def list_job_shadows(
    job_id: UUID,
    principal: Annotated[ClientApp | None, Depends(current_client_or_admin)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ShadowsOut:
    """List the eval shadows fanned out for a parent job. Admin sees any
    parent's shadows; a client sees only its own."""
    job = await session.get(Job, job_id)
    if job is None or (principal is not None and job.client_app_id != principal.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, _JOB_NOT_FOUND)
    rows = (
        await session.scalars(
            select(JobShadow)
            .where(JobShadow.parent_job_id == job.id)
            .order_by(JobShadow.created_at.asc())
        )
    ).all()
    return ShadowsOut(
        parent_job_id=job.id,
        shadows=[ShadowOut.model_validate(s) for s in rows],
    )


class EvalLinkOut(BaseModel):
    job_id: UUID
    eval_url: str
    eval_token: str
    expires_at: datetime


@router.post("/{job_id}/eval-link")
async def create_eval_link(
    job_id: UUID,
    principal: Annotated[ClientApp | None, Depends(current_client_or_admin)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EvalLinkOut:
    """Mint a single-use scoring token for a job (owner or admin). Hand the
    returned URL + token to a credential-less rater (e.g. a portal link)."""
    job = await session.get(Job, job_id)
    if job is None or (principal is not None and job.client_app_id != principal.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, _JOB_NOT_FOUND)
    raw, expires = await mint_eval_token(session, job)
    base = get_settings().public_base_url.rstrip("/")
    return EvalLinkOut(
        job_id=job.id, eval_url=f"{base}/jobs/{job.id}/eval", eval_token=raw, expires_at=expires
    )


class EvalSubmitIn(BaseModel):
    eval_token: str
    score: int = Field(ge=1, le=5)
    note: str | None = Field(default=None, max_length=500)


class EvalSubmitOut(BaseModel):
    job_id: UUID
    score: int
    recorded: bool


@router.post("/{job_id}/eval")
async def submit_eval(
    job_id: UUID,
    body: EvalSubmitIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> EvalSubmitOut:
    """Submit a 1-5 score for a job using its single-use eval token. No bearer
    auth — the token in the body is the credential."""
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _JOB_NOT_FOUND)
    try:
        await redeem_eval_token(
            session, job, raw_token=body.eval_token, score=body.score, note=body.note
        )
    except EvalTokenError as e:
        raise HTTPException(e.status, e.message) from e
    return EvalSubmitOut(job_id=job.id, score=body.score, recorded=True)


@router.delete("/{job_id}")
async def cancel_job(
    job_id: UUID,
    client: Annotated[ClientApp, Depends(current_client)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobOut:
    job = await session.get(Job, job_id)
    if job is None or job.client_app_id != client.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _JOB_NOT_FOUND)
    if job.status == JobStatus.RUNNING.value:
        # A live worker still owns the job → refuse. But if the RQ record is
        # gone (worker host crashed, AbandonedJobError fired, TTL expired),
        # the row is orphaned and safe to flip → cancelled.
        try:
            RQJob.fetch(str(job.id), connection=get_redis())
        except NoSuchJobError:
            job.status = JobStatus.CANCELLED.value
            await session.commit()
            await session.refresh(job)
            return JobOut.from_job(job)
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
