"""Conduct MCP server — the remote tool surface for Claude custom connectors.

A FastMCP (Streamable HTTP, stateless) app exposing a small set of job tools.
It's mounted at /mcp by main.py and gated by an OAuth bearer token: the ASGI
middleware below resolves the token to a ClientApp and stashes it in a
contextvar, so every tool acts as that client (jobs are attributed to it).

Tools open their own DB sessions (they run outside the FastAPI request /
dependency system). create_job always enqueues async — the natural pattern
for a phone: create, then poll get_job.
"""

from __future__ import annotations

import contextvars
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import select

from config.settings import get_settings
from db.session import SessionLocal
from eval.scoring import apply_score
from eval.shadow_runner import enqueue_shadows_for_parent
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus, Sensitivity
from oauth_provider import resolve_access_token
from prompt_loader import PromptNotFoundError
from providers.registry import ProviderRegistry
from providers.resident import is_resident
from routing.engine import RoutingDecision, SensitivityViolation, decide
from worker.executor import execute_job, is_judge_job
from worker.queue import DEFAULT_JOB_TIMEOUT_S, get_queue
from worker.runner import run_job

# Set per-request by the OAuth middleware; read by the tools.
_principal: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "mcp_principal", default=None
)

# The API's provider registry, handed in from the app lifespan. Lets sync-
# eligible jobs run inline in this process. None until the lifespan sets it
# (and in tests unless explicitly provided).
_provider_registry: ProviderRegistry | None = None

_BAD_JOB_UUID = "job_id is not a valid UUID"
_NO_JOB_FOR_ID = "no job with that id"

#: Same list as routes/jobs._CLOUD_PROVIDERS — duplicated here to keep mcp_server
#: importable without pulling in the FastAPI route module.
_CLOUD_PROVIDERS = ("anthropic", "bedrock")


def _cloud_providers_for_principal(client: ClientApp) -> frozenset[str]:
    if _provider_registry is None:
        return frozenset()
    return frozenset(
        name
        for name in _CLOUD_PROVIDERS
        if _provider_registry.has_for_client(client, name)
    )


def set_provider_registry(registry: ProviderRegistry) -> None:
    global _provider_registry
    _provider_registry = registry


def _sync_eligible(decision: RoutingDecision) -> bool:
    """The API can run a job inline only for cloud models or resident local
    models (the worker owns Ollama swaps for everything else)."""
    if decision.provider != "ollama":
        return True
    return is_resident(decision.model)


def _transport_security() -> TransportSecuritySettings:
    """The SDK's DNS-rebinding guard defaults to localhost-only, which 421s
    every request once we're behind a public host. Allow our public origin
    (from CONDUCT_PUBLIC_URL) plus localhost for local dev."""
    host = urlparse(get_settings().public_base_url.rstrip("/")).netloc or "localhost:8000"
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[host, f"{host}:*", "localhost:*", "127.0.0.1:*", "[::1]:*"],
        allowed_origins=[
            get_settings().public_base_url.rstrip("/"),
            "http://localhost:*",
            "http://127.0.0.1:*",
        ],
    )


mcp = FastMCP(
    "Conduct",
    stateless_http=True,
    streamable_http_path="/",
    json_response=False,
    transport_security=_transport_security(),
)


def _client_app_id() -> UUID:
    principal = _principal.get()
    if principal is None:
        raise ValueError("not authenticated")
    return principal["client_app_id"]


def _job_summary(job: Job) -> dict[str, Any]:
    return {
        "job_id": str(job.id),
        "task_type": job.task_type,
        "status": job.status,
        "model_used": job.model_used or None,
        "created_at": job.created_at.isoformat(),
    }


def _job_detail(job: Job) -> dict[str, Any]:
    return {
        **_job_summary(job),
        "prompt": job.prompt,
        "response": job.response or None,
        "error": job.error or None,
        "tokens_in": job.tokens_in,
        "tokens_out": job.tokens_out,
        "cost_usd": float(job.cost_usd) if isinstance(job.cost_usd, Decimal) else job.cost_usd,
        "latency_ms": job.latency_ms,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        # Echo the typed-input bag so clients can confirm what the server
        # received (e.g. that a judge's target_job_id actually arrived).
        "inputs": job.inputs or None,
        # Media tasks return URLs instead of text. Clients render audio/video
        # by fetching `media_url` from the API's /output handler; text tasks
        # leave it null.
        "media_url": job.media_url,
    }


async def _create_media_job(
    *, client, rule, task_type, prompt, system_prompt, inputs
) -> dict[str, Any]:
    """Media task path — skip the routing engine, enqueue onto the
    conduct-media queue with the longer media job timeout."""
    from worker.queue import (  # noqa: PLC0415
        DEFAULT_MEDIA_JOB_TIMEOUT_S,
        get_media_queue,
    )

    async with SessionLocal() as session:
        job = Job(
            client_app_id=client.id,
            task_type=task_type,
            sensitivity=rule.sensitivity,
            priority=5,
            prompt=prompt,
            system_prompt=system_prompt,
            model_requested="",
            status=JobStatus.PENDING.value,
            inputs=inputs,
            job_metadata={},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = str(job.id)

    get_media_queue().enqueue(
        run_job, job_id, job_id=job_id, job_timeout=DEFAULT_MEDIA_JOB_TIMEOUT_S
    )
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING.value,
        "note": "queued on conduct-media — call get_job(job_id) for the result",
    }


@mcp.tool()
async def list_task_types() -> list[dict[str, Any]]:
    """List the task types this Conduct instance can run, with the model each
    routes to and its sensitivity floor. Use these as the `task_type` for
    create_job."""
    async with SessionLocal() as session:
        rules = (
            await session.scalars(
                select(RoutingRule)
                .where(RoutingRule.is_archived.is_(False))
                .order_by(RoutingRule.task_type)
            )
        ).all()
    return [
        {
            "task_type": r.task_type,
            "preferred_model": r.preferred_model,
            "sensitivity": r.sensitivity,
        }
        for r in rules
    ]


@mcp.tool()
async def list_jobs(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    """List your recent jobs, newest first. Optionally filter by status
    (pending, running, complete, failed, cancelled)."""
    client_app_id = _client_app_id()
    capped = min(max(limit, 1), 100)
    async with SessionLocal() as session:
        stmt = (
            select(Job)
            .where(Job.client_app_id == client_app_id)
            .order_by(Job.created_at.desc())
            .limit(capped)
        )
        if status:
            stmt = stmt.where(Job.status == status)
        jobs = (await session.scalars(stmt)).all()
    return [_job_summary(j) for j in jobs]


@mcp.tool()
async def get_job(job_id: str) -> dict[str, Any]:
    """Get a single job's status and result (response, tokens, cost) by id."""
    client_app_id = _client_app_id()
    try:
        target = UUID(job_id)
    except ValueError as e:
        raise ValueError(_BAD_JOB_UUID) from e
    async with SessionLocal() as session:
        job = await session.get(Job, target)
        if job is None or job.client_app_id != client_app_id:
            raise ValueError(_NO_JOB_FOR_ID)
        return _job_detail(job)


@mcp.tool()
async def list_shadows(job_id: str) -> dict[str, Any]:
    """Get the eval shadows for one of your jobs — the side-by-side responses
    from candidate models the routing rule sampled in parallel with the
    primary. Each shadow has its own `model`, `response`, `latency_ms`,
    `cost_usd`, and `status`. Useful for comparing models on the same input
    before calling submit_eval on the parent job."""
    client_app_id = _client_app_id()
    try:
        target = UUID(job_id)
    except ValueError as e:
        raise ValueError(_BAD_JOB_UUID) from e
    async with SessionLocal() as session:
        job = await session.get(Job, target)
        if job is None or job.client_app_id != client_app_id:
            raise ValueError(_NO_JOB_FOR_ID)
        rows = (
            await session.scalars(
                select(JobShadow)
                .where(JobShadow.parent_job_id == job.id)
                .order_by(JobShadow.created_at.asc())
            )
        ).all()
        return {
            "parent_job_id": str(job.id),
            "shadows": [
                {
                    "shadow_id": str(s.id),
                    "model": s.model,
                    "provider": s.provider,
                    "status": s.status,
                    "response": s.response or None,
                    "error": s.error or None,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cost_usd": (
                        float(s.cost_usd) if isinstance(s.cost_usd, Decimal) else s.cost_usd
                    ),
                    "latency_ms": s.latency_ms,
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                }
                for s in rows
            ],
        }


@mcp.tool()
async def create_job(
    task_type: str,
    prompt: str,
    system_prompt: str = "",
    sensitivity: str | None = None,
    force_shadows: bool = False,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a job. Fast tasks (cloud or resident local models) run inline
    and the result is returned directly; heavier ones return status=pending —
    poll get_job(job_id) for those. `task_type` should be one from
    list_task_types. `sensitivity` (public/internal/confidential) may raise the
    floor, never lower it. Set `force_shadows=true` to fan out every eligible
    eval shadow for THIS request regardless of the rule's sampling rate —
    useful when you specifically want a side-by-side model comparison.

    `inputs` is the typed-input bag. It flows through for any task type — most
    text tasks ignore it, but two consumers rely on it:

    - **judge** (`task_type="judge"`): `{"mode": "pointwise"|"pairwise"|"panel",
      "target_job_id": "<job_or_shadow_uuid>", "apply_to_target": true,
      "dimensions": [...]}`. Carrying a `target_job_id` marks the job a judge —
      it runs the judge executor on the async worker (status=pending, poll
      get_job), not a plain inline completion. See judging.md.
    - **media** (image/video/audio/mux): reference upstream jobs by id — the
      worker resolves each id to a local file via the upstream job's
      `media_url`, no HTTP round-trip. image→video:
      `{"source_image_job_id": "<image_job_uuid>"}`. video+audio→mux:
      `{"source_video_job_id": "<vid>", "source_audio_job_id": "<music>"}`.
      The legacy `source_<kind>_url` form still works but is deprecated.

    `get_job` echoes the stored `inputs` back so you can confirm what arrived."""
    client_app_id = _client_app_id()
    settings = get_settings()
    inputs = inputs or {}
    try:
        requested = Sensitivity(sensitivity) if sensitivity else None
    except ValueError as e:
        raise ValueError(f"invalid sensitivity {sensitivity!r}") from e

    async with SessionLocal() as session:
        client = await session.get(ClientApp, client_app_id)
        rule = await session.scalar(
            select(RoutingRule).where(
                RoutingRule.task_type == task_type,
                RoutingRule.is_archived.is_(False),
            )
        )

        # Media tasks (image/video/audio/mux) skip the text routing engine
        # and go straight onto the conduct-media queue. Mirror the HTTP
        # path in routes/jobs.py.
        if rule is not None and rule.media_kind != "text":
            return await _create_media_job(
                client=client,
                rule=rule,
                task_type=task_type,
                prompt=prompt,
                system_prompt=system_prompt,
                inputs=inputs,
            )

        effective_request = requested or (
            Sensitivity(rule.sensitivity) if rule else Sensitivity.INTERNAL
        )
        try:
            decision = decide(
                sensitivity=effective_request,
                model_requested=None,
                allow_cloud_for_internal=client.allow_cloud_for_internal,
                rule=rule,
                default_model=settings.default_model,
                default_sensitive_model=settings.default_sensitive_model,
                available_cloud_providers=_cloud_providers_for_principal(client),
            )
        except SensitivityViolation as e:
            raise ValueError(str(e)) from e

        job = Job(
            client_app_id=client_app_id,
            task_type=task_type,
            sensitivity=decision.effective_sensitivity.value,
            priority=5,
            prompt=prompt,
            system_prompt=system_prompt,
            model_requested="",
            status=JobStatus.PENDING.value,
            inputs=inputs,
            job_metadata={"force_shadows": True} if force_shadows else {},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = str(job.id)

        registry = _provider_registry
        # Judge jobs (they carry a target_job_id) must go async: the judge
        # executor only lives on the worker path. Running them inline would
        # treat the bare rubric as a plain completion. Mirrors routes/jobs.py's
        # _should_enqueue. See worker.executor.execute_judge_job.
        if (
            not is_judge_job(inputs)
            and _sync_eligible(decision)
            and registry is not None
            and registry.has(decision.provider)
        ):
            # Run inline (cloud or resident-local model) and return the answer
            # in one call. Eval shadows still fan out async afterward.
            try:
                await execute_job(
                    job=job,
                    decision=decision,
                    client=client,
                    providers=registry,
                    session=session,
                )
            except PromptNotFoundError as e:
                job.status = JobStatus.FAILED.value
                job.error = f"prompt resolution failed: {e}"
                await session.commit()
            if job.status == JobStatus.COMPLETE.value:
                await enqueue_shadows_for_parent(
                    parent_job=job, rule=rule, client=client, session=session
                )
            await session.refresh(job)
            return _job_detail(job)

    # Async path: the worker owns this model (non-resident local). It runs the
    # primary and fans out eval shadows itself.
    get_queue().enqueue(run_job, job_id, job_id=job_id, job_timeout=DEFAULT_JOB_TIMEOUT_S)
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING.value,
        "note": "queued — call get_job(job_id) for the result",
    }


async def _resolve_eval_target(session, target, client_app_id) -> str:
    """Resolve a score target to 'job' or 'shadow', enforcing ownership.

    UUIDs don't collide across the two tables, so we probe in order — Job first
    (the common case), then JobShadow (whose ownership follows its parent).
    Raises ValueError(_NO_JOB_FOR_ID) if not found or not owned by the caller.
    """
    job = await session.get(Job, target)
    if job is not None:
        if job.client_app_id != client_app_id:
            raise ValueError(_NO_JOB_FOR_ID)
        return "job"
    shadow = await session.get(JobShadow, target)
    if shadow is None:
        raise ValueError(_NO_JOB_FOR_ID)
    parent = await session.get(Job, shadow.parent_job_id)
    if parent is None or parent.client_app_id != client_app_id:
        raise ValueError(_NO_JOB_FOR_ID)
    return "shadow"


@mcp.tool()
async def submit_eval(
    job_id: str, score: int | None = None, note: str = "",
    scores: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Record a quality score for one of your jobs OR one of its eval shadows.
    `job_id` accepts either a parent job UUID (from `create_job` / `list_jobs`)
    or a shadow UUID (from `list_shadows[*].shadow_id`) — the server figures out
    which one it is and you don't need to disambiguate.

    Provide an overall `score` (1-5) and/or a named-dimension `scores` map (e.g.
    {"correctness": 5, "format": 3, "craft": 4}) when you want to rate distinct
    qualities separately. Interpret any freeform human feedback ("that was
    hilarious", "too generic") into a score yourself: 1=very poor, 2=poor,
    3=acceptable, 4=good, 5=excellent. `note` is an optional short rationale.

    Authorization for shadow scores follows the parent job's ownership —
    same model as `list_shadows`. The response's `kind` field tells you
    whether you scored a `job` or a `shadow`."""
    client_app_id = _client_app_id()
    if score is not None and not 1 <= score <= 5:
        raise ValueError("score must be an integer 1-5")
    try:
        target = UUID(job_id)
    except ValueError as e:
        raise ValueError(_BAD_JOB_UUID) from e

    principal = _principal.get() or {}
    reviewer = principal.get("client_app_name", "mcp")
    async with SessionLocal() as session:
        kind = await _resolve_eval_target(session, target, client_app_id)
        result = await apply_score(
            session, target, score=score, scores=scores,
            reviewer=reviewer, note=note or None, via="mcp",
        )
    recorded = (result or (None, []))[1]
    overall = recorded[-1]["score"] if recorded else score
    return {"job_id": job_id, "kind": kind, "score": overall, "scores": scores, "recorded": True}


# --- OAuth-gated ASGI wrapper ---


def _bearer_from_scope(scope: dict) -> str:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            header = value.decode("latin-1")
            if header.lower().startswith("bearer "):
                return header[7:].strip()
    return ""


def _resource_metadata_url() -> str:
    base = get_settings().public_base_url.rstrip("/")
    return f"{base}/.well-known/oauth-protected-resource"


async def _send_unauthorized(send) -> None:
    challenge = f'Bearer resource_metadata="{_resource_metadata_url()}"'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", challenge.encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})


class OAuthMiddleware:
    """Pure-ASGI bearer-token gate (kept pure so SSE streaming isn't buffered).
    Resolves the token to a ClientApp and exposes it via the contextvar."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = _bearer_from_scope(scope)
        principal = None
        if token:
            async with SessionLocal() as session:
                client_app = await resolve_access_token(session, token)
                if client_app is not None:
                    principal = {
                        "client_app_id": client_app.id,
                        "client_app_name": client_app.name,
                    }
        if principal is None:
            await _send_unauthorized(send)
            return

        reset = _principal.set(principal)
        try:
            await self.app(scope, receive, send)
        finally:
            _principal.reset(reset)


def build_mcp_app():
    """The /mcp ASGI app: FastMCP's streamable-HTTP transport behind the
    OAuth gate. Mounted by main.py; its session manager is started in the
    app lifespan."""
    return OAuthMiddleware(mcp.streamable_http_app())
