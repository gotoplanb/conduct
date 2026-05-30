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
from worker.executor import execute_job
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
    }


@mcp.tool()
async def list_task_types() -> list[dict[str, Any]]:
    """List the task types this Conduct instance can run, with the model each
    routes to and its sensitivity floor. Use these as the `task_type` for
    create_job."""
    async with SessionLocal() as session:
        rules = (
            await session.scalars(select(RoutingRule).order_by(RoutingRule.task_type))
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
) -> dict[str, Any]:
    """Create a job. Fast tasks (cloud or resident local models) run inline
    and the result is returned directly; heavier ones return status=pending —
    poll get_job(job_id) for those. `task_type` should be one from
    list_task_types. `sensitivity` (public/internal/confidential) may raise the
    floor, never lower it. Set `force_shadows=true` to fan out every eligible
    eval shadow for THIS request regardless of the rule's sampling rate —
    useful when you specifically want a side-by-side model comparison."""
    client_app_id = _client_app_id()
    settings = get_settings()
    try:
        requested = Sensitivity(sensitivity) if sensitivity else None
    except ValueError as e:
        raise ValueError(f"invalid sensitivity {sensitivity!r}") from e

    async with SessionLocal() as session:
        client = await session.get(ClientApp, client_app_id)
        rule = await session.scalar(
            select(RoutingRule).where(RoutingRule.task_type == task_type)
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
                cloud_available=(
                    _provider_registry is not None
                    and _provider_registry.has_for_client(client, "anthropic")
                ),
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
            job_metadata={"force_shadows": True} if force_shadows else {},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = str(job.id)

        registry = _provider_registry
        if _sync_eligible(decision) and registry is not None and registry.has(decision.provider):
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


@mcp.tool()
async def submit_eval(job_id: str, score: int, note: str = "") -> dict[str, Any]:
    """Record a 1-5 quality score for one of your completed jobs. Interpret any
    freeform human feedback ("that was hilarious", "too generic") into a score
    yourself before calling: 1=very poor, 2=poor, 3=acceptable, 4=good,
    5=excellent. `note` is an optional short rationale."""
    client_app_id = _client_app_id()
    if not 1 <= score <= 5:
        raise ValueError("score must be an integer 1-5")
    try:
        target = UUID(job_id)
    except ValueError as e:
        raise ValueError(_BAD_JOB_UUID) from e

    principal = _principal.get() or {}
    reviewer = principal.get("client_app_name", "mcp")
    async with SessionLocal() as session:
        job = await session.get(Job, target)
        if job is None or job.client_app_id != client_app_id:
            raise ValueError(_NO_JOB_FOR_ID)
        await apply_score(
            session, target, score=score, reviewer=reviewer, note=note or None, via="mcp"
        )
    return {"job_id": job_id, "score": score, "recorded": True}


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
