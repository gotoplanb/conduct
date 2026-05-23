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
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select

from config.settings import get_settings
from db.session import SessionLocal
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus, Sensitivity
from oauth_provider import resolve_access_token
from routing.engine import SensitivityViolation, decide
from worker.queue import DEFAULT_JOB_TIMEOUT_S, get_queue
from worker.runner import run_job

# Set per-request by the OAuth middleware; read by the tools.
_principal: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "mcp_principal", default=None
)

mcp = FastMCP("Conduct", stateless_http=True, streamable_http_path="/", json_response=False)


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
        raise ValueError("job_id is not a valid UUID") from e
    async with SessionLocal() as session:
        job = await session.get(Job, target)
        if job is None or job.client_app_id != client_app_id:
            raise ValueError("no job with that id")
        return _job_detail(job)


@mcp.tool()
async def create_job(
    task_type: str, prompt: str, system_prompt: str = "", sensitivity: str | None = None
) -> dict[str, Any]:
    """Create a job. It runs asynchronously; poll get_job(job_id) for the
    result. `task_type` should be one from list_task_types. `sensitivity`
    (public/internal/confidential) may raise the floor, never lower it."""
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
            job_metadata={},
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = str(job.id)

    get_queue().enqueue(run_job, job_id, job_id=job_id, job_timeout=DEFAULT_JOB_TIMEOUT_S)
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING.value,
        "note": "queued — call get_job(job_id) for the result",
    }


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
