"""HTMX + Alpine + Tailwind UI for browsing Conduct state locally.

Cookie-based admin auth: POST /ui/login with the admin key stores it in a
httponly cookie; subsequent GETs read the cookie. Intentionally simple —
this UI is for local-network operator use, not external exposure.
"""

from __future__ import annotations

import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import generate_api_key, hash_api_key
from config.settings import get_settings
from db.session import get_session
from eval.rollup import compute_rollup
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow

ADMIN_COOKIE = "conduct_admin"
LOGIN_PATH = "/ui/login"
JOBS_PATH = "/ui/jobs"
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)


def _require_admin_cookie(value: str | None) -> bool:
    if not value:
        return False
    return hmac.compare_digest(value, get_settings().admin_key)


def _redirect_to_login() -> RedirectResponse:
    return RedirectResponse(url=LOGIN_PATH, status_code=status.HTTP_303_SEE_OTHER)


async def admin_session(
    conduct_admin: Annotated[str | None, Cookie()] = None,
) -> None:
    """Cookie-based admin guard. Used as a dep on every authed UI route.

    Raises a 303 instead of a 401 so browsers transparently bounce to the
    login page rather than rendering a JSON error blob.
    """
    if not _require_admin_cookie(conduct_admin):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": LOGIN_PATH},
        )


def _grafana_trace_url(job: Job) -> str | None:
    """Build a Grafana Explore URL pre-loaded with a TraceQL query for this
    job's spans. Time range is bracketed around the job's actual run window
    so Tempo's lookup is fast even on busy days. Returns None when Grafana
    isn't configured."""
    base = get_settings().grafana_base_url.rstrip("/")
    if not base:
        return None

    # Pad ±60s around the job. created_at always exists; completed_at may
    # not (for still-running or never-started jobs) — fall back to "now".
    created = job.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    completed = job.completed_at or datetime.now(UTC)
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=UTC)
    from_ms = int((created - timedelta(seconds=60)).timestamp() * 1000)
    to_ms = int((completed + timedelta(seconds=60)).timestamp() * 1000)

    explore = {
        "datasource": "Tempo",
        "queries": [
            {
                "refId": "A",
                "queryType": "traceql",
                "query": f'{{.job.id="{job.id}"}}',
            }
        ],
        "range": {"from": str(from_ms), "to": str(to_ms)},
    }
    return f"{base}/explore?orgId=1&left={quote(json.dumps(explore))}"


def _humanize_age(dt: datetime) -> str:
    now = datetime.now(UTC)
    delta = now - dt.replace(tzinfo=UTC) if dt.tzinfo is None else now - dt
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


# ---------- auth ----------


@router.get("", response_class=HTMLResponse)
async def root_redirect(
    conduct_admin: Annotated[str | None, Cookie()] = None,
) -> RedirectResponse:
    target = JOBS_PATH if _require_admin_cookie(conduct_admin) else LOGIN_PATH
    return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {})


@router.post("/login", response_model=None)
async def login(
    request: Request, admin_key: Annotated[str, Form()]
) -> HTMLResponse | RedirectResponse:
    if not hmac.compare_digest(admin_key, get_settings().admin_key):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid admin key."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    resp = RedirectResponse(url=JOBS_PATH, status_code=status.HTTP_303_SEE_OTHER)
    # Cookie is HttpOnly to keep it out of JS; `secure` is opt-in via env
    # (UI_COOKIE_SECURE=true) since local dev is HTTP. Set it to true once
    # you're behind HTTPS (ngrok / reverse proxy) — otherwise the cookie
    # silently won't be sent on the HTTPS leg.
    resp.set_cookie(
        key=ADMIN_COOKIE,
        value=admin_key,
        httponly=True,
        secure=get_settings().ui_cookie_secure,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # one week
    )
    return resp


@router.post("/logout")
async def logout() -> RedirectResponse:
    resp = RedirectResponse(url=LOGIN_PATH, status_code=status.HTTP_303_SEE_OTHER)
    resp.delete_cookie(ADMIN_COOKIE)
    return resp


# ---------- jobs ----------


async def _load_jobs(
    session: AsyncSession,
    *,
    task_type: str | None,
    job_status: str | None,
    q: str | None,
    limit: int = 200,
) -> tuple[list[dict], list[str]]:
    """Returns (job rows for template, distinct task_types for filter)."""
    stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
    if task_type:
        stmt = stmt.where(Job.task_type == task_type)
    if job_status:
        stmt = stmt.where(Job.status == job_status)
    if q:
        stmt = stmt.where(Job.prompt.ilike(f"%{q}%"))
    rows = (await session.scalars(stmt)).all()

    client_ids = {j.client_app_id for j in rows}
    clients = {
        c.id: c.name
        for c in (
            await session.scalars(select(ClientApp).where(ClientApp.id.in_(client_ids)))
        ).all()
    } if client_ids else {}

    jobs_view = [
        {
            "id": str(j.id),
            "task_type": j.task_type,
            "status": j.status,
            "model_used": j.model_used,
            "client_name": clients.get(j.client_app_id, "?"),
            "latency_ms": j.latency_ms,
            "cost_usd": float(j.cost_usd) if j.cost_usd is not None else None,
            "created_at": j.created_at,
            "created_rel": _humanize_age(j.created_at),
        }
        for j in rows
    ]
    task_types = sorted({j.task_type for j in rows})
    return jobs_view, task_types


@router.get("/jobs", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def jobs_page(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    job_status: Annotated[str | None, Query(alias="status")] = None,
    q: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    jobs, task_types = await _load_jobs(
        session, task_type=task_type, job_status=job_status, q=q
    )
    return templates.TemplateResponse(
        request,
        "jobs_list.html",
        {
            "jobs": jobs,
            "task_types": task_types,
            "task_type": task_type,
            "status": job_status,
        },
    )


@router.get("/jobs/partial", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def jobs_partial(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    job_status: Annotated[str | None, Query(alias="status")] = None,
    q: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    jobs, _ = await _load_jobs(session, task_type=task_type, job_status=job_status, q=q)
    return templates.TemplateResponse(request, "_jobs_table.html", {"jobs": jobs})


@router.get("/jobs/{job_id}", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def job_detail(
    request: Request,
    job_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    client = await session.get(ClientApp, job.client_app_id)
    shadows = (
        await session.scalars(
            select(JobShadow)
            .where(JobShadow.parent_job_id == job.id)
            .order_by(JobShadow.created_at.asc())
        )
    ).all()
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            "job": job,
            "client_name": client.name if client else "?",
            "shadows": [
                {
                    "model": s.model,
                    "status": s.status,
                    "latency_ms": s.latency_ms,
                    "cost_usd": float(s.cost_usd) if s.cost_usd is not None else None,
                    "response": s.response,
                }
                for s in shadows
            ],
            "metadata_json": json.dumps(job.job_metadata or {}, indent=2, default=str),
            "grafana_trace_url": _grafana_trace_url(job),
        },
    )


# ---------- clients ----------


async def _load_clients(session: AsyncSession) -> list[dict]:
    rows = (await session.scalars(select(ClientApp).order_by(ClientApp.created_at))).all()
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "is_active": c.is_active,
            "rate_limit_per_minute": c.rate_limit_per_minute,
            "allow_cloud_for_internal": c.allow_cloud_for_internal,
            "notes": c.notes,
            "created_at": c.created_at,
            "key_created_at": c.key_created_at,
            "key_created_rel": _humanize_age(c.key_created_at),
        }
        for c in rows
    ]


async def _render_clients(
    request: Request,
    session: AsyncSession,
    *,
    new_key: dict | None = None,
    flash: str | None = None,
    error: str | None = None,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    clients = await _load_clients(session)
    return templates.TemplateResponse(
        request,
        "clients_list.html",
        {"clients": clients, "new_key": new_key, "flash": flash, "error": error},
        status_code=status_code,
    )


@router.get("/clients", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def clients_page(
    request: Request, session: Annotated[AsyncSession, Depends(get_session)]
) -> HTMLResponse:
    return await _render_clients(request, session)


@router.post("/clients", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def clients_create(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    name: Annotated[str, Form()],
    notes: Annotated[str, Form()] = "",
    rate_limit_per_minute: Annotated[str, Form()] = "",
    allow_cloud_for_internal: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    name = name.strip()
    if not name:
        return await _render_clients(
            request, session, error="Name is required.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    rate = int(rate_limit_per_minute) if rate_limit_per_minute.strip() else None
    raw_key = generate_api_key()
    client = ClientApp(
        name=name,
        api_key_hash=hash_api_key(raw_key),
        notes=notes.strip(),
        rate_limit_per_minute=rate,
        allow_cloud_for_internal=allow_cloud_for_internal,
    )
    session.add(client)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return await _render_clients(
            request, session, error=f"A client named {name!r} already exists.",
            status_code=status.HTTP_409_CONFLICT,
        )
    return await _render_clients(
        request,
        session,
        new_key={"name": name, "api_key": raw_key, "action": "created"},
    )


@router.post(
    "/clients/{client_id}/rotate",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_rotate(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error="Client not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    raw_key = generate_api_key()
    client.api_key_hash = hash_api_key(raw_key)
    client.key_created_at = datetime.now(UTC)
    await session.commit()
    return await _render_clients(
        request,
        session,
        new_key={"name": client.name, "api_key": raw_key, "action": "rotated"},
    )


@router.post(
    "/clients/{client_id}/toggle",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_toggle(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error="Client not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    client.is_active = not client.is_active
    state = "active" if client.is_active else "inactive"
    await session.commit()
    return await _render_clients(
        request, session, flash=f"{client.name} is now {state}."
    )


# ---------- eval ----------


async def _compute_eval(
    session: AsyncSession, *, task_type: str, days: int
) -> list[dict]:
    """Thin wrapper around the shared rollup helper. Kept as a local symbol
    so the route handlers don't need to know about eval.rollup directly."""
    return await compute_rollup(session, task_type=task_type, days=days)


async def _known_task_types(session: AsyncSession) -> list[str]:
    """Routing rules + any task_types that have appeared on jobs."""
    rules = [r.task_type for r in (await session.scalars(select(RoutingRule))).all()]
    job_types = [
        t for (t,) in (await session.execute(select(Job.task_type).distinct())).all() if t
    ]
    return sorted(set(rules) | set(job_types))


@router.get("/eval", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def eval_page(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
    days: Annotated[int, Query(ge=1, le=365)] = 7,
) -> HTMLResponse:
    task_types = await _known_task_types(session)
    selected = task_type or (task_types[0] if task_types else "")
    models = await _compute_eval(session, task_type=selected, days=days) if selected else []
    return templates.TemplateResponse(
        request,
        "eval_compare.html",
        {
            "task_types": task_types,
            "task_type": selected,
            "days": days,
            "models": models,
        },
    )


@router.get("/eval/partial", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def eval_partial(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str, Query()],
    days: Annotated[int, Query(ge=1, le=365)] = 7,
) -> HTMLResponse:
    models = await _compute_eval(session, task_type=task_type, days=days)
    return templates.TemplateResponse(request, "_eval_table.html", {"models": models})
