"""HTMX + Alpine + Tailwind UI for browsing Conduct state locally.

Cookie-based admin auth: POST /ui/login with the admin key stores it in a
httponly cookie; subsequent GETs read the cookie. Intentionally simple —
this UI is for local-network operator use, not external exposure.
"""

from __future__ import annotations

import hmac
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from db.session import get_session
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from routes.eval import _aggregate_metadata_scores

ADMIN_COOKIE = "conduct_admin"
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)


def _require_admin_cookie(value: str | None) -> bool:
    if not value:
        return False
    return hmac.compare_digest(value, get_settings().admin_key)


def _redirect_to_login() -> RedirectResponse:
    return RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)


async def admin_session(
    conduct_admin: str | None = Cookie(default=None),
) -> None:
    """Cookie-based admin guard. Used as a dep on every authed UI route.

    Raises a 303 instead of a 401 so browsers transparently bounce to the
    login page rather than rendering a JSON error blob.
    """
    if not _require_admin_cookie(conduct_admin):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/ui/login"},
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
async def root_redirect(conduct_admin: str | None = Cookie(default=None)) -> RedirectResponse:
    target = "/ui/jobs" if _require_admin_cookie(conduct_admin) else "/ui/login"
    return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {})


@router.post("/login", response_model=None)
async def login(request: Request, admin_key: str = Form(...)) -> HTMLResponse | RedirectResponse:
    if not hmac.compare_digest(admin_key, get_settings().admin_key):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid admin key."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    resp = RedirectResponse(url="/ui/jobs", status_code=status.HTTP_303_SEE_OTHER)
    # Local-network tool; httponly is enough. secure=False because we're HTTP.
    resp.set_cookie(
        key=ADMIN_COOKIE,
        value=admin_key,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # one week
    )
    return resp


@router.post("/logout")
async def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/ui/login", status_code=status.HTTP_303_SEE_OTHER)
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
    task_type: str | None = Query(default=None),
    job_status: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
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
    task_type: str | None = Query(default=None),
    job_status: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    jobs, _ = await _load_jobs(session, task_type=task_type, job_status=job_status, q=q)
    return templates.TemplateResponse(request, "_jobs_table.html", {"jobs": jobs})


@router.get("/jobs/{job_id}", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def job_detail(
    request: Request,
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
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


# ---------- eval ----------


async def _compute_eval(
    session: AsyncSession, *, task_type: str, days: int
) -> list[dict]:
    """Mirrors /eval/compare but shaped for the template. Keeps the UI from
    needing to call the JSON API internally."""
    from sqlalchemy import case, func

    from models.types import JobStatus

    since = datetime.now(UTC) - timedelta(days=days)
    job_is_complete = case((Job.status == JobStatus.COMPLETE.value, 1), else_=0)
    job_is_failed = case((Job.status == JobStatus.FAILED.value, 1), else_=0)
    shadow_is_complete = case((JobShadow.status == JobStatus.COMPLETE.value, 1), else_=0)
    shadow_is_failed = case((JobShadow.status == JobStatus.FAILED.value, 1), else_=0)

    job_rows = (
        await session.execute(
            select(
                Job.model_used,
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
            .where(Job.task_type == task_type, Job.created_at >= since, Job.model_used != "")
            .group_by(Job.model_used)
        )
    ).all()

    shadow_rows = (
        await session.execute(
            select(
                JobShadow.model,
                func.count(),
                func.sum(shadow_is_complete),
                func.sum(shadow_is_failed),
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
            .where(Job.task_type == task_type, JobShadow.created_at >= since)
            .group_by(JobShadow.model)
        )
    ).all()

    rolled: dict[str, dict] = defaultdict(
        lambda: {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_sum": 0.0,
            "tokens_count": 0,
            "cost_total": 0.0,
        }
    )
    for source in (job_rows, shadow_rows):
        for model, attempts, successes, failures, avg_lat, avg_tok, total_cost in source:
            e = rolled[model]
            attempts_i = int(attempts or 0)
            successes_i = int(successes or 0)
            e["attempts"] += attempts_i
            e["successes"] += successes_i
            e["failures"] += int(failures or 0)
            e["cost_total"] += float(total_cost or 0)
            if avg_lat is not None and successes_i:
                e["latency_sum"] += float(avg_lat) * successes_i
                e["latency_count"] += successes_i
            if avg_tok is not None and successes_i:
                e["tokens_sum"] += float(avg_tok) * successes_i
                e["tokens_count"] += successes_i

    # Score rollup
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

    out: list[dict] = []
    for model, e in sorted(rolled.items(), key=lambda kv: -kv[1]["attempts"]):
        successes = e["successes"]
        out.append(
            {
                "model": model,
                "job_count": e["attempts"],
                "failure_rate": (e["failures"] / e["attempts"]) if e["attempts"] else 0.0,
                "avg_latency_ms": (
                    e["latency_sum"] / e["latency_count"] if e["latency_count"] else None
                ),
                "avg_tokens_out": (
                    e["tokens_sum"] / e["tokens_count"] if e["tokens_count"] else None
                ),
                "cost_per_job_usd": (e["cost_total"] / successes) if successes else 0.0,
                "avg_score": avg_scores.get(model),
                "score_count": score_counts.get(model, 0),
            }
        )
    return out


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
    task_type: str | None = Query(default=None),
    days: int = Query(default=7, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
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
    task_type: str = Query(...),
    days: int = Query(default=7, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    models = await _compute_eval(session, task_type=task_type, days=days)
    return templates.TemplateResponse(request, "_eval_table.html", {"models": models})
