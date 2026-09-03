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
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import generate_api_key, hash_api_key
from config.settings import get_settings
from db.session import get_session
from eval.rollup import compute_rollup
from eval.scoring import apply_score, score_state
from models.client import ClientApp
from models.job import Job
from models.oauth import OAuthClient
from models.prompt import Prompt, PromptVersion
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus
from oauth_provider import hash_secret, new_client_id, new_client_secret
from secrets_box import SecretsKeyMissing, encrypt

ADMIN_COOKIE = "conduct_admin"
LOGIN_PATH = "/ui/login"
JOBS_PATH = "/ui/jobs"
_CLIENT_NOT_FOUND = "Client not found."
_NAME_REQUIRED = "Name is required."
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)


def _require_admin_cookie(value: str | None) -> bool:
    if not value:
        return False
    return hmac.compare_digest(value, get_settings().admin_key)


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

    # Pad ±60s around the job. created_at always exists and is tz-aware
    # (the columns are DateTime(timezone=True)); completed_at may not exist
    # (still-running or never-started jobs) — fall back to "now".
    created = job.created_at
    completed = job.completed_at or datetime.now(UTC)
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


def _safe_next(next_url: str | None) -> str | None:
    """Only allow same-site redirects: a path starting with a single '/'.
    Blocks '//evil.com' and scheme-relative / absolute URLs (open redirect)."""
    if not next_url or not next_url.startswith("/") or next_url.startswith("//"):
        return None
    return next_url


@router.get("/login", response_class=HTMLResponse)
async def login_form(
    request: Request, next: Annotated[str | None, Query()] = None
) -> HTMLResponse:
    return templates.TemplateResponse(request, "login.html", {"next": _safe_next(next) or ""})


@router.post("/login", response_model=None)
async def login(
    request: Request,
    admin_key: Annotated[str, Form()],
    next: Annotated[str, Form()] = "",
) -> HTMLResponse | RedirectResponse:
    if not hmac.compare_digest(admin_key, get_settings().admin_key):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid admin key.", "next": _safe_next(next) or ""},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    resp = RedirectResponse(
        url=_safe_next(next) or JOBS_PATH, status_code=status.HTTP_303_SEE_OTHER
    )
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
            "anthropic_api_key_set_at": c.anthropic_api_key_set_at,
            "has_anthropic_key": c.anthropic_api_key_encrypted is not None,
            "bedrock_creds_set_at": c.bedrock_creds_set_at,
            "has_bedrock_creds": c.bedrock_creds_encrypted is not None,
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
            request, session, error=_NAME_REQUIRED,
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
            request, session, error=_CLIENT_NOT_FOUND,
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
            request, session, error=_CLIENT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
    client.is_active = not client.is_active
    state = "active" if client.is_active else "inactive"
    await session.commit()
    return await _render_clients(
        request, session, flash=f"{client.name} is now {state}."
    )


@router.post(
    "/clients/{client_id}/edit",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_edit(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    name: Annotated[str, Form()],
    notes: Annotated[str, Form()] = "",
    rate_limit_per_minute: Annotated[str, Form()] = "",
    allow_cloud_for_internal: Annotated[bool, Form()] = False,
) -> HTMLResponse:
    """Update the editable fields on an existing client. Mirrors the JSON
    `PATCH /clients/{id}` semantics but driven by an HTML form."""
    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error=_CLIENT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
    name = name.strip()
    if not name:
        return await _render_clients(
            request, session, error=_NAME_REQUIRED,
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    rate = int(rate_limit_per_minute) if rate_limit_per_minute.strip() else None
    client.name = name
    client.notes = notes.strip()
    client.rate_limit_per_minute = rate
    client.allow_cloud_for_internal = allow_cloud_for_internal
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return await _render_clients(
            request, session, error=f"A client named {name!r} already exists.",
            status_code=status.HTTP_409_CONFLICT,
        )
    return await _render_clients(
        request, session, flash=f"Updated {client.name}."
    )


@router.post(
    "/clients/{client_id}/anthropic-key",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_set_anthropic_key(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    api_key: Annotated[str, Form()],
) -> HTMLResponse:
    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error=_CLIENT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
    key = api_key.strip()
    if not key:
        return await _render_clients(
            request, session, error="API key is required.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    try:
        client.anthropic_api_key_encrypted = encrypt(key)
    except SecretsKeyMissing:
        return await _render_clients(
            request,
            session,
            error="CONDUCT_SECRETS_KEY is not configured — cannot store the key.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    client.anthropic_api_key_set_at = datetime.now(UTC)
    await session.commit()
    return await _render_clients(
        request, session, flash=f"Anthropic key set for {client.name}."
    )


@router.post(
    "/clients/{client_id}/anthropic-key/clear",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_clear_anthropic_key(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error=_CLIENT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
    client.anthropic_api_key_encrypted = None
    client.anthropic_api_key_set_at = None
    await session.commit()
    return await _render_clients(
        request, session, flash=f"Anthropic key cleared for {client.name}."
    )


@router.post(
    "/clients/{client_id}/bedrock-creds",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_set_bedrock_creds(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    region: Annotated[str, Form()],
    bearer_token: Annotated[str, Form()] = "",
    access_key_id: Annotated[str, Form()] = "",
    secret_access_key: Annotated[str, Form()] = "",
) -> HTMLResponse:
    import json as _json  # noqa: PLC0415

    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error=_CLIENT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
    bearer_token = bearer_token.strip()
    access_key_id = access_key_id.strip()
    secret_access_key = secret_access_key.strip()
    region = region.strip()
    if not region:
        return await _render_clients(
            request, session,
            error="Region is required.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    has_bearer = bool(bearer_token)
    has_pair = bool(access_key_id and secret_access_key)
    if has_bearer and has_pair:
        return await _render_clients(
            request, session,
            error="Provide either a bearer token OR access key + secret, not both.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if not has_bearer and not has_pair:
        return await _render_clients(
            request, session,
            error="Provide either a bearer token OR access key id + secret access key.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    payload: dict[str, str] = {"region": region}
    if has_bearer:
        payload["bearer_token"] = bearer_token
    else:
        payload["access_key_id"] = access_key_id
        payload["secret_access_key"] = secret_access_key
    blob = _json.dumps(payload)
    try:
        client.bedrock_creds_encrypted = encrypt(blob)
    except SecretsKeyMissing:
        return await _render_clients(
            request,
            session,
            error="CONDUCT_SECRETS_KEY is not configured — cannot store the creds.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    client.bedrock_creds_set_at = datetime.now(UTC)
    await session.commit()
    return await _render_clients(
        request, session, flash=f"Bedrock creds set for {client.name}."
    )


@router.post(
    "/clients/{client_id}/bedrock-creds/clear",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def clients_clear_bedrock_creds(
    request: Request,
    client_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    client = await session.get(ClientApp, client_id)
    if client is None:
        return await _render_clients(
            request, session, error=_CLIENT_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
        )
    client.bedrock_creds_encrypted = None
    client.bedrock_creds_set_at = None
    await session.commit()
    return await _render_clients(
        request, session, flash=f"Bedrock creds cleared for {client.name}."
    )


# ---------- connectors (OAuth clients for the MCP server) ----------

DEFAULT_CONNECTOR_REDIRECT = "https://claude.ai/api/mcp/auth_callback"


async def _load_connectors(session: AsyncSession) -> list[dict]:
    rows = (await session.scalars(select(OAuthClient).order_by(OAuthClient.created_at))).all()
    app_ids = {c.client_app_id for c in rows}
    apps = (
        {
            a.id: a.name
            for a in (
                await session.scalars(select(ClientApp).where(ClientApp.id.in_(app_ids)))
            ).all()
        }
        if app_ids
        else {}
    )
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "client_id": c.client_id,
            "client_app_name": apps.get(c.client_app_id, "?"),
            "redirect_uris": c.redirect_uris or [],
            "is_active": c.is_active,
            "created_rel": _humanize_age(c.created_at),
        }
        for c in rows
    ]


def _connector_instructions() -> dict:
    base = get_settings().public_base_url.rstrip("/")
    return {
        "base_url": base,
        "mcp_url": f"{base}/mcp",
        "authorize_url": f"{base}/oauth/authorize",
        "token_url": f"{base}/oauth/token",
    }


async def _render_connectors(
    request: Request,
    session: AsyncSession,
    *,
    new_pair: dict | None = None,
    flash: str | None = None,
    error: str | None = None,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    apps = (await session.scalars(select(ClientApp).order_by(ClientApp.name))).all()
    return templates.TemplateResponse(
        request,
        "connectors_list.html",
        {
            "connectors": await _load_connectors(session),
            "client_apps": [{"id": str(a.id), "name": a.name} for a in apps],
            "instructions": _connector_instructions(),
            "default_redirect": DEFAULT_CONNECTOR_REDIRECT,
            "new_pair": new_pair,
            "flash": flash,
            "error": error,
        },
        status_code=status_code,
    )


def _parse_redirect_uris(raw: str) -> list[str]:
    uris = [u.strip() for u in raw.replace(",", "\n").splitlines() if u.strip()]
    return uris or [DEFAULT_CONNECTOR_REDIRECT]


@router.get("/connectors", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def connectors_page(
    request: Request, session: Annotated[AsyncSession, Depends(get_session)]
) -> HTMLResponse:
    return await _render_connectors(request, session)


@router.post("/connectors", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def connectors_create(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    name: Annotated[str, Form()],
    client_app_id: Annotated[str, Form()],
    redirect_uris: Annotated[str, Form()] = "",
) -> HTMLResponse:
    name = name.strip()
    if not name:
        return await _render_connectors(
            request, session, error=_NAME_REQUIRED, status_code=status.HTTP_400_BAD_REQUEST
        )
    try:
        app_uuid = UUID(client_app_id)
    except ValueError:
        return await _render_connectors(
            request, session, error="Pick a client to bind to.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    app = await session.get(ClientApp, app_uuid)
    if app is None:
        return await _render_connectors(
            request, session, error=_CLIENT_NOT_FOUND, status_code=status.HTTP_404_NOT_FOUND
        )

    raw_secret = new_client_secret()
    client_id = new_client_id()
    session.add(
        OAuthClient(
            client_id=client_id,
            client_secret_hash=hash_secret(raw_secret),
            name=name,
            client_app_id=app.id,
            redirect_uris=_parse_redirect_uris(redirect_uris),
            created_by="ui",
        )
    )
    await session.commit()
    return await _render_connectors(
        request,
        session,
        new_pair={
            "name": name,
            "client_id": client_id,
            "client_secret": raw_secret,
            "action": "created",
        },
    )


@router.post(
    "/connectors/{connector_id}/rotate-secret",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def connectors_rotate(
    request: Request,
    connector_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    connector = await session.get(OAuthClient, connector_id)
    if connector is None:
        return await _render_connectors(
            request, session, error="Connector not found.", status_code=status.HTTP_404_NOT_FOUND
        )
    raw_secret = new_client_secret()
    connector.client_secret_hash = hash_secret(raw_secret)
    await session.commit()
    return await _render_connectors(
        request,
        session,
        new_pair={
            "name": connector.name,
            "client_id": connector.client_id,
            "client_secret": raw_secret,
            "action": "rotated",
        },
    )


@router.post(
    "/connectors/{connector_id}/toggle",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def connectors_toggle(
    request: Request,
    connector_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    connector = await session.get(OAuthClient, connector_id)
    if connector is None:
        return await _render_connectors(
            request, session, error="Connector not found.", status_code=status.HTTP_404_NOT_FOUND
        )
    connector.is_active = not connector.is_active
    state = "active" if connector.is_active else "inactive"
    await session.commit()
    return await _render_connectors(
        request, session, flash=f"{connector.name} is now {state}."
    )


# ---------- tasks (read-only config view) ----------


def _shadow_summary(rule: RoutingRule) -> list[str]:
    """Human-readable 'model @ rate' strings for a rule's shadow models."""
    return [f"{s.get('model')} @ {s.get('rate')}" for s in (rule.eval_shadow_models or [])]


async def _load_tasks(session: AsyncSession) -> list[dict]:
    """Assemble one view object per task_type: its routing rule (if any) plus
    the shared prompt and any per-client overrides. Read-only — this is the
    operator's map of 'what tasks exist and who has customized them'."""
    rules = {r.task_type: r for r in (await session.scalars(select(RoutingRule))).all()}
    prompts = (await session.scalars(select(Prompt))).all()

    client_ids = {p.client_id for p in prompts if p.client_id is not None}
    clients = (
        {
            c.id: c.name
            for c in (
                await session.scalars(select(ClientApp).where(ClientApp.id.in_(client_ids)))
            ).all()
        }
        if client_ids
        else {}
    )

    version_rows = (
        await session.execute(
            select(PromptVersion.task_type, PromptVersion.client_id, func.count()).group_by(
                PromptVersion.task_type, PromptVersion.client_id
            )
        )
    ).all()
    vcounts = {(tt, cid): n for tt, cid, n in version_rows}

    task_types = sorted(set(rules) | {p.task_type for p in prompts})
    by_task: dict[str, dict] = {}
    for tt in task_types:
        rule = rules.get(tt)
        by_task[tt] = {
            "task_type": tt,
            "rule": (
                {
                    "preferred_model": rule.preferred_model,
                    "fallback_model": rule.fallback_model,
                    "sensitivity": rule.sensitivity,
                    "max_tokens": rule.max_tokens,
                    "notes": rule.notes,
                    "shadows": _shadow_summary(rule),
                }
                if rule
                else None
            ),
            "prompts": [],
        }

    for p in prompts:
        scope = clients.get(p.client_id, "shared") if p.client_id else "shared"
        by_task[p.task_type]["prompts"].append(
            {
                "scope": scope,
                "key": str(p.client_id) if p.client_id else "shared",
                "client_id": str(p.client_id) if p.client_id else "",
                "is_shared": p.client_id is None,
                "bytes": len(p.content.encode("utf-8")),
                "updated_rel": _humanize_age(p.updated_at),
                "updated_by": p.updated_by or "—",
                "versions": vcounts.get((p.task_type, p.client_id), 0),
                "content": p.content,
            }
        )

    for tt in task_types:
        by_task[tt]["prompts"].sort(key=lambda e: (not e["is_shared"], e["scope"]))
    return [by_task[tt] for tt in task_types]


@router.get("/tasks", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def tasks_page(
    request: Request, session: Annotated[AsyncSession, Depends(get_session)]
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "tasks_list.html", {"tasks": await _load_tasks(session)}
    )


@router.get(
    "/tasks/{task_type}/history",
    response_class=HTMLResponse,
    dependencies=[Depends(admin_session)],
)
async def task_history(
    request: Request,
    task_type: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client_id: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    stmt = select(PromptVersion).where(PromptVersion.task_type == task_type)
    if client_id:
        stmt = stmt.where(PromptVersion.client_id == UUID(client_id))
    else:
        stmt = stmt.where(PromptVersion.client_id.is_(None))
    versions = (
        await session.scalars(stmt.order_by(PromptVersion.edited_at.desc()).limit(50))
    ).all()
    rows = [
        {
            "id": v.id,
            "edited_rel": _humanize_age(v.edited_at),
            "edited_at": v.edited_at,
            "edited_by": v.edited_by or "—",
            "bytes": len(v.content.encode("utf-8")),
        }
        for v in versions
    ]
    return templates.TemplateResponse(request, "_task_history.html", {"versions": rows})


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


# ---------- eval review (human scoring) ----------


def _candidate(*, target_id, model: str, response: str, meta: dict | None, is_production: bool):
    return {
        "target_id": str(target_id),
        "model": model or "?",
        "response": response or "",
        "is_production": is_production,
        "score": score_state((meta or {}).get("quality_scores", [])),
    }


async def _load_review(session: AsyncSession, task_type: str, *, limit: int = 10) -> list[dict]:
    """Group completed jobs (with their completed shadows) for a task_type so
    the reviewer sees one prompt and every model's answer side by side. Newest
    jobs first; capped at `limit` parent jobs."""
    rows = (
        await session.execute(
            select(JobShadow, Job)
            .join(Job, Job.id == JobShadow.parent_job_id)
            .where(
                Job.task_type == task_type,
                Job.status == JobStatus.COMPLETE.value,
                JobShadow.status == JobStatus.COMPLETE.value,
            )
            .order_by(Job.created_at.desc())
        )
    ).all()

    grouped: dict[str, dict] = {}
    order: list[str] = []
    for shadow, job in rows:
        key = str(job.id)
        if key not in grouped:
            if len(order) >= limit:
                continue
            order.append(key)
            grouped[key] = {
                "job_id": key,
                "prompt": job.prompt,
                "created_rel": _humanize_age(job.created_at),
                "candidates": [
                    _candidate(
                        target_id=job.id,
                        model=job.model_used,
                        response=job.response,
                        meta=job.job_metadata,
                        is_production=True,
                    )
                ],
            }
        grouped[key]["candidates"].append(
            _candidate(
                target_id=shadow.id,
                model=shadow.model,
                response=shadow.response,
                meta=shadow.shadow_metadata,
                is_production=False,
            )
        )
    return [grouped[k] for k in order]


@router.get("/eval/review", response_class=HTMLResponse, dependencies=[Depends(admin_session)])
async def eval_review_page(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    task_type: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    task_types = await _known_task_types(session)
    selected = task_type or (task_types[0] if task_types else "")
    jobs = await _load_review(session, selected) if selected else []
    return templates.TemplateResponse(
        request,
        "eval_review.html",
        {"task_types": task_types, "task_type": selected, "jobs": jobs},
    )


@router.post(
    "/eval/review/score", response_class=HTMLResponse, dependencies=[Depends(admin_session)]
)
async def eval_review_score(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    target_id: Annotated[str, Form()],
    score: Annotated[int, Form()],
    note: Annotated[str, Form()] = "",
) -> HTMLResponse:
    if not 1 <= score <= 5:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "score must be 1-5")
    result = await apply_score(
        session, UUID(target_id), score=score, reviewer="ui", note=note or None
    )
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no job or shadow with that id")
    _, scores = result
    return templates.TemplateResponse(
        request,
        "_review_score.html",
        {"target_id": target_id, "score": score_state(scores)},
    )
