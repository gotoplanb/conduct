"""OAuth 2.0 authorization-server HTTP endpoints.

Discovery + authorize (with an admin-approved consent screen) + token. The
security logic lives in `oauth_provider`; this module is request parsing,
redirects, and rendering. Designed for Claude custom connectors: manual
client pair, authorization-code grant with PKCE, refresh tokens.
"""

from __future__ import annotations

from base64 import b64decode
from pathlib import Path
from typing import Annotated
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Cookie, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from models.client import ClientApp
from oauth_provider import (
    ACCESS_TOKEN_TTL,
    OAuthError,
    authenticate_client,
    authorization_server_metadata,
    get_active_client,
    issue_authorization_code,
    protected_resource_metadata,
    redeem_authorization_code,
    redirect_uri_allowed,
    refresh_token_grant,
)
from routes.ui import _require_admin_cookie

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

router = APIRouter(tags=["oauth"], include_in_schema=False)


# --- discovery (unauthenticated) ---


@router.get("/.well-known/oauth-authorization-server")
async def authorization_server_discovery() -> JSONResponse:
    return JSONResponse(authorization_server_metadata())


@router.get("/.well-known/oauth-protected-resource")
async def protected_resource_discovery() -> JSONResponse:
    return JSONResponse(protected_resource_metadata())


# --- redirect helpers ---


def _append_query(url: str, params: dict[str, str]) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urlencode({k: v for k, v in params.items() if v})}"


def _redirect_error(
    redirect_uri: str, error: str, state: str, description: str = ""
) -> RedirectResponse:
    params = {"error": error, "error_description": description, "state": state}
    return RedirectResponse(
        _append_query(redirect_uri, params), status_code=status.HTTP_303_SEE_OTHER
    )


def _error_page(request: Request, message: str) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "oauth_error.html", {"message": message}, status_code=status.HTTP_400_BAD_REQUEST
    )


# --- authorize ---


@router.get("/oauth/authorize", response_class=HTMLResponse, response_model=None)
async def authorize_form(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    response_type: str = "",
    client_id: str = "",
    redirect_uri: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "S256",
    scope: str = "",
    state: str = "",
    conduct_admin: Annotated[str | None, Cookie()] = None,
) -> HTMLResponse | RedirectResponse:
    client = await get_active_client(session, client_id)
    # If we can't trust the client/redirect, never redirect — show an error.
    if client is None or not redirect_uri_allowed(client, redirect_uri):
        return _error_page(request, "Unknown client_id or unregistered redirect_uri.")
    if response_type != "code":
        return _redirect_error(redirect_uri, "unsupported_response_type", state)
    if code_challenge_method != "S256" or not code_challenge:
        return _redirect_error(redirect_uri, "invalid_request", state, "PKCE S256 required")

    # Approval = being logged in as admin. Bounce through login, then back here.
    if not _require_admin_cookie(conduct_admin):
        next_url = f"{request.url.path}?{request.url.query}"
        return RedirectResponse(
            f"/ui/login?next={quote(next_url, safe='')}", status_code=status.HTTP_303_SEE_OTHER
        )

    app = await session.get(ClientApp, client.client_app_id)
    return templates.TemplateResponse(
        request,
        "oauth_consent.html",
        {
            "client_name": client.name,
            "client_app_name": app.name if app else "?",
            "scope": scope or "mcp",
            "fields": {
                "response_type": response_type,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method,
                "scope": scope,
                "state": state,
            },
        },
    )


@router.post("/oauth/authorize", response_model=None)
async def authorize_submit(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    client_id: Annotated[str, Form()],
    redirect_uri: Annotated[str, Form()],
    code_challenge: Annotated[str, Form()],
    decision: Annotated[str, Form()] = "",
    code_challenge_method: Annotated[str, Form()] = "S256",
    scope: Annotated[str, Form()] = "",
    state: Annotated[str, Form()] = "",
    conduct_admin: Annotated[str | None, Cookie()] = None,
) -> HTMLResponse | RedirectResponse:
    if not _require_admin_cookie(conduct_admin):
        return _error_page(request, "Admin session required to approve.")
    client = await get_active_client(session, client_id)
    if client is None or not redirect_uri_allowed(client, redirect_uri):
        return _error_page(request, "Unknown client_id or unregistered redirect_uri.")
    if decision != "approve":
        return _redirect_error(redirect_uri, "access_denied", state)

    code = await issue_authorization_code(
        session,
        client=client,
        redirect_uri=redirect_uri,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        scope=scope,
    )
    return RedirectResponse(
        _append_query(redirect_uri, {"code": code, "state": state}),
        status_code=status.HTTP_303_SEE_OTHER,
    )


# --- token ---


def _client_creds(request: Request, form) -> tuple[str, str]:
    """Pull client_id/secret from HTTP Basic auth, falling back to the body."""
    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            decoded = b64decode(header[6:]).decode("utf-8")
            cid, _, secret = decoded.partition(":")
            return cid, secret
        except ValueError:  # UnicodeDecodeError is a ValueError subclass
            return "", ""
    return form.get("client_id", ""), form.get("client_secret", "")


def _token_error(e: OAuthError) -> JSONResponse:
    code = (
        status.HTTP_401_UNAUTHORIZED
        if e.error == "invalid_client"
        else status.HTTP_400_BAD_REQUEST
    )
    return JSONResponse(
        {"error": e.error, "error_description": e.description},
        status_code=code,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.post("/oauth/token", response_model=None)
async def token(
    request: Request, session: Annotated[AsyncSession, Depends(get_session)]
) -> JSONResponse:
    form = await request.form()
    grant_type = form.get("grant_type", "")
    client_id, client_secret = _client_creds(request, form)

    try:
        client = await authenticate_client(session, client_id, client_secret)
        if grant_type == "authorization_code":
            tok = await redeem_authorization_code(
                session,
                client=client,
                code=form.get("code", ""),
                redirect_uri=form.get("redirect_uri", ""),
                code_verifier=form.get("code_verifier", ""),
            )
        elif grant_type == "refresh_token":
            tok = await refresh_token_grant(
                session, client=client, refresh_token=form.get("refresh_token", "")
            )
        else:
            raise OAuthError("unsupported_grant_type", f"{grant_type!r} is not supported")
    except OAuthError as e:
        return _token_error(e)

    return JSONResponse(
        {
            "access_token": tok.raw_access_token,
            "token_type": "Bearer",
            "expires_in": int(ACCESS_TOKEN_TTL.total_seconds()),
            "refresh_token": tok.raw_refresh_token,
            "scope": tok.scope,
        },
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )
