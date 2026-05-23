"""OAuth 2.0 authorization-server core for Conduct.

Hand-rolled, deliberately minimal: authorization-code grant with mandatory
PKCE (S256) plus refresh tokens. No external auth lib — just stdlib crypto.
All secrets/codes/tokens are stored as SHA-256 hashes; raw values exist only
in transit. Every token resolves to a ClientApp, which is how MCP-created
jobs get attributed.

The HTTP layer (routes/oauth.py) handles request parsing and redirects; this
module owns the security-sensitive logic so it can be unit-tested directly.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from base64 import urlsafe_b64encode
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from models.client import ClientApp
from models.oauth import OAuthAuthorizationCode, OAuthClient, OAuthToken

CLIENT_ID_PREFIX = "cdtc_"
CLIENT_SECRET_PREFIX = "cdts_"
ACCESS_TOKEN_PREFIX = "cdt_at_"
REFRESH_TOKEN_PREFIX = "cdt_rt_"

AUTH_CODE_TTL = timedelta(minutes=5)
ACCESS_TOKEN_TTL = timedelta(hours=1)
REFRESH_TOKEN_TTL = timedelta(days=30)

DEFAULT_SCOPE = "mcp"


class OAuthError(Exception):
    """Raised for OAuth protocol failures. `error` is the RFC 6749 code; the
    HTTP layer renders it as the standard JSON error body."""

    def __init__(self, error: str, description: str = "") -> None:
        super().__init__(f"{error}: {description}")
        self.error = error
        self.description = description


# --- crypto helpers ---


def _hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def new_client_id() -> str:
    return f"{CLIENT_ID_PREFIX}{secrets.token_urlsafe(16)}"


def new_client_secret() -> str:
    return f"{CLIENT_SECRET_PREFIX}{secrets.token_urlsafe(32)}"


def hash_secret(raw: str) -> str:
    return _hash(raw)


def verify_pkce(verifier: str, challenge: str, method: str) -> bool:
    """Only S256 is supported (plain is disallowed for security)."""
    if method != "S256" or not verifier or not challenge:
        return False
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return hmac.compare_digest(expected, challenge)


# --- client + redirect validation ---


async def get_active_client(session: AsyncSession, client_id: str) -> OAuthClient | None:
    client = await session.scalar(
        select(OAuthClient).where(OAuthClient.client_id == client_id)
    )
    if client is None or not client.is_active:
        return None
    return client


def redirect_uri_allowed(client: OAuthClient, redirect_uri: str) -> bool:
    return redirect_uri in (client.redirect_uris or [])


async def authenticate_client(
    session: AsyncSession, client_id: str, client_secret: str
) -> OAuthClient:
    """Confirm a client_id/secret pair. Raises OAuthError(invalid_client)."""
    client = await get_active_client(session, client_id)
    if client is None or not client_secret:
        raise OAuthError("invalid_client", "unknown or inactive client")
    if not hmac.compare_digest(client.client_secret_hash, _hash(client_secret)):
        raise OAuthError("invalid_client", "bad client secret")
    return client


# --- authorization codes ---


async def issue_authorization_code(
    session: AsyncSession,
    *,
    client: OAuthClient,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    scope: str,
) -> str:
    """Mint and persist an auth code; returns the raw code (shown once)."""
    raw_code = secrets.token_urlsafe(32)
    session.add(
        OAuthAuthorizationCode(
            code_hash=_hash(raw_code),
            client_id=client.client_id,
            client_app_id=client.client_app_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            scope=scope or DEFAULT_SCOPE,
            expires_at=datetime.now(UTC) + AUTH_CODE_TTL,
        )
    )
    await session.commit()
    return raw_code


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


async def redeem_authorization_code(
    session: AsyncSession,
    *,
    client: OAuthClient,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> OAuthToken:
    """Validate a code + PKCE verifier and exchange it for a token pair.
    Single-use: the code is marked used. Raises OAuthError on any mismatch."""
    row = await session.scalar(
        select(OAuthAuthorizationCode).where(OAuthAuthorizationCode.code_hash == _hash(code))
    )
    if row is None or row.used:
        raise OAuthError("invalid_grant", "code is invalid or already used")
    if row.client_id != client.client_id:
        raise OAuthError("invalid_grant", "code was issued to a different client")
    if _aware(row.expires_at) < datetime.now(UTC):
        raise OAuthError("invalid_grant", "code has expired")
    if not hmac.compare_digest(row.redirect_uri, redirect_uri):
        raise OAuthError("invalid_grant", "redirect_uri mismatch")
    if not verify_pkce(code_verifier, row.code_challenge, row.code_challenge_method):
        raise OAuthError("invalid_grant", "PKCE verification failed")

    row.used = True
    token = _create_token(session, client, row.scope)
    await session.commit()
    return token


# --- tokens ---


def _create_token(session: AsyncSession, client: OAuthClient, scope: str) -> OAuthToken:
    raw_access = f"{ACCESS_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    raw_refresh = f"{REFRESH_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    now = datetime.now(UTC)
    token = OAuthToken(
        access_token_hash=_hash(raw_access),
        refresh_token_hash=_hash(raw_refresh),
        client_id=client.client_id,
        client_app_id=client.client_app_id,
        scope=scope or DEFAULT_SCOPE,
        access_expires_at=now + ACCESS_TOKEN_TTL,
        refresh_expires_at=now + REFRESH_TOKEN_TTL,
    )
    session.add(token)
    # Stash raw values on the instance so the caller can serialize them once.
    token.raw_access_token = raw_access  # type: ignore[attr-defined]
    token.raw_refresh_token = raw_refresh  # type: ignore[attr-defined]
    return token


async def refresh_token_grant(
    session: AsyncSession, *, client: OAuthClient, refresh_token: str
) -> OAuthToken:
    """Rotate a refresh token: the old one is revoked and a new pair issued."""
    row = await session.scalar(
        select(OAuthToken).where(OAuthToken.refresh_token_hash == _hash(refresh_token))
    )
    if row is None or row.revoked:
        raise OAuthError("invalid_grant", "refresh token is invalid or revoked")
    if row.client_id != client.client_id:
        raise OAuthError("invalid_grant", "refresh token belongs to another client")
    if row.refresh_expires_at is None or _aware(row.refresh_expires_at) < datetime.now(UTC):
        raise OAuthError("invalid_grant", "refresh token has expired")

    row.revoked = True
    token = _create_token(session, client, row.scope)
    await session.commit()
    return token


async def resolve_access_token(session: AsyncSession, raw_access: str) -> ClientApp | None:
    """Resource-server entry point: map a bearer access token to its ClientApp,
    or None if missing/expired/revoked/inactive. Used by the MCP server."""
    if not raw_access:
        return None
    row = await session.scalar(
        select(OAuthToken).where(OAuthToken.access_token_hash == _hash(raw_access))
    )
    if row is None or row.revoked or _aware(row.access_expires_at) < datetime.now(UTC):
        return None
    # Deactivating the connector (OAuthClient) is a kill switch for all its
    # tokens, even ones not yet expired.
    connector = await get_active_client(session, row.client_id)
    if connector is None:
        return None
    client_app = await session.get(ClientApp, row.client_app_id)
    if client_app is None or not client_app.is_active:
        return None
    return client_app


# --- discovery metadata ---


def authorization_server_metadata() -> dict:
    base = get_settings().public_base_url.rstrip("/")
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "scopes_supported": [DEFAULT_SCOPE],
    }


def protected_resource_metadata() -> dict:
    base = get_settings().public_base_url.rstrip("/")
    return {
        "resource": f"{base}/mcp",
        "authorization_servers": [base],
        "scopes_supported": [DEFAULT_SCOPE],
    }
