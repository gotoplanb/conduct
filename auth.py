import hashlib
import hmac
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from db.session import get_session
from models.client import ClientApp

API_KEY_PREFIX = "cdt_"
_bearer = HTTPBearer(auto_error=False)
_MISSING_BEARER = "missing bearer token"


def generate_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


async def current_client(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> ClientApp:
    if credentials is None or not credentials.credentials:
        raise _unauthorized(_MISSING_BEARER)
    digest = hash_api_key(credentials.credentials)
    client = await session.scalar(select(ClientApp).where(ClientApp.api_key_hash == digest))
    if client is None:
        raise _unauthorized("invalid api key")
    if not client.is_active:
        raise _unauthorized("client is inactive")
    return client


def admin_only(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    settings = get_settings()
    if credentials is None or not credentials.credentials:
        raise _unauthorized(_MISSING_BEARER)
    if not hmac.compare_digest(credentials.credentials, settings.admin_key):
        raise _unauthorized("admin auth required")


async def current_client_or_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> ClientApp | None:
    """Resolve a bearer token to its ClientApp, or None if it's the admin key.
    Lets a route serve both an owning client (scoped to its own rows) and the
    admin (unscoped). Raises 403 if the token is neither."""
    if credentials is None or not credentials.credentials:
        raise _unauthorized(_MISSING_BEARER)
    if hmac.compare_digest(credentials.credentials, get_settings().admin_key):
        return None
    digest = hash_api_key(credentials.credentials)
    client = await session.scalar(select(ClientApp).where(ClientApp.api_key_hash == digest))
    if client is None:
        raise _unauthorized("invalid api key")
    if not client.is_active:
        raise _unauthorized("client is inactive")
    return client
