"""Shared fixtures for the integration / route tests.

The pattern:
- One outer transaction per test, bound to a fresh connection
- The FastAPI app's `get_session` dep is overridden to yield sessions bound to
  that connection with `join_transaction_mode="create_savepoint"`, so any
  `session.commit()` inside a route releases a savepoint rather than committing
- At test teardown we roll back the outer transaction — everything the test
  wrote is gone

Providers + redis are replaced with test doubles via dependency overrides
and monkeypatching. The lifespan is disabled (we set up providers manually)
to keep test boot fast and free of OTel instrumentation side effects.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any
from uuid import uuid4

import fakeredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from auth import generate_api_key, hash_api_key
from config.settings import get_settings
from db.session import engine, get_session
from models.client import ClientApp

# --- DB: per-test transaction with savepoints ---


@pytest_asyncio.fixture
async def db_conn() -> AsyncIterator[AsyncConnection]:
    async with engine.connect() as conn:
        trans = await conn.begin()
        try:
            yield conn
        finally:
            await trans.rollback()


@pytest_asyncio.fixture
async def db_session(db_conn: AsyncConnection) -> AsyncIterator[AsyncSession]:
    """A session bound to the test transaction. Tests use this to seed data
    directly; the app uses its own (also bound to db_conn) via dep override."""
    async with AsyncSession(
        bind=db_conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    ) as session:
        yield session


# --- Test doubles: redis + provider registry ---


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> fakeredis.FakeRedis:
    """Replace worker.queue.get_redis with a fake. rate_limit + queue both
    call get_redis, so this covers both."""
    r = fakeredis.FakeRedis()

    import worker.queue as wq

    monkeypatch.setattr(wq, "_redis", r)
    monkeypatch.setattr(wq, "_queue", None)  # force re-create against fake
    monkeypatch.setattr(wq, "_shadow_queue", None)
    return r


class _StubProvider:
    """Minimal provider double — tests that need real behavior should
    install respx routes against this class's HTTP calls instead, or
    override the specific method on the instance."""

    def __init__(self, name: str = "ollama") -> None:
        self.name = name

    async def complete(self, **kwargs: Any) -> Any:
        from providers.base import ProviderResponse

        return ProviderResponse(
            response="stub response",
            tokens_in=10,
            tokens_out=5,
            cost_usd=Decimal("0"),
            latency_ms=42,
            model_used=kwargs.get("model", ""),
            provider=self.name,
        )

    async def list_models(self) -> list[dict]:
        return [
            {"name": "llama3.3:70b", "size": 42_000_000_000, "modified_at": "2026-01-01T00:00:00Z"},
            {"name": "llama3.2:3b", "size": 2_000_000_000, "modified_at": "2026-01-01T00:00:00Z"},
        ]

    async def list_loaded(self) -> list[dict]:
        return [{"name": "llama3.3:70b"}]

    async def load(self, model: str, keep_alive: int | str | None = None) -> None:
        return None

    async def unload(self, model: str) -> None:
        return None


@pytest.fixture
def stub_registry() -> Any:
    from providers.registry import ProviderRegistry

    reg = ProviderRegistry()
    reg.register(_StubProvider("ollama"))
    reg.register(_StubProvider("anthropic"))
    return reg


# --- FastAPI app + client ---


@pytest_asyncio.fixture
async def app(db_conn: AsyncConnection, stub_registry: Any, fake_redis: Any):
    """Boot the FastAPI app with overridden deps. We bypass the lifespan by
    using ASGITransport with `lifespan='off'` in the AsyncClient — and
    manually attach the provider registry to app.state so deps that read it
    still work."""
    from deps import get_provider_registry
    from main import app as fastapi_app

    fastapi_app.state.providers = stub_registry

    async def override_get_session() -> AsyncIterator[AsyncSession]:
        async with AsyncSession(
            bind=db_conn,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        ) as s:
            yield s

    fastapi_app.dependency_overrides[get_session] = override_get_session
    fastapi_app.dependency_overrides[get_provider_registry] = lambda: stub_registry
    try:
        yield fastapi_app
    finally:
        fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --- Auth helpers ---


@pytest.fixture
def admin_token() -> str:
    """The configured admin key. Tests authenticate as admin with this."""
    return get_settings().admin_key


@pytest.fixture
def admin_headers(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest_asyncio.fixture
async def seeded_client(db_session: AsyncSession) -> tuple[ClientApp, str]:
    """Insert a client app inside the test transaction; return (row, raw_key).
    Each test gets a fresh client with a unique name."""
    raw_key = generate_api_key()
    name = f"test-client-{uuid4().hex[:8]}"
    client = ClientApp(
        name=name,
        api_key_hash=hash_api_key(raw_key),
        notes="pytest",
        rate_limit_per_minute=None,
        allow_cloud_for_internal=False,
    )
    db_session.add(client)
    await db_session.commit()
    await db_session.refresh(client)
    return client, raw_key


@pytest_asyncio.fixture
async def cloud_client(db_session: AsyncSession) -> tuple[ClientApp, str]:
    """A client that opted into cloud for internal sensitivity."""
    raw_key = generate_api_key()
    name = f"test-cloud-{uuid4().hex[:8]}"
    client = ClientApp(
        name=name,
        api_key_hash=hash_api_key(raw_key),
        notes="pytest cloud",
        rate_limit_per_minute=None,
        allow_cloud_for_internal=True,
    )
    db_session.add(client)
    await db_session.commit()
    await db_session.refresh(client)
    return client, raw_key


@pytest.fixture
def client_headers(seeded_client: tuple[ClientApp, str]) -> dict[str, str]:
    _, raw = seeded_client
    return {"Authorization": f"Bearer {raw}"}


# --- Quietly suppress the test-irrelevant warnings -----------------------


def _hash_token_for_lookup(raw: str) -> str:
    """Convenience for direct DB lookups in tests if needed."""
    return hashlib.sha256(raw.encode()).hexdigest()
