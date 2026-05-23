"""MCP server — tools (run against the test transaction) and the OAuth ASGI
gate. The transport itself is the SDK's responsibility; here we cover our
tool logic, client scoping, and the bearer-token middleware."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import mcp_server
from mcp_server import (
    OAuthMiddleware,
    create_job,
    get_job,
    list_jobs,
    list_task_types,
)
from models.client import ClientApp
from models.job import Job
from models.oauth import OAuthClient, OAuthToken
from models.routing import RoutingRule
from models.types import JobStatus
from oauth_provider import hash_secret


@pytest.fixture
def task_type() -> str:
    return f"mcp-{uuid4().hex[:8]}"


@pytest_asyncio.fixture
def mcp_sessionmaker(db_conn, monkeypatch):
    """Point mcp_server.SessionLocal at the test transaction so tools (which
    open their own sessions) see test-seeded data and roll back with it."""
    maker = async_sessionmaker(
        bind=db_conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
        class_=AsyncSession,
    )
    monkeypatch.setattr(mcp_server, "SessionLocal", maker)
    return maker


@pytest.fixture
def fake_queue(monkeypatch):
    class _Q:
        def __init__(self) -> None:
            self.calls: list = []

        def enqueue(self, *a, **k) -> None:
            self.calls.append((a, k))

    q = _Q()
    monkeypatch.setattr(mcp_server, "get_queue", lambda: q)
    return q


@contextlib.contextmanager
def principal(client_app: ClientApp):
    token = mcp_server._principal.set(
        {"client_app_id": client_app.id, "client_app_name": client_app.name}
    )
    try:
        yield
    finally:
        mcp_server._principal.reset(token)


async def _add_job(db, *, client_id, task_type, status=JobStatus.COMPLETE.value) -> Job:
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="internal",
        prompt="hi",
        status=status,
        model_used="llama3.3:70b",
        response="hello",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


# --- tools ---


async def test_list_task_types(db_session, seeded_client, task_type, mcp_sessionmaker) -> None:
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="llama3.3:70b",
            fallback_model="claude-sonnet-4-5",
            sensitivity="internal",
        )
    )
    await db_session.commit()
    with principal(seeded_client[0]):
        out = await list_task_types()
    assert any(r["task_type"] == task_type for r in out)


async def test_list_jobs_is_scoped_to_client(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    mine, _ = seeded_client
    other = ClientApp(name=f"other-{uuid4().hex[:6]}", api_key_hash=uuid4().hex)
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)

    await _add_job(db_session, client_id=mine.id, task_type=task_type)
    await _add_job(db_session, client_id=other.id, task_type=task_type)

    with principal(mine):
        out = await list_jobs(limit=50)
    assert len(out) == 1
    assert all(j["task_type"] == task_type for j in out)


async def test_get_job_own(db_session, seeded_client, task_type, mcp_sessionmaker) -> None:
    job = await _add_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    with principal(seeded_client[0]):
        out = await get_job(str(job.id))
    assert out["job_id"] == str(job.id)
    assert out["response"] == "hello"


async def test_get_job_other_client_is_hidden(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    other = ClientApp(name=f"other-{uuid4().hex[:6]}", api_key_hash=uuid4().hex)
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)
    job = await _add_job(db_session, client_id=other.id, task_type=task_type)

    with principal(seeded_client[0]), pytest.raises(ValueError, match="no job"):
        await get_job(str(job.id))


async def test_get_job_bad_uuid(seeded_client, mcp_sessionmaker) -> None:
    with principal(seeded_client[0]), pytest.raises(ValueError, match="UUID"):
        await get_job("not-a-uuid")


async def test_create_job_enqueues(
    db_session, seeded_client, task_type, mcp_sessionmaker, fake_queue
) -> None:
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="llama3.3:70b",
            fallback_model="claude-sonnet-4-5",
            sensitivity="internal",
        )
    )
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(task_type=task_type, prompt="Write a bio")
    assert out["status"] == JobStatus.PENDING.value
    assert len(fake_queue.calls) == 1

    job = await db_session.get(Job, UUID(out["job_id"]))
    assert job is not None
    assert job.client_app_id == seeded_client[0].id
    assert job.task_type == task_type


async def test_create_job_invalid_sensitivity(
    seeded_client, task_type, mcp_sessionmaker, fake_queue
) -> None:
    with principal(seeded_client[0]), pytest.raises(ValueError, match="invalid sensitivity"):
        await create_job(task_type=task_type, prompt="x", sensitivity="banana")


# --- OAuth ASGI middleware ---


async def _drive(mw, scope) -> list[dict]:
    sent: list[dict] = []

    async def send(message) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    await mw(scope, receive, send)
    return sent


async def test_mcp_middleware_rejects_without_token(mcp_sessionmaker) -> None:
    called = {"v": False}

    async def inner(scope, receive, send) -> None:
        called["v"] = True

    sent = await _drive(OAuthMiddleware(inner), {"type": "http", "headers": []})
    assert called["v"] is False
    assert sent[0]["status"] == 401
    assert any(h[0] == b"www-authenticate" for h in sent[0]["headers"])


async def test_mcp_middleware_passes_with_valid_token(
    db_session, seeded_client, mcp_sessionmaker
) -> None:
    capp, _ = seeded_client
    raw = "cdt_at_mw_example"
    db_session.add(
        OAuthClient(
            client_id="cdtc_mw",
            client_secret_hash=hash_secret("s"),
            name="mw",
            client_app_id=capp.id,
            redirect_uris=[],
        )
    )
    db_session.add(
        OAuthToken(
            access_token_hash=hash_secret(raw),
            client_id="cdtc_mw",
            client_app_id=capp.id,
            scope="mcp",
            access_expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
    )
    await db_session.commit()

    seen: dict = {}

    async def inner(scope, receive, send) -> None:
        seen["principal"] = mcp_server._principal.get()

    scope = {"type": "http", "headers": [(b"authorization", f"Bearer {raw}".encode())]}
    await _drive(OAuthMiddleware(inner), scope)
    assert seen["principal"]["client_app_id"] == capp.id


def test_transport_security_allows_public_host(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        mcp_server,
        "get_settings",
        lambda: SimpleNamespace(public_base_url="https://conduct.ngrok.app"),
    )
    sec = mcp_server._transport_security()
    assert sec.enable_dns_rebinding_protection is True
    assert "conduct.ngrok.app" in sec.allowed_hosts
    assert "localhost:*" in sec.allowed_hosts


async def test_mcp_middleware_passes_non_http(mcp_sessionmaker) -> None:
    called = {"v": False}

    async def inner(scope, receive, send) -> None:
        called["v"] = True

    await OAuthMiddleware(inner)({"type": "lifespan"}, None, None)
    assert called["v"] is True
