"""Coverage gaps in mcp_server.py: the media-task path, sensitivity refusal,
inline prompt-resolution failure, auth guards, and the bearer-header parse
branches. Fixtures mirror test_mcp.py (fixtures don't cross test modules)."""

from __future__ import annotations

import contextlib
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import mcp_server
from mcp_server import (
    OAuthMiddleware,
    _bearer_from_scope,
    _sync_eligible,
    create_job,
    list_jobs,
    submit_eval,
)
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus, Sensitivity
from routing.engine import RoutingDecision


@pytest.fixture
def task_type() -> str:
    return f"mcpgap-{uuid4().hex[:8]}"


@pytest_asyncio.fixture
def mcp_sessionmaker(db_conn, monkeypatch):
    maker = async_sessionmaker(
        bind=db_conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
        class_=AsyncSession,
    )
    monkeypatch.setattr(mcp_server, "SessionLocal", maker)
    return maker


class _RecordingQueue:
    def __init__(self) -> None:
        self.calls: list = []

    def enqueue(self, *a, **k) -> None:
        self.calls.append((a, k))


@pytest.fixture
def fake_media_queue(monkeypatch):
    # _create_media_job imports get_media_queue from worker.queue at call time,
    # so the patch must land on that module, not on mcp_server.
    import worker.queue as wq

    q = _RecordingQueue()
    monkeypatch.setattr(wq, "get_media_queue", lambda: q)
    return q


@pytest.fixture
def no_registry():
    prev = mcp_server._provider_registry
    mcp_server.set_provider_registry(None)
    yield
    mcp_server.set_provider_registry(prev)


@pytest.fixture
def sync_registry(stub_registry):
    mcp_server.set_provider_registry(stub_registry)
    yield stub_registry
    mcp_server.set_provider_registry(None)


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


# --- small guards ---------------------------------------------------------


def test_sync_eligible_for_non_ollama_provider() -> None:
    decision = RoutingDecision(
        model="claude-haiku-4-5", provider="anthropic",
        fallback_model=None, fallback_provider=None,
        effective_sensitivity=Sensitivity.PUBLIC, max_tokens=100, reason="test",
    )
    assert _sync_eligible(decision) is True


async def test_tool_without_principal_is_unauthenticated(mcp_sessionmaker) -> None:
    with pytest.raises(ValueError, match="not authenticated"):
        await list_jobs()


async def test_submit_eval_bad_uuid(db_session, seeded_client, mcp_sessionmaker) -> None:
    with principal(seeded_client[0]), pytest.raises(ValueError, match="UUID"):
        await submit_eval("not-a-uuid", 3)


async def test_list_jobs_status_filter(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    await _add_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    await _add_job(
        db_session, client_id=seeded_client[0].id, task_type=task_type,
        status=JobStatus.PENDING.value,
    )
    with principal(seeded_client[0]):
        out = await list_jobs(status=JobStatus.PENDING.value)
    assert len(out) == 1
    assert out[0]["status"] == JobStatus.PENDING.value


# --- create_job branches --------------------------------------------------


async def test_create_job_media_rule_enqueues_on_media_queue(
    db_session, seeded_client, task_type, mcp_sessionmaker, fake_media_queue
) -> None:
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="sdxl",
            fallback_model="",
            sensitivity="internal",
            media_kind="image",
        )
    )
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(
            task_type=task_type, prompt="a castle at dusk", inputs={"style": "ghibli"}
        )

    assert out["status"] == JobStatus.PENDING.value
    assert "conduct-media" in out["note"]
    # The job row itself isn't observable through the savepoint harness here
    # (the media path commits in a session nested under the tool's own, which
    # the outer close rolls back) — assert on the enqueue payload instead.
    from worker.queue import DEFAULT_MEDIA_JOB_TIMEOUT_S

    assert len(fake_media_queue.calls) == 1
    args, kwargs = fake_media_queue.calls[0]
    assert args[1] == out["job_id"]
    assert kwargs["job_id"] == out["job_id"]
    assert kwargs["job_timeout"] == DEFAULT_MEDIA_JOB_TIMEOUT_S


async def test_create_job_sensitivity_violation_raises(
    db_session, seeded_client, task_type, mcp_sessionmaker, no_registry
) -> None:
    # Cloud-only rule with no provider registry -> no cloud creds are visible
    # to routing, so neither model is allowed and decide() refuses.
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="claude-sonnet-4-5",
            fallback_model="claude-haiku-4-5",
            sensitivity="internal",
        )
    )
    await db_session.commit()

    with principal(seeded_client[0]), pytest.raises(ValueError, match="sensitivity"):
        await create_job(task_type=task_type, prompt="x")


async def test_create_job_inline_prompt_resolution_failure_marks_failed(
    db_session, seeded_client, task_type, mcp_sessionmaker, sync_registry,
    fake_redis, monkeypatch,
) -> None:
    # Sync-eligible path with no library prompt and no system_prompt override:
    # the inline executor's PromptNotFoundError must land on the job as a
    # structured failure, not escape the tool.
    monkeypatch.setattr(mcp_server, "is_resident", lambda m: True)
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="llama3.2:3b",
            fallback_model="llama3.3:70b",
            sensitivity="public",
        )
    )
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(task_type=task_type, prompt="hi")

    assert out["status"] == JobStatus.FAILED.value
    assert "prompt resolution failed" in out["error"]
    job = await db_session.get(Job, UUID(out["job_id"]))
    assert job.status == JobStatus.FAILED.value


# --- bearer parsing + middleware ------------------------------------------


def test_bearer_from_scope_ignores_other_headers() -> None:
    scope = {"headers": [(b"x-api-key", b"nope"), (b"content-type", b"application/json")]}
    assert _bearer_from_scope(scope) == ""


def test_bearer_from_scope_ignores_non_bearer_authorization() -> None:
    scope = {"headers": [(b"authorization", b"Basic dXNlcjpwdw==")]}
    assert _bearer_from_scope(scope) == ""


async def test_middleware_rejects_unknown_token(db_session, mcp_sessionmaker) -> None:
    # A bearer token that resolves to no ClientApp gets the same 401 as no
    # token at all (with the resource-metadata challenge).
    sent: list[dict] = []

    async def send(message) -> None:
        sent.append(message)

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    called = {"v": False}

    async def inner(scope, receive, send) -> None:
        called["v"] = True

    scope = {"type": "http", "headers": [(b"authorization", b"Bearer cdt_at_bogus")]}
    await OAuthMiddleware(inner)(scope, receive, send)
    assert called["v"] is False
    assert sent[0]["status"] == 401
