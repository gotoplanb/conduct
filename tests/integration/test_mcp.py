"""MCP server — tools (run against the test transaction) and the OAuth ASGI
gate. The transport itself is the SDK's responsibility; here we cover our
tool logic, client scoping, and the bearer-token middleware."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import mcp_server
from mcp_server import (
    OAuthMiddleware,
    create_job,
    get_job,
    list_jobs,
    list_shadows,
    list_task_types,
    submit_eval,
)
from models.client import ClientApp
from models.job import Job
from models.oauth import OAuthClient, OAuthToken
from models.prompt import Prompt
from models.routing import RoutingRule
from models.shadow import JobShadow
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


async def _add_shadow(db, *, parent_job_id, model, provider="ollama") -> JobShadow:
    from decimal import Decimal as _Decimal

    s = JobShadow(
        parent_job_id=parent_job_id,
        model=model,
        provider=provider,
        status=JobStatus.COMPLETE.value,
        response=f"resp-{model}",
        tokens_in=10,
        tokens_out=5,
        cost_usd=_Decimal("0.001"),
        latency_ms=100,
        completed_at=datetime.now(UTC),
    )
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def test_list_shadows_returns_all_for_owner(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    job = await _add_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    await _add_shadow(db_session, parent_job_id=job.id, model="gemma4:e4b")
    await _add_shadow(
        db_session, parent_job_id=job.id, model="claude-haiku-4-5", provider="anthropic"
    )
    with principal(seeded_client[0]):
        out = await list_shadows(str(job.id))
    assert out["parent_job_id"] == str(job.id)
    models = sorted(s["model"] for s in out["shadows"])
    assert models == ["claude-haiku-4-5", "gemma4:e4b"]
    haiku = next(s for s in out["shadows"] if s["model"] == "claude-haiku-4-5")
    assert haiku["provider"] == "anthropic"
    assert haiku["response"] == "resp-claude-haiku-4-5"
    assert haiku["cost_usd"] == 0.001


async def test_list_shadows_other_client_hidden(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    other = ClientApp(name=f"other-{uuid4().hex[:6]}", api_key_hash=uuid4().hex)
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)
    job = await _add_job(db_session, client_id=other.id, task_type=task_type)
    await _add_shadow(db_session, parent_job_id=job.id, model="gemma4:e4b")

    with principal(seeded_client[0]), pytest.raises(ValueError, match="no job"):
        await list_shadows(str(job.id))


async def test_list_shadows_bad_uuid(seeded_client, mcp_sessionmaker) -> None:
    with principal(seeded_client[0]), pytest.raises(ValueError, match="UUID"):
        await list_shadows("not-a-uuid")


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


async def test_submit_eval_records_score(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    job = await _add_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    with principal(seeded_client[0]):
        out = await submit_eval(str(job.id), 5, "hilarious")
    assert out["recorded"] is True
    assert out["kind"] == "job"
    await db_session.refresh(job)
    scores = job.job_metadata["quality_scores"]
    assert scores[-1]["score"] == 5
    assert scores[-1]["via"] == "mcp"


async def test_submit_eval_out_of_range(seeded_client, mcp_sessionmaker) -> None:
    with principal(seeded_client[0]), pytest.raises(ValueError, match="1-5"):
        await submit_eval(str(uuid4()), 9)


async def test_submit_eval_other_client_hidden(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    other = ClientApp(name=f"other-{uuid4().hex[:6]}", api_key_hash=uuid4().hex)
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)
    job = await _add_job(db_session, client_id=other.id, task_type=task_type)
    with principal(seeded_client[0]), pytest.raises(ValueError, match="no job"):
        await submit_eval(str(job.id), 4)


async def test_submit_eval_on_shadow_records_score(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    """Owner can score one of their own job's shadows by shadow_id."""
    job = await _add_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    shadow = await _add_shadow(
        db_session, parent_job_id=job.id, model="claude-sonnet-4-6", provider="anthropic"
    )
    with principal(seeded_client[0]):
        out = await submit_eval(str(shadow.id), 4, "sonnet was the funniest")
    assert out["kind"] == "shadow"
    assert out["recorded"] is True
    await db_session.refresh(shadow)
    scores = shadow.shadow_metadata["quality_scores"]
    assert scores[-1]["score"] == 4
    assert scores[-1]["note"] == "sonnet was the funniest"
    assert scores[-1]["via"] == "mcp"


async def test_submit_eval_on_shadow_of_other_clients_job_hidden(
    db_session, seeded_client, task_type, mcp_sessionmaker
) -> None:
    """A shadow's auth piggybacks on its parent: scoring someone else's
    shadow looks just like an unknown UUID."""
    other = ClientApp(name=f"other-{uuid4().hex[:6]}", api_key_hash=uuid4().hex)
    db_session.add(other)
    await db_session.commit()
    await db_session.refresh(other)
    other_job = await _add_job(db_session, client_id=other.id, task_type=task_type)
    other_shadow = await _add_shadow(
        db_session, parent_job_id=other_job.id, model="gemma4:e4b"
    )
    with principal(seeded_client[0]), pytest.raises(ValueError, match="no job"):
        await submit_eval(str(other_shadow.id), 5)


async def test_submit_eval_unknown_uuid(seeded_client, mcp_sessionmaker) -> None:
    """A UUID that matches neither a job nor a shadow → no job error."""
    with principal(seeded_client[0]), pytest.raises(ValueError, match="no job"):
        await submit_eval(str(uuid4()), 3)


@pytest.fixture
def sync_registry(stub_registry):
    mcp_server.set_provider_registry(stub_registry)
    yield stub_registry
    mcp_server.set_provider_registry(None)


async def test_create_job_runs_sync_for_eligible_model(
    db_session, seeded_client, task_type, mcp_sessionmaker, sync_registry, fake_redis, monkeypatch
) -> None:
    # Make the routed model sync-eligible (resident) and give it a prompt + a
    # shadow so we can assert both the inline result and the async fan-out.
    monkeypatch.setattr(mcp_server, "is_resident", lambda m: True)
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="llama3.2:3b",
            fallback_model="llama3.3:70b",
            sensitivity="public",
            eval_shadow_models=[{"model": "llama3.3:70b", "rate": 1.0}],
        )
    )
    db_session.add(Prompt(task_type=task_type, client_id=None, content="be funny"))
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(task_type=task_type, prompt="joke please")

    # Result came back inline, not pending.
    assert out["status"] == JobStatus.COMPLETE.value
    assert out["response"] == "stub response"
    # The eval shadow still fanned out asynchronously.
    job = await db_session.get(Job, UUID(out["job_id"]))
    shadows = (
        await db_session.scalars(
            select(JobShadow).where(JobShadow.parent_job_id == job.id)
        )
    ).all()
    assert len(shadows) == 1
    assert shadows[0].model == "llama3.3:70b"


async def test_create_job_force_shadows_bypasses_rate(
    db_session, seeded_client, task_type, mcp_sessionmaker, sync_registry, fake_redis, monkeypatch
) -> None:
    # Rate=0 means no shadow would normally fan out — force_shadows=True must
    # override that.
    monkeypatch.setattr(mcp_server, "is_resident", lambda m: True)
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="llama3.2:3b",
            fallback_model="llama3.3:70b",
            sensitivity="public",
            eval_shadow_models=[{"model": "llama3.3:70b", "rate": 0.0}],
        )
    )
    db_session.add(Prompt(task_type=task_type, client_id=None, content="x"))
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(task_type=task_type, prompt="p", force_shadows=True)
    job = await db_session.get(Job, UUID(out["job_id"]))
    assert (job.job_metadata or {}).get("force_shadows") is True
    shadows = (
        await db_session.scalars(
            select(JobShadow).where(JobShadow.parent_job_id == job.id)
        )
    ).all()
    assert len(shadows) == 1
    assert shadows[0].model == "llama3.3:70b"


async def test_create_job_async_when_not_sync_eligible(
    db_session, seeded_client, task_type, mcp_sessionmaker, sync_registry, fake_queue, monkeypatch
) -> None:
    # Registry is set, but the model isn't resident → must still enqueue async.
    monkeypatch.setattr(mcp_server, "is_resident", lambda m: False)
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="llama3.3:70b",
            fallback_model="llama3.3:70b",
            sensitivity="internal",
        )
    )
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(task_type=task_type, prompt="x")
    assert out["status"] == JobStatus.PENDING.value
    assert len(fake_queue.calls) == 1


async def test_create_judge_job_always_enqueues_async(
    db_session, seeded_client, task_type, mcp_sessionmaker, sync_registry, fake_queue, monkeypatch
) -> None:
    # Regression for #19: a judge job (inputs carry a target_job_id) must run on
    # the worker even when the routed model is sync-eligible (resident) — the
    # judge executor only lives on the async path. Running inline would treat
    # the bare rubric as a plain completion.
    monkeypatch.setattr(mcp_server, "is_resident", lambda m: True)
    target = await _add_job(db_session, client_id=seeded_client[0].id, task_type=task_type)
    # Judge-ness comes from inputs.target_job_id, not the task_type literal, so
    # the per-test task_type keeps us off the seeded shared judge config.
    db_session.add(
        RoutingRule(
            task_type=task_type,
            preferred_model="gemma4:e4b",
            fallback_model="gemma4:e4b",
            sensitivity="internal",
            sampling="deterministic",
        )
    )
    db_session.add(Prompt(task_type=task_type, client_id=None, content="be a judge"))
    await db_session.commit()

    with principal(seeded_client[0]):
        out = await create_job(
            task_type=task_type,
            prompt="(judge)",
            inputs={"mode": "pointwise", "target_job_id": str(target.id)},
        )

    # Queued, not run inline — and the typed inputs are echoed back on get_job.
    assert out["status"] == JobStatus.PENDING.value
    assert len(fake_queue.calls) == 1
    with principal(seeded_client[0]):
        detail = await get_job(out["job_id"])
    assert detail["inputs"]["target_job_id"] == str(target.id)


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


@pytest.mark.parametrize(
    ("scope_in", "path_out"),
    [
        ({"type": "http", "path": "/mcp", "raw_path": b"/mcp"}, "/mcp/"),
        ({"type": "http", "path": "/mcp/", "raw_path": b"/mcp/"}, "/mcp/"),
        ({"type": "http", "path": "/mcpx", "raw_path": b"/mcpx"}, "/mcpx"),
        ({"type": "http", "path": "/jobs", "raw_path": b"/jobs"}, "/jobs"),
    ],
)
async def test_mcp_trailing_slash_rewrite(scope_in: dict, path_out: str) -> None:
    from main import MCPTrailingSlashRewrite

    seen: dict = {}

    async def inner(scope, receive, send) -> None:
        seen["path"] = scope["path"]
        seen["raw_path"] = scope["raw_path"]

    await MCPTrailingSlashRewrite(inner)(scope_in, None, None)
    assert seen["path"] == path_out
    assert seen["raw_path"] == path_out.encode()


async def test_mcp_trailing_slash_rewrite_ignores_non_http() -> None:
    from main import MCPTrailingSlashRewrite

    called = {"v": False}

    async def inner(scope, receive, send) -> None:
        called["v"] = True

    await MCPTrailingSlashRewrite(inner)({"type": "lifespan"}, None, None)
    assert called["v"] is True
