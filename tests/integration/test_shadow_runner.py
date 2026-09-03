"""Tests for the shadow RQ entry point (eval.shadow_runner).

`_run_async` is the dispatch shell around execute_shadow: it re-fetches the
shadow from a worker session, guards against missing/already-run rows, swaps
the local model in if needed, and derives max_tokens + sampling from the
parent's routing rule. execute_shadow itself is covered elsewhere
(test_fanout_and_shadow.py), so it's stubbed here to observe what the
dispatcher hands it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import eval.shadow_runner as shadow_runner
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus
from providers.registry import ProviderRegistry


class _OllamaStub:
    """Ollama double for the swap logic: scripted list_loaded + recorded
    load calls, optionally failing either."""

    name = "ollama"

    def __init__(
        self, *, loaded: list[str] | None = None,
        list_raises: bool = False, load_raises: bool = False,
    ) -> None:
        self._loaded = loaded or []
        self._list_raises = list_raises
        self._load_raises = load_raises
        self.load_calls: list[str] = []

    async def list_loaded(self) -> list[dict]:
        if self._list_raises:
            raise RuntimeError("ollama /ps unavailable")
        return [{"name": m} for m in self._loaded]

    async def load(self, model: str, keep_alive: Any = None) -> None:
        if self._load_raises:
            raise RuntimeError("out of memory")
        self.load_calls.append(model)


@pytest.fixture
def worker_maker(db_conn, monkeypatch):
    """Bind the runner's worker session factory to the test transaction."""
    maker = async_sessionmaker(
        bind=db_conn, expire_on_commit=False,
        join_transaction_mode="create_savepoint", class_=AsyncSession,
    )
    monkeypatch.setattr(shadow_runner, "get_worker_session_maker", lambda: maker)
    return maker


@pytest.fixture
def ollama_stub(monkeypatch) -> _OllamaStub:
    stub = _OllamaStub()
    reg = ProviderRegistry()
    reg.register(stub)
    monkeypatch.setattr(shadow_runner, "_get_providers", lambda: reg)
    stub.registry = reg
    return stub


@pytest.fixture
def captured_execute(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def _capture(**kwargs):
        calls.append(kwargs)
        return kwargs["shadow"]

    monkeypatch.setattr(shadow_runner, "execute_shadow", _capture)
    return calls


async def _seed_parent(db, client_id, *, status=JobStatus.COMPLETE.value) -> Job:
    job = Job(
        client_app_id=client_id, task_type=f"shadow_task_{uuid4().hex[:8]}",
        prompt="hello", sensitivity="internal", status=status,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _seed_shadow(
    db, parent_id, *, model="gemma4:e4b", provider="ollama",
    status=JobStatus.PENDING.value,
) -> JobShadow:
    shadow = JobShadow(
        parent_job_id=parent_id, model=model, provider=provider, status=status,
    )
    db.add(shadow)
    await db.commit()
    await db.refresh(shadow)
    return shadow


# --- _get_providers (module-global registry bootstrap) ---------------------


def test_get_providers_registers_anthropic_when_key_present(monkeypatch) -> None:
    monkeypatch.setattr(shadow_runner, "_providers", None)
    monkeypatch.setattr(
        shadow_runner, "get_settings",
        lambda: SimpleNamespace(
            ollama_base_url="http://localhost:11434", ollama_num_ctx=2048,
            anthropic_api_key="sk-test",
        ),
    )
    reg = shadow_runner._get_providers()
    assert reg.has("ollama")
    assert reg.has("anthropic")
    # Cached: the second call must not rebuild (settings are only read once).
    assert shadow_runner._get_providers() is reg


def test_get_providers_local_only_without_anthropic_key(monkeypatch) -> None:
    monkeypatch.setattr(shadow_runner, "_providers", None)
    monkeypatch.setattr(
        shadow_runner, "get_settings",
        lambda: SimpleNamespace(
            ollama_base_url="http://localhost:11434", ollama_num_ctx=2048,
            anthropic_api_key="",
        ),
    )
    reg = shadow_runner._get_providers()
    assert reg.has("ollama")
    assert not reg.has("anthropic")


# --- run_shadow (sync RQ entry) --------------------------------------------


def test_run_shadow_parses_id_and_runs_async(monkeypatch) -> None:
    seen: list[UUID] = []

    async def _fake(shadow_id: UUID) -> None:
        seen.append(shadow_id)

    monkeypatch.setattr(shadow_runner, "_run_async", _fake)
    sid = uuid4()
    shadow_runner.run_shadow(str(sid))
    assert seen == [sid]


# --- _run_async guards ------------------------------------------------------


async def test_missing_shadow_is_a_noop(
    worker_maker, ollama_stub, captured_execute, caplog
) -> None:
    with caplog.at_level(logging.WARNING, logger="eval.shadow_runner"):
        await shadow_runner._run_async(uuid4())
    assert any("non-existent shadow" in r.message for r in caplog.records)
    assert captured_execute == []


async def test_non_pending_shadow_is_skipped(
    db_session: AsyncSession, seeded_client, worker_maker, ollama_stub, captured_execute
) -> None:
    parent = await _seed_parent(db_session, seeded_client[0].id)
    shadow = await _seed_shadow(db_session, parent.id, status=JobStatus.COMPLETE.value)

    await shadow_runner._run_async(shadow.id)

    await db_session.refresh(shadow)
    assert shadow.status == JobStatus.COMPLETE.value  # untouched
    assert captured_execute == []


async def test_missing_parent_marks_shadow_failed(
    db_session: AsyncSession, seeded_client, worker_maker, ollama_stub, captured_execute
) -> None:
    # A genuinely orphaned shadow can't be inserted through the FK, but the
    # runner must still tolerate one (e.g. a parent hard-deleted out of band).
    # Drop the FK inside the test transaction — rolled back at teardown.
    rows = await db_session.execute(text(
        "SELECT conname FROM pg_constraint "
        "WHERE conrelid = 'job_shadows'::regclass AND contype = 'f'"
    ))
    for (conname,) in rows:
        await db_session.execute(
            text(f'ALTER TABLE job_shadows DROP CONSTRAINT "{conname}"')
        )
    await db_session.commit()
    shadow = await _seed_shadow(db_session, uuid4())

    await shadow_runner._run_async(shadow.id)

    await db_session.refresh(shadow)
    assert shadow.status == JobStatus.FAILED.value
    assert shadow.error == "parent job missing"
    assert captured_execute == []


# --- local model swap -------------------------------------------------------


async def test_swap_failure_marks_shadow_failed(
    db_session: AsyncSession, seeded_client, worker_maker, monkeypatch, captured_execute
) -> None:
    stub = _OllamaStub(loaded=[], load_raises=True)
    reg = ProviderRegistry()
    reg.register(stub)
    monkeypatch.setattr(shadow_runner, "_get_providers", lambda: reg)

    parent = await _seed_parent(db_session, seeded_client[0].id)
    shadow = await _seed_shadow(db_session, parent.id, model="huge:70b")

    await shadow_runner._run_async(shadow.id)

    await db_session.refresh(shadow)
    assert shadow.status == JobStatus.FAILED.value
    assert "shadow model swap failed" in shadow.error
    assert "out of memory" in shadow.error
    assert captured_execute == []  # never dispatched


async def test_dispatch_uses_rule_max_tokens_and_sampling(
    db_session: AsyncSession, seeded_client, worker_maker, monkeypatch, captured_execute
) -> None:
    stub = _OllamaStub(loaded=["gemma4:e4b"])  # already resident: no swap
    reg = ProviderRegistry()
    reg.register(stub)
    monkeypatch.setattr(shadow_runner, "_get_providers", lambda: reg)

    parent = await _seed_parent(db_session, seeded_client[0].id)
    db_session.add(RoutingRule(
        task_type=parent.task_type, preferred_model="gemma4:e4b",
        fallback_model="gemma4:e4b", sensitivity="internal",
        max_tokens=512, sampling="deterministic",
    ))
    await db_session.commit()
    shadow = await _seed_shadow(db_session, parent.id, model="gemma4:e4b")

    await shadow_runner._run_async(shadow.id)

    assert stub.load_calls == []  # matched the loaded set — no double-load
    assert len(captured_execute) == 1
    call = captured_execute[0]
    assert call["shadow"].id == shadow.id
    assert call["parent"].id == parent.id
    assert call["client"].id == seeded_client[0].id
    assert call["max_tokens"] == 512
    # Rule-level deterministic sampling reaches the executor.
    assert call["temperature"] == 0.0
    assert call["deterministic_seed"] is True
    assert call["providers"] is reg


async def test_cloud_shadow_skips_swap_entirely(
    db_session: AsyncSession, seeded_client, worker_maker, ollama_stub, captured_execute
) -> None:
    # A cloud shadow never touches Ollama load state.
    parent = await _seed_parent(db_session, seeded_client[0].id)
    shadow = await _seed_shadow(
        db_session, parent.id, model="claude-haiku-4-5", provider="anthropic",
    )

    await shadow_runner._run_async(shadow.id)

    assert ollama_stub.load_calls == []
    assert len(captured_execute) == 1
    assert captured_execute[0]["shadow"].id == shadow.id


async def test_dispatch_defaults_and_swaps_cold_model(
    db_session: AsyncSession, seeded_client, worker_maker, monkeypatch, captured_execute
) -> None:
    # list_loaded failing must not block dispatch — treat as nothing loaded
    # and swap the model in.
    stub = _OllamaStub(list_raises=True)
    reg = ProviderRegistry()
    reg.register(stub)
    monkeypatch.setattr(shadow_runner, "_get_providers", lambda: reg)

    parent = await _seed_parent(db_session, seeded_client[0].id)  # no rule seeded
    shadow = await _seed_shadow(db_session, parent.id, model="cold:1b")

    await shadow_runner._run_async(shadow.id)

    assert stub.load_calls == ["cold:1b"]
    assert len(captured_execute) == 1
    call = captured_execute[0]
    assert call["max_tokens"] == 1000  # no rule → default
    # No rule → balanced sampling: moderate temperature, varying seed.
    assert call["temperature"] == 0.7
    assert call["deterministic_seed"] is False
