"""Focused tests for the fan-out + shadow-executor flow.

These exercise the code paths that the DB-prompt-resolver refactor
touched — the route-level tests go through HTTP but skip these specific
helpers, so they need direct coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

import eval.shadow_executor as shadow_executor
from eval.fanout import (
    FanoutValidationError,
    run_fanout_secondaries,
    validate_fanout_targets,
)
from eval.shadow_executor import execute_shadow
from models.job import Job
from models.prompt import Prompt
from models.shadow import JobShadow
from models.types import JobStatus
from providers.base import ProviderError, ProviderResponse
from providers.registry import ProviderRegistry


class _Stub:
    """Provider double that returns a canned response. Set `fail=True` to
    raise ProviderError on the first call."""

    def __init__(self, name: str = "anthropic", *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append(kwargs)
        if self.fail:
            raise ProviderError("simulated")
        return ProviderResponse(
            response="ok",
            tokens_in=4,
            tokens_out=2,
            cost_usd=Decimal("0.001"),
            latency_ms=11,
            model_used=kwargs.get("model", ""),
            provider=self.name,
        )


@pytest.fixture
def stub_providers() -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register(_Stub("anthropic"))
    reg.register(_Stub("ollama"))
    return reg


@pytest_asyncio.fixture
async def parent_job(db_session, seeded_client):
    """A minimal Job row that can serve as a fan-out parent."""
    c, _ = seeded_client
    job = Job(
        client_app_id=c.id,
        task_type=f"fanout_task_{uuid4().hex[:6]}",
        prompt="hello",
        sensitivity="public",
        status=JobStatus.PENDING.value,
        priority=0,
        model_requested="claude-haiku-4-5",
        system_prompt="",
    )
    db_session.add(job)
    # Seed a shared prompt so the resolver finds it.
    db_session.add(
        Prompt(task_type=job.task_type, client_id=None, content="be brief")
    )
    await db_session.commit()
    await db_session.refresh(job)
    return job, c


class _CargoStub:
    """Returns a Cargo project JSON manifest (for code-gen shadow artifacts)."""

    def __init__(self, response: str) -> None:
        self.name = "ollama"
        self._response = response

    async def complete(self, **kwargs: Any) -> ProviderResponse:
        return ProviderResponse(
            response=self._response, tokens_in=1, tokens_out=1,
            cost_usd=Decimal("0"), latency_ms=1,
            model_used=kwargs.get("model", ""), provider=self.name,
        )


_CARGO_MANIFEST = (
    '{"files": {"Cargo.toml": "[package]\\nname=\\"x\\"\\nversion=\\"0.1.0\\"'
    '\\nedition=\\"2021\\"\\n", "src/lib.rs": "pub fn f(){}\\n"}}'
)


async def test_execute_shadow_stores_cargo_artifact(
    db_session, parent_job, tmp_path, monkeypatch
) -> None:
    # Parent opted into a Cargo artifact (#35): its shadows materialize one too,
    # so they can be code_eval'd as candidates for composite preference pairs.
    job, c = parent_job
    job.inputs = {"artifact": "cargo"}
    await db_session.commit()
    monkeypatch.setattr(
        shadow_executor, "get_settings", lambda: SimpleNamespace(tts_output_dir=str(tmp_path))
    )
    shadow = JobShadow(
        parent_job_id=job.id, model="alt:1b", provider="ollama",
        status=JobStatus.PENDING.value,
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    reg = ProviderRegistry()
    reg.register(_CargoStub(_CARGO_MANIFEST))
    result = await execute_shadow(
        shadow=shadow, parent=job, client=c, max_tokens=100, providers=reg, session=db_session,
    )

    assert result.status == JobStatus.COMPLETE.value
    art = (result.shadow_metadata or {}).get("artifact")
    assert art and art["url"] == f"/output/{shadow.id}.tar"
    assert art["files"] == ["Cargo.toml", "src/lib.rs"]
    assert (tmp_path / f"{shadow.id}.tar").is_file()


async def test_execute_shadow_malformed_artifact_marks_failed(
    db_session, parent_job, tmp_path, monkeypatch
) -> None:
    job, c = parent_job
    job.inputs = {"artifact": "cargo"}
    await db_session.commit()
    monkeypatch.setattr(
        shadow_executor, "get_settings", lambda: SimpleNamespace(tts_output_dir=str(tmp_path))
    )
    shadow = JobShadow(
        parent_job_id=job.id, model="alt:1b", provider="ollama",
        status=JobStatus.PENDING.value,
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    reg = ProviderRegistry()
    reg.register(_CargoStub("just prose, no project"))  # no parseable Cargo project
    result = await execute_shadow(
        shadow=shadow, parent=job, client=c, max_tokens=100, providers=reg, session=db_session,
    )

    assert result.status == JobStatus.FAILED.value
    assert result.shadow_metadata["artifact"]["error"] == "no_files"


def test_validate_fanout_targets_accepts_cloud() -> None:
    # cloud models are always callable from the API path
    validate_fanout_targets(["claude-haiku-4-5", "claude-sonnet-4-5"])


def test_validate_fanout_targets_rejects_non_resident_local() -> None:
    with pytest.raises(FanoutValidationError):
        # 70b is not in RESIDENT_MODELS by default — must go through the worker
        validate_fanout_targets(["llama3.3:70b"])


async def test_execute_shadow_records_version_id(
    db_session, parent_job, stub_providers
) -> None:
    job, c = parent_job
    shadow = JobShadow(
        parent_job_id=job.id,
        model="claude-haiku-4-5",
        provider="anthropic",
        status=JobStatus.PENDING.value,
        shadow_metadata={"source": "test"},
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    result = await execute_shadow(
        shadow=shadow,
        parent=job,
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
    )

    assert result.status == JobStatus.COMPLETE.value
    assert result.response == "ok"
    assert result.tokens_in == 4
    # Prompt resolution metadata should be recorded — source = shared:<task>
    # and the resolved version_id should be None (we never wrote a version row
    # in the fixture).
    assert "prompt" in (result.shadow_metadata or {})
    assert result.shadow_metadata["prompt"]["source"] == f"shared:{job.task_type}"


async def test_execute_shadow_uses_system_prompt_override(
    db_session, parent_job, stub_providers
) -> None:
    """If the parent job carries a system_prompt, skip the DB lookup and
    record `source: request_override`."""
    job, c = parent_job
    job.system_prompt = "OVERRIDE"
    await db_session.commit()

    shadow = JobShadow(
        parent_job_id=job.id,
        model="claude-haiku-4-5",
        provider="anthropic",
        status=JobStatus.PENDING.value,
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    result = await execute_shadow(
        shadow=shadow,
        parent=job,
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
    )
    assert result.shadow_metadata["prompt"]["source"] == "request_override"
    assert result.shadow_metadata["prompt"]["version_id"] is None
    # The provider should have received the override, not the DB content.
    stub = stub_providers.get("anthropic")
    assert any(call["system_prompt"] == "OVERRIDE" for call in stub.calls)


async def test_execute_shadow_records_provider_failure(
    db_session, parent_job
) -> None:
    """A shadow that hits a ProviderError should land as FAILED with the
    error message — not raise back to the caller."""
    job, c = parent_job
    failing = ProviderRegistry()
    failing.register(_Stub("anthropic", fail=True))

    shadow = JobShadow(
        parent_job_id=job.id,
        model="claude-haiku-4-5",
        provider="anthropic",
        status=JobStatus.PENDING.value,
    )
    db_session.add(shadow)
    await db_session.commit()
    await db_session.refresh(shadow)

    result = await execute_shadow(
        shadow=shadow,
        parent=job,
        client=c,
        max_tokens=100,
        providers=failing,
        session=db_session,
    )
    assert result.status == JobStatus.FAILED.value
    assert "simulated" in (result.error or "")


async def test_run_fanout_secondaries_single_target(
    db_session, parent_job, stub_providers
) -> None:
    """Drive the full fan-out path with one shadow on the shared session —
    without a session_factory the shadows run sequentially, which is what
    makes this safe under the single-connection savepoint fixture."""
    job, c = parent_job
    results = await run_fanout_secondaries(
        parent=job,
        secondary_models=["claude-haiku-4-5"],
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
    )
    assert len(results) == 1
    assert results[0].status == JobStatus.COMPLETE.value
    # Parent metadata should now point at the fan-out shadow id.
    await db_session.refresh(job)
    assert "fanout" in (job.job_metadata or {})
    assert len(job.job_metadata["fanout"]["shadow_ids"]) == 1
    # `ran_at` should be ISO-parseable.
    datetime.fromisoformat(job.job_metadata["fanout"]["ran_at"]).astimezone(UTC)


async def test_run_fanout_secondaries_empty_targets_noop(
    db_session, parent_job, stub_providers
) -> None:
    job, c = parent_job
    out = await run_fanout_secondaries(
        parent=job,
        secondary_models=[],
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
    )
    assert out == []


async def test_run_fanout_secondaries_multi_target_sequential(
    db_session, parent_job, stub_providers
) -> None:
    # No session_factory → shadows run one at a time on the shared session.
    job, c = parent_job
    results = await run_fanout_secondaries(
        parent=job,
        secondary_models=["claude-haiku-4-5", "claude-sonnet-5"],
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
    )
    assert [r.status for r in results] == [JobStatus.COMPLETE.value] * 2
    await db_session.refresh(job)
    assert len(job.job_metadata["fanout"]["shadow_ids"]) == 2


async def test_run_fanout_secondaries_awaits_primary_before_shadows(
    db_session, parent_job, stub_providers
) -> None:
    # Sequential mode still honors the primary — it must finish before any
    # shadow starts, since both share the session.
    job, c = parent_job
    order: list[str] = []

    async def _primary() -> None:
        order.append("primary")

    stub = stub_providers.get("anthropic")
    orig_complete = stub.complete

    async def _complete(**kwargs: Any) -> ProviderResponse:
        order.append("shadow")
        return await orig_complete(**kwargs)

    stub.complete = _complete
    await run_fanout_secondaries(
        parent=job,
        secondary_models=["claude-haiku-4-5"],
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
        primary=_primary(),
    )
    assert order == ["primary", "shadow"]


async def test_run_fanout_secondaries_empty_targets_still_awaits_primary(
    db_session, parent_job, stub_providers
) -> None:
    job, c = parent_job
    ran = False

    async def _primary() -> None:
        nonlocal ran
        ran = True

    out = await run_fanout_secondaries(
        parent=job,
        secondary_models=[],
        client=c,
        max_tokens=100,
        providers=stub_providers,
        session=db_session,
        primary=_primary(),
    )
    assert out == []
    assert ran


async def test_run_fanout_secondaries_primary_provider_error_propagates(
    db_session, parent_job, stub_providers
) -> None:
    # Shadows swallow their own ProviderErrors (FAILED rows); one escaping the
    # batch can only come from the primary, and it must reach the route's
    # 502 translation rather than being eaten here.
    job, c = parent_job

    async def _primary() -> None:
        raise ProviderError("primary blew up")

    with pytest.raises(ProviderError, match="primary blew up"):
        await run_fanout_secondaries(
            parent=job,
            secondary_models=["claude-haiku-4-5"],
            client=c,
            max_tokens=100,
            providers=stub_providers,
            session=db_session,
            primary=_primary(),
        )


async def test_run_fanout_secondaries_concurrent_sessions(stub_providers) -> None:
    """The production path: primary + N shadows in one asyncio.gather, each
    shadow on its own session from the factory. The savepoint fixture pins
    everything to one connection, which cannot serve concurrent queries — so
    this test commits real rows via SessionLocal and cleans them up itself."""
    import secrets

    from sqlalchemy import delete

    from db.session import SessionLocal
    from models.client import ClientApp

    async with SessionLocal() as setup:
        c = ClientApp(
            name=f"fanout-conc-{uuid4().hex[:8]}",
            api_key_hash=secrets.token_hex(32),
        )
        setup.add(c)
        await setup.flush()
        job = Job(
            client_app_id=c.id,
            task_type=f"fanout_conc_{uuid4().hex[:6]}",
            prompt="hello",
            system_prompt="sys",  # skip DB prompt resolution
            sensitivity="public",
            status=JobStatus.PENDING.value,
            priority=0,
        )
        setup.add(job)
        await setup.commit()
        job_id, client_id = job.id, c.id

    ran = {"primary": False}

    async def _primary() -> None:
        ran["primary"] = True

    try:
        async with SessionLocal() as session:
            live_job = await session.get(Job, job_id)
            live_client = await session.get(ClientApp, client_id)
            results = await run_fanout_secondaries(
                parent=live_job,
                secondary_models=["claude-haiku-4-5", "claude-sonnet-5"],
                client=live_client,
                max_tokens=100,
                providers=stub_providers,
                session=session,
                primary=_primary(),
                session_factory=SessionLocal,
            )
            assert ran["primary"]
            assert {r.status for r in results} == {JobStatus.COMPLETE.value}
            assert len({r.id for r in results}) == 2
            await session.refresh(live_job)
            assert len(live_job.job_metadata["fanout"]["shadow_ids"]) == 2
    finally:
        async with SessionLocal() as cleanup:
            await cleanup.execute(
                delete(JobShadow).where(JobShadow.parent_job_id == job_id)
            )
            await cleanup.execute(delete(Job).where(Job.id == job_id))
            await cleanup.execute(delete(ClientApp).where(ClientApp.id == client_id))
            await cleanup.commit()
