"""Focused tests for the fan-out + shadow-executor flow.

These exercise the code paths that the DB-prompt-resolver refactor
touched — the route-level tests go through HTTP but skip these specific
helpers, so they need direct coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

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
    """Drive the full fan-out path with one shadow. (Multi-shadow concurrent
    runs share a session and trip the SQLAlchemy savepoint fixture; in
    real deployments each gather'd task gets its own connection via the
    pool, which the test fixture intentionally pins to a single conn for
    rollback safety.)"""
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
