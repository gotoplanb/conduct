"""Parallel fan-out: run a job against the primary plus extra eval targets
concurrently, persist non-primary results as JobShadow rows.

Used by the API path when the request specifies `fanout`. All targets must
be directly callable from the API (cloud or resident-local Ollama models)
because the worker's queue-and-swap path doesn't support concurrent calls.
The route validates this constraint before invoking.

Concurrent execution requires a `session_factory` so each shadow gets its
own DB session — sharing one AsyncSession across gathered tasks corrupts
the session (flush/commit interleaving).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from eval.shadow_executor import execute_shadow
from models.client import ClientApp
from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus
from observability.tracing import get_tracer
from providers.base import ProviderError
from providers.registry import ProviderRegistry, is_cloud, provider_for_model
from providers.resident import is_resident

log = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class FanoutValidationError(ValueError):
    """A fan-out target requires the worker queue (non-resident local model)."""


def validate_fanout_targets(targets: list[str]) -> None:
    """All targets must be directly callable from the API path."""
    for model in targets:
        if is_cloud(model):
            continue
        if is_resident(model):
            continue
        raise FanoutValidationError(
            f"fanout target {model!r} is non-resident local — "
            "must be cloud or in RESIDENT_MODELS"
        )


async def run_fanout_secondaries(
    *,
    parent: Job,
    secondary_models: list[str],
    client: ClientApp,
    max_tokens: int,
    providers: ProviderRegistry,
    session: AsyncSession,
    temperature: float | None = None,
    deterministic_seed: bool = False,
    primary: Coroutine[Any, Any, Any] | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> list[JobShadow]:
    """Create JobShadow rows for each secondary, run them, return the rows.

    An AsyncSession cannot be used from concurrent tasks, so concurrency is
    gated on `session_factory`: when provided, each shadow re-fetches its row
    in its own session and everything (including `primary`, which keeps the
    shared `session` as its sole user) runs in one gather. Without a factory,
    the primary and each shadow run sequentially on the shared session.
    `parent` stays attached to the shared session throughout — the shadow
    tasks only read its already-loaded attributes.
    """
    if not secondary_models:
        if primary is not None:
            await primary
        return []

    shadows: list[JobShadow] = []
    for model in secondary_models:
        shadow = JobShadow(
            parent_job_id=parent.id,
            model=model,
            provider=provider_for_model(model),
            status=JobStatus.PENDING.value,
            shadow_metadata={"source": "fanout"},
        )
        session.add(shadow)
        shadows.append(shadow)
    await session.flush()
    await session.commit()

    async def _run(s: JobShadow, run_session: AsyncSession) -> JobShadow:
        return await execute_shadow(
            shadow=s,
            parent=parent,
            client=client,
            max_tokens=max_tokens,
            providers=providers,
            session=run_session,
            temperature=temperature,
            deterministic_seed=deterministic_seed,
        )

    async def _run_own_session(s: JobShadow) -> JobShadow:
        async with session_factory() as own:
            live = await own.get(JobShadow, s.id)
            return await _run(live, own)

    with _tracer.start_as_current_span("conduct.fanout") as span:
        span.set_attribute("fanout.parent_job_id", str(parent.id))
        span.set_attribute("fanout.target_count", len(shadows))
        started = perf_counter()
        try:
            if session_factory is not None:
                head = [primary] if primary is not None else []
                out = await asyncio.gather(
                    *head, *(_run_own_session(s) for s in shadows)
                )
                results = out[len(head):]
            else:
                if primary is not None:
                    await primary
                results = [await _run(s, session) for s in shadows]
        except ProviderError as e:
            span.record_exception(e)
            raise
        span.set_attribute("fanout.duration_s", perf_counter() - started)

    # Update parent metadata to point at the fanout shadows so callers can
    # walk back to them without an extra query.
    parent.job_metadata = {
        **(parent.job_metadata or {}),
        "fanout": {
            "shadow_ids": [str(s.id) for s in shadows],
            "ran_at": datetime.now(UTC).isoformat(),
        },
    }
    await session.commit()
    return results
