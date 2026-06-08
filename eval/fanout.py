"""Parallel fan-out: run a job against the primary plus extra eval targets
concurrently, persist non-primary results as JobShadow rows.

Used by the API path when the request specifies `fanout`. All targets must
be directly callable from the API (cloud or resident-local Ollama models)
because the worker's queue-and-swap path doesn't support concurrent calls.
The route validates this constraint before invoking.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from time import perf_counter

from sqlalchemy.ext.asyncio import AsyncSession

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
) -> list[JobShadow]:
    """Create JobShadow rows for each secondary, run them all in parallel,
    return the resulting rows. The caller is responsible for running the
    primary's execute_job in the same asyncio.gather batch."""
    if not secondary_models:
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

    async def _run(s: JobShadow) -> JobShadow:
        return await execute_shadow(
            shadow=s,
            parent=parent,
            client=client,
            max_tokens=max_tokens,
            providers=providers,
            session=session,
            temperature=temperature,
            deterministic_seed=deterministic_seed,
        )

    with _tracer.start_as_current_span("conduct.fanout") as span:
        span.set_attribute("fanout.parent_job_id", str(parent.id))
        span.set_attribute("fanout.target_count", len(shadows))
        started = perf_counter()
        try:
            results = await asyncio.gather(*(_run(s) for s in shadows))
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
