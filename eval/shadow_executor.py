"""Shadow-job executor — runs an enqueued JobShadow against its target model.

Mirrors `worker/executor.py` for the shadow case, with the differences:
  - Reads parent job for prompt content; doesn't accept a routing decision
    (the shadow's model is fixed at enqueue time).
  - No client-facing return value; results land in the JobShadow row only.
  - No ClientAppUsage bump (shadows don't count against the client's quota).
  - No fallback handling — a shadow's failure is informational, not customer-
    facing, so we record the error and move on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter

from opentelemetry.trace import Status, StatusCode
from sqlalchemy.ext.asyncio import AsyncSession

from models.client import ClientApp
from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus
from observability.tracing import get_tracer
from prompt_loader import resolve_prompt
from providers.base import ProviderError
from providers.registry import ProviderRegistry
from routing.engine import derive_seed

_tracer = get_tracer(__name__)


async def execute_shadow(
    *,
    shadow: JobShadow,
    parent: Job,
    client: ClientApp,
    max_tokens: int,
    providers: ProviderRegistry,
    session: AsyncSession,
    temperature: float | None = None,
    deterministic_seed: bool = False,
) -> JobShadow:
    """Run a shadow job to completion, persist the result, return the row.

    `temperature`/`deterministic_seed` come from the rule's sampling profile so
    the shadow generates under the same regime as the primary — otherwise a
    deterministic task's A/B would compare a seeded primary against a
    differently-sampled shadow."""

    client_name = client.name
    with _tracer.start_as_current_span("conduct.shadow") as span:
        span.set_attribute("shadow.id", str(shadow.id))
        span.set_attribute("shadow.parent_job_id", str(parent.id))
        span.set_attribute("shadow.model", shadow.model)
        span.set_attribute("shadow.provider", shadow.provider)
        span.set_attribute("job.task_type", parent.task_type)
        span.set_attribute("job.client_app", client_name)

        if parent.system_prompt:
            system_prompt = parent.system_prompt
            resolved = None
        else:
            resolved = await resolve_prompt(session, parent.task_type, client_name=client_name)
            system_prompt = resolved.content

        shadow.status = JobStatus.RUNNING.value
        shadow.started_at = datetime.now(UTC)
        await session.commit()

        seed = (
            derive_seed(parent.prompt, system_prompt) if deterministic_seed else None
        )
        provider = providers.get_for_client(client, shadow.provider)
        started = perf_counter()
        try:
            response = await provider.complete(
                prompt=parent.prompt,
                model=shadow.model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            shadow.status = JobStatus.COMPLETE.value
            shadow.response = response.response
            shadow.tokens_in = response.tokens_in
            shadow.tokens_out = response.tokens_out
            shadow.cost_usd = response.cost_usd
            shadow.latency_ms = response.latency_ms
        except ProviderError as e:
            shadow.status = JobStatus.FAILED.value
            shadow.error = f"{type(e).__name__}: {e}"
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))

        shadow.completed_at = datetime.now(UTC)
        shadow.shadow_metadata = {
            **(shadow.shadow_metadata or {}),
            "prompt": (
                {"source": "request_override", "version_id": None}
                if resolved is None
                else {"source": resolved.source, "version_id": resolved.version_id}
            ),
            "duration_s": perf_counter() - started,
        }
        await session.commit()
        await session.refresh(shadow)

        span.set_attribute("shadow.status", shadow.status)
        return shadow
