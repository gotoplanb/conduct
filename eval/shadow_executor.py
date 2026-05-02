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

from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus
from observability.tracing import get_tracer
from prompt_loader import PromptResolver, get_prompt_resolver
from providers.base import ProviderError
from providers.registry import ProviderRegistry

_tracer = get_tracer(__name__)


async def execute_shadow(
    *,
    shadow: JobShadow,
    parent: Job,
    client_name: str,
    max_tokens: int,
    providers: ProviderRegistry,
    session: AsyncSession,
    prompt_resolver: PromptResolver | None = None,
) -> JobShadow:
    """Run a shadow job to completion, persist the result, return the row."""

    with _tracer.start_as_current_span("conduct.shadow") as span:
        span.set_attribute("shadow.id", str(shadow.id))
        span.set_attribute("shadow.parent_job_id", str(parent.id))
        span.set_attribute("shadow.model", shadow.model)
        span.set_attribute("shadow.provider", shadow.provider)
        span.set_attribute("job.task_type", parent.task_type)
        span.set_attribute("job.client_app", client_name)

        resolver = prompt_resolver or get_prompt_resolver()

        if parent.system_prompt:
            system_prompt = parent.system_prompt
            prompt_path: str | None = None
            prompt_hash: str | None = None
        else:
            resolved = resolver.resolve(parent.task_type, client_name=client_name)
            system_prompt = resolved.content
            prompt_path = resolved.path
            prompt_hash = resolved.git_hash

        shadow.status = JobStatus.RUNNING.value
        shadow.started_at = datetime.now(UTC)
        await session.commit()

        provider = providers.get(shadow.provider)
        started = perf_counter()
        try:
            response = await provider.complete(
                prompt=parent.prompt,
                model=shadow.model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
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
            "prompt": {
                "source": "request_override" if parent.system_prompt else "library",
                "path": prompt_path,
                "git_hash": prompt_hash,
            },
            "duration_s": perf_counter() - started,
        }
        await session.commit()
        await session.refresh(shadow)

        span.set_attribute("shadow.status", shadow.status)
        return shadow
