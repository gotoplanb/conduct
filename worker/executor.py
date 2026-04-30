"""Job execution — shared between the sync API path (P3) and the RQ worker (P4).

The executor:
  1. Resolves the prompt (system_prompt override > prompt library)
  2. Calls the primary provider; on ProviderError, retries via fallback if eligible
  3. Updates the Job row in place with results, tokens, cost, latency, status
  4. Bumps ClientAppUsage for the day
  5. Emits OTel spans (`conduct.job` root, `conduct.inference` child) and
     Prometheus metrics
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from time import perf_counter

from opentelemetry.trace import Status, StatusCode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.client import ClientAppUsage
from models.job import Job
from models.types import JobStatus
from observability.metrics import record_fallback, record_job_completion
from observability.tracing import get_tracer
from prompt_loader import PromptResolver, get_prompt_resolver
from providers.base import BaseProvider, ProviderError, ProviderResponse
from providers.registry import ProviderRegistry
from retry.base import FailureContext, FailureHandler, HandlerAction
from retry.static import StaticFailureHandler
from routing.engine import RoutingDecision

# Process-level lock — protects the API process if it ever runs local sync.
_local_inference_lock = asyncio.Lock()

_tracer = get_tracer(__name__)

# Default failure handler — swap to TriageFailureHandler in lifespan when v2 is ready.
_default_failure_handler: FailureHandler = StaticFailureHandler()


async def _call_provider(
    provider: BaseProvider,
    *,
    prompt: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    is_local: bool,
) -> ProviderResponse:
    with _tracer.start_as_current_span("conduct.inference") as span:
        span.set_attribute("model.used", model)
        span.set_attribute("model.provider", provider.name)
        if is_local:
            async with _local_inference_lock:
                response = await provider.complete(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
        else:
            response = await provider.complete(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
        span.set_attribute("tokens.in", response.tokens_in)
        span.set_attribute("tokens.out", response.tokens_out)
        span.set_attribute("cost.usd", float(response.cost_usd))
        span.set_attribute("latency.ms", response.latency_ms)
        return response


async def execute_job(
    *,
    job: Job,
    decision: RoutingDecision,
    client_name: str,
    providers: ProviderRegistry,
    session: AsyncSession,
    prompt_resolver: PromptResolver | None = None,
    failure_handler: FailureHandler | None = None,
) -> Job:
    handler = failure_handler or _default_failure_handler
    with _tracer.start_as_current_span("conduct.job") as job_span:
        job_span.set_attribute("job.id", str(job.id))
        job_span.set_attribute("job.task_type", job.task_type)
        job_span.set_attribute("job.sensitivity", job.sensitivity)
        job_span.set_attribute("job.client_app", client_name)
        job_span.set_attribute("job.priority", job.priority)
        job_span.set_attribute("model.requested", job.model_requested or "")
        job_span.set_attribute("routing.reason", decision.reason)

        started = perf_counter()
        resolver = prompt_resolver or get_prompt_resolver()

        if job.system_prompt:
            system_prompt = job.system_prompt
            prompt_path: str | None = None
            prompt_hash: str | None = None
        else:
            resolved = resolver.resolve(job.task_type, client_name=client_name)
            system_prompt = resolved.content
            prompt_path = resolved.path
            prompt_hash = resolved.git_hash

        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        job.model_requested = job.model_requested or decision.model
        # Set model_used eagerly so failed jobs are still attributable for
        # per-model failure-rate analytics. Updated on fallback below.
        job.model_used = decision.model
        await session.commit()

        response: ProviderResponse | None = None
        error_message = ""
        used_fallback = False
        primary_is_local = decision.provider == "ollama"
        primary_provider = providers.get(decision.provider)

        try:
            response = await _call_provider(
                primary_provider,
                prompt=job.prompt,
                model=decision.model,
                system_prompt=system_prompt,
                max_tokens=decision.max_tokens,
                is_local=primary_is_local,
            )
        except ProviderError as primary_err:
            job_span.add_event(
                "primary_failed",
                {"error.type": type(primary_err).__name__, "error.message": str(primary_err)},
            )
            ctx = FailureContext(
                error_type=type(primary_err).__name__,
                error_message=str(primary_err),
                job_task_type=job.task_type,
                job_sensitivity=job.sensitivity,
                decision=decision,
                available_providers=frozenset(providers.names),
            )
            handler_decision = await handler.on_provider_error(ctx)
            job_span.set_attribute("failure_handler.action", handler_decision.action.value)

            if handler_decision.action == HandlerAction.FALLBACK:
                fb_provider_name = handler_decision.target_provider or decision.fallback_provider
                fb_model = handler_decision.target_model or decision.fallback_model
                record_fallback(
                    from_provider=decision.provider,
                    to_provider=fb_provider_name,
                    reason=type(primary_err).__name__,
                )
                fb_provider = providers.get(fb_provider_name)
                fb_is_local = fb_provider_name == "ollama"
                # Attribute the fallback attempt to its model regardless of outcome.
                job.model_used = fb_model
                try:
                    response = await _call_provider(
                        fb_provider,
                        prompt=job.prompt,
                        model=fb_model,
                        system_prompt=system_prompt,
                        max_tokens=decision.max_tokens,
                        is_local=fb_is_local,
                    )
                    used_fallback = True
                except ProviderError as fb_err:
                    error_message = (
                        f"primary {type(primary_err).__name__}: {primary_err}; "
                        f"fallback {type(fb_err).__name__}: {fb_err}"
                    )
                    job_span.record_exception(fb_err)
            else:
                # action == FAIL (v1) or v2 actions not yet implemented
                error_message = f"{type(primary_err).__name__}: {primary_err}"
                job_span.record_exception(primary_err)

        job.completed_at = datetime.now(UTC)
        job.job_metadata = {
            **(job.job_metadata or {}),
            "routing": {
                "reason": decision.reason,
                "effective_sensitivity": decision.effective_sensitivity.value,
                "used_fallback": used_fallback,
            },
            "prompt": {
                "source": "request_override" if job.system_prompt else "library",
                "path": prompt_path,
                "git_hash": prompt_hash,
            },
        }

        if response is not None:
            job.status = JobStatus.COMPLETE.value
            job.response = response.response
            job.model_used = response.model_used
            job.tokens_in = response.tokens_in
            job.tokens_out = response.tokens_out
            job.cost_usd = response.cost_usd
            job.latency_ms = response.latency_ms
            await _bump_usage(session, job)
        else:
            job.status = JobStatus.FAILED.value
            job.error = error_message
            job_span.set_status(Status(StatusCode.ERROR, error_message[:200]))

        await session.commit()
        await session.refresh(job)

        duration_s = perf_counter() - started
        job_span.set_attribute("job.status", job.status)
        job_span.set_attribute("model.used", job.model_used or "")
        job_span.set_attribute("job.duration_s", duration_s)

        record_job_completion(
            status=job.status,
            task_type=job.task_type,
            model=job.model_used or "",
            client_app=client_name,
            duration_s=duration_s,
            tokens_in=job.tokens_in or 0,
            tokens_out=job.tokens_out or 0,
            cost_usd=float(job.cost_usd or 0),
        )
        return job


async def _bump_usage(session: AsyncSession, job: Job) -> None:
    today: date = (job.completed_at or datetime.now(UTC)).date()
    row = await session.scalar(
        select(ClientAppUsage).where(
            ClientAppUsage.client_app_id == job.client_app_id,
            ClientAppUsage.date == today,
        )
    )
    tokens_in = job.tokens_in or 0
    tokens_out = job.tokens_out or 0
    cost = job.cost_usd or 0
    if row is None:
        session.add(
            ClientAppUsage(
                client_app_id=job.client_app_id,
                date=today,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                job_count=1,
                cost_usd=cost,
            )
        )
    else:
        row.tokens_in += tokens_in
        row.tokens_out += tokens_out
        row.job_count += 1
        row.cost_usd += cost
