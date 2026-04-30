"""RQ job entry point.

`run_job(job_id_str)` is registered with RQ. On dequeue, it wraps the async
executor in a fresh asyncio loop. Provider registry is process-global (built
once per worker boot); DB sessions are per-job via NullPool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID

from sqlalchemy import select

from config.settings import get_settings
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.types import JobStatus, Sensitivity
from observability.metrics import record_job_completion, record_model_swap
from observability.tracing import get_tracer
from providers.anthropic import AnthropicProvider
from providers.ollama import OllamaProvider
from providers.registry import ProviderRegistry
from routing.engine import SensitivityViolation, decide
from worker.db import get_worker_session_maker
from worker.executor import execute_job

log = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

_providers: ProviderRegistry | None = None


def _get_providers() -> ProviderRegistry:
    global _providers
    if _providers is None:
        settings = get_settings()
        registry = ProviderRegistry()
        registry.register(OllamaProvider(base_url=settings.ollama_base_url))
        if settings.anthropic_api_key:
            registry.register(AnthropicProvider(api_key=settings.anthropic_api_key))
        _providers = registry
    return _providers


def run_job(job_id_str: str) -> None:
    """Sync RQ entry point. Wraps async logic in a fresh event loop."""
    asyncio.run(_run_async(UUID(job_id_str)))


async def _run_async(job_id: UUID) -> None:
    providers = _get_providers()
    settings = get_settings()
    SessionMaker = get_worker_session_maker()

    with _tracer.start_as_current_span("conduct.worker.dispatch") as dispatch_span:
        dispatch_span.set_attribute("job.id", str(job_id))

        async with SessionMaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                log.warning("worker dequeued non-existent job %s", job_id)
                dispatch_span.set_attribute("dispatch.outcome", "missing")
                return

            # Skip jobs that aren't pending — they may have been cancelled or
            # already processed if the queue replayed.
            if job.status != JobStatus.PENDING.value:
                log.info("worker skipping job %s with status=%s", job_id, job.status)
                dispatch_span.set_attribute("dispatch.outcome", f"skip:{job.status}")
                return

            dispatch_span.set_attribute("job.task_type", job.task_type)
            dispatch_span.set_attribute("job.sensitivity", job.sensitivity)

            client = await session.get(ClientApp, job.client_app_id)
            rule = await session.scalar(
                select(RoutingRule).where(RoutingRule.task_type == job.task_type)
            )

            try:
                decision = decide(
                    sensitivity=Sensitivity(job.sensitivity),
                    model_requested=job.model_requested or None,
                    allow_cloud_for_internal=client.allow_cloud_for_internal,
                    rule=rule,
                    default_model=settings.default_model,
                    default_sensitive_model=settings.default_sensitive_model,
                )
            except SensitivityViolation as e:
                job.status = JobStatus.FAILED.value
                job.error = f"routing: {e}"
                await session.commit()
                dispatch_span.set_attribute("dispatch.outcome", "sensitivity_violation")
                dispatch_span.record_exception(e)
                record_job_completion(
                    status=JobStatus.FAILED.value,
                    task_type=job.task_type,
                    model="",
                    client_app=client.name,
                )
                return

            dispatch_span.set_attribute("model.target", decision.model)
            dispatch_span.set_attribute("model.provider", decision.provider)

            # Local model swap, if needed. Worker is the only component that does this.
            if decision.provider == "ollama":
                ollama = providers.get("ollama")
                with _tracer.start_as_current_span("conduct.worker.swap") as swap_span:
                    swap_span.set_attribute("model.target", decision.model)
                    try:
                        loaded = await ollama.list_loaded()
                        loaded_names = {m["name"] for m in loaded}
                    except Exception:
                        loaded_names = set()
                    swap_span.set_attribute("model.already_loaded", decision.model in loaded_names)
                    if decision.model not in loaded_names:
                        from_model = next(iter(loaded_names), "")
                        swap_span.set_attribute("model.from", from_model)
                        t0 = time.perf_counter()
                        try:
                            await ollama.load(decision.model)
                        except Exception as e:
                            job.status = JobStatus.FAILED.value
                            job.error = f"model swap failed: {e!r}"
                            # Attribute the failure to the model we tried to load.
                            job.model_used = decision.model
                            await session.commit()
                            swap_span.record_exception(e)
                            dispatch_span.set_attribute("dispatch.outcome", "swap_failed")
                            record_job_completion(
                                status=JobStatus.FAILED.value,
                                task_type=job.task_type,
                                model=decision.model,
                                client_app=client.name,
                            )
                            return
                        swap_s = time.perf_counter() - t0
                        swap_span.set_attribute("swap.duration_s", swap_s)
                        record_model_swap(
                            from_model=from_model,
                            to_model=decision.model,
                            duration_s=swap_s,
                        )
                        job.job_metadata = {
                            **(job.job_metadata or {}),
                            "model_swap_ms": int(swap_s * 1000),
                        }

            await execute_job(
                job=job,
                decision=decision,
                client_name=client.name,
                providers=providers,
                session=session,
            )
            dispatch_span.set_attribute("dispatch.outcome", "executed")
