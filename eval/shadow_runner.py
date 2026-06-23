"""RQ entry point for shadow jobs and the post-completion enqueue helper.

`run_shadow(shadow_id_str)` is the function registered with RQ on the
`conduct-shadow` queue. `enqueue_shadows_for_parent` is called by the worker
immediately after a parent job completes successfully — it consults the
planner, creates JobShadow rows, and enqueues a runner call for each.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from eval.shadow_executor import execute_shadow
from eval.shadow_planner import plan_shadows
from models.client import ClientApp
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus, Sensitivity
from observability.tracing import get_tracer
from providers.anthropic import AnthropicProvider
from providers.ollama import OllamaProvider
from providers.registry import ProviderRegistry
from routing.engine import rule_sampling, sampling_params
from worker.db import get_worker_session_maker

log = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

# Span attribute key used at each early-return / completion point of the
# shadow dispatch flow.
_DISPATCH_OUTCOME = "dispatch.outcome"

_providers: ProviderRegistry | None = None


def _get_providers() -> ProviderRegistry:
    global _providers
    if _providers is None:
        settings = get_settings()
        registry = ProviderRegistry()
        registry.register(OllamaProvider(
        base_url=settings.ollama_base_url, num_ctx=settings.ollama_num_ctx,
    ))
        if settings.anthropic_api_key:
            registry.register(AnthropicProvider(api_key=settings.anthropic_api_key))
        _providers = registry
    return _providers


async def today_cost_by_model(session: AsyncSession) -> dict[str, Decimal]:
    """Sum of cost_usd grouped by shadow model for the current UTC day."""
    start_of_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = (
        await session.execute(
            select(JobShadow.model, func.coalesce(func.sum(JobShadow.cost_usd), 0))
            .where(JobShadow.created_at >= start_of_day)
            .group_by(JobShadow.model)
        )
    ).all()
    return {model: Decimal(str(total)) for model, total in rows}


def should_fan_out_shadows(job: Job) -> bool:
    """Whether to fan out eval shadows for a finished primary. Normally we only
    shadow a COMPLETE primary (nothing to compare against otherwise). But when
    `force_shadows` was requested the goal is the cross-model comparison itself
    (a bench run), so a FAILED primary must not take the whole fan-out with it
    (#36) — the other candidates still need to run."""
    if job.status == JobStatus.COMPLETE.value:
        return True
    return bool((job.job_metadata or {}).get("force_shadows", False))


async def enqueue_shadows_for_parent(
    *,
    parent_job: Job,
    rule: RoutingRule | None,
    client: ClientApp,
    session: AsyncSession,
) -> list[UUID]:
    """Plan + persist + enqueue shadows. Returns the IDs of created shadows."""
    from worker.queue import DEFAULT_JOB_TIMEOUT_S, get_shadow_queue

    cost_by_model = await today_cost_by_model(session)
    force_all = bool((parent_job.job_metadata or {}).get("force_shadows", False))
    plans = plan_shadows(
        rule=rule,
        sensitivity=Sensitivity(parent_job.sensitivity),
        allow_cloud_for_internal=client.allow_cloud_for_internal,
        primary_model=parent_job.model_used or "",
        today_cost_by_model=cost_by_model,
        rng=random.Random(),
        force_all=force_all,
    )
    if not plans:
        return []

    created: list[UUID] = []
    for plan in plans:
        shadow = JobShadow(
            parent_job_id=parent_job.id,
            model=plan.model,
            provider=plan.provider,
            status=JobStatus.PENDING.value,
        )
        session.add(shadow)
        await session.flush()  # populate shadow.id without ending the txn
        created.append(shadow.id)
    await session.commit()

    queue = get_shadow_queue()
    for shadow_id in created:
        queue.enqueue(
            run_shadow,
            str(shadow_id),
            job_timeout=DEFAULT_JOB_TIMEOUT_S,
        )
    return created


def run_shadow(shadow_id_str: str) -> None:
    """Sync RQ entry point. Wraps async logic in a fresh event loop."""
    asyncio.run(_run_async(UUID(shadow_id_str)))


async def _run_async(shadow_id: UUID) -> None:
    providers = _get_providers()
    SessionMaker = get_worker_session_maker()

    with _tracer.start_as_current_span("conduct.worker.shadow_dispatch") as span:
        span.set_attribute("shadow.id", str(shadow_id))

        async with SessionMaker() as session:
            shadow = await session.get(JobShadow, shadow_id)
            if shadow is None:
                log.warning("shadow runner dequeued non-existent shadow %s", shadow_id)
                span.set_attribute(_DISPATCH_OUTCOME, "missing")
                return
            if shadow.status != JobStatus.PENDING.value:
                log.info(
                    "shadow runner skipping shadow %s with status=%s", shadow_id, shadow.status
                )
                span.set_attribute(_DISPATCH_OUTCOME, f"skip:{shadow.status}")
                return

            parent = await session.get(Job, shadow.parent_job_id)
            if parent is None:
                shadow.status = JobStatus.FAILED.value
                shadow.error = "parent job missing"
                await session.commit()
                span.set_attribute(_DISPATCH_OUTCOME, "parent_missing")
                return

            client = await session.get(ClientApp, parent.client_app_id)
            rule = await session.scalar(
                select(RoutingRule).where(
                    RoutingRule.task_type == parent.task_type,
                    RoutingRule.is_archived.is_(False),
                )
            )
            max_tokens = rule.max_tokens if rule else 1000
            # Match the primary's sampling regime so a deterministic task's
            # shadow is seeded the same way (fair A/B). Rule-level only —
            # shadows don't carry the parent's per-request sampling override.
            shadow_temperature, shadow_deterministic_seed = sampling_params(
                rule_sampling(rule)
            )

            # Local model swap, if needed. Same logic as the main worker — we
            # don't double-load if the model already matches.
            if shadow.provider == "ollama":
                ollama = providers.get("ollama")
                try:
                    loaded = await ollama.list_loaded()
                    loaded_names = {m["name"] for m in loaded}
                except Exception:
                    loaded_names = set()
                if shadow.model not in loaded_names:
                    try:
                        await ollama.load(shadow.model)
                    except Exception as e:
                        shadow.status = JobStatus.FAILED.value
                        shadow.error = f"shadow model swap failed: {e!r}"
                        await session.commit()
                        span.set_attribute(_DISPATCH_OUTCOME, "swap_failed")
                        return

            await execute_shadow(
                shadow=shadow,
                parent=parent,
                client=client,
                max_tokens=max_tokens,
                providers=providers,
                session=session,
                temperature=shadow_temperature,
                deterministic_seed=shadow_deterministic_seed,
            )
            span.set_attribute(_DISPATCH_OUTCOME, "executed")


# Re-export so RQ can find it as `eval.shadow_runner.run_shadow` and so callers
# inside the package can import without worrying about layout.
_ = enqueue_shadows_for_parent
