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

# Span attribute key used at each early-return / completion point of the
# worker dispatch flow.
_DISPATCH_OUTCOME = "dispatch.outcome"

_providers: ProviderRegistry | None = None


def _get_providers() -> ProviderRegistry:
    global _providers
    if _providers is None:
        import os  # noqa: PLC0415

        settings = get_settings()
        registry = ProviderRegistry()
        registry.register(OllamaProvider(base_url=settings.ollama_base_url))
        if settings.anthropic_api_key:
            registry.register(AnthropicProvider(api_key=settings.anthropic_api_key))
        # Media providers — same defaults as the API lifespan's
        # _register_media_providers. Failures are non-fatal so text jobs
        # work even if a media daemon is down.
        try:
            from providers.comfyui import ComfyUIProvider  # noqa: PLC0415

            registry.register_media(
                ComfyUIProvider(
                    base_url=os.environ.get(
                        "COMFYUI_BASE_URL", "http://host.docker.internal:8188"
                    )
                )
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from providers.acestep import ACEStepProvider  # noqa: PLC0415

            registry.register_media(
                ACEStepProvider(
                    base_url=os.environ.get(
                        "ACESTEP_BASE_URL", "http://host.docker.internal:8002"
                    )
                )
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from providers.ffmpeg_mux import FFmpegMuxProvider  # noqa: PLC0415

            registry.register_media(FFmpegMuxProvider())
        except Exception:  # noqa: BLE001
            pass
        _providers = registry
    return _providers


def run_job(job_id_str: str) -> None:
    """Sync RQ entry point. Wraps async logic in a fresh event loop."""
    asyncio.run(_run_async(UUID(job_id_str)))


async def _decide_or_fail(
    job: Job,
    client: ClientApp,
    rule: RoutingRule | None,
    settings,
    session,
    dispatch_span,
):
    """Run the routing engine. On SensitivityViolation, mark the job failed,
    emit metrics, and return None so the caller bails out."""
    try:
        return decide(
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
        dispatch_span.set_attribute(_DISPATCH_OUTCOME, "sensitivity_violation")
        dispatch_span.record_exception(e)
        record_job_completion(
            status=JobStatus.FAILED.value,
            task_type=job.task_type,
            model="",
            client_app=client.name,
        )
        return None


async def _swap_ollama_if_needed(
    decision,
    providers: ProviderRegistry,
    job: Job,
    client_name: str,
    session,
    dispatch_span,
) -> bool:
    """Ensure the target Ollama model is loaded. Returns True on success or
    when no swap was required; False when a swap was attempted but failed
    (caller should bail). Marks the job failed and records metrics on
    failure."""
    if decision.provider != "ollama":
        return True
    ollama = providers.get("ollama")
    with _tracer.start_as_current_span("conduct.worker.swap") as swap_span:
        swap_span.set_attribute("model.target", decision.model)
        try:
            loaded = await ollama.list_loaded()
            loaded_names = {m["name"] for m in loaded}
        except Exception:
            loaded_names = set()
        swap_span.set_attribute("model.already_loaded", decision.model in loaded_names)
        if decision.model in loaded_names:
            return True

        from_model = next(iter(loaded_names), "")
        swap_span.set_attribute("model.from", from_model)
        t0 = time.perf_counter()
        try:
            await ollama.load(decision.model)
        except Exception as e:
            job.status = JobStatus.FAILED.value
            job.error = f"model swap failed: {e!r}"
            job.model_used = decision.model  # for per-model failure attribution
            await session.commit()
            swap_span.record_exception(e)
            dispatch_span.set_attribute(_DISPATCH_OUTCOME, "swap_failed")
            record_job_completion(
                status=JobStatus.FAILED.value,
                task_type=job.task_type,
                model=decision.model,
                client_app=client_name,
            )
            return False
        swap_s = time.perf_counter() - t0
        swap_span.set_attribute("swap.duration_s", swap_s)
        record_model_swap(
            from_model=from_model, to_model=decision.model, duration_s=swap_s
        )
        job.job_metadata = {
            **(job.job_metadata or {}),
            "model_swap_ms": int(swap_s * 1000),
        }
        return True


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
                dispatch_span.set_attribute(_DISPATCH_OUTCOME, "missing")
                return

            # Skip jobs that aren't pending — they may have been cancelled or
            # already processed if the queue replayed.
            if job.status != JobStatus.PENDING.value:
                log.info("worker skipping job %s with status=%s", job_id, job.status)
                dispatch_span.set_attribute(_DISPATCH_OUTCOME, f"skip:{job.status}")
                return

            dispatch_span.set_attribute("job.task_type", job.task_type)
            dispatch_span.set_attribute("job.sensitivity", job.sensitivity)

            client = await session.get(ClientApp, job.client_app_id)

            # TTS jobs go through a completely separate executor — no routing,
            # no provider fallback, no model swap. They write audio to disk
            # and return a URL.
            if job.task_type == "tts":
                from tts.executor import execute_tts  # noqa: PLC0415

                await execute_tts(job=job, client=client, session=session)
                dispatch_span.set_attribute(_DISPATCH_OUTCOME, "tts_executed")
                return

            rule = await session.scalar(
                select(RoutingRule).where(
                    RoutingRule.task_type == job.task_type,
                    RoutingRule.is_archived.is_(False),
                )
            )

            # Media-task branch: image/video/audio/mux routes go through
            # execute_media_job and write Job.media_url. No routing engine
            # involvement (no fallback model, no fan-out shadows) — the
            # provider directly invokes the daemon API.
            if rule is not None and rule.media_kind != "text":
                from worker.executor import execute_media_job  # noqa: PLC0415

                provider_name, workflow_template, extra_params = _media_dispatch_for_rule(rule)
                output_dir = getattr(settings, "tts_output_dir", "/app/output")
                await execute_media_job(
                    job=job,
                    media_provider_name=provider_name,
                    media_kind=rule.media_kind,
                    workflow_template=workflow_template,
                    providers=providers,
                    output_dir=output_dir,
                    session=session,
                    extra_params=extra_params,
                )
                dispatch_span.set_attribute(_DISPATCH_OUTCOME, "media_executed")
                # Media tasks don't fan out eval shadows (yet) — comparing
                # generated videos is a different problem from comparing
                # text completions, and it's out of scope for this branch.
                return

            decision = await _decide_or_fail(
                job, client, rule, settings, session, dispatch_span
            )
            if decision is None:
                return

            dispatch_span.set_attribute("model.target", decision.model)
            dispatch_span.set_attribute("model.provider", decision.provider)

            if not await _swap_ollama_if_needed(
                decision, providers, job, client.name, session, dispatch_span
            ):
                return

            await execute_job(
                job=job,
                decision=decision,
                client=client,
                providers=providers,
                session=session,
            )
            dispatch_span.set_attribute(_DISPATCH_OUTCOME, "executed")

            # Plan + enqueue shadow jobs after a successful original. Failed
            # originals don't shadow — there's nothing meaningful to compare
            # against if the primary couldn't produce a response.
            if job.status == JobStatus.COMPLETE.value:
                from eval.shadow_runner import enqueue_shadows_for_parent

                shadow_ids = await enqueue_shadows_for_parent(
                    parent_job=job,
                    rule=rule,
                    client=client,
                    session=session,
                )
                if shadow_ids:
                    dispatch_span.set_attribute("shadows.enqueued", len(shadow_ids))


def _media_dispatch_for_rule(rule: RoutingRule) -> tuple[str, str, dict]:
    """Map a media RoutingRule to (provider_name, workflow_template, params).

    For ComfyUI-backed kinds (image/video), the rule's `preferred_model`
    holds the workflow template name (e.g. 'wander_scene_image'). For
    audio, the provider is 'acestep' and there's no template. For mux,
    the provider is 'ffmpeg_mux' with no template. The
    `eval_shadow_models[0]` slot (if present) can carry default-params
    overrides as a `{}` dict — that's how rule authors pin width/height/
    fps without touching code; ignored for now to keep this commit
    focused.
    """
    if rule.media_kind in ("image", "video"):
        return "comfyui", rule.preferred_model, {}
    if rule.media_kind == "audio":
        return "acestep", "", {}
    if rule.media_kind == "mux":
        return "ffmpeg_mux", "", {}
    raise ValueError(f"unknown media_kind: {rule.media_kind!r}")
