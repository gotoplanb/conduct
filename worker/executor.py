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
import json
import logging
from datetime import UTC, date, datetime
from time import perf_counter
from uuid import UUID

from opentelemetry.trace import Status, StatusCode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eval.scoring import apply_score
from models.client import ClientApp, ClientAppUsage
from models.job import Job
from models.shadow import JobShadow
from models.types import JobStatus, Sensitivity, stricter
from observability.metrics import record_fallback, record_job_completion
from observability.tracing import get_tracer
from prompt_loader import ResolvedPrompt, resolve_prompt
from providers.base import BaseProvider, ProviderError, ProviderResponse
from providers.registry import ProviderRegistry
from retry.base import FailureContext, FailureHandler, HandlerAction
from retry.static import StaticFailureHandler
from routing.engine import RoutingDecision, derive_seed

# Process-level lock — protects the API process if it ever runs local sync.
_local_inference_lock = asyncio.Lock()

_tracer = get_tracer(__name__)

# Default failure handler — swap to TriageFailureHandler in lifespan when v2 is ready.
_default_failure_handler: FailureHandler = StaticFailureHandler()

_SPAN_MODEL_USED = "model.used"
_SPAN_JOB_ID = "job.id"
_SPAN_TASK_TYPE = "job.task_type"

log = logging.getLogger(__name__)

# Media chaining inputs that historically held a URL. Workers no longer fetch
# them over HTTP — the canonical contract is `source_<kind>_job_id`, which
# the worker resolves to a local file path before invoking the provider.
# Listed here to gate the deprecation warning on the URL alias.
_MEDIA_SOURCE_URL_KEYS = frozenset(
    {"source_image_url", "source_audio_url", "source_video_url"}
)


async def _resolve_one_job_id_ref(
    key: str, ref_id, session: AsyncSession
) -> str:
    """Look up the referenced upstream Job and return a worker-local file
    path translated from its media_url. Raises ValueError with an operator-
    readable message on any failure."""
    try:
        ref_uuid = UUID(str(ref_id))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{key}={ref_id!r} is not a valid UUID: {e}") from e
    ref_job = await session.get(Job, ref_uuid)
    if ref_job is None:
        raise ValueError(f"{key}={ref_id}: referenced job not found")
    if not ref_job.media_url:
        raise ValueError(
            f"{key}={ref_id}: referenced job has no media_url "
            f"(status={ref_job.status})"
        )
    # media_url is stored as "/output/<uuid>.<ext>"; the worker mounts the
    # same directory at /app/output so we translate to a local filesystem
    # path that providers can hand straight to ffmpeg / ComfyUI's LoadImage
    # node without any HTTP round-trip.
    if ref_job.media_url.startswith("/output/"):
        return ref_job.media_url.replace("/output/", "/app/output/", 1)
    return ref_job.media_url


def _warn_on_deprecated_url_inputs(
    inputs: dict, resolved_url_keys: set[str]
) -> None:
    for k in inputs:
        if k in _MEDIA_SOURCE_URL_KEYS and k not in resolved_url_keys:
            log.warning(
                "media input %s uses deprecated URL form; "
                "switch to %s pointing at the upstream job's id",
                k,
                k[: -len("_url")] + "_job_id",
            )


async def _resolve_media_input_refs(
    inputs: dict, session: AsyncSession
) -> dict:
    """Translate `source_<kind>_job_id` keys → `source_<kind>_url` pointing
    at a worker-local file path, so providers stay oblivious to the DB.

    Job-id refs are the canonical chaining contract (issues #12, #14). The
    older URL form is accepted for one release as a deprecated alias and
    logs a warning; when both forms are present the job-id wins.

    A bad ref (unknown job_id, or referenced job has no media_url yet) raises
    ValueError — execute_media_job's except-Exception block surfaces it as
    Job.error, so /ui/jobs shows the operator exactly what was wrong.
    """
    if not inputs:
        return inputs or {}
    resolved = dict(inputs)
    job_id_keys = [
        k for k in inputs
        if k.startswith("source_") and k.endswith("_job_id")
    ]
    resolved_url_keys: set[str] = set()
    for key in job_id_keys:
        # source_image_job_id → source_image_url (strip "_job_id", append "_url")
        url_key = key[: -len("_job_id")] + "_url"
        resolved[url_key] = await _resolve_one_job_id_ref(
            key, inputs[key], session
        )
        resolved.pop(key, None)
        resolved_url_keys.add(url_key)

    _warn_on_deprecated_url_inputs(inputs, resolved_url_keys)
    return resolved


def _sampling_for(
    decision: RoutingDecision, prompt: str, system_prompt: str
) -> tuple[float, int | None]:
    """Resolve the (temperature, seed) to send for this job. The seed is
    derived from the input only when the profile is deterministic — otherwise
    None so the provider varies each call."""
    seed = derive_seed(prompt, system_prompt) if decision.deterministic_seed else None
    return decision.temperature, seed


async def _call_provider(
    provider: BaseProvider,
    *,
    prompt: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    is_local: bool,
    temperature: float | None = None,
    seed: int | None = None,
) -> ProviderResponse:
    with _tracer.start_as_current_span("conduct.inference") as span:
        span.set_attribute(_SPAN_MODEL_USED, model)
        span.set_attribute("model.provider", provider.name)
        if temperature is not None:
            span.set_attribute("sampling.temperature", temperature)
        span.set_attribute("sampling.seed", seed if seed is not None else -1)
        if is_local:
            async with _local_inference_lock:
                response = await provider.complete(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
        else:
            response = await provider.complete(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
        span.set_attribute("tokens.in", response.tokens_in)
        span.set_attribute("tokens.out", response.tokens_out)
        span.set_attribute("cost.usd", float(response.cost_usd))
        span.set_attribute("latency.ms", response.latency_ms)
        return response


async def _resolve_prompt_for_job(
    job: Job, client_name: str, session: AsyncSession
) -> tuple[str, ResolvedPrompt | None]:
    """Pick the prompt content for this job. A non-empty job.system_prompt
    overrides the library (no DB lookup, no version capture).

    Returns (content, resolved) — `resolved` is None when system_prompt was
    used so callers can branch on "came from request override" vs "came
    from the prompts table"."""
    if job.system_prompt:
        return job.system_prompt, None
    resolved = await resolve_prompt(session, job.task_type, client_name=client_name)
    return resolved.content, resolved


async def _try_fallback(
    *,
    job: Job,
    decision: RoutingDecision,
    primary_err: ProviderError,
    handler: FailureHandler,
    providers: ProviderRegistry,
    client: ClientApp,
    system_prompt: str,
    job_span,
) -> tuple[ProviderResponse | None, bool, str]:
    """Consult the FailureHandler and, if it returns FALLBACK, attempt the
    fallback model. Returns (response, used_fallback, error_message).
    Updates job.model_used so per-model attribution stays accurate."""
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

    if handler_decision.action != HandlerAction.FALLBACK:
        # action == FAIL (v1) or v2 actions not yet implemented
        job_span.record_exception(primary_err)
        return None, False, f"{type(primary_err).__name__}: {primary_err}"

    fb_provider_name = handler_decision.target_provider or decision.fallback_provider
    fb_model = handler_decision.target_model or decision.fallback_model
    record_fallback(
        from_provider=decision.provider,
        to_provider=fb_provider_name,
        reason=type(primary_err).__name__,
    )
    job.model_used = fb_model
    temperature, seed = _sampling_for(decision, job.prompt, system_prompt)
    try:
        response = await _call_provider(
            providers.get_for_client(client, fb_provider_name),
            prompt=job.prompt,
            model=fb_model,
            system_prompt=system_prompt,
            max_tokens=decision.max_tokens,
            is_local=fb_provider_name == "ollama",
            temperature=temperature,
            seed=seed,
        )
        return response, True, ""
    except ProviderError as fb_err:
        job_span.record_exception(fb_err)
        return None, False, (
            f"primary {type(primary_err).__name__}: {primary_err}; "
            f"fallback {type(fb_err).__name__}: {fb_err}"
        )


async def execute_job(
    *,
    job: Job,
    decision: RoutingDecision,
    client: ClientApp,
    providers: ProviderRegistry,
    session: AsyncSession,
    failure_handler: FailureHandler | None = None,
) -> Job:
    handler = failure_handler or _default_failure_handler
    client_name = client.name
    with _tracer.start_as_current_span("conduct.job") as job_span:
        job_span.set_attribute(_SPAN_JOB_ID, str(job.id))
        job_span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        job_span.set_attribute("job.sensitivity", job.sensitivity)
        job_span.set_attribute("job.client_app", client_name)
        job_span.set_attribute("job.priority", job.priority)
        job_span.set_attribute("model.requested", job.model_requested or "")
        job_span.set_attribute("routing.reason", decision.reason)

        started = perf_counter()
        system_prompt, resolved = await _resolve_prompt_for_job(job, client_name, session)

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

        temperature, seed = _sampling_for(decision, job.prompt, system_prompt)
        try:
            response = await _call_provider(
                providers.get_for_client(client, decision.provider),
                prompt=job.prompt,
                model=decision.model,
                system_prompt=system_prompt,
                max_tokens=decision.max_tokens,
                is_local=decision.provider == "ollama",
                temperature=temperature,
                seed=seed,
            )
        except ProviderError as primary_err:
            job_span.add_event(
                "primary_failed",
                {"error.type": type(primary_err).__name__, "error.message": str(primary_err)},
            )
            response, used_fallback, error_message = await _try_fallback(
                job=job,
                decision=decision,
                primary_err=primary_err,
                handler=handler,
                providers=providers,
                client=client,
                system_prompt=system_prompt,
                job_span=job_span,
            )

        job.completed_at = datetime.now(UTC)
        job.job_metadata = {
            **(job.job_metadata or {}),
            "routing": {
                "reason": decision.reason,
                "effective_sensitivity": decision.effective_sensitivity.value,
                "used_fallback": used_fallback,
            },
            "prompt": (
                {"source": "request_override", "version_id": None}
                if resolved is None
                else {
                    "source": resolved.source,
                    "version_id": resolved.version_id,
                }
            ),
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
        job_span.set_attribute(_SPAN_MODEL_USED, job.model_used or "")
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


# ----------------------------------------------------------------------------
# LLM-as-a-judge (Phase 1: single-model pointwise) — see GitHub issue #17
# ----------------------------------------------------------------------------
# A judge job scores another job's output. It's an ordinary text task (routed
# through the engine, so it gets a model + sensitivity + the deterministic
# sampling profile from its rule) distinguished only by carrying a
# `target_job_id` in its inputs. Conduct provides the PRIMITIVE (resolve the
# target, build a rubric prompt, parse a structured verdict); the client
# orchestrates (it submits the judge job and reads the verdict). Scores are
# written to the target only on an explicit opt-in (`apply_to_target`), never
# as a hidden lifecycle hook.

_JUDGE_OUTPUT_INSTRUCTION = (
    "\n\nReturn ONLY a JSON object — no prose, no markdown fences — of the form:\n"
    '{"score": <integer 1-5>, "rationale": "<one concise sentence>"}'
)


def is_judge_job(inputs: dict | None) -> bool:
    """A job is a judge job iff its inputs name a target to evaluate. Keying on
    `target_job_id` keeps judging a pure primitive — any task_type with a
    deterministic rule can be a judge — without a schema change. (A future
    explicit `task_kind` on the rule could replace this sniff.)"""
    return bool((inputs or {}).get("target_job_id"))


def _parse_judge_verdict(text: str) -> tuple[int, str]:
    """Pull (score, rationale) out of the judge model's output. Tolerant of
    surrounding prose and ```json fences: we take the outermost {...} block.
    Raises ValueError on anything unparseable so the caller fails the job
    loudly rather than recording a bogus score."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    score = int(obj["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"judge score {score} out of range 1-5")
    return score, str(obj.get("rationale", ""))


async def _resolve_judge_target(
    target_id: UUID, session: AsyncSession
) -> tuple[str | None, str, str, str]:
    """Return (prompt, response, sensitivity, kind) for the judged target.
    Tries Job first, then JobShadow (whose prompt + sensitivity come from its
    parent). Returns (None, ...) if neither matches."""
    job = await session.get(Job, target_id)
    if job is not None:
        return job.prompt, job.response, job.sensitivity, "job"

    shadow = await session.get(JobShadow, target_id)
    if shadow is not None:
        parent = await session.get(Job, shadow.parent_job_id)
        prompt = parent.prompt if parent else ""
        sensitivity = parent.sensitivity if parent else Sensitivity.CONFIDENTIAL.value
        return prompt, shadow.response, sensitivity, "shadow"

    return None, "", "", ""


async def _fail_judge(job: Job, session: AsyncSession, message: str, client_name: str) -> Job:
    job.status = JobStatus.FAILED.value
    job.error = message
    job.completed_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(job)
    record_job_completion(
        status=job.status, task_type=job.task_type, model=job.model_used or "",
        client_app=client_name,
    )
    return job


async def execute_judge_job(
    *,
    job: Job,
    decision: RoutingDecision,
    client: ClientApp,
    providers: ProviderRegistry,
    session: AsyncSession,
) -> Job:
    """Score the target named by job.inputs['target_job_id'] against the judge
    task's rubric (its resolved prompt), using the routed model under the
    rule's (deterministic) sampling. Stores the verdict on the judge job; if
    inputs['apply_to_target'] is set, also appends the score to the target via
    the normal quality-score lane (via='judge')."""
    client_name = client.name
    with _tracer.start_as_current_span("conduct.judge") as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        started = perf_counter()

        inputs = job.inputs or {}
        mode = inputs.get("mode", "pointwise")
        apply_to_target = bool(inputs.get("apply_to_target", False))
        span.set_attribute("judge.mode", mode)

        if mode != "pointwise":
            return await _fail_judge(
                job, session,
                f"judge mode {mode!r} not supported (Phase 1 is pointwise only)",
                client_name,
            )

        try:
            target_id = UUID(str(inputs.get("target_job_id")))
        except (ValueError, TypeError):
            return await _fail_judge(
                job, session,
                f"target_job_id={inputs.get('target_job_id')!r} is not a valid UUID",
                client_name,
            )

        resolved_target = await _resolve_judge_target(target_id, session)
        target_prompt, target_response, target_sensitivity, target_kind = resolved_target
        if target_prompt is None:
            return await _fail_judge(
                job, session, f"target_job_id={target_id}: not found", client_name
            )

        # Sensitivity guard: the judge sees the target's content, so the judge
        # must have routed to a model at least as strict as the target requires.
        # (The engine already vetted the judge's own sensitivity; this catches a
        # judge job created looser than its target — e.g. a public judge over a
        # confidential job, which could send that content to a cloud model.)
        target_sens = Sensitivity(target_sensitivity)
        if stricter(target_sens, decision.effective_sensitivity) != decision.effective_sensitivity:
            return await _fail_judge(
                job, session,
                f"target is {target_sens.value} but the judge resolved to "
                f"{decision.effective_sensitivity.value}; submit the judge job with "
                f"sensitivity>={target_sens.value} so it routes to an allowed model",
                client_name,
            )

        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        job.model_requested = job.model_requested or decision.model
        job.model_used = decision.model
        await session.commit()

        rubric, _resolved = await _resolve_prompt_for_job(job, client_name, session)
        system_prompt = rubric + _JUDGE_OUTPUT_INSTRUCTION
        user_prompt = (
            f"ORIGINAL PROMPT:\n{target_prompt}\n\n"
            f"RESPONSE TO EVALUATE:\n{target_response}\n\n"
            "Score the response per the rubric."
        )

        temperature, seed = _sampling_for(decision, user_prompt, system_prompt)
        try:
            response = await _call_provider(
                providers.get_for_client(client, decision.provider),
                prompt=user_prompt,
                model=decision.model,
                system_prompt=system_prompt,
                max_tokens=decision.max_tokens,
                is_local=decision.provider == "ollama",
                temperature=temperature,
                seed=seed,
            )
        except ProviderError as e:
            return await _fail_judge(job, session, f"judge provider error: {e}", client_name)

        try:
            score, rationale = _parse_judge_verdict(response.response)
        except (ValueError, KeyError) as e:  # JSONDecodeError is a ValueError subclass
            return await _fail_judge(
                job, session, f"could not parse judge verdict: {e}", client_name
            )

        applied = False
        if apply_to_target:
            applied = (
                await apply_score(
                    session, target_id, score=score, reviewer=decision.model,
                    note=rationale, via="judge",
                )
                is not None
            )

        job.status = JobStatus.COMPLETE.value
        job.response = json.dumps({"score": score, "rationale": rationale})
        job.model_used = response.model_used
        job.tokens_in = response.tokens_in
        job.tokens_out = response.tokens_out
        job.cost_usd = response.cost_usd
        job.latency_ms = response.latency_ms
        job.completed_at = datetime.now(UTC)
        job.job_metadata = {
            **(job.job_metadata or {}),
            "judge": {
                "mode": mode,
                "target_job_id": str(target_id),
                "target_kind": target_kind,
                "score": score,
                "rationale": rationale,
                "applied_to_target": applied,
            },
        }
        await _bump_usage(session, job)
        await session.commit()
        await session.refresh(job)

        span.set_attribute("judge.score", score)
        span.set_attribute("judge.applied_to_target", applied)
        record_job_completion(
            status=job.status, task_type=job.task_type, model=job.model_used or "",
            client_app=client_name, duration_s=perf_counter() - started,
            tokens_in=job.tokens_in or 0, tokens_out=job.tokens_out or 0,
            cost_usd=float(job.cost_usd or 0),
        )
        return job


async def execute_media_job(
    *,
    job: Job,
    media_provider_name: str,
    media_kind: str,
    workflow_template: str,
    providers: ProviderRegistry,
    output_dir: str,
    session: AsyncSession,
    extra_params: dict | None = None,
) -> Job:
    """Media-path counterpart to execute_job. Calls the named
    BaseMediaProvider via `produce()` and writes Job.media_url + metadata.

    `media_kind` is for observability — we set it as a span attribute so
    Tempo searches can filter `media.kind=video` etc. The actual dispatch
    is driven by `media_provider_name` (which the routing engine derived
    from the rule).

    `workflow_template` is the ComfyUI-specific knob (which JSON to load).
    For non-ComfyUI providers it's a no-op param. Passed via the
    provider's `params` so each provider decides what to honor.
    """
    from providers.media_base import BaseMediaProvider  # noqa: PLC0415

    media: BaseMediaProvider = providers.get_media(media_provider_name)
    client_name = (await session.get(ClientApp, job.client_app_id)).name

    with _tracer.start_as_current_span("conduct.media_job") as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        span.set_attribute("job.client_app", client_name)
        span.set_attribute("media.kind", media_kind)
        span.set_attribute("media.provider", media_provider_name)
        span.set_attribute("media.workflow_template", workflow_template or "")

        started = perf_counter()
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        job.model_used = workflow_template or media_provider_name
        await session.commit()

        params = dict(extra_params or {})
        if workflow_template:
            params.setdefault("workflow_template", workflow_template)

        try:
            resolved_inputs = await _resolve_media_input_refs(
                job.inputs or {}, session
            )
            response = await media.produce(
                prompt=job.prompt,
                inputs=resolved_inputs,
                output_dir=output_dir,
                output_basename=str(job.id),
                params=params,
            )
        except Exception as e:  # noqa: BLE001 — surface every failure to the row
            job.status = JobStatus.FAILED.value
            job.error = f"{type(e).__name__}: {e}"
            job.completed_at = datetime.now(UTC)
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            await session.commit()
            await session.refresh(job)
            duration_s = perf_counter() - started
            record_job_completion(
                status=job.status,
                task_type=job.task_type,
                model=job.model_used or "",
                client_app=client_name,
                duration_s=duration_s,
                tokens_in=0, tokens_out=0, cost_usd=0.0,
            )
            return job

        job.status = JobStatus.COMPLETE.value
        job.completed_at = datetime.now(UTC)
        job.latency_ms = response.latency_ms
        job.cost_usd = response.cost_usd
        job.model_used = response.model_used
        job.media_url = response.url_path
        # Mirror the existing routing metadata block; media-specific bits
        # land under metadata['media'] so they're queryable in /ui/jobs.
        job.job_metadata = {
            **(job.job_metadata or {}),
            "media": {
                "mime_type": response.mime_type,
                "width": response.width,
                "height": response.height,
                "duration_s": response.duration_s,
                "provider": response.provider,
                "extra": response.extra,
            },
        }
        await _bump_usage(session, job)
        await session.commit()
        await session.refresh(job)

        duration_s = perf_counter() - started
        span.set_attribute("job.status", job.status)
        span.set_attribute(_SPAN_MODEL_USED, job.model_used or "")
        span.set_attribute("job.duration_s", duration_s)

        record_job_completion(
            status=job.status,
            task_type=job.task_type,
            model=job.model_used or "",
            client_app=client_name,
            duration_s=duration_s,
            tokens_in=0, tokens_out=0,
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
