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
import contextlib
import json
import logging
import re
import statistics
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import UUID

from opentelemetry.trace import Status, StatusCode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codegen.artifact import ArtifactError, build_cargo_artifact
from codegen.build_client import (
    BuildServiceError,
    RustBuildClient,
    compile_summary,
    counterexample,
    mutation_summary,
    summarize_tests,
)
from codegen.deps import CratesIndexClient, run_deps_check
from codegen.structural import analyze_tar
from codegen.train_client import DpoTrainClient, TrainServiceError
from config.settings import get_settings
from eval.scoring import apply_pairwise_preference, apply_score
from models.client import ClientApp, ClientAppUsage
from models.job import Job
from models.routing import RoutingRule
from models.shadow import JobShadow
from models.types import JobStatus, Sensitivity, stricter
from observability.metrics import record_fallback, record_job_completion
from observability.tracing import get_tracer
from prompt_loader import ResolvedPrompt, resolve_prompt
from providers.base import BaseProvider, ProviderError, ProviderResponse
from providers.registry import ProviderRegistry, is_cloud, provider_for_model
from providers.resident import pin_resident_models, unload_resident_models
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
_SPAN_CLIENT_APP = "job.client_app"
_SPAN_JUDGE = "conduct.judge"
_SPAN_JUDGE_MODE = "judge.mode"

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


def _store_code_artifact(job: Job, job_span) -> None:
    """If the job opted into a code artifact (``inputs.artifact == "cargo"``),
    parse its response into a Cargo project and store it as a tarball (#23).

    Malformed output is a **structured failure**: the job flips to FAILED with a
    stable reason under ``metadata['artifact']`` — never a crash, mirroring the
    judge parser's "never record a bogus result" stance. A plain text job
    (no opt-in) is untouched."""
    if (job.inputs or {}).get("artifact") != "cargo":
        return
    output_dir = getattr(get_settings(), "tts_output_dir", "/app/output")
    try:
        meta = build_cargo_artifact(job.response, job_id=job.id, output_dir=output_dir)
    except ArtifactError as e:
        job.status = JobStatus.FAILED.value
        job.error = f"artifact: {e.reason}"
        job.job_metadata = {
            **(job.job_metadata or {}),
            "artifact": {"error": e.reason, "detail": e.detail},
        }
        job_span.set_status(Status(StatusCode.ERROR, f"artifact: {e.reason}"))
        job_span.set_attribute("artifact.error", e.reason)
        return
    job.media_url = meta["url"]
    job.job_metadata = {**(job.job_metadata or {}), "artifact": meta}
    job_span.set_attribute("artifact.format", meta["format"])
    job_span.set_attribute("artifact.file_count", meta["file_count"])


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
        job_span.set_attribute(_SPAN_CLIENT_APP, client_name)
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
            _store_code_artifact(job, job_span)
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

_PAIRWISE_OUTPUT_INSTRUCTION = (
    "\n\nYou are comparing two responses, A and B, to the same prompt. Decide "
    "which one better satisfies the rubric (or 'tie' if genuinely equal). Return "
    "ONLY a JSON object — no prose, no markdown fences — of the form:\n"
    '{"winner": "A" | "B" | "tie", "rationale": "<one concise sentence>"}'
)


def _judge_output_instruction(dimensions: list[str] | None) -> str:
    """The output-format instruction appended to the rubric. With explicit
    dimensions (the caller's vocabulary, #18) the judge scores each one;
    otherwise a single overall score."""
    if not dimensions:
        return _JUDGE_OUTPUT_INSTRUCTION
    fields = ", ".join(f'"{d}": <integer 1-5>' for d in dimensions)
    names = ", ".join(dimensions)
    # The rubric above may name its own criteria; the caller's dimensions
    # override them. Be emphatic so the model uses these exact keys, not the
    # rubric's, otherwise the verdict parser rejects the mismatch.
    return (
        "\n\nIMPORTANT: ignore any criteria named above. Score the response on "
        "EXACTLY these dimensions and no others: " + names + ".\n"
        "Return ONLY a JSON object — no prose, no markdown fences — using these "
        "exact keys:\n"
        '{"scores": {' + fields + '}, "rationale": "<one concise sentence>"}'
    )


def is_judge_job(inputs: dict | None) -> bool:
    """A job is a judge job iff its inputs name a target to evaluate. Keying on
    `target_job_id` keeps judging a pure primitive — any task_type with a
    deterministic rule can be a judge — without a schema change. (A future
    explicit `task_kind` on the rule could replace this sniff.)"""
    return bool((inputs or {}).get("target_job_id"))


def _parse_judge_verdict(
    text: str, dimensions: list[str] | None = None
) -> tuple[int, dict[str, int] | None, str]:
    """Pull (overall, dimension_scores, rationale) from the judge's output.
    Tolerant of surrounding prose and ```json fences (outermost {...} block).
    With `dimensions`, expects a `scores` map covering each named dimension and
    derives the overall as their mean; without, expects a single `score`.
    Raises ValueError/KeyError on anything unparseable so the caller fails the
    job loudly rather than recording a bogus score."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    rationale = str(obj.get("rationale", ""))

    if dimensions:
        raw = obj.get("scores") or {}
        # Match dimension keys case-/whitespace-insensitively — local models
        # routinely re-case ("Humor") or pad keys. We still require every
        # requested dimension to be present (never invent a score).
        folded = {str(k).strip().lower(): v for k, v in raw.items()}
        scores: dict[str, int] = {}
        for d in dimensions:
            key = d.strip().lower()
            if key not in folded:
                raise KeyError(d)  # model omitted a requested dimension
            v = int(folded[key])
            if not 1 <= v <= 5:
                raise ValueError(f"dimension {d} score {v} out of range 1-5")
            scores[d] = v
        overall = round(sum(scores.values()) / len(scores))
        return overall, scores, rationale

    score = int(obj["score"])
    if not 1 <= score <= 5:
        raise ValueError(f"judge score {score} out of range 1-5")
    return score, None, rationale


def _parse_pairwise_verdict(text: str) -> tuple[str, str]:
    """Pull (winner, rationale) from a pairwise verdict. winner ∈ {A, B, TIE}.
    Same fence/prose tolerance as the pointwise parser."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    winner = str(obj.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        raise ValueError(f"pairwise winner {winner!r} not in A/B/tie")
    return winner, str(obj.get("rationale", ""))


def _reconcile_pairwise(
    w1: str, w2: str, target_id: UUID, against_id: UUID
) -> tuple[UUID | None, UUID | None, bool, bool]:
    """Reconcile the two order-swapped verdicts into (chosen, rejected, tie,
    position_consistent). Call 1 presented A=target, B=against; call 2 swapped
    them (A=against, B=target). A confident result needs BOTH calls to name the
    same job as winner — otherwise the judge has position bias on this pair and
    we record a tie (no DPO pair)."""
    win1 = {"A": target_id, "B": against_id}.get(w1)  # None on TIE
    win2 = {"A": against_id, "B": target_id}.get(w2)
    if win1 is not None and win1 == win2:
        rejected = against_id if win1 == target_id else target_id
        return win1, rejected, False, True
    return None, None, True, False


def _pairwise_user_prompt(original_prompt: str, resp_a: str, resp_b: str) -> str:
    return (
        f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
        f"RESPONSE A:\n{resp_a}\n\n"
        f"RESPONSE B:\n{resp_b}\n\n"
        "Which response better satisfies the rubric?"
    )


def _judge_sensitivity_block(decision: RoutingDecision, target_sensitivity: str) -> str | None:
    """Return an error message if the judge resolved looser than the target it
    must read (the judge sees the target's content); else None."""
    target_sens = Sensitivity(target_sensitivity)
    if stricter(target_sens, decision.effective_sensitivity) != decision.effective_sensitivity:
        return (
            f"target is {target_sens.value} but the judge resolved to "
            f"{decision.effective_sensitivity.value}; submit the judge job with "
            f"sensitivity>={target_sens.value} so it routes to an allowed model"
        )
    return None


async def _resolve_and_guard_target(
    target_id: UUID, decision: RoutingDecision, session: AsyncSession
) -> tuple[str, str, str, str, str]:
    """Resolve a judged target and apply the sensitivity guard in one step.
    Returns (prompt, response, kind, producer_model, error). A non-empty error
    means the caller should fail the judge job."""
    prompt, response, sensitivity, kind, producer = await _resolve_judge_target(
        target_id, session
    )
    if prompt is None:
        return "", "", "", "", f"target {target_id}: not found"
    block = _judge_sensitivity_block(decision, sensitivity)
    if block:
        return "", "", "", "", block
    return prompt, response, kind, producer, ""


async def _run_judge_model(
    decision: RoutingDecision,
    client: ClientApp,
    providers: ProviderRegistry,
    *,
    system_prompt: str,
    user_prompt: str,
) -> ProviderResponse:
    """Call the judge model under the decision's (deterministic) sampling."""
    temperature, seed = _sampling_for(decision, user_prompt, system_prompt)
    return await _call_provider(
        providers.get_for_client(client, decision.provider),
        prompt=user_prompt,
        model=decision.model,
        system_prompt=system_prompt,
        max_tokens=decision.max_tokens,
        is_local=decision.provider == "ollama",
        temperature=temperature,
        seed=seed,
    )


async def _resolve_judge_target(
    target_id: UUID, session: AsyncSession
) -> tuple[str | None, str, str, str, str]:
    """Return (prompt, response, sensitivity, kind, producer_model) for the
    judged target. Tries Job first, then JobShadow (whose prompt + sensitivity
    come from its parent). `producer_model` is the model that generated the
    output — used by the panel to exclude self-preference. Returns (None, ...)
    if neither matches."""
    job = await session.get(Job, target_id)
    if job is not None:
        return job.prompt, job.response, job.sensitivity, "job", job.model_used or ""

    shadow = await session.get(JobShadow, target_id)
    if shadow is not None:
        parent = await session.get(Job, shadow.parent_job_id)
        prompt = parent.prompt if parent else ""
        sensitivity = parent.sensitivity if parent else Sensitivity.CONFIDENTIAL.value
        return prompt, shadow.response, sensitivity, "shadow", shadow.model or ""

    return None, "", "", "", ""


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


async def _mark_judge_running(job: Job, decision: RoutingDecision, session: AsyncSession) -> None:
    job.status = JobStatus.RUNNING.value
    job.started_at = datetime.now(UTC)
    job.model_requested = job.model_requested or decision.model
    job.model_used = decision.model
    await session.commit()


async def _finish_judge(
    job: Job,
    responses: list[ProviderResponse],
    session: AsyncSession,
    client_name: str,
    started: float,
) -> None:
    """Shared completion path: roll up tokens/cost across the judge's model
    call(s), mark complete, bump usage, emit metrics. job.response +
    job_metadata['judge'] must already be set by the caller."""
    job.status = JobStatus.COMPLETE.value
    job.model_used = responses[-1].model_used
    job.tokens_in = sum(r.tokens_in for r in responses)
    job.tokens_out = sum(r.tokens_out for r in responses)
    job.cost_usd = sum((r.cost_usd for r in responses), Decimal("0"))
    job.latency_ms = sum(r.latency_ms for r in responses)
    job.completed_at = datetime.now(UTC)
    await _bump_usage(session, job)
    await session.commit()
    await session.refresh(job)
    record_job_completion(
        status=job.status, task_type=job.task_type, model=job.model_used or "",
        client_app=client_name, duration_s=perf_counter() - started,
        tokens_in=job.tokens_in or 0, tokens_out=job.tokens_out or 0,
        cost_usd=float(job.cost_usd or 0),
    )


async def execute_judge_job(
    *,
    job: Job,
    decision: RoutingDecision,
    client: ClientApp,
    providers: ProviderRegistry,
    session: AsyncSession,
    rule: RoutingRule | None = None,
) -> Job:
    """Dispatch a judge job by inputs['mode'] (default 'pointwise'). A judge job
    scores or compares other jobs' outputs through the routed (deterministic)
    model; 'panel' fans out across a jury of models from the rule. See #17."""
    mode = (job.inputs or {}).get("mode", "pointwise")
    if mode == "pointwise":
        return await _execute_pointwise_judge(
            job=job, decision=decision, client=client, providers=providers, session=session
        )
    if mode == "pairwise":
        return await _execute_pairwise_judge(
            job=job, decision=decision, client=client, providers=providers, session=session
        )
    if mode == "panel":
        return await _execute_panel_judge(
            job=job, decision=decision, client=client, providers=providers,
            session=session, rule=rule,
        )
    return await _fail_judge(
        job, session,
        f"judge mode {mode!r} not supported (use 'pointwise', 'pairwise', or 'panel')",
        client.name,
    )


async def _execute_pointwise_judge(
    *, job: Job, decision: RoutingDecision, client: ClientApp,
    providers: ProviderRegistry, session: AsyncSession,
) -> Job:
    """Score one target against the rubric → {score, rationale}. On
    apply_to_target, append the score to the target via the quality-score lane
    (via='judge')."""
    client_name = client.name
    with _tracer.start_as_current_span(_SPAN_JUDGE) as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        span.set_attribute(_SPAN_JUDGE_MODE, "pointwise")
        started = perf_counter()
        inputs = job.inputs or {}
        apply_to_target = bool(inputs.get("apply_to_target", False))
        dimensions = inputs.get("dimensions")  # optional named dimensions (#18)

        try:
            target_id = UUID(str(inputs.get("target_job_id")))
        except (ValueError, TypeError):
            return await _fail_judge(
                job, session,
                f"target_job_id={inputs.get('target_job_id')!r} is not a valid UUID",
                client_name,
            )

        target_prompt, target_response, target_kind, _producer, err = (
            await _resolve_and_guard_target(target_id, decision, session)
        )
        if err:
            return await _fail_judge(job, session, err, client_name)

        await _mark_judge_running(job, decision, session)
        rubric, _resolved = await _resolve_prompt_for_job(job, client_name, session)
        system_prompt = rubric + _judge_output_instruction(dimensions)
        user_prompt = (
            f"ORIGINAL PROMPT:\n{target_prompt}\n\n"
            f"RESPONSE TO EVALUATE:\n{target_response}\n\n"
            "Score the response per the rubric."
        )

        try:
            response = await _run_judge_model(
                decision, client, providers, system_prompt=system_prompt, user_prompt=user_prompt
            )
        except ProviderError as e:
            return await _fail_judge(job, session, f"judge provider error: {e}", client_name)
        try:
            score, dim_scores, rationale = _parse_judge_verdict(response.response, dimensions)
        except (ValueError, KeyError) as e:  # JSONDecodeError is a ValueError subclass
            return await _fail_judge(
                job, session,
                f"could not parse judge verdict: {e} | raw: {(response.response or '')[:300]!r}",
                client_name,
            )

        applied = False
        if apply_to_target:
            applied = (
                await apply_score(
                    session, target_id, score=score, scores=dim_scores,
                    reviewer=decision.model, note=rationale, via="judge",
                )
                is not None
            )

        # Self-contained verdict (carries mode + applied_to_target) in both the
        # response and metadata.judge, so a JSONL reader needn't cross-reference
        # inputs to know what it's parsing (#20).
        verdict = {
            "mode": "pointwise", "score": score, "scores": dim_scores,
            "rationale": rationale, "applied_to_target": applied,
        }
        job.response = json.dumps(verdict)
        job.job_metadata = {
            **(job.job_metadata or {}),
            "judge": {
                **verdict, "target_job_id": str(target_id), "target_kind": target_kind,
            },
        }
        span.set_attribute("judge.score", score)
        span.set_attribute("judge.applied_to_target", applied)
        await _finish_judge(job, [response], session, client_name, started)
        return job


async def _execute_pairwise_judge(
    *, job: Job, decision: RoutingDecision, client: ClientApp,
    providers: ProviderRegistry, session: AsyncSession,
) -> Job:
    """Compare two targets against the rubric with ORDER-SWAP: judge A-vs-B and
    B-vs-A, then require both calls to agree on the winner. Agreement →
    chosen/rejected; disagreement → tie (position bias detected, no DPO pair).
    On apply_to_target, record the preference on both participants."""
    client_name = client.name
    with _tracer.start_as_current_span(_SPAN_JUDGE) as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        span.set_attribute(_SPAN_JUDGE_MODE, "pairwise")
        started = perf_counter()
        inputs = job.inputs or {}
        apply_to_target = bool(inputs.get("apply_to_target", False))

        try:
            target_id = UUID(str(inputs.get("target_job_id")))
            against_id = UUID(str(inputs.get("against_job_id")))
        except (ValueError, TypeError):
            return await _fail_judge(
                job, session,
                "pairwise judging requires a valid target_job_id and against_job_id",
                client_name,
            )
        if target_id == against_id:
            return await _fail_judge(
                job, session, "target_job_id and against_job_id must differ", client_name
            )

        tp, tr, tkind, _p1, err = await _resolve_and_guard_target(target_id, decision, session)
        if err:
            return await _fail_judge(job, session, err, client_name)
        _ap, ar, akind, _p2, err = await _resolve_and_guard_target(against_id, decision, session)
        if err:
            return await _fail_judge(job, session, err, client_name)

        await _mark_judge_running(job, decision, session)
        rubric, _resolved = await _resolve_prompt_for_job(job, client_name, session)
        system_prompt = rubric + _PAIRWISE_OUTPUT_INSTRUCTION

        try:
            # The two responses are compared against the target's prompt (for
            # primary-vs-shadow both share it). Call 2 swaps A/B to cancel
            # position bias.
            r1 = await _run_judge_model(
                decision, client, providers, system_prompt=system_prompt,
                user_prompt=_pairwise_user_prompt(tp, tr, ar),
            )
            r2 = await _run_judge_model(
                decision, client, providers, system_prompt=system_prompt,
                user_prompt=_pairwise_user_prompt(tp, ar, tr),
            )
        except ProviderError as e:
            return await _fail_judge(job, session, f"judge provider error: {e}", client_name)
        try:
            w1, rat1 = _parse_pairwise_verdict(r1.response)
            w2, rat2 = _parse_pairwise_verdict(r2.response)
        except (ValueError, KeyError) as e:
            return await _fail_judge(
                job, session, f"could not parse pairwise verdict: {e}", client_name
            )

        chosen, rejected, tie, consistent = _reconcile_pairwise(w1, w2, target_id, against_id)

        applied = False
        if apply_to_target and not tie:
            await apply_pairwise_preference(
                session, chosen, against_job_id=rejected, outcome="win", judge_job_id=job.id
            )
            await apply_pairwise_preference(
                session, rejected, against_job_id=chosen, outcome="loss", judge_job_id=job.id
            )
            applied = True

        # Full verdict in BOTH response and metadata.judge so a reader of the
        # stored response can tell a true tie from a position flip, and read the
        # two swap rationales (#20). metadata adds the target/against id+kind.
        verdict = {
            "mode": "pairwise",
            "chosen_job_id": str(chosen) if chosen else None,
            "rejected_job_id": str(rejected) if rejected else None,
            "tie": tie, "position_consistent": consistent,
            "rationale_ab": rat1, "rationale_ba": rat2, "applied_to_target": applied,
        }
        job.response = json.dumps(verdict)
        job.job_metadata = {
            **(job.job_metadata or {}),
            "judge": {
                **verdict, "target_job_id": str(target_id), "against_job_id": str(against_id),
                "target_kind": tkind, "against_kind": akind,
            },
        }
        span.set_attribute("judge.tie", tie)
        span.set_attribute("judge.position_consistent", consistent)
        await _finish_judge(job, [r1, r2], session, client_name, started)
        return job


# --- panel / jury (Phase 3) -----------------------------------------------


def _model_family(model: str) -> str:
    """Coarse vendor/family key for self-preference exclusion. Strips cloud geo
    prefixes + the version/size tail, leaving the leading name token:
    gemma4:e4b→gemma, llama3.2:3b→llama, qwen3.5:9b→qwen,
    us.anthropic.claude-haiku-4-5→claude, mistral-small3.2→mistral."""
    name = re.sub(r"^(us|eu|apac)\.", "", model.lower())
    name = re.sub(r"^anthropic\.", "", name)
    m = re.match(r"[a-z]+", name)
    return m.group(0) if m else name


def _panelist_allowed(
    model: str, sensitivity: Sensitivity, allow_cloud_for_internal: bool
) -> bool:
    """Mirror the routing engine's sensitivity gate for a single panelist (the
    panelist will read the target's content, so a confidential target bars
    cloud jurors)."""
    if not is_cloud(model):
        return True
    if sensitivity == Sensitivity.CONFIDENTIAL:
        return False
    if sensitivity == Sensitivity.INTERNAL:
        return allow_cloud_for_internal
    return True


def _resolve_min_panel_n(inputs: dict, rule: RoutingRule | None) -> int | None:
    """The panel quorum floor (#21): per-request inputs.min_panel_n overrides the
    rule-level default. Returns None when neither is set (no floor). A value of
    <= 1 is treated as no floor (a single survivor is already the default-OK
    case; the zero-survivor path fails on its own)."""
    raw = inputs.get("min_panel_n")
    if raw is None:
        raw = rule.min_panel_n if rule is not None else None
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 1 else None


def _rule_panel(rule: RoutingRule | None) -> list[str]:
    """Default panel for a judge rule: its preferred model + its
    eval_shadow_models (repurposed as the jury, since judges don't shadow)."""
    if rule is None:
        return []
    panel = [rule.preferred_model]
    panel += [
        s["model"]
        for s in (rule.eval_shadow_models or [])
        if isinstance(s, dict) and s.get("model")
    ]
    return panel


def _build_panel(
    candidates: list[str], producer_model: str,
    sensitivity: Sensitivity, allow_cloud_for_internal: bool,
) -> tuple[list[str], dict[str, str]]:
    """From candidate models, drop duplicates, the target's producer FAMILY
    (self-preference), and any model the target's sensitivity disallows.
    Returns (eligible, excluded={model: reason})."""
    producer_fam = _model_family(producer_model) if producer_model else ""
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    seen: set[str] = set()
    for m in candidates:
        if m in seen:
            continue
        seen.add(m)
        if producer_fam and _model_family(m) == producer_fam:
            excluded[m] = f"self-preference (same family as producer: {producer_fam})"
        elif not _panelist_allowed(m, sensitivity, allow_cloud_for_internal):
            excluded[m] = f"sensitivity ({sensitivity.value})"
        else:
            eligible.append(m)
    return eligible, excluded


async def _run_panelist(
    model: str, decision: RoutingDecision, client: ClientApp, providers: ProviderRegistry,
    *, system_prompt: str, user_prompt: str,
) -> ProviderResponse:
    provider_name = provider_for_model(model)
    is_local = provider_name == "ollama"
    if is_local:
        # Best-effort load (a no-op for resident models); a real failure
        # surfaces on the call below and the panelist is skipped.
        with contextlib.suppress(Exception):
            await providers.get("ollama").load(model)
    temperature, seed = _sampling_for(decision, user_prompt, system_prompt)
    return await _call_provider(
        providers.get_for_client(client, provider_name),
        prompt=user_prompt, model=model, system_prompt=system_prompt,
        max_tokens=decision.max_tokens, is_local=is_local,
        temperature=temperature, seed=seed,
    )


async def _run_panel(
    eligible: list[str], decision: RoutingDecision, client: ClientApp,
    providers: ProviderRegistry, system_prompt: str, user_prompt: str,
    dimensions: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[ProviderResponse]]:
    """Run every eligible panelist; a panelist that errors or returns an
    unparseable verdict is recorded as a failure and skipped (a jury tolerates
    a missing juror). Returns (scored, failures, responses)."""
    scored: list[dict] = []
    failures: list[dict] = []
    responses: list[ProviderResponse] = []
    for m in eligible:
        try:
            resp = await _run_panelist(
                m, decision, client, providers, system_prompt=system_prompt, user_prompt=user_prompt
            )
        except ProviderError as e:
            failures.append({"model": m, "error": f"provider: {e}"[:160]})
            continue
        try:
            score, dim_scores, rationale = _parse_judge_verdict(resp.response, dimensions)
        except (ValueError, KeyError) as e:
            failures.append({"model": m, "error": f"parse: {e}"[:160]})
            continue
        responses.append(resp)
        scored.append({"model": m, "score": score, "scores": dim_scores, "rationale": rationale})
    return scored, failures, responses


def _aggregate_panel(scored: list[dict], dimensions: list[str] | None = None) -> dict:
    """Aggregate panelist scores by MEDIAN (robust to an outlier juror). Spread
    >= 2 flags a contested verdict worth a human's eyes. With dimensions, also
    medians each named dimension across the jurors (#18)."""
    scores = [p["score"] for p in scored]
    spread = max(scores) - min(scores)
    agg = {
        "score": int(round(statistics.median(scores))),
        "n": len(scores), "min": min(scores), "max": max(scores),
        "spread": spread, "disagreement": spread >= 2, "scores": scores,
    }
    if dimensions:
        dim_medians: dict[str, int] = {}
        for d in dimensions:
            vals = [p["scores"][d] for p in scored if p.get("scores") and d in p["scores"]]
            if vals:
                dim_medians[d] = int(round(statistics.median(vals)))
        agg["dimensions"] = dim_medians
    return agg


async def _execute_panel_judge(
    *, job: Job, decision: RoutingDecision, client: ClientApp,
    providers: ProviderRegistry, session: AsyncSession, rule: RoutingRule | None,
) -> Job:
    """Jury of diverse models pointwise-scoring one target → median. Excludes
    the producer's family (self-preference) and sensitivity-disallowed jurors."""
    client_name = client.name
    with _tracer.start_as_current_span(_SPAN_JUDGE) as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        span.set_attribute(_SPAN_JUDGE_MODE, "panel")
        started = perf_counter()
        inputs = job.inputs or {}
        apply_to_target = bool(inputs.get("apply_to_target", False))
        dimensions = inputs.get("dimensions")  # optional named dimensions (#18)

        try:
            target_id = UUID(str(inputs.get("target_job_id")))
        except (ValueError, TypeError):
            return await _fail_judge(
                job, session,
                f"target_job_id={inputs.get('target_job_id')!r} is not a valid UUID",
                client_name,
            )

        target_prompt, target_response, target_kind, producer, err = (
            await _resolve_and_guard_target(target_id, decision, session)
        )
        if err:
            return await _fail_judge(job, session, err, client_name)

        candidates = inputs.get("panel") or _rule_panel(rule)
        if not candidates:
            return await _fail_judge(
                job, session,
                "panel mode needs panel models (inputs.panel or the judge "
                "rule's eval_shadow_models)",
                client_name,
            )
        eligible, excluded = _build_panel(
            candidates, producer, decision.effective_sensitivity, client.allow_cloud_for_internal
        )
        if not eligible:
            return await _fail_judge(
                job, session, f"no eligible panelists after exclusions: {excluded}", client_name
            )

        await _mark_judge_running(job, decision, session)
        rubric, _resolved = await _resolve_prompt_for_job(job, client_name, session)
        system_prompt = rubric + _judge_output_instruction(dimensions)
        user_prompt = (
            f"ORIGINAL PROMPT:\n{target_prompt}\n\n"
            f"RESPONSE TO EVALUATE:\n{target_response}\n\n"
            "Score the response per the rubric."
        )

        scored, failures, responses = await _run_panel(
            eligible, decision, client, providers, system_prompt, user_prompt, dimensions
        )
        if not scored:
            return await _fail_judge(
                job, session, f"all {len(eligible)} panelists failed: {failures}", client_name
            )

        # Quorum floor (#21): when min_panel_n is set (request overrides rule),
        # a jury that lost too many jurors fails loudly rather than writing a
        # degraded low-n "median" to the target. Unset/<=1 = today's behavior.
        min_panel_n = _resolve_min_panel_n(inputs, rule)
        if min_panel_n is not None and len(scored) < min_panel_n:
            return await _fail_judge(
                job, session,
                f"panel quorum not met: {len(scored)} of {min_panel_n} required jurors "
                f"scored (failures: {failures}, excluded: {excluded})",
                client_name,
            )

        agg = _aggregate_panel(scored, dimensions)
        applied = False
        if apply_to_target:
            applied = (
                await apply_score(
                    session, target_id, score=agg["score"], scores=agg.get("dimensions"),
                    reviewer="panel:" + ",".join(p["model"] for p in scored),
                    note=f"panel median {agg['score']} of {agg['n']} {agg['scores']}",
                    via="judge-panel",
                )
                is not None
            )

        # One full verdict, written to BOTH response and metadata.judge so a
        # downstream reader of the stored response can see why a juror dropped
        # (#20) — panelists/failures/excluded/spread, not just the median.
        verdict = {
            "mode": "panel", "score": agg["score"], "dimensions": agg.get("dimensions"),
            "n": agg["n"], "min_panel_n": min_panel_n,
            "spread": agg["spread"], "disagreement": agg["disagreement"],
            "panelists": scored, "failures": failures, "excluded": excluded,
            "applied_to_target": applied,
        }
        job.response = json.dumps(verdict)
        job.job_metadata = {
            **(job.job_metadata or {}),
            "judge": {
                **verdict, "target_job_id": str(target_id), "target_kind": target_kind,
            },
        }
        span.set_attribute("judge.score", agg["score"])
        span.set_attribute("judge.panel_n", agg["n"])
        span.set_attribute("judge.disagreement", agg["disagreement"])
        await _finish_judge(job, responses, session, client_name, started)
        return job


# --- code_eval: deterministic compile/test gate (P1.3, #25) ----------------

_VIA_CODE_EVAL = "code-eval"
# The compile dimension is binary; map onto the 1-5 quality lane so it rolls up
# alongside human/judge scores (#18). Pass -> 5, fail -> 1.
_COMPILE_PASS = 5
_COMPILE_FAIL = 1
# cargo-mutants is slow (compiles+tests each mutant); give it room. The service
# kills a run at _MUTANTS_TIMEOUT_S; the client waits a bit longer.
_MUTANTS_TIMEOUT_S = 300
_CODE_EVAL_CLIENT_TIMEOUT_S = 600.0


async def _load_code_eval_target(
    session: AsyncSession, inputs: dict, output_dir: str
) -> tuple[UUID, Path]:
    """Parse inputs.target_job_id, resolve it to a Job or one of its shadows
    (#35), and locate its artifact tarball. Raises ValueError on any miss
    (caller turns it into a structured code_eval failure)."""
    try:
        target_id = UUID(str(inputs.get("target_job_id")))
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid target_job_id {inputs.get('target_job_id')!r}") from e
    target = await session.get(Job, target_id) or await session.get(JobShadow, target_id)
    return target_id, _resolve_code_eval_target(target, output_dir)


def _resolve_code_eval_target(target: Job | JobShadow | None, output_dir: str) -> Path:
    """Locate a target's stored Cargo artifact tarball (#23) on the shared output
    dir. Works for a Job (job_metadata) or a JobShadow (shadow_metadata, #35),
    keyed on the artifact's url. Raises ValueError with a clear reason on a miss."""
    if target is None:
        raise ValueError("target job not found")
    meta = getattr(target, "job_metadata", None) or getattr(target, "shadow_metadata", None) or {}
    url = (meta.get("artifact") or {}).get("url")
    if not url:
        raise ValueError(f"target {target.id} has no code artifact")
    tar_path = Path(output_dir) / Path(url).name
    if not tar_path.is_file():
        raise ValueError(f"artifact file missing at {tar_path}")
    return tar_path


def _rate_to_score(rate: float) -> int:
    """Map a test pass-rate [0,1] onto the 1-5 quality lane: 0->1 ... 1.0->5."""
    return max(1, min(5, round(1 + 4 * rate)))


def _suite_test_target(files: dict) -> str | None:
    """Derive the cargo integration-test target from a suite's single
    tests/<name>.rs file, so its pass-rate is measured in isolation."""
    stems = [Path(p).stem for p in files if p.startswith("tests/") and p.endswith(".rs")]
    return stems[0] if len(stems) == 1 else None


async def _run_suite(
    bc: RustBuildClient, tar_bytes: bytes, suite: dict
) -> tuple[str, int, dict]:
    """Run one harness-authored suite (overlay its files, `cargo test` the
    target) and return (dimension, score, detail). Property suites capture the
    proptest counterexample on failure."""
    dim = str(suite.get("dimension") or "tests")
    files = suite.get("files") or {}
    report = await bc.build(
        tar_bytes, ["test"], overlay_files=files, test_target=_suite_test_target(files)
    )
    summ = summarize_tests(report, "test")
    score = _rate_to_score(summ["pass_rate"]) if summ["total"] else _COMPILE_FAIL
    detail = {k: summ[k] for k in ("passed", "failed", "total", "pass_rate", "ok", "timed_out")}
    if suite.get("property") and not summ["ok"]:
        ce = counterexample(report, "test")
        if ce:
            detail["counterexample"] = ce
    return dim, score, detail


async def _run_code_eval_builds(
    bc: RustBuildClient, tar_bytes: bytes, commands: list[str], inputs: dict
) -> tuple[dict, dict, dict]:
    """Run the compile gate, then each harness suite (only if it compiled —
    `cargo test` would just re-fail to compile otherwise). Returns
    (compile_summary, dimensions, suites_detail)."""
    summary = compile_summary(await bc.build(tar_bytes, commands))
    dimensions = {"compile": _COMPILE_PASS if summary["success"] else _COMPILE_FAIL}
    suites_detail: dict[str, dict] = {}
    if summary["success"]:
        for suite in inputs.get("suites") or []:
            dim, score, detail = await _run_suite(bc, tar_bytes, suite)
            dimensions[dim] = score
            suites_detail[dim] = detail
    return summary, dimensions, suites_detail


async def _run_mutation_dimension(
    bc: RustBuildClient, tar_bytes: bytes, compiled: bool
) -> tuple[int | None, dict]:
    """Mutation testing against the crate's OWN tests (#28). cargo-mutants needs
    a passing baseline, so we skip (NA) when it didn't compile, has no
    model-authored tests, or those tests fail. Returns (score|None, detail);
    score is None when the dimension is NA (don't record a bogus number)."""
    if not compiled:
        return None, {"skipped": "did not compile"}
    model_tests = summarize_tests(await bc.build(tar_bytes, ["test"]), "test")
    if model_tests["total"] == 0:
        return None, {"skipped": "no model-authored tests"}
    if not model_tests["ok"]:
        return None, {"skipped": "model tests failing", "passed": model_tests["passed"],
                      "failed": model_tests["failed"]}
    mut = mutation_summary(await bc.build(tar_bytes, ["mutants"], timeout_s=_MUTANTS_TIMEOUT_S))
    if mut["timed_out"]:
        return None, {"skipped": "mutants timed out", **mut}
    if mut["mutants_tested"] == 0:
        return None, {"skipped": "no viable mutants", **mut}
    return _rate_to_score(mut["kill_rate"]), {**mut, "model_tests": model_tests["total"]}


async def _run_structural_dimension(
    bc: RustBuildClient, tar_bytes: bytes, inputs: dict, compiled: bool
) -> tuple[int, dict]:
    """Structural / AST regression check (#29): an AST scan (unsafe, happy-path
    panics, signature drift) plus clippy. Clean -> 5, any violation -> 1. The AST
    scan runs even on non-compiling code (tree-sitter is error-tolerant); clippy
    only runs when it compiled."""
    ast = analyze_tar(tar_bytes, inputs.get("signatures") or [])
    clippy_ok = True
    clippy_detail: dict = {"ran": False}
    if compiled:
        report = await bc.build(tar_bytes, ["clippy"])
        r = report.results.get("clippy")
        clippy_ok = bool(r and r.ok)
        diags = [ln.strip() for ln in ((r.stderr if r else "") or "").splitlines() if ln.strip()]
        clippy_detail = {"ran": True, "ok": clippy_ok, "diagnostics": diags[:20]}
    ok = ast["ok"] and clippy_ok
    return (_COMPILE_PASS if ok else _COMPILE_FAIL), {"ast": ast, "clippy": clippy_detail}


def _code_eval_note(summary: dict, suites: dict) -> str:
    parts = ["compiled" if summary["success"] else "compile failed"]
    parts += [f"{dim} {d['passed']}/{d['total']}" for dim, d in suites.items()]
    return "; ".join(parts)[:300]


async def _run_deps_dimension(
    tar_bytes: bytes, inputs: dict, index_client: CratesIndexClient | None
) -> dict | None:
    """Opt-in (`inputs.check_deps`) hallucinated-dependency check (#27). Returns
    the deps detail, or None when not requested / no Cargo.toml. Network issues
    leave individual crates `unresolved` rather than failing the job."""
    if not inputs.get("check_deps"):
        return None
    return await run_deps_check(tar_bytes, index_client or CratesIndexClient())


async def execute_code_eval_job(
    *, job: Job, client: ClientApp, session: AsyncSession,
    build_client: RustBuildClient | None = None,
    index_client: CratesIndexClient | None = None,
) -> Job:
    """Deterministic code evaluator (#25). Mirrors the judge as a submittable
    primitive — carries a `target_job_id`, runs async, writes back to the target
    only on `apply_to_target` — but needs **no model**: it ships the target's
    Cargo artifact to Watchtower's rust-build sandbox and records a `compile`
    dimension under a distinct `via="code-eval"` so deterministic truth never
    mixes with probabilistic LLM-judge scores."""
    client_name = client.name
    inputs = job.inputs or {}
    apply_to_target = bool(inputs.get("apply_to_target"))
    commands = inputs.get("commands") or ["check"]
    job.model_used = _VIA_CODE_EVAL  # attribution for failed-job analytics

    with _tracer.start_as_current_span("conduct.code_eval") as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        span.set_attribute(_SPAN_CLIENT_APP, client_name)

        settings = get_settings()
        try:
            target_id, tar_path = await _load_code_eval_target(
                session, inputs, settings.tts_output_dir
            )
        except ValueError as e:
            return await _fail_judge(job, session, f"code_eval: {e}", client_name)

        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        await session.commit()

        bc = build_client or RustBuildClient(
            settings.rust_build_url, timeout_s=_CODE_EVAL_CLIENT_TIMEOUT_S
        )
        tar_bytes = tar_path.read_bytes()
        started = perf_counter()
        deps_detail = await _run_deps_dimension(tar_bytes, inputs, index_client)
        try:
            summary, dimensions, suites_detail = await _run_code_eval_builds(
                bc, tar_bytes, commands, inputs
            )
            mut_score, mutation_detail = (
                await _run_mutation_dimension(bc, tar_bytes, summary["success"])
                if inputs.get("check_mutants") else (None, None)
            )
            struct_score, structural_detail = (
                await _run_structural_dimension(bc, tar_bytes, inputs, summary["success"])
                if inputs.get("check_structural") else (None, None)
            )
        except BuildServiceError as e:
            return await _fail_judge(job, session, f"code_eval: {e}", client_name)

        if deps_detail is not None:
            dimensions["deps"] = _COMPILE_PASS if deps_detail["ok"] else _COMPILE_FAIL
        if mut_score is not None:
            dimensions["mutation"] = mut_score
        if struct_score is not None:
            dimensions["structural"] = struct_score

        applied = False
        if apply_to_target:
            applied = (
                await apply_score(
                    session, target_id, scores=dimensions, reviewer=_VIA_CODE_EVAL,
                    note=_code_eval_note(summary, suites_detail), via=_VIA_CODE_EVAL,
                )
                is not None
            )

        verdict = {
            "mode": "code_eval", "target_job_id": str(target_id), "commands": commands,
            "compile": summary, "suites": suites_detail, "deps": deps_detail,
            "mutation": mutation_detail, "structural": structural_detail,
            "dimensions": dimensions, "applied_to_target": applied,
        }
        job.response = json.dumps(verdict)
        job.job_metadata = {**(job.job_metadata or {}), "code_eval": verdict}
        job.status = JobStatus.COMPLETE.value
        job.completed_at = datetime.now(UTC)
        job.latency_ms = int((perf_counter() - started) * 1000)
        span.set_attribute("code_eval.compile_success", summary["success"])
        span.set_attribute("code_eval.dimensions", json.dumps(dimensions))
        span.set_attribute("code_eval.applied_to_target", applied)
        await session.commit()
        await session.refresh(job)
        record_job_completion(
            status=job.status, task_type=job.task_type, model=_VIA_CODE_EVAL,
            client_app=client_name,
        )
        return job


# --- dpo_fine_tune: thin provider over the external MLX training sidecar (#45) -

_VIA_DPO = "dpo-fine-tune"


async def execute_dpo_fine_tune_job(
    *, job: Job, client: ClientApp, session: AsyncSession,
    providers: ProviderRegistry | None = None,
    train_client: DpoTrainClient | None = None,
) -> Job:
    """DPO fine-tune as a thin Conduct task (#45). Like the media/code_eval
    providers, Conduct carries no ML stack: it pulls the calling client's OWN
    preference pairs (the #41 scoped export), POSTs them to the external MLX
    training sidecar, and records the resulting servable tag + provenance. The
    heavy training lives entirely in the sidecar (native on the M5).

    Honors the declined-#44 line: Conduct dispatches + records; it does not
    train. A missing/failed sidecar fails the job loudly — no phantom tag."""
    from eval.datasets import iter_preferences  # noqa: PLC0415 - avoid import cycle

    client_name = client.name
    inputs = job.inputs or {}
    base_model = inputs.get("base_model")
    job.model_used = _VIA_DPO

    with _tracer.start_as_current_span("conduct.dpo_fine_tune") as span:
        span.set_attribute(_SPAN_JOB_ID, str(job.id))
        span.set_attribute(_SPAN_TASK_TYPE, job.task_type)
        span.set_attribute(_SPAN_CLIENT_APP, client_name)

        if not base_model:
            return await _fail_judge(job, session, "dpo_fine_tune: inputs.base_model is required",
                                     client_name)

        # Pull the caller's OWN preference pairs — same shape /datasets/preferences
        # returns, scoped by client_app_id so a tenant only trains on its data.
        pairs = await iter_preferences(
            session,
            task_type=inputs.get("source_task_type"),
            method=inputs.get("method", "composite"),
            min_gap=float(inputs.get("min_gap", 2.0)),
            label_dim=inputs.get("label_dim"),
            limit=int(inputs.get("limit", 1000)),
            client_app_id=job.client_app_id,
        )
        if not pairs:
            return await _fail_judge(
                job, session, "dpo_fine_tune: no preference pairs to train on "
                f"(source_task_type={inputs.get('source_task_type')!r})", client_name)

        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        await session.commit()

        # Free the resident serving models (~37GB) so MLX training has the GPU
        # memory it needs — the box can't hold the pinned set + a training peak
        # at once. Conduct owns this (the client never stops Ollama). Re-pinned
        # in `finally` so serving is restored whether training succeeds or not.
        ollama = providers.get("ollama") if providers and providers.has("ollama") else None
        if ollama is not None:
            await unload_resident_models(ollama)

        tc = train_client or DpoTrainClient(get_settings().dpo_train_url)
        started = perf_counter()
        try:
            try:
                result = await tc.train(
                    base_model=base_model, pairs=pairs, training=inputs.get("training"),
                )
            except TrainServiceError as e:
                return await _fail_judge(job, session, f"dpo_fine_tune: {e}", client_name)

            job.model_used = result.tag  # the new servable tag is the job's result
            job.status = JobStatus.COMPLETE.value
            job.completed_at = datetime.now(UTC)
            job.latency_ms = int((perf_counter() - started) * 1000)
            training_meta = {**result.as_metadata(), "base_model": base_model,
                             "pairs_submitted": len(pairs)}
            job.response = json.dumps(
                {"mode": "dpo_fine_tune", "tag": result.tag, **training_meta}
            )
            job.job_metadata = {**(job.job_metadata or {}), "training": training_meta}
            span.set_attribute("dpo.tag", result.tag)
            span.set_attribute("dpo.pairs", len(pairs))
            await session.commit()
            await session.refresh(job)
            record_job_completion(
                status=job.status, task_type=job.task_type, model=result.tag,
                client_app=client_name,
            )
            return job
        finally:
            # Restore the resident serving set regardless of outcome.
            if ollama is not None:
                await pin_resident_models(ollama)


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
        span.set_attribute(_SPAN_CLIENT_APP, client_name)
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
