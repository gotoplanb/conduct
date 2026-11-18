"""Worker entry for TTS jobs.

Mirrors `worker/executor.py` for LLM jobs but writes audio to disk instead
of a text response. The Job row's `response` field becomes the URL where
the MP3 can be fetched (the caller polls /jobs/{id}, reads `response`,
then GETs that URL).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from opentelemetry.trace import Status, StatusCode
from sqlalchemy.ext.asyncio import AsyncSession

from config.pricing import get_pricing
from config.settings import get_settings
from models.client import ClientApp
from models.job import Job
from models.types import JobStatus
from observability.metrics import record_job_completion
from observability.tracing import get_tracer
from secrets_box import SecretDecryptError, SecretsKeyMissing, decrypt
from tts import elevenlabs_engine
from tts.piper_engine import TTSEngineError, synthesize_to_mp3
from tts.voices import ELEVENLABS

log = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


async def execute_tts(
    *,
    job: Job,
    client: ClientApp,
    session: AsyncSession,
) -> Job:
    """Run a TTS job to completion and persist the result on the Job row."""
    settings = get_settings()
    client_name = client.name
    voice = job.model_requested or settings.tts_default_voice
    engine = (job.job_metadata or {}).get("tts_engine", "piper")
    voices_dir = Path(settings.tts_voices_dir).resolve()
    output_dir = Path(settings.tts_output_dir).resolve()
    output_path = output_dir / f"{job.id}.mp3"
    url_path = f"/output/{job.id}.mp3"

    with _tracer.start_as_current_span("conduct.tts") as span:
        span.set_attribute("job.id", str(job.id))
        span.set_attribute("job.task_type", job.task_type)
        span.set_attribute("job.client_app", client_name)
        span.set_attribute("tts.voice", voice)
        span.set_attribute("tts.engine", engine)
        span.set_attribute("tts.input_chars", len(job.prompt))

        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        job.model_used = voice
        await session.commit()

        started = perf_counter()
        stats: dict = {}
        try:
            if engine == ELEVENLABS:
                stats = await _synthesize_elevenlabs(job, client, voice, output_path)
            else:
                stats = synthesize_to_mp3(
                    text=job.prompt,
                    voice_name=voice,
                    voices_dir=voices_dir,
                    output_path=output_path,
                )
            job.status = JobStatus.COMPLETE.value
            job.response = url_path
            job.tokens_in = len(job.prompt)
            job.tokens_out = stats["mp3_bytes"]
            job.latency_ms = int(stats["total_s"] * 1000)
            job.cost_usd = stats.get("cost_usd", Decimal("0"))
            job.job_metadata = {
                **(job.job_metadata or {}),
                "tts": {
                    "engine": engine,
                    "voice": voice,
                    "output_path": str(output_path),
                    **{k: v for k, v in stats.items() if k != "cost_usd"},
                },
            }
        except TTSEngineError as e:
            job.status = JobStatus.FAILED.value
            job.error = f"{type(e).__name__}: {e}"
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))

        job.completed_at = datetime.now(UTC)
        if engine == ELEVENLABS and job.status == JobStatus.COMPLETE.value:
            # Cloud TTS costs money — roll it into the client's daily usage
            # the way cloud LLM jobs do. Local piper stays out of the rollup.
            from worker.executor import _bump_usage  # noqa: PLC0415 — heavy module

            await _bump_usage(session, job)
        await session.commit()
        await session.refresh(job)

        duration_s = perf_counter() - started
        span.set_attribute("job.status", job.status)
        span.set_attribute("job.duration_s", duration_s)

        record_job_completion(
            status=job.status,
            task_type=job.task_type,
            model=voice,
            client_app=client_name,
            duration_s=duration_s,
            tokens_in=len(job.prompt),
            tokens_out=stats["mp3_bytes"] if job.status == JobStatus.COMPLETE.value else 0,
            cost_usd=float(job.cost_usd or 0),
        )
        return job


async def _synthesize_elevenlabs(
    job: Job, client: ClientApp, voice_id: str, output_path: Path
) -> dict:
    """Call ElevenLabs with the client's own key; price by characters.

    The submit route already refused clients without a key, but the key can
    be cleared between submit and execution — fail the job cleanly then.
    """
    settings = get_settings()
    if not client.elevenlabs_api_key_encrypted:
        raise TTSEngineError("no ElevenLabs API key configured for this client")
    try:
        api_key = decrypt(client.elevenlabs_api_key_encrypted)
    except (SecretsKeyMissing, SecretDecryptError) as e:
        raise TTSEngineError(f"cannot decrypt ElevenLabs key: {e}") from e

    stats = await elevenlabs_engine.synthesize_to_mp3(
        text=job.prompt,
        voice_id=voice_id,
        api_key=api_key,
        output_path=output_path,
        model_id=settings.elevenlabs_model_id,
        output_format=settings.elevenlabs_output_format,
        base_url=settings.elevenlabs_base_url,
        timeout_s=settings.elevenlabs_timeout_s,
    )
    # ElevenLabs bills per input character; pricing.yaml carries the rate as
    # per-1M "input tokens" with characters standing in for tokens.
    cost = get_pricing().cost("elevenlabs", settings.elevenlabs_model_id, len(job.prompt), 0)
    return {**stats, "model_id": settings.elevenlabs_model_id, "cost_usd": cost}
