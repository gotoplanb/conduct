"""Worker entry for TTS jobs.

Mirrors `worker/executor.py` for LLM jobs but writes audio to disk instead
of a text response. The Job row's `response` field becomes the URL where
the MP3 can be fetched (the caller polls /jobs/{id}, reads `response`,
then GETs that URL).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from opentelemetry.trace import Status, StatusCode
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from models.client import ClientApp
from models.job import Job
from models.types import JobStatus
from observability.metrics import record_job_completion
from observability.tracing import get_tracer
from tts.piper_engine import TTSEngineError, synthesize_to_mp3

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
    voices_dir = Path(settings.tts_voices_dir).resolve()
    output_dir = Path(settings.tts_output_dir).resolve()
    output_path = output_dir / f"{job.id}.mp3"
    url_path = f"/output/{job.id}.mp3"

    with _tracer.start_as_current_span("conduct.tts") as span:
        span.set_attribute("job.id", str(job.id))
        span.set_attribute("job.task_type", job.task_type)
        span.set_attribute("job.client_app", client_name)
        span.set_attribute("tts.voice", voice)
        span.set_attribute("tts.input_chars", len(job.prompt))

        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(UTC)
        job.model_used = voice
        await session.commit()

        started = perf_counter()
        try:
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
            job.job_metadata = {
                **(job.job_metadata or {}),
                "tts": {
                    "voice": voice,
                    "synth_s": stats["synth_s"],
                    "encode_s": stats["encode_s"],
                    "mp3_bytes": stats["mp3_bytes"],
                    "output_path": str(output_path),
                },
            }
        except TTSEngineError as e:
            job.status = JobStatus.FAILED.value
            job.error = f"{type(e).__name__}: {e}"
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)[:200]))

        job.completed_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(job)

        duration_s = perf_counter() - started
        span.set_attribute("job.status", job.status)
        span.set_attribute("job.duration_s", duration_s)

        # Reuse the LLM job-completion metrics; cost is always 0 for local TTS.
        record_job_completion(
            status=job.status,
            task_type=job.task_type,
            model=voice,
            client_app=client_name,
            duration_s=duration_s,
            tokens_in=len(job.prompt),
            tokens_out=stats["mp3_bytes"] if job.status == JobStatus.COMPLETE.value else 0,
            cost_usd=0.0,
        )
        return job
