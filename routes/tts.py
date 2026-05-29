"""Text-to-speech routes.

POST /tts — accept a chunk of text, enqueue a TTS job, return 202 + a poll URL.
GET /output/{filename} — serve a generated MP3 file by name.

The `output/` directory is the shared delivery point. Another machine on the
local network rsyncs from there and stitches chunks into a full audiobook.
"""

from __future__ import annotations

import hmac
import logging
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from db.session import get_session
from models.client import ClientApp
from models.job import Job
from models.types import JobStatus, Sensitivity
from rate_limit import rate_limited_client
from worker.queue import DEFAULT_JOB_TIMEOUT_S, get_queue
from worker.runner import run_job

_bearer = HTTPBearer(auto_error=False)


async def _admin_via_bearer_or_cookie(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    conduct_admin: str | None = Cookie(default=None),
) -> None:
    """Admin gate that accepts either the standard Bearer header (for
    CLI / sync agents) or the UI session cookie (for the audio player)."""
    admin_key = get_settings().admin_key
    if credentials and credentials.credentials and hmac.compare_digest(
        credentials.credentials, admin_key
    ):
        return
    if conduct_admin and hmac.compare_digest(conduct_admin, admin_key):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin auth required")

log = logging.getLogger(__name__)

# UUID-shaped filenames only — keeps the output path-safe even if a future
# refactor lets clients pass names. Today the worker generates them.
_OUTPUT_FILENAME = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.mp3$"
)

tts_router = APIRouter(prefix="/tts", tags=["tts"])
output_router = APIRouter(prefix="/output", tags=["tts"])


class TTSCreateIn(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = Field(default=None, max_length=100)


@tts_router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_tts(
    body: TTSCreateIn,
    client: Annotated[ClientApp, Depends(rate_limited_client)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JSONResponse:
    settings = get_settings()
    if len(body.text) > settings.tts_max_chars:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"text exceeds TTS_MAX_CHARS={settings.tts_max_chars}; "
            "split into smaller chunks (caller is responsible for stitching)",
        )
    voice = body.voice or settings.tts_default_voice

    job = Job(
        client_app_id=client.id,
        task_type="tts",
        sensitivity=Sensitivity.PUBLIC.value,  # local-only synth, no external send
        priority=5,
        prompt=body.text,
        system_prompt="",
        model_requested=voice,
        status=JobStatus.PENDING.value,
        job_metadata={},
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    try:
        get_queue().enqueue(
            run_job,
            str(job.id),
            job_id=str(job.id),
            job_timeout=DEFAULT_JOB_TIMEOUT_S,
        )
    except Exception as e:
        log.exception("failed to enqueue tts job %s", job.id)
        job.status = JobStatus.FAILED.value
        job.error = f"enqueue failed: {e}"
        await session.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "queue backend unavailable"
        ) from e

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": str(job.id),
            "status": JobStatus.PENDING.value,
            "poll_url": f"/jobs/{job.id}",
            "expected_output_url": f"/output/{job.id}.mp3",
            "voice": voice,
        },
    )


@output_router.get("/{filename}", dependencies=[Depends(_admin_via_bearer_or_cookie)])
async def get_output(filename: str) -> FileResponse:
    """Serve a generated MP3. Admin-auth — these are private outputs."""
    if not _OUTPUT_FILENAME.match(filename):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid filename")
    settings = get_settings()
    output_dir = Path(settings.tts_output_dir).resolve()
    file_path = (output_dir / filename).resolve()
    # Belt-and-suspenders: even though the regex restricts the shape, make
    # sure the resolved path lives under output_dir.
    try:
        file_path.relative_to(output_dir)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid path") from e
    if not file_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


# Combined router for main.py to mount as one symbol.
__all__ = ["tts_router", "output_router"]
