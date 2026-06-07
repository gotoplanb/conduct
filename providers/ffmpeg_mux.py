"""FFmpeg mux media provider — the composition primitive.

Conduct dispatches `mux` task kind through this provider. It takes a video
file URL + an audio file URL, runs ffmpeg with `-c:v copy -c:a aac
-shortest`, and writes the result. No model, no GPU, just subprocess.

This is the "lego brick" that lets Wanderer (and any other Conduct client)
build a final video out of independent image/video/audio task outputs:

    Wanderer engine:
      vid = create_job(task_type="wander_scene_video", ...)
      aud = create_job(task_type="wander_scene_music", ...)
      mux = create_job(task_type="wander_scene_assemble",
                       inputs={"source_video_url": vid.url,
                               "source_audio_url": aud.url})

The mux task doesn't know whether the video came from Wan or an iPhone
camera, or whether the audio came from ACE-Step or a podcast recording —
it just composes whatever URLs it's handed. That keeps the primitive
clean and reusable beyond the wander use case.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import urllib.parse
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from providers.media_base import BaseMediaProvider, MediaResponse

log = logging.getLogger(__name__)

# Resolve ffmpeg lazily so unit tests can patch without messing with PATH.
_DEFAULT_FFMPEG = "ffmpeg"


class FFmpegMuxProvider(BaseMediaProvider):
    name = "ffmpeg_mux"

    def __init__(self, *, ffmpeg_path: str = _DEFAULT_FFMPEG) -> None:
        self._ffmpeg = ffmpeg_path

    async def produce(
        self,
        *,
        prompt: str,
        inputs: dict[str, Any],
        output_dir: str,
        output_basename: str,
        params: dict[str, Any] | None = None,
    ) -> MediaResponse:
        """Mux video + audio into one MP4.

        Requires:
          - inputs["source_video_url"]: video file URL or path
          - inputs["source_audio_url"]: audio file URL or path

        Params:
          - container (default "mp4")
          - audio_codec (default "aac")
          - video_passthrough (default True — copy video stream rather than
            re-encode, makes this near-instant for already-H.264 sources)
        """
        params = params or {}
        video_src = inputs.get("source_video_url")
        audio_src = inputs.get("source_audio_url")
        if not video_src or not audio_src:
            raise ValueError(
                "ffmpeg_mux requires inputs.source_video_url and "
                "inputs.source_audio_url (both URLs or paths)"
            )

        # Stage the inputs locally so ffmpeg gets readable file paths even
        # when the inputs were http(s) URLs (from cross-job references).
        # `_stage_source` returns (path, did_we_create_it) — the second
        # tuple element drives the cleanup pass: only files we downloaded
        # ourselves get removed; caller-owned paths are left alone.
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        video_local, video_owned = await _stage_source(video_src, out_dir, f"{output_basename}-v")
        audio_local, audio_owned = await _stage_source(audio_src, out_dir, f"{output_basename}-a")

        container = params.get("container", "mp4")
        out_path = out_dir / f"{output_basename}.{container}"

        cmd = [
            self._ffmpeg,
            "-y",  # overwrite — Conduct's worker owns the output dir
            "-i", str(video_local),
            "-i", str(audio_local),
            "-c:v", "copy" if params.get("video_passthrough", True) else "libx264",
            "-c:a", params.get("audio_codec", "aac"),
            "-shortest",
            str(out_path),
        ]

        started = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        latency_ms = int((time.perf_counter() - started) * 1000)

        if proc.returncode != 0:
            tail = stderr.decode("utf-8", errors="replace")[-800:]
            raise RuntimeError(
                f"ffmpeg mux failed (exit={proc.returncode}): {tail}"
            )

        # Clean up only the intermediates *we* downloaded. Caller-owned
        # paths (already-local files passed by path) stay untouched —
        # they may be other Conduct jobs' outputs that need to remain
        # readable for /ui/jobs/{id} rendering.
        for staged, owned in ((video_local, video_owned), (audio_local, audio_owned)):
            if not owned:
                continue
            with contextlib.suppress(OSError):
                Path(staged).unlink()

        return MediaResponse(
            file_path=str(out_path),
            url_path=f"/output/{out_path.name}",
            mime_type=f"video/{container}" if container in ("mp4", "webm") else "video/mp4",
            width=None,  # let ffprobe at view time if anyone needs it
            height=None,
            duration_s=None,
            latency_ms=latency_ms,
            cost_usd=Decimal("0"),
            model_used="ffmpeg",
            provider=self.name,
            extra={
                "video_source": str(video_src),
                "audio_source": str(audio_src),
                "container": container,
            },
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


async def _stage_source(src: str, out_dir: Path, basename: str) -> tuple[Path, bool]:
    """Resolve `src` to (local_path, we_created_it). http(s) URLs get
    downloaded; everything else is treated as caller-owned and not copied.

    The `we_created_it` flag drives FFmpegMuxProvider's cleanup pass — we
    only delete files we ourselves materialized, so caller-owned paths
    survive the mux."""
    if src.startswith(("http://", "https://")):
        suffix = Path(urllib.parse.urlparse(src).path).suffix or ".bin"
        local = out_dir / f"{basename}{suffix}"
        async with httpx.AsyncClient(timeout=120.0) as c:
            r = await c.get(src)
            r.raise_for_status()
            local.write_bytes(r.content)
        return local, True
    if src.startswith("file://"):
        return Path(urllib.parse.urlparse(src).path), False
    return Path(src), False
