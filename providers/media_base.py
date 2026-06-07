"""BaseMediaProvider — counterpart to BaseProvider for non-text outputs.

The text-side `BaseProvider.complete()` returns a `ProviderResponse` whose
core payload is the generated string. Media tasks (image, video, audio,
mux) produce a *file* — Conduct stores the file on disk and exposes a URL
via the worker's already-existing `/output/{file}` static handler (same
path TTS jobs have used since day one).

`BaseMediaProvider.produce()` is the shared shape. Concrete impls so far:

- `ComfyUIProvider` — image + video, via ComfyUI's REST API.
- `ACEStepProvider` — audio (music), via ACE-Step's OpenRouter-style API.
- `FFmpegMuxProvider` — composes a video + audio into one MP4. No model.

Routing dispatch picks the right impl by reading `RoutingRule.media_kind`
(text/image/video/audio/mux). See `routing/engine.py` for the dispatch
logic and `worker/runner.py` for how the worker invokes this path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class MediaResponse:
    """Return shape of `BaseMediaProvider.produce()`. The worker writes
    `media_url` and (where applicable) latency/cost onto the Job row.

    `file_path` is the worker-local absolute path; `url_path` is what the
    Conduct API serves (e.g. `/output/{job.id}.mp4`). The two are kept
    distinct so different deployments can move the storage backend
    (S3, NFS, etc.) without touching providers.
    """

    file_path: str
    url_path: str
    mime_type: str
    # Image/video carry dimensions; audio leaves them None.
    width: int | None
    height: int | None
    # Audio/video carry a duration; image leaves it None.
    duration_s: float | None
    latency_ms: int
    cost_usd: Decimal
    model_used: str
    provider: str
    # Provider-specific extras that don't fit the common shape (workflow id,
    # seed, sampler params, etc.). Lands in `Job.metadata['media']`.
    extra: dict[str, Any]


class BaseMediaProvider(ABC):
    """Abstract media provider. Subclasses pick a `media_kind` they serve
    (a single provider may serve multiple kinds — e.g. ComfyUI does image
    AND video — by branching internally on the task_type or workflow)."""

    name: str

    @abstractmethod
    async def produce(
        self,
        *,
        prompt: str,
        inputs: dict[str, Any],
        output_dir: str,
        output_basename: str,
        params: dict[str, Any] | None = None,
    ) -> MediaResponse:
        """Generate media and write it to `output_dir/output_basename.<ext>`.

        - `prompt`: text description (always passed, even when the task is
          mostly driven by `inputs` — e.g. video tasks use it for motion
          guidance on top of the source image).
        - `inputs`: typed bag from `Job.inputs` (e.g.
          `{"source_image_url": "/output/abc.png"}`).
        - `output_dir`: worker-local writable dir.
        - `output_basename`: usually `str(job.id)` — provider appends extension.
        - `params`: optional knob bag (width/height/fps/seed/etc.). Routing
          rules can pin these via the `notes`/`metadata` slot; not enforced
          here.
        """
        ...
