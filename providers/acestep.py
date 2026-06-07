"""ACE-Step media provider — audio (music) via OpenRouter-style HTTP.

Conduct dispatches `audio` task kind through this provider. The underlying
ACE-Step 1.5 service exposes an OpenAI-compatible `/v1/chat/completions`
endpoint that returns a generated audio clip as a base64 data URL on the
assistant message.

One-time setup: the ACE-Step server needs the DiT + LM model loaded before
it can generate. The provider issues a `/v1/init` call lazily on the first
`produce()` invocation (and caches the success so subsequent calls skip the
round-trip). The /health endpoint reports models_initialized=true after
that, so a launchd-restart of the daemon will trigger a re-init on the
next produce().

Reachability: defaults to `http://host.docker.internal:8002` to match the
launchd service we set up on the host. Override with `ACESTEP_BASE_URL`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from providers.media_base import BaseMediaProvider, MediaResponse

log = logging.getLogger(__name__)


_DEFAULT_MODEL = "acestep-v15-xl-turbo"
_DEFAULT_LM_MODEL = "acestep-5Hz-lm-1.7B"
_DEFAULT_TIMEOUT_S = 1800  # 30 min — first-time init can pull a few GB
_INIT_RETRY_S = 5.0


class ACEStepProvider(BaseMediaProvider):
    name = "acestep"

    def __init__(
        self,
        *,
        base_url: str = "http://host.docker.internal:8002",
        model: str = _DEFAULT_MODEL,
        lm_model: str = _DEFAULT_LM_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._lm_model = lm_model
        self._timeout_s = timeout_s
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def produce(
        self,
        *,
        prompt: str,
        inputs: dict[str, Any],
        output_dir: str,
        output_basename: str,
        params: dict[str, Any] | None = None,
    ) -> MediaResponse:
        """Generate audio from `prompt` and write it under
        `output_dir/output_basename.<ext>`.

        Params honored (with sane defaults):
          - `temperature` (default 0.85)
          - `vocal_language` (default "unknown")
          - `instrumental` (default True — Wanderer-style ambient tracks)
          - `audio_format` (default "mp3"; ACE-Step also handles wav)
        """
        params = params or {}
        async with httpx.AsyncClient(
            base_url=self._base, timeout=httpx.Timeout(60.0, read=self._timeout_s)
        ) as client:
            await self._ensure_initialized(client)
            started = time.perf_counter()
            audio_bytes, audio_format = await self._chat_completion(
                client, prompt, params
            )
        latency_ms = int((time.perf_counter() - started) * 1000)

        suffix = f".{audio_format}"
        out_path = Path(output_dir) / f"{output_basename}{suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)

        return MediaResponse(
            file_path=str(out_path),
            url_path=f"/output/{out_path.name}",
            mime_type=_mime_from_format(audio_format),
            width=None,
            height=None,
            # ACE-Step encodes track length into the response metadata but
            # the OpenRouter shape doesn't surface it cleanly — leave the
            # field None; downstream code can ffprobe if exact duration matters.
            duration_s=None,
            latency_ms=latency_ms,
            cost_usd=Decimal("0"),  # local model
            model_used=self._model,
            provider=self.name,
            extra={"audio_format": audio_format},
        )

    # ------------------------------------------------------------------
    # API mechanics
    # ------------------------------------------------------------------

    async def _ensure_initialized(self, client: httpx.AsyncClient) -> None:
        """Lazy init — first call POSTs /v1/init; subsequent calls fast-path.
        We hold a lock so concurrent fan-out shadow jobs don't double-init."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            # Optimization: if the daemon already has models loaded from a
            # prior process (e.g. launchd restart), /health says so and we
            # can skip the slow /v1/init call.
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    data = r.json().get("data") or {}
                    if data.get("models_initialized") and data.get("llm_initialized"):
                        self._initialized = True
                        return
            except httpx.HTTPError:
                pass  # /v1/init below will handle the real surface

            body = {
                "model": self._model,
                "slot": 1,
                "init_llm": True,
                "lm_model_path": self._lm_model,
            }
            r = await client.post("/v1/init", json=body)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"ACE-Step /v1/init {r.status_code}: {r.text[:500]}"
                )
            self._initialized = True

    async def _chat_completion(
        self, client: httpx.AsyncClient, prompt: str, params: dict[str, Any]
    ) -> tuple[bytes, str]:
        audio_format = params.get("audio_format", "mp3")
        body = {
            "model": f"acestep/{self._model}",
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["audio"],
            "audio": {"format": audio_format, "voice": ""},
            "temperature": params.get("temperature", 0.85),
        }
        # ACE-Step picks instrumental vs vocal from the prompt's tags by
        # default but the body's `instrumental` flag wins when present.
        if params.get("instrumental") is not None:
            body["instrumental"] = bool(params["instrumental"])
        if vlang := params.get("vocal_language"):
            body["vocal_language"] = vlang

        r = await client.post("/v1/chat/completions", json=body)
        if r.status_code >= 400:
            raise RuntimeError(
                f"ACE-Step /v1/chat/completions {r.status_code}: {r.text[:500]}"
            )
        return _extract_audio(r.json(), audio_format)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _extract_audio(chat_resp: dict, fmt: str) -> tuple[bytes, str]:
    """Pull the base64 audio from ACE-Step's OpenRouter-style response.

    Shape (verified during the smoke):
      choices[0].message.audio: [{type, audio_url: {url: "data:audio/...;base64,..."}}]
    The smoke also discovered the audio could be a dict in some impls; we
    accept either shape so this provider doesn't break on upstream churn.
    """
    choices = chat_resp.get("choices") or []
    if not choices:
        raise RuntimeError("ACE-Step response had no choices")
    msg = choices[0].get("message") or {}
    items = msg.get("audio") or []
    if isinstance(items, dict):
        items = [items]
    for item in items:
        url_obj = item.get("audio_url") or {}
        data = url_obj.get("url") or item.get("data") or url_obj.get("data")
        if isinstance(data, str) and data.startswith("data:"):
            header, b64 = data.split(",", 1)
            return base64.b64decode(b64), _format_from_header(header, fmt)
    raise RuntimeError(
        f"ACE-Step response did not include base64 audio (got: {list(msg.keys())!r})"
    )


def _format_from_header(header: str, fallback: str) -> str:
    """`data:audio/mpeg;base64` → `mp3` (so the file gets .mp3 not .mpeg)."""
    if "mpeg" in header or "mp3" in header:
        return "mp3"
    if "wav" in header:
        return "wav"
    if "ogg" in header:
        return "ogg"
    if "flac" in header:
        return "flac"
    return fallback


def _mime_from_format(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }.get(fmt, "audio/mpeg")
