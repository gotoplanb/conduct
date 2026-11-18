"""ElevenLabs TTS engine (#52) — optional cloud engine behind the voice registry.

A VoiceAlias with `engine='elevenlabs'` stores the ElevenLabs voice id in
`voice_file`. Synthesis is one POST to `/v1/text-to-speech/{voice_id}`, which
returns MP3 bytes directly (output_format=mp3_*), so — unlike Piper — there is
no ffmpeg step. The file lands at the same `output/{job.id}.mp3` path, so
`/output` serving and every downstream consumer are engine-agnostic.

The API key is per-client (encrypted on `client_apps`); this module never
reads settings for it and never logs it. Egress gating (sensitivity) happens
at submit time in routes/tts.py — by the time a job reaches here it is
already cleared to leave the box.
"""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter

import httpx

from tts.piper_engine import TTSEngineError

# ElevenLabs voice ids are short alphanumeric tokens (e.g. JBFqnCBsd6RMkjVDRZzb).
# Deliberately loose on length so a format change upstream doesn't reject
# real ids; the point is to catch a piper stem or a typo'd logical name
# (both contain '-' / '_') at registration instead of at synthesis.
_VOICE_ID = re.compile(r"^[A-Za-z0-9]{10,64}$")


def is_valid_voice_id(voice_id: str) -> bool:
    return bool(_VOICE_ID.match(voice_id))


async def synthesize_to_mp3(
    *,
    text: str,
    voice_id: str,
    api_key: str,
    output_path: Path,
    model_id: str,
    output_format: str,
    base_url: str,
    timeout_s: float,
) -> dict:
    """Synthesize `text` with an ElevenLabs voice and write MP3 to `output_path`.

    Returns timing/size metadata in the same shape as the Piper engine plus
    the ElevenLabs request id (useful for `previous_request_ids` stitching
    and for matching a job to ElevenLabs' own usage history).
    """
    if not is_valid_voice_id(voice_id):
        raise TTSEngineError(f"malformed ElevenLabs voice id {voice_id!r}")

    started = perf_counter()
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s
        ) as client:
            r = await client.post(
                f"/v1/text-to-speech/{voice_id}",
                params={"output_format": output_format},
                headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                json={"text": text, "model_id": model_id},
            )
    except httpx.HTTPError as e:
        raise TTSEngineError(f"ElevenLabs request failed: {type(e).__name__}") from e

    if r.status_code >= 400:
        # Body is ElevenLabs' JSON error detail (quota_exceeded, invalid key,
        # unknown voice…). It never contains our key, so it's safe to surface.
        raise TTSEngineError(f"ElevenLabs {r.status_code}: {r.text[:300]}")
    if not r.content:
        raise TTSEngineError("ElevenLabs returned an empty audio body")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(r.content)
    total_s = perf_counter() - started
    return {
        "synth_s": total_s,
        "encode_s": 0.0,
        "total_s": total_s,
        "mp3_bytes": len(r.content),
        "request_id": r.headers.get("request-id"),
    }
