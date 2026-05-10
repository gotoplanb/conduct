"""Piper TTS engine — synthesizes a chunk of text into an MP3 file.

Piper outputs WAV; we pipe through ffmpeg to MP3 because audiobook chunks
are typically 5-10x smaller as MP3 (matters when the user's stitching
machine is rsync'ing them across a LAN).

Voice files live in `settings.tts_voices_dir`. Each voice is two files —
`<name>.onnx` (the model) and `<name>.onnx.json` (config). They're not
shipped with this repo; users download them once via `make download-voice`
or grab them from huggingface.co/rhasspy/piper-voices.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from time import perf_counter

log = logging.getLogger(__name__)


class TTSEngineError(Exception):
    """Anything went wrong producing audio."""


class VoiceNotFound(TTSEngineError):
    """The requested voice file isn't present in tts_voices_dir."""


_voice_cache: dict[str, object] = {}


def _load_voice(voice_name: str, voices_dir: Path) -> object:
    """Load a Piper voice, caching the in-memory model so subsequent jobs
    reuse it. ONNX session creation is the slow part."""
    if voice_name in _voice_cache:
        return _voice_cache[voice_name]

    onnx_path = voices_dir / f"{voice_name}.onnx"
    config_path = voices_dir / f"{voice_name}.onnx.json"
    if not onnx_path.is_file() or not config_path.is_file():
        raise VoiceNotFound(
            f"voice {voice_name!r} not found in {voices_dir} "
            f"(need {onnx_path.name} and {config_path.name})"
        )

    # Lazy import — piper pulls in onnxruntime + numpy, ~80MB. Don't pay that
    # at module import time if no TTS jobs ever run.
    from piper.voice import PiperVoice  # noqa: PLC0415

    voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))
    _voice_cache[voice_name] = voice
    return voice


def synthesize_to_mp3(
    *,
    text: str,
    voice_name: str,
    voices_dir: Path,
    output_path: Path,
) -> dict:
    """Generate an MP3 at `output_path` from `text` using `voice_name`.

    Returns a small metadata dict with timing + sizes — the caller writes
    these into the Job row.
    """
    started = perf_counter()
    voice = _load_voice(voice_name, voices_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = Path(wav_tmp.name)

    try:
        with wave.open(str(wav_path), "wb") as wav_file:
            # synthesize_wav sets channels/sample-rate/sample-width on the wav
            # object before writing frames, so we don't need to pre-configure.
            voice.synthesize_wav(text, wav_file)  # type: ignore[attr-defined]
        synth_s = perf_counter() - started

        # WAV → MP3 via ffmpeg. -y overwrites existing, -loglevel error keeps
        # the worker logs clean. 96k bitrate is plenty for spoken word.
        encode_started = perf_counter()
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise TTSEngineError(f"ffmpeg failed: {result.stderr.strip()[:200]}")
        encode_s = perf_counter() - encode_started
    finally:
        wav_path.unlink(missing_ok=True)

    return {
        "synth_s": synth_s,
        "encode_s": encode_s,
        "total_s": perf_counter() - started,
        "mp3_bytes": output_path.stat().st_size,
    }
