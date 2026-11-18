"""ElevenLabs TTS engine (#52): request shape, error surfacing, voice ids."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from models.types import Sensitivity
from tts import elevenlabs_engine
from tts.piper_engine import TTSEngineError
from tts.voices import cloud_tts_block_reason

_BASE = "https://api.elevenlabs.test"
_VOICE = "JBFqnCBsd6RMkjVDRZzb"


async def _synth(tmp_path: Path, voice_id: str = _VOICE) -> dict:
    return await elevenlabs_engine.synthesize_to_mp3(
        text="hello there",
        voice_id=voice_id,
        api_key="xi-test-key",
        output_path=tmp_path / "out.mp3",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        base_url=_BASE,
        timeout_s=5,
    )


@pytest.mark.parametrize(
    ("voice_id", "ok"),
    [
        (_VOICE, True),
        ("en_US-amy-medium", False),  # a piper stem registered under the wrong engine
        ("ops-manager", False),
        ("short", False),
        ("", False),
    ],
)
def test_voice_id_validation(voice_id: str, ok: bool) -> None:
    assert elevenlabs_engine.is_valid_voice_id(voice_id) is ok


@respx.mock
async def test_synthesize_writes_mp3_and_sends_expected_request(tmp_path: Path) -> None:
    route = respx.post(f"{_BASE}/v1/text-to-speech/{_VOICE}").mock(
        return_value=httpx.Response(
            200, content=b"ID3fake-mp3", headers={"request-id": "req-123"}
        )
    )
    stats = await _synth(tmp_path)

    assert (tmp_path / "out.mp3").read_bytes() == b"ID3fake-mp3"
    assert stats["mp3_bytes"] == len(b"ID3fake-mp3")
    assert stats["request_id"] == "req-123"
    req = route.calls.last.request
    assert req.headers["xi-api-key"] == "xi-test-key"
    assert req.url.params["output_format"] == "mp3_44100_128"
    assert b'"model_id":"eleven_multilingual_v2"' in req.content.replace(b" ", b"")


@respx.mock
async def test_api_error_surfaces_status_and_detail(tmp_path: Path) -> None:
    respx.post(f"{_BASE}/v1/text-to-speech/{_VOICE}").mock(
        return_value=httpx.Response(401, json={"detail": {"status": "invalid_api_key"}})
    )
    with pytest.raises(TTSEngineError, match="401.*invalid_api_key"):
        await _synth(tmp_path)
    assert not (tmp_path / "out.mp3").exists()


@respx.mock
async def test_transport_error_is_wrapped(tmp_path: Path) -> None:
    respx.post(f"{_BASE}/v1/text-to-speech/{_VOICE}").mock(
        side_effect=httpx.ConnectError("boom")
    )
    with pytest.raises(TTSEngineError, match="ConnectError"):
        await _synth(tmp_path)


@respx.mock
async def test_empty_body_is_an_error(tmp_path: Path) -> None:
    respx.post(f"{_BASE}/v1/text-to-speech/{_VOICE}").mock(
        return_value=httpx.Response(200, content=b"")
    )
    with pytest.raises(TTSEngineError, match="empty"):
        await _synth(tmp_path)


async def test_malformed_voice_id_rejected_before_any_request(tmp_path: Path) -> None:
    with pytest.raises(TTSEngineError, match="malformed"):
        await _synth(tmp_path, voice_id="en_US-amy-medium")


# --- egress gate --------------------------------------------------------------


def _client(*, key: bool = True, cloud_internal: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        elevenlabs_api_key_encrypted="ct" if key else None,
        allow_cloud_for_internal=cloud_internal,
    )


@pytest.mark.parametrize(
    ("engine", "sensitivity", "client", "blocked_by"),
    [
        ("piper", Sensitivity.CONFIDENTIAL, _client(key=False), None),
        ("elevenlabs", Sensitivity.PUBLIC, _client(), None),
        ("elevenlabs", Sensitivity.CONFIDENTIAL, _client(cloud_internal=True), "confidential"),
        ("elevenlabs", Sensitivity.INTERNAL, _client(), "allow_cloud_for_internal"),
        ("elevenlabs", Sensitivity.INTERNAL, _client(cloud_internal=True), None),
        ("elevenlabs", Sensitivity.PUBLIC, _client(key=False), "no ElevenLabs API key"),
    ],
)
def test_cloud_tts_gate(engine, sensitivity, client, blocked_by) -> None:
    reason = cloud_tts_block_reason(engine, sensitivity, client)
    if blocked_by is None:
        assert reason is None
    else:
        assert reason is not None
        assert blocked_by in reason


def test_malformed_elevenlabs_ids_flags_only_bad_cloud_rows() -> None:
    from models.voice import VoiceAlias
    from tts.voices import malformed_elevenlabs_ids

    aliases = [
        VoiceAlias(name="ok", engine="elevenlabs", voice_file=_VOICE),
        VoiceAlias(name="bad", engine="elevenlabs", voice_file="en_US-amy-medium"),
        VoiceAlias(name="local", engine="piper", voice_file="en_US-amy-medium"),
    ]
    assert malformed_elevenlabs_ids(aliases) == [("bad", "en_US-amy-medium")]
