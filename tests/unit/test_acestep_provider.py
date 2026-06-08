"""Unit tests for ACEStepProvider.

The HTTP shape comes straight from the live smoke I ran (verified mp3
data URL in the OpenRouter choices[0].message.audio list)."""

from __future__ import annotations

import base64
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from providers.acestep import ACEStepProvider


def _chat_response(audio_bytes: bytes, mime: str = "audio/mpeg") -> dict:
    """Shape matches the response captured during the live smoke test."""
    data_url = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode()}"
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "acestep/acestep-v15-xl-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "## Metadata\nCaption: chiptune\nDuration: 34s",
                    "audio": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": data_url},
                        }
                    ],
                },
            }
        ],
    }


@pytest.fixture
def provider():
    return ACEStepProvider(base_url="http://ace.test", timeout_s=10)


@pytest.mark.asyncio
@respx.mock
async def test_first_call_initializes_then_generates(provider, tmp_path) -> None:
    """First produce() POSTs /v1/init then /v1/chat/completions. The init
    only happens once even across multiple calls (idempotency check)."""
    init_route = respx.post("http://ace.test/v1/init").mock(
        return_value=httpx.Response(200, json={"data": {"loaded_model": "x"}})
    )
    # Force the /health fast-path to report "not initialized" so we go
    # through /v1/init.
    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": False, "llm_initialized": False}
        })
    )
    chat_route = respx.post("http://ace.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(b"MP3DATA"))
    )

    r = await provider.produce(
        prompt="8-bit chiptune, ominous",
        inputs={},
        output_dir=str(tmp_path),
        output_basename="job-1",
    )
    assert r.mime_type == "audio/mpeg"
    assert r.url_path == "/output/job-1.mp3"
    assert r.cost_usd == Decimal("0")
    assert Path(r.file_path).read_bytes() == b"MP3DATA"
    assert init_route.call_count == 1
    assert chat_route.call_count == 1

    # Second call must skip /v1/init
    await provider.produce(
        prompt="upbeat 8-bit",
        inputs={},
        output_dir=str(tmp_path),
        output_basename="job-2",
    )
    assert init_route.call_count == 1, "init must not re-fire on subsequent calls"


@pytest.mark.asyncio
@respx.mock
async def test_health_fast_path_skips_init_when_already_loaded(
    provider, tmp_path
) -> None:
    """If the daemon is already initialized (e.g. survived from a previous
    process), /health surfaces that and the provider skips the slow init."""
    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": True, "llm_initialized": True}
        })
    )
    init_route = respx.post("http://ace.test/v1/init").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.post("http://ace.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(b"x"))
    )

    await provider.produce(
        prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x"
    )
    assert init_route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_response_accepts_dict_shape_for_audio(provider, tmp_path) -> None:
    """The smoke discovered some impls return audio as a dict instead of a
    list. Provider tolerates either shape so upstream churn doesn't break
    media dispatch."""
    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": True, "llm_initialized": True}
        })
    )
    dict_shape = _chat_response(b"AUDIO")
    dict_shape["choices"][0]["message"]["audio"] = dict_shape["choices"][0]["message"]["audio"][0]
    respx.post("http://ace.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=dict_shape)
    )

    r = await provider.produce(
        prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x"
    )
    assert Path(r.file_path).read_bytes() == b"AUDIO"


@pytest.mark.asyncio
@respx.mock
async def test_chat_completion_error_raises(provider, tmp_path) -> None:
    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": True, "llm_initialized": True}
        })
    )
    respx.post("http://ace.test/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(RuntimeError, match="ACE-Step /v1/chat/completions 500"):
        await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x"
        )


@pytest.mark.asyncio
@respx.mock
async def test_missing_audio_in_response_raises(provider, tmp_path) -> None:
    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": True, "llm_initialized": True}
        })
    )
    respx.post("http://ace.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "txt only"}}]
        })
    )
    with pytest.raises(RuntimeError, match="did not include base64 audio"):
        await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x"
        )


@pytest.mark.asyncio
@respx.mock
async def test_request_sends_bounded_audio_config_duration(provider, tmp_path) -> None:
    """Regression for the OOM crash: the provider MUST send a bounded
    `audio_config.duration`. ACE-Step reads generation settings from
    `audio_config` only — with duration unset it auto-selects up to 600s, and a
    long pick (a 215s auto-pick was observed spiking ~80GB of unified memory)
    OOMs the host when the resident video model is also loaded. Also pins that
    instrumental/format ride inside audio_config (where the adapter reads them)
    and that the old top-level `audio` key is gone."""
    import json as _json

    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": True, "llm_initialized": True}
        })
    )
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.clear()
        captured.update(_json.loads(request.content))
        return httpx.Response(200, json=_chat_response(b"x"))

    respx.post("http://ace.test/v1/chat/completions").mock(side_effect=_capture)

    # Default: bounded duration present, instrumental defaults True, no stray
    # top-level `audio` key.
    await provider.produce(
        prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="d1"
    )
    ac = captured["audio_config"]
    assert ac["duration"] == 30.0
    assert ac["instrumental"] is True
    assert ac["format"] == "mp3"
    assert "audio" not in captured

    # Explicit params override the defaults.
    await provider.produce(
        prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="d2",
        params={"duration": 12.5, "instrumental": False, "audio_format": "wav"},
    )
    ac = captured["audio_config"]
    assert ac["duration"] == 12.5
    assert ac["instrumental"] is False
    assert ac["format"] == "wav"


@pytest.mark.asyncio
@respx.mock
async def test_wav_format_extension_used(provider, tmp_path) -> None:
    """Audio format flows from params through to filename suffix."""
    respx.get("http://ace.test/health").mock(
        return_value=httpx.Response(200, json={
            "data": {"models_initialized": True, "llm_initialized": True}
        })
    )
    respx.post("http://ace.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_response(b"WAV", mime="audio/wav"))
    )
    r = await provider.produce(
        prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
        params={"audio_format": "wav"},
    )
    assert r.url_path == "/output/x.wav"
    assert r.mime_type == "audio/wav"
