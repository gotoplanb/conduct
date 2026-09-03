"""Coverage gaps in the provider modules: Ollama admin endpoints + error
mapping, ACE-Step init edge cases, ComfyUI failure surfaces, and the
resident-model best-effort pin/unload paths."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

import providers.resident as resident
from providers.acestep import (
    ACEStepProvider,
    _extract_audio,
    _format_from_header,
)
from providers.base import ProviderError, ProviderTimeout
from providers.comfyui import (
    ComfyUIProvider,
    _duration_from_params,
    _load_image_source,
    _set_input,
)
from providers.ollama import OllamaProvider
from providers.resident import (
    PIN_FOREVER,
    pin_resident_models,
    reconcile_resident_models,
    resume_residency,
    unload_resident_models,
)

OLLAMA = "http://localhost:11434"
ACE = "http://ace.test"
COMFY = "http://comfy.test"


# --- Ollama ---------------------------------------------------------------


async def test_ollama_system_prompt_lands_in_payload() -> None:
    captured: dict = {}
    with respx.mock(base_url=OLLAMA) as mock:

        def _capture(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "x", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        await OllamaProvider(base_url=OLLAMA).complete(
            prompt="hi", model="m", system_prompt="be terse"
        )
    assert captured["system"] == "be terse"


async def test_ollama_timeout_maps_to_provider_timeout() -> None:
    with respx.mock(base_url=OLLAMA) as mock:
        mock.post("/api/generate").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(ProviderTimeout, match="timeout"):
            await OllamaProvider(base_url=OLLAMA, timeout_s=1.0).complete(
                prompt="hi", model="m"
            )


async def test_ollama_connect_error_maps_to_provider_error() -> None:
    with respx.mock(base_url=OLLAMA) as mock:
        mock.post("/api/generate").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="request failed"):
            await OllamaProvider(base_url=OLLAMA).complete(prompt="hi", model="m")


async def test_ollama_list_models_returns_tags() -> None:
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/tags").mock(
            return_value=httpx.Response(
                200, json={"models": [{"name": "llama3.3:70b"}, {"name": "gemma4:e4b"}]}
            )
        )
        models = await OllamaProvider(base_url=OLLAMA).list_models()
    assert [m["name"] for m in models] == ["llama3.3:70b", "gemma4:e4b"]


async def test_ollama_list_loaded_returns_ps() -> None:
    with respx.mock(base_url=OLLAMA) as mock:
        mock.get("/api/ps").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "gemma4:e4b"}]})
        )
        loaded = await OllamaProvider(base_url=OLLAMA).list_loaded()
    assert loaded == [{"name": "gemma4:e4b"}]


async def test_ollama_load_default_omits_keep_alive_and_options() -> None:
    # No keep_alive and no num_ctx configured -> neither key may be injected
    # (Ollama would otherwise treat their presence as an explicit setting).
    captured: dict = {}
    with respx.mock(base_url=OLLAMA) as mock:

        def _capture(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        await OllamaProvider(base_url=OLLAMA).load("m")
    assert "keep_alive" not in captured
    assert "options" not in captured


async def test_ollama_unload_posts_keep_alive_zero() -> None:
    captured: dict = {}
    with respx.mock(base_url=OLLAMA) as mock:

        def _capture(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "", "done": True})

        mock.post("/api/generate").mock(side_effect=_capture)
        await OllamaProvider(base_url=OLLAMA).unload("m")
    assert captured["keep_alive"] == 0
    assert captured["model"] == "m"


# --- ACE-Step -------------------------------------------------------------


@pytest.fixture
def ace():
    return ACEStepProvider(base_url=ACE, timeout_s=10)


def _audio_response(url: str) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "audio": [{"audio_url": {"url": url}}]}}
        ]
    }


@respx.mock
async def test_acestep_second_waiter_skips_init_under_lock(ace) -> None:
    # A concurrent produce() that lost the init race must re-check under the
    # lock and return without any HTTP (no routes are mocked -> any call fails).
    async with httpx.AsyncClient(base_url=ACE) as client:
        async with ace._init_lock:
            waiter = asyncio.create_task(ace._ensure_initialized(client))
            await asyncio.sleep(0)  # waiter reaches the lock and blocks
            ace._initialized = True  # the race winner finished init
        await waiter
    assert ace._initialized is True
    assert len(respx.calls) == 0


@respx.mock
async def test_acestep_health_non_200_falls_through_to_init(ace, tmp_path) -> None:
    respx.get(f"{ACE}/health").mock(return_value=httpx.Response(503, text="warming"))
    init_route = respx.post(f"{ACE}/v1/init").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.post(f"{ACE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_audio_response("data:audio/mpeg;base64,QUJD")
        )
    )
    await ace.produce(prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x")
    assert init_route.call_count == 1


@respx.mock
async def test_acestep_health_transport_error_falls_through_to_init(ace, tmp_path) -> None:
    respx.get(f"{ACE}/health").mock(side_effect=httpx.ConnectError("down"))
    init_route = respx.post(f"{ACE}/v1/init").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.post(f"{ACE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_audio_response("data:audio/mpeg;base64,QUJD")
        )
    )
    await ace.produce(prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x")
    assert init_route.call_count == 1


@respx.mock
async def test_acestep_init_error_raises(ace, tmp_path) -> None:
    respx.get(f"{ACE}/health").mock(return_value=httpx.Response(500))
    respx.post(f"{ACE}/v1/init").mock(return_value=httpx.Response(500, text="no gpu"))
    with pytest.raises(RuntimeError, match="/v1/init 500"):
        await ace.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x"
        )


@respx.mock
async def test_acestep_vocal_language_and_bpm_ride_in_audio_config(ace, tmp_path) -> None:
    respx.get(f"{ACE}/health").mock(
        return_value=httpx.Response(
            200, json={"data": {"models_initialized": True, "llm_initialized": True}}
        )
    )
    captured: dict = {}

    def _capture(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_audio_response("data:audio/mpeg;base64,QUJD"))

    respx.post(f"{ACE}/v1/chat/completions").mock(side_effect=_capture)
    await ace.produce(
        prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
        params={"vocal_language": "ja", "bpm": 92.0, "instrumental": False},
    )
    ac = captured["audio_config"]
    assert ac["vocal_language"] == "ja"
    assert ac["bpm"] == 92  # coerced to int for the adapter
    assert ac["instrumental"] is False


def test_acestep_extract_audio_no_choices_raises() -> None:
    with pytest.raises(RuntimeError, match="no choices"):
        _extract_audio({"choices": []}, "mp3")


def test_acestep_extract_audio_non_data_url_raises() -> None:
    # An http(s) URL isn't inline base64 — the item is skipped and the miss
    # surfaces as the structured "no base64 audio" error.
    resp = _audio_response("http://ace.test/files/track.mp3")
    with pytest.raises(RuntimeError, match="did not include base64 audio"):
        _extract_audio(resp, "mp3")


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("data:audio/ogg;base64", "ogg"),
        ("data:audio/flac;base64", "flac"),
        ("data:audio/x-unknown;base64", "mp3"),  # falls back to requested format
    ],
)
def test_acestep_format_from_header(header: str, expected: str) -> None:
    assert _format_from_header(header, "mp3") == expected


# --- ComfyUI --------------------------------------------------------------


@pytest.fixture
def comfy():
    return ComfyUIProvider(base_url=COMFY, timeout_s=10)


@respx.mock
async def test_comfyui_enqueue_error_raises(comfy, tmp_path) -> None:
    respx.post(f"{COMFY}/prompt").mock(
        return_value=httpx.Response(400, text="invalid workflow")
    )
    with pytest.raises(RuntimeError, match="/prompt error 400"):
        await comfy.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={"workflow_template": "wander_scene_image"},
        )


@respx.mock
async def test_comfyui_no_usable_output_items_raises(comfy, tmp_path) -> None:
    # A terminal history entry whose output node has no filename-bearing item
    # (text-only outputs, malformed items) must fail loudly, not write junk.
    prompt_id = "no-items"
    respx.post(f"{COMFY}/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": prompt_id})
    )
    entry = {
        prompt_id: {
            "status": {"status_str": "success"},
            "outputs": {"8": {"text": ["caption"], "images": [{"nofile": 1}, 42]}},
        }
    }
    respx.get(f"{COMFY}/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json=entry)
    )
    with pytest.raises(RuntimeError, match="produced no output items"):
        await comfy.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={"workflow_template": "wander_scene_image"},
        )


async def test_comfyui_stage_inputs_requires_source_image_url(comfy) -> None:
    meta = {"inject": {"source_image": {"node": "1", "input": "image"}}}
    with pytest.raises(ValueError, match="source_image_url"):
        await comfy._stage_inputs({}, {}, meta)


def test_comfyui_collect_outputs_honors_output_node() -> None:
    entry = {
        "outputs": {
            "8": {"images": [{"filename": "intermediate.png"}]},
            "17": {"images": [{"filename": "final.png"}]},
        }
    }
    out = ComfyUIProvider._collect_outputs(entry, {"output_node": "17"})
    assert [i["filename"] for i in out] == ["final.png"]


def test_comfyui_set_input_missing_node_raises() -> None:
    with pytest.raises(KeyError, match="missing node"):
        _set_input({"1": {"inputs": {}}}, {"node": "99", "input": "text"}, "v")


@respx.mock
async def test_comfyui_load_image_source_http() -> None:
    respx.get("http://img.test/stills/frame.png").mock(
        return_value=httpx.Response(200, content=b"IMGBYTES")
    )
    data, name = await _load_image_source("http://img.test/stills/frame.png")
    assert data == b"IMGBYTES"
    assert name == "frame.png"


@respx.mock
async def test_comfyui_load_image_source_http_no_filename_defaults() -> None:
    respx.get("http://img.test/").mock(return_value=httpx.Response(200, content=b"I"))
    _data, name = await _load_image_source("http://img.test/")
    assert name == "source.png"


def test_comfyui_duration_none_without_length_or_fps() -> None:
    assert _duration_from_params({"length": 49}, "video/mp4") is None
    assert _duration_from_params({}, "video/mp4") is None


# --- resident helpers -----------------------------------------------------


class _FlakyOllama:
    """load/unload succeed except for the configured failing model names."""

    def __init__(self, fail: set[str]) -> None:
        self._fail = fail
        self.loads: list[tuple[str, int | None]] = []
        self.unloads: list[str] = []

    async def list_loaded(self) -> list[dict]:
        return []

    async def load(self, model, keep_alive=None) -> None:
        if model in self._fail:
            raise RuntimeError("pull failed")
        self.loads.append((model, keep_alive))

    async def unload(self, model) -> None:
        if model in self._fail:
            raise RuntimeError("not loaded")
        self.unloads.append(model)


def _set_residents(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(resident, "resident_model_names", lambda: names)


async def test_pin_resident_models_skips_failures(monkeypatch) -> None:
    # One un-pinnable model must not abort worker boot; the rest still pin.
    _set_residents(monkeypatch, ["good:1", "broken:2", "good:3"])
    ollama = _FlakyOllama(fail={"broken:2"})
    pinned = await pin_resident_models(ollama)
    assert pinned == ["good:1", "good:3"]
    assert ollama.loads == [("good:1", PIN_FOREVER), ("good:3", PIN_FOREVER)]


async def test_unload_resident_models_best_effort(monkeypatch) -> None:
    _set_residents(monkeypatch, ["good:1", "broken:2"])
    ollama = _FlakyOllama(fail={"broken:2"})
    unloaded = await unload_resident_models(ollama)
    assert unloaded == ["good:1"]
    assert ollama.unloads == ["good:1"]


async def test_reconcile_repin_failure_is_logged_not_raised(monkeypatch) -> None:
    _set_residents(monkeypatch, ["good:1", "broken:2"])
    ollama = _FlakyOllama(fail={"broken:2"})
    resume_residency()
    repinned = await reconcile_resident_models(ollama)
    assert repinned == ["good:1"]
