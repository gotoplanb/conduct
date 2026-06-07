"""Unit tests for ComfyUIProvider.

The ComfyUI HTTP API is mocked via respx (already a dev dep). Tests cover:
- happy path (image + video templates)
- workflow injection (prompt + params + source image)
- failure surfaces (ComfyUI status_str='error' → RuntimeError, not infinite poll)
- timeout
- output retrieval URL shape
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from providers.comfyui import WORKFLOWS_DIR, ComfyUIProvider


@pytest.fixture
def provider(tmp_path):
    # tmp output_dir so file writes stay scoped to the test.
    return ComfyUIProvider(base_url="http://comfy.test", timeout_s=10)


def _history_complete(prompt_id, filename="conduct_wander_scene_00001_.png", node_id="8"):
    return {
        prompt_id: {
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                node_id: {
                    "images": [
                        {"filename": filename, "subfolder": "", "type": "output"}
                    ]
                }
            },
        }
    }


def _history_video(prompt_id):
    return {
        prompt_id: {
            "status": {"status_str": "success"},
            "outputs": {
                "17": {
                    "video": [
                        {"filename": "conduct_wander_video_00001_.mp4",
                         "subfolder": "", "type": "output"}
                    ]
                }
            },
        }
    }


def _history_failed(prompt_id):
    return {
        prompt_id: {
            "status": {"status_str": "error", "messages": [["execution_error", {}]]},
            "outputs": {},
        }
    }


@pytest.mark.asyncio
@respx.mock
async def test_image_workflow_happy_path(provider, tmp_path) -> None:
    """SDXL image workflow → ComfyUI returns a single PNG → provider writes
    it under output_dir and reports the URL path."""
    prompt_id = "test-prompt-image"
    respx.post("http://comfy.test/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": prompt_id, "number": 0})
    )
    respx.get(f"http://comfy.test/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json=_history_complete(prompt_id))
    )
    respx.get("http://comfy.test/view").mock(
        return_value=httpx.Response(200, content=b"PNGDATA")
    )

    result = await provider.produce(
        prompt="misty mountain village",
        inputs={},
        output_dir=str(tmp_path),
        output_basename="job-abc",
        params={"workflow_template": "wander_scene_image", "width": 1024, "height": 768},
    )

    assert result.provider == "comfyui"
    assert result.mime_type == "image/png"
    assert result.url_path == "/output/job-abc.png"
    assert result.width == 1024 and result.height == 768
    assert result.cost_usd == Decimal("0")
    assert Path(result.file_path).read_bytes() == b"PNGDATA"
    assert result.extra["workflow_template"] == "wander_scene_image"
    assert result.extra["prompt_id"] == prompt_id


@pytest.mark.asyncio
@respx.mock
async def test_workflow_injection_writes_prompt_into_named_node(
    provider, tmp_path
) -> None:
    """Confirm the workflow we POST has the prompt landed in the node the
    template's _meta.inject map names — regression-pins the injection logic."""
    prompt_id = "test-injection"
    posted = []

    def _capture(request):
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": prompt_id})

    respx.post("http://comfy.test/prompt").mock(side_effect=_capture)
    respx.get(f"http://comfy.test/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json=_history_complete(prompt_id))
    )
    respx.get("http://comfy.test/view").mock(
        return_value=httpx.Response(200, content=b"x")
    )

    await provider.produce(
        prompt="THE PROMPT",
        inputs={},
        output_dir=str(tmp_path),
        output_basename="x",
        params={"workflow_template": "wander_scene_image"},
    )

    # Node 3 (CLIPTextEncode positive) carries the prompt per the template's
    # _meta.inject.prompt mapping.
    assert posted[0]["prompt"]["3"]["inputs"]["text"] == "THE PROMPT"


@pytest.mark.asyncio
@respx.mock
async def test_video_workflow_uploads_source_image(provider, tmp_path) -> None:
    """Video workflow requires inputs.source_image_url. Provider must POST
    it to /upload/image so ComfyUI's LoadImage node can find it."""
    # Create a real file at a worker-local path the provider will read.
    src = tmp_path / "still.png"
    src.write_bytes(b"FAKEPNG")

    prompt_id = "test-video"
    upload_called = {"count": 0}

    def _upload(request):
        upload_called["count"] += 1
        return httpx.Response(200, json={"name": "uploaded.png", "subfolder": "", "type": "input"})

    respx.post("http://comfy.test/upload/image").mock(side_effect=_upload)
    respx.post("http://comfy.test/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": prompt_id})
    )
    respx.get(f"http://comfy.test/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json=_history_video(prompt_id))
    )
    respx.get("http://comfy.test/view").mock(
        return_value=httpx.Response(200, content=b"VIDDATA")
    )

    result = await provider.produce(
        prompt="gentle breeze",
        inputs={"source_image_url": str(src)},
        output_dir=str(tmp_path),
        output_basename="vid",
        params={
            "workflow_template": "wander_scene_video",
            "width": 720, "height": 480, "length": 49, "fps": 16,
        },
    )

    assert upload_called["count"] == 1
    assert result.mime_type == "video/mp4"
    assert result.url_path == "/output/vid.mp4"
    assert result.duration_s == pytest.approx(49 / 16)


@pytest.mark.asyncio
@respx.mock
async def test_workflow_error_surfaces_as_runtime_error(provider, tmp_path) -> None:
    """Failed workflows mark status_str=error with empty outputs. Without
    this branch the poll loop would hang waiting for outputs forever — same
    bug I hit during the FP8 smoke test."""
    prompt_id = "test-error"
    respx.post("http://comfy.test/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": prompt_id})
    )
    respx.get(f"http://comfy.test/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json=_history_failed(prompt_id))
    )

    with pytest.raises(RuntimeError, match="failed"):
        await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={"workflow_template": "wander_scene_image"},
        )


@pytest.mark.asyncio
@respx.mock
async def test_polling_timeout_raises(provider, tmp_path) -> None:
    """If ComfyUI never reports done within timeout_s, raise TimeoutError —
    don't let a wedged workflow tie up the worker forever."""
    provider._timeout_s = 0.5  # force timeout fast
    prompt_id = "test-timeout"
    respx.post("http://comfy.test/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": prompt_id})
    )
    # History returns "still running" forever.
    respx.get(f"http://comfy.test/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json={})
    )

    with pytest.raises(TimeoutError):
        await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={"workflow_template": "wander_scene_image"},
        )


@pytest.mark.asyncio
async def test_missing_workflow_template_raises(provider, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={"workflow_template": "does_not_exist"},
        )


@pytest.mark.asyncio
async def test_missing_workflow_template_param_raises(provider, tmp_path) -> None:
    with pytest.raises(ValueError, match="workflow_template"):
        await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={},
        )


def test_bundled_workflows_are_valid_json_with_meta() -> None:
    """Every committed workflow has the `_meta` block we depend on for
    injection. Regression-pins the templates against accidental edits."""
    found = list(WORKFLOWS_DIR.glob("*.json"))
    assert found, "expected bundled workflow templates"
    for path in found:
        wf = json.loads(path.read_text())
        meta = wf.get("_meta")
        assert meta, f"{path.name} missing _meta"
        assert "inject" in meta, f"{path.name} _meta missing inject"
        assert "output_node" in meta, f"{path.name} _meta missing output_node"


@pytest.mark.asyncio
@respx.mock
async def test_poll_succeeds_after_a_few_misses(provider, tmp_path) -> None:
    """Provider tolerates the entry being missing from /history for a few
    polls (ComfyUI takes a moment to register the job in its db). Don't
    error out — just keep polling."""
    prompt_id = "test-eventual"
    respx.post("http://comfy.test/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": prompt_id})
    )
    miss_then_hit = [
        httpx.Response(200, json={}),  # not yet registered
        httpx.Response(200, json={prompt_id: {"status": {"status_str": "running"}}}),
        httpx.Response(200, json=_history_complete(prompt_id)),
    ]
    respx.get(f"http://comfy.test/history/{prompt_id}").mock(
        side_effect=miss_then_hit
    )
    respx.get("http://comfy.test/view").mock(
        return_value=httpx.Response(200, content=b"P")
    )

    # Shorten the poll interval so the test runs in well under a second.
    import providers.comfyui as comfyui_mod
    original = comfyui_mod._POLL_INTERVAL_S
    comfyui_mod._POLL_INTERVAL_S = 0.01
    try:
        r = await provider.produce(
            prompt="x", inputs={}, output_dir=str(tmp_path), output_basename="x",
            params={"workflow_template": "wander_scene_image"},
        )
    finally:
        comfyui_mod._POLL_INTERVAL_S = original
    assert r.url_path == "/output/x.png"
