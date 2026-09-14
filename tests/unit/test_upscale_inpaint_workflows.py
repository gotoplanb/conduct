"""Unit tests for the upscale_image / inpaint_image workflow templates.

Same respx-mock approach as test_comfyui_provider.py: no GPU, no ComfyUI.
The live path (real ComfyUI run) is tests/integration/test_comfyui_live.py,
gated behind CONDUCT_LIVE_COMFYUI=1.
"""

from __future__ import annotations

import json

import pytest
import respx
from httpx import Response

from providers.comfyui import WORKFLOWS_DIR, ComfyUIProvider

TEMPLATES = ("upscale_image", "inpaint_image")


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_loads_and_meta_targets_exist(name):
    provider = ComfyUIProvider(base_url="http://comfy.test")
    workflow, meta = provider._load_workflow(name)
    assert "_meta" not in workflow
    for key, target in meta["inject"].items():
        node = workflow.get(str(target["node"]))
        assert node is not None, f"{name}: inject {key!r} references missing node"
    assert str(meta["output_node"]) in workflow


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_parses_as_strict_json(name):
    # A trailing comma or comment would take the whole task type down at
    # produce() time; catch it at test time instead.
    raw = (WORKFLOWS_DIR / f"{name}.json").read_text()
    json.loads(raw)


def _mock_comfy(mock, captured, filename="conduct_out_00001_.png", node="9"):
    def _capture(request):
        captured["workflow"] = json.loads(request.content)["prompt"]
        return Response(200, json={"prompt_id": "p1"})

    mock.post("/upload/image").mock(return_value=Response(200, json={"name": "staged"}))
    mock.post("/prompt").mock(side_effect=_capture)
    mock.get("/history/p1").mock(
        return_value=Response(200, json={
            "p1": {
                "status": {"status_str": "success"},
                "outputs": {node: {"images": [
                    {"filename": filename, "subfolder": "", "type": "output"}
                ]}},
            }
        })
    )
    mock.get(url__regex=r".*/view.*").mock(
        return_value=Response(200, content=b"png-bytes")
    )


async def test_inpaint_injects_pad_prompt_and_seed(tmp_path):
    src = tmp_path / "cover.png"
    src.write_bytes(b"src")
    provider = ComfyUIProvider(base_url="http://comfy.test", timeout_s=10)
    captured: dict = {}

    with respx.mock(base_url="http://comfy.test") as mock:
        _mock_comfy(mock, captured, node="9")
        resp = await provider.produce(
            prompt="continue the night-sky painting seamlessly",
            inputs={"source_image_url": str(src)},
            output_dir=str(tmp_path),
            output_basename="job1",
            params={
                "workflow_template": "inpaint_image",
                "left": 1800,
                "seed": 7,
                "negative_prompt": "seams",
            },
        )

    wf = captured["workflow"]
    assert wf["3"]["inputs"]["left"] == 1800
    assert wf["3"]["inputs"]["right"] == 0  # default holds for unspecified sides
    assert wf["4"]["inputs"]["text"] == "continue the night-sky painting seamlessly"
    assert wf["5"]["inputs"]["text"] == "seams"
    assert wf["7"]["inputs"]["seed"] == 7
    assert wf["7"]["inputs"]["denoise"] == 1.0
    # Source image was staged and wired into LoadImage.
    assert wf["2"]["inputs"]["image"].startswith("conduct_")
    # Effective params (defaults + overrides) are echoed for reproducibility.
    assert resp.extra["params"]["seed"] == 7
    assert resp.extra["params"]["left"] == 1800
    assert resp.extra["params"]["feathering"] == 40


async def test_upscale_injects_target_size_and_model(tmp_path):
    src = tmp_path / "art.png"
    src.write_bytes(b"src")
    provider = ComfyUIProvider(base_url="http://comfy.test", timeout_s=10)
    captured: dict = {}

    with respx.mock(base_url="http://comfy.test") as mock:
        _mock_comfy(mock, captured, node="5")
        resp = await provider.produce(
            prompt="",
            inputs={"source_image_url": str(src)},
            output_dir=str(tmp_path),
            output_basename="job2",
            params={
                "workflow_template": "upscale_image",
                "width": 3663,
                "height": 0,
            },
        )

    wf = captured["workflow"]
    assert wf["4"]["inputs"]["width"] == 3663
    assert wf["4"]["inputs"]["height"] == 0
    assert wf["2"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"
    assert resp.extra["params"]["width"] == 3663
    assert resp.extra["params"]["model_name"] == "RealESRGAN_x4plus.pth"


async def test_upscale_defaults_keep_native_4x_output(tmp_path):
    # No width/height → ImageScale gets 0/0, ComfyUI's passthrough mode.
    src = tmp_path / "art.png"
    src.write_bytes(b"src")
    provider = ComfyUIProvider(base_url="http://comfy.test", timeout_s=10)
    captured: dict = {}

    with respx.mock(base_url="http://comfy.test") as mock:
        _mock_comfy(mock, captured, node="5")
        await provider.produce(
            prompt="",
            inputs={"source_image_url": str(src)},
            output_dir=str(tmp_path),
            output_basename="job3",
            params={"workflow_template": "upscale_image"},
        )

    wf = captured["workflow"]
    assert wf["4"]["inputs"]["width"] == 0
    assert wf["4"]["inputs"]["height"] == 0
