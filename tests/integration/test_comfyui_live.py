"""Live ComfyUI integration for the upscale_image / inpaint_image workflows.

Skipped unless CONDUCT_LIVE_COMFYUI=1 — these hit the real ComfyUI daemon
(COMFYUI_BASE_URL, default http://localhost:8188) and run actual models, so
they cost GPU-minutes and require RealESRGAN_x4plus.pth + SDXL base to be
installed. Run explicitly:

    CONDUCT_LIVE_COMFYUI=1 uv run pytest tests/integration/test_comfyui_live.py -v
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest

from providers.comfyui import ComfyUIProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("CONDUCT_LIVE_COMFYUI") != "1",
    reason="live ComfyUI tests are opt-in (CONDUCT_LIVE_COMFYUI=1)",
)

_BASE = os.environ.get("COMFYUI_BASE_URL", "http://localhost:8188")


def _tiny_png(path: Path, width=64, height=64) -> Path:
    """A real decodable PNG (solid dark blue) with zero dependencies."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    raw = b"".join(
        b"\x00" + b"\x20\x30\x60" * width for _ in range(height)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return path


def _png_size(data: bytes) -> tuple[int, int]:
    w, h = struct.unpack(">II", data[16:24])
    return w, h


async def test_live_upscale_hits_exact_target(tmp_path):
    src = _tiny_png(tmp_path / "src.png")
    provider = ComfyUIProvider(base_url=_BASE, timeout_s=600)
    resp = await provider.produce(
        prompt="",
        inputs={"source_image_url": str(src)},
        output_dir=str(tmp_path),
        output_basename="live_upscale",
        params={"workflow_template": "upscale_image", "width": 200, "height": 0},
    )
    data = Path(resp.file_path).read_bytes()
    w, h = _png_size(data)
    assert (w, h) == (200, 200)


async def test_live_outpaint_extends_left(tmp_path):
    src = _tiny_png(tmp_path / "src2.png", width=128, height=128)
    provider = ComfyUIProvider(base_url=_BASE, timeout_s=600)
    resp = await provider.produce(
        prompt="deep blue night sky, smooth gradient",
        inputs={"source_image_url": str(src)},
        output_dir=str(tmp_path),
        output_basename="live_outpaint",
        params={
            "workflow_template": "inpaint_image",
            "left": 64,
            "seed": 7,
            "steps": 12,  # keep the live test cheap; quality irrelevant here
        },
    )
    data = Path(resp.file_path).read_bytes()
    w, h = _png_size(data)
    assert (w, h) == (192, 128)
    assert resp.extra["params"]["seed"] == 7
