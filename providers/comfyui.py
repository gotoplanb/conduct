"""ComfyUI media provider — image + video via REST.

Conduct dispatches `image` and `video` task kinds through this provider.
Image goes through SDXL (+ LoRA), video goes through Wan 2.2 14B I2V. The
two share the API mechanics; the difference is purely which workflow JSON
template gets loaded and how the outputs are saved.

Workflow templates live in `conduct/comfy_workflows/{name}.json` and ship
with the conduct image. Each template has a `_meta` block that points at
which node IDs receive the prompt / negative prompt / inputs / params, so
this provider is template-agnostic — adding a new task type is "drop a JSON
in `comfy_workflows/`, add a routing rule that names it".

Reachability: by default the provider talks to
`http://host.docker.internal:8188` (the macOS Docker bridge to the host
ComfyUI launchd service). Override with `COMFYUI_BASE_URL` when running
on Linux or a Mac mini cluster setup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
import uuid
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from providers.media_base import BaseMediaProvider, MediaResponse

log = logging.getLogger(__name__)

# Bundled workflow templates ship next to the conduct package — same path
# inside and outside the container.
WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "comfy_workflows"

# How often the provider polls /history while a job is running. Workflow
# generation runs from 20 s (SDXL image) to many minutes (Wan I2V); a 2 s
# cadence keeps the overhead negligible vs the model time while still
# completing image tasks promptly.
_POLL_INTERVAL_S = 2.0

# Provider-level timeout. The worker has its own RQ job timeout (longer for
# the conduct-media queue), so this is a defense-in-depth — kill the poll
# loop if ComfyUI got wedged.
_DEFAULT_TIMEOUT_S = 3600  # 1 hour


class ComfyUIProvider(BaseMediaProvider):
    name = "comfyui"

    def __init__(
        self,
        *,
        base_url: str = "http://host.docker.internal:8188",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client_id = uuid.uuid4().hex

    async def produce(
        self,
        *,
        prompt: str,
        inputs: dict[str, Any],
        output_dir: str,
        output_basename: str,
        params: dict[str, Any] | None = None,
    ) -> MediaResponse:
        """Run a workflow on ComfyUI and write the output under
        `output_dir/output_basename.<ext>`.

        Required params/inputs:
          - `workflow_template` (in `params` or `inputs`) — which JSON file
            under `comfy_workflows/` to load. Required for every call.
          - For video tasks: `inputs["source_image_url"]` — a URL or path
            ComfyUI can read. The provider fetches it once and uploads it
            via `/upload/image` so ComfyUI's `LoadImage` node can see it.
        """
        params = params or {}
        template_name = params.get("workflow_template") or inputs.get("workflow_template")
        if not template_name:
            raise ValueError(
                "ComfyUIProvider requires `workflow_template` in params or inputs"
            )

        workflow, meta = self._load_workflow(template_name)
        await self._stage_inputs(inputs, workflow, meta)
        self._inject(workflow, meta, prompt, params)

        started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url=self._base, timeout=httpx.Timeout(60.0, read=self._timeout_s)
        ) as client:
            prompt_id = await self._enqueue(client, workflow)
            history_entry = await self._poll_for_completion(client, prompt_id)
            outputs = self._collect_outputs(history_entry, meta)
            if not outputs:
                raise RuntimeError(
                    f"ComfyUI workflow {template_name!r} produced no output items"
                )

            # First output wins. Multi-output workflows (e.g. ones that save
            # intermediates) can be supported by extending MediaResponse with a
            # list, but for the wander_* templates each produces one artifact.
            head = outputs[0]
            data = await self._fetch(client, head)
        latency_ms = int((time.perf_counter() - started) * 1000)

        # Choose extension based on filename ComfyUI gave us (it knows mp4
        # vs png vs whatever the SaveVideo / SaveImage node decided).
        suffix = Path(head["filename"]).suffix or ".bin"
        out_path = Path(output_dir) / f"{output_basename}{suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)

        mime = _mime_from_suffix(suffix)
        return MediaResponse(
            file_path=str(out_path),
            url_path=f"/output/{out_path.name}",
            mime_type=mime,
            width=params.get("width"),
            height=params.get("height"),
            duration_s=_duration_from_params(params, mime),
            latency_ms=latency_ms,
            cost_usd=Decimal("0"),  # local model, no per-token cost
            model_used=template_name,
            provider=self.name,
            extra={"prompt_id": prompt_id, "workflow_template": template_name},
        )

    # ------------------------------------------------------------------
    # Workflow loading + injection
    # ------------------------------------------------------------------

    def _load_workflow(self, name: str) -> tuple[dict, dict]:
        """Read the named workflow template and return (workflow, meta).
        The `_meta` block is stripped from the workflow before posting —
        ComfyUI rejects unknown top-level keys."""
        path = WORKFLOWS_DIR / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"workflow template not found at {path} "
                f"(available: {[p.name for p in WORKFLOWS_DIR.glob('*.json')]})"
            )
        loaded = json.loads(path.read_text())
        meta = loaded.pop("_meta", {})
        return deepcopy(loaded), meta

    def _inject(
        self, workflow: dict, meta: dict, prompt: str, params: dict[str, Any]
    ) -> None:
        """Apply the prompt + params to the workflow per the template's
        `_meta.inject` map. Each entry says "this user-supplied value lands
        in node X's input field Y", so the provider stays agnostic to which
        nodes a given workflow uses for prompting."""
        defaults = dict(meta.get("default_params") or {})
        defaults.update(params)
        injection_map = meta.get("inject") or {}

        for key, target in injection_map.items():
            if key == "prompt":
                _set_input(workflow, target, prompt)
                continue
            if key == "negative_prompt":
                _set_input(
                    workflow,
                    target,
                    params.get("negative_prompt") or meta.get("default_negative") or "",
                )
                continue
            # Anything else comes from params (width/height/fps/seed/...).
            if key in defaults:
                _set_input(workflow, target, defaults[key])

    async def _stage_inputs(
        self, inputs: dict[str, Any], workflow: dict, meta: dict
    ) -> None:
        """If the workflow needs a source image (video tasks), fetch it once
        and upload to ComfyUI's input dir under a deterministic name so the
        `LoadImage` node can pick it up."""
        injection_map = meta.get("inject") or {}
        if "source_image" not in injection_map:
            return
        src = inputs.get("source_image_url")
        if not src:
            raise ValueError(
                "this workflow expects `inputs.source_image_url` (URL or path "
                "to an image to animate)"
            )
        # Resolve src to bytes — accept http(s) URLs, file:// URLs, and bare
        # paths. The TTS executor's `/output/` URLs land here as bare paths
        # when the worker passes them through.
        image_bytes, suggested_name = await _load_image_source(src)
        # Stage as a stable name based on the URL — repeat calls reuse it.
        src_hash = uuid.uuid5(uuid.NAMESPACE_URL, str(src)).hex
        suffix = Path(suggested_name).suffix or ".png"
        staged_name = f"conduct_{src_hash}{suffix}"
        async with httpx.AsyncClient(base_url=self._base, timeout=60.0) as client:
            files = {"image": (staged_name, image_bytes, "application/octet-stream")}
            data = {"overwrite": "true", "type": "input"}
            r = await client.post("/upload/image", files=files, data=data)
            r.raise_for_status()
        _set_input(workflow, injection_map["source_image"], staged_name)

    # ------------------------------------------------------------------
    # API mechanics
    # ------------------------------------------------------------------

    async def _enqueue(self, client: httpx.AsyncClient, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": self._client_id}
        r = await client.post("/prompt", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"ComfyUI /prompt error {r.status_code}: {r.text[:500]}")
        data = r.json()
        return data["prompt_id"]

    async def _poll_for_completion(
        self, client: httpx.AsyncClient, prompt_id: str
    ) -> dict:
        """Poll /history/{id} until the entry has outputs (success) or
        status_str='error' (failure). The smoke-test work surfaced this:
        without checking status_str, a failed workflow would silently poll
        forever waiting for outputs that never come."""
        deadline = time.monotonic() + self._timeout_s
        while True:
            entry = await self._fetch_history_entry(client, prompt_id)
            done = _terminal_state(entry, prompt_id)
            if done is not None:
                return done
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"ComfyUI workflow {prompt_id} did not complete within "
                    f"{self._timeout_s}s"
                )
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def _fetch_history_entry(
        self, client: httpx.AsyncClient, prompt_id: str
    ) -> dict | None:
        r = await client.get(f"/history/{prompt_id}")
        r.raise_for_status()
        return r.json().get(prompt_id)

    @staticmethod
    def _collect_outputs(entry: dict, meta: dict) -> list[dict]:
        """The workflow's `_meta.output_node` (when present) constrains which
        node we read from. Otherwise, take the first output node with any
        usable items."""
        output_node = meta.get("output_node")
        outputs = entry.get("outputs") or {}
        candidates: list[dict] = []
        for node_id, node_out in outputs.items():
            if output_node and str(node_id) != str(output_node):
                continue
            candidates.extend(_items_from_node(node_out))
        return candidates

    async def _fetch(self, client: httpx.AsyncClient, item: dict) -> bytes:
        q = urllib.parse.urlencode({
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
        r = await client.get(f"/view?{q}")
        r.raise_for_status()
        return r.content


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _terminal_state(entry: dict | None, prompt_id: str) -> dict | None:
    """Return the history entry if it's done (success or failure-as-runtime
    error). Returns None if still in flight so the caller keeps polling."""
    if not entry:
        return None
    if entry.get("outputs"):
        return entry
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        msgs = status.get("messages") or []
        raise RuntimeError(f"ComfyUI workflow {prompt_id} failed: {msgs!r}")
    return None


def _items_from_node(node_out: dict) -> list[dict]:
    """Pull out every output item (video/gifs/images) carrying a filename."""
    found: list[dict] = []
    for kind in ("video", "gifs", "images"):
        for item in node_out.get(kind) or []:
            if isinstance(item, dict) and "filename" in item:
                found.append(item)
    return found


def _set_input(workflow: dict, target: dict, value: Any) -> None:
    """`target` is `{"node": "<id>", "input": "<input_name>"}`."""
    node = workflow.get(str(target["node"]))
    if node is None:
        raise KeyError(
            f"workflow injection target {target!r} references missing node"
        )
    node.setdefault("inputs", {})[target["input"]] = value


async def _load_image_source(src: str) -> tuple[bytes, str]:
    """Resolve `src` to (bytes, suggested_filename). Accepts http(s)
    URLs, file:// URLs, and bare worker-local paths."""
    if src.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.get(src)
            r.raise_for_status()
            name = Path(urllib.parse.urlparse(src).path).name or "source.png"
            return r.content, name
    p = Path(urllib.parse.urlparse(src).path) if src.startswith("file://") else Path(src)
    return p.read_bytes(), p.name


def _mime_from_suffix(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".gif": "image/gif",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
    }.get(suffix.lower(), "application/octet-stream")


def _duration_from_params(params: dict[str, Any], mime: str) -> float | None:
    """Best-effort duration for video tasks. ffprobe would be authoritative
    but adds a subprocess hop per call; using length/fps from params is
    close enough for the Job row and matches what the worker logs."""
    if not mime.startswith("video/"):
        return None
    length = params.get("length")
    fps = params.get("fps")
    if length and fps:
        return float(length) / float(fps)
    return None
