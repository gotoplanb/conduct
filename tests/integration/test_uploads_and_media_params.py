"""POST /uploads ingestion + the media-job plumbing that consumes it.

The upload endpoint is how external source art (e.g. an existing book
cover) enters the media pipeline; the executor pieces here are the
`/output/` URL translation and the `inputs.params` per-request knobs.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

import routes.uploads as uploads_route
from config.settings import get_settings
from models.job import Job
from models.types import JobStatus
from providers.media_base import BaseMediaProvider, MediaResponse
from providers.registry import ProviderRegistry
from worker.executor import _resolve_media_input_refs, execute_media_job

# --- POST /uploads ----------------------------------------------------------


async def test_upload_png_lands_in_output_uploads(client, client_headers) -> None:
    r = await client.post(
        "/uploads",
        files={"file": ("cover art.png", b"\x89PNG fake bytes", "image/png")},
        headers=client_headers,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["bytes"] == len(b"\x89PNG fake bytes")
    # Server-minted UUID name; the client's filename (with its space) is gone.
    assert body["source_url"].startswith("/output/uploads/")
    assert body["source_url"].endswith(".png")
    assert " " not in body["source_url"]

    on_disk = Path(get_settings().tts_output_dir) / "uploads" / Path(body["source_url"]).name
    assert on_disk.read_bytes() == b"\x89PNG fake bytes"
    on_disk.unlink()  # keep the shared output dir clean between runs


async def test_upload_rejects_unknown_suffix(client, client_headers) -> None:
    r = await client.post(
        "/uploads",
        files={"file": ("evil.svg", b"<svg/>", "image/svg+xml")},
        headers=client_headers,
    )
    assert r.status_code == 415


async def test_upload_rejects_empty_file(client, client_headers) -> None:
    r = await client.post(
        "/uploads",
        files={"file": ("empty.png", b"", "image/png")},
        headers=client_headers,
    )
    assert r.status_code == 400


async def test_upload_rejects_oversize(client, client_headers, monkeypatch) -> None:
    monkeypatch.setattr(uploads_route, "_MAX_BYTES", 8)
    r = await client.post(
        "/uploads",
        files={"file": ("big.png", b"123456789", "image/png")},
        headers=client_headers,
    )
    assert r.status_code == 413


async def test_upload_requires_client_auth(client) -> None:
    r = await client.post(
        "/uploads",
        files={"file": ("a.png", b"x", "image/png")},
    )
    assert r.status_code in (401, 403)


# --- GET /output with a client key ------------------------------------------


async def test_output_accepts_client_key(client, client_headers, tmp_path) -> None:
    # Media clients fetch their results from /output — client keys must pass
    # the gate (admin-only before this; see _admin_or_client_auth).
    from uuid import uuid4 as _uuid4

    name = f"{_uuid4()}.png"
    out = Path(get_settings().tts_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_bytes(b"png!")
    try:
        r = await client.get(f"/output/{name}", headers=client_headers)
        assert r.status_code == 200
        assert r.content == b"png!"
    finally:
        (out / name).unlink()


async def test_output_rejects_bogus_client_key(client) -> None:
    r = await client.get(
        "/output/00000000-0000-0000-0000-000000000000.png",
        headers={"Authorization": "Bearer cdt_bogus"},
    )
    assert r.status_code == 403


# --- executor: /output/ source translation ----------------------------------


async def test_output_url_source_translates_without_deprecation_warning(
    db_session: AsyncSession, caplog
) -> None:
    with caplog.at_level(logging.WARNING, logger="worker.executor"):
        resolved = await _resolve_media_input_refs(
            {"source_image_url": "/output/uploads/abc.png"}, db_session
        )
    assert resolved["source_image_url"] == "/app/output/uploads/abc.png"
    assert "deprecated" not in caplog.text


async def test_foreign_url_source_still_warns(
    db_session: AsyncSession, caplog
) -> None:
    with caplog.at_level(logging.WARNING, logger="worker.executor"):
        resolved = await _resolve_media_input_refs(
            {"source_image_url": "https://elsewhere.test/a.png"}, db_session
        )
    assert resolved["source_image_url"] == "https://elsewhere.test/a.png"
    assert "deprecated" in caplog.text


# --- executor: inputs.params → provider params ------------------------------


class _ParamCapture(BaseMediaProvider):
    name = "capture"

    def __init__(self) -> None:
        self.params: dict | None = None

    async def produce(self, *, prompt, inputs, output_dir, output_basename, params=None):
        self.params = dict(params or {})
        return MediaResponse(
            file_path=f"{output_dir}/{output_basename}.png",
            url_path=f"/output/{output_basename}.png",
            mime_type="image/png",
            width=None, height=None, duration_s=None,
            latency_ms=1, cost_usd=Decimal("0"),
            model_used="upscale_image", provider=self.name,
            extra={},
        )


async def test_request_params_override_rule_params_but_not_template(
    db_session: AsyncSession, seeded_client
) -> None:
    job = Job(
        client_app_id=seeded_client[0].id,
        task_type="upscale_image",
        sensitivity="internal",
        prompt="",
        inputs={
            "source_image_url": "/output/uploads/art.png",
            # workflow_template here must lose to the routing rule's choice.
            "params": {"width": 3663, "seed": 9, "workflow_template": "evil"},
        },
        status=JobStatus.PENDING.value,
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    capture = _ParamCapture()
    reg = ProviderRegistry()
    reg.register_media(capture)

    await execute_media_job(
        job=job,
        media_provider_name="capture",
        media_kind="image",
        workflow_template="upscale_image",
        providers=reg,
        output_dir="/tmp/does-not-matter",
        session=db_session,
        extra_params={"width": 1, "upscale_method": "lanczos"},
    )

    assert capture.params is not None
    assert capture.params["width"] == 3663  # request beats rule default
    assert capture.params["upscale_method"] == "lanczos"  # rule default survives
    assert capture.params["seed"] == 9
    assert capture.params["workflow_template"] == "upscale_image"  # rule wins
