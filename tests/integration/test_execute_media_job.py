"""Integration tests for worker.executor.execute_media_job.

The executor wraps a BaseMediaProvider call: it sets job.status to running,
calls provider.produce(), writes Job.media_url + metadata['media'] on
success, captures exceptions as Job.error on failure. These tests pin the
contract via a stub provider; the real ComfyUI/ACE-Step paths get their
own provider-level tests under tests/unit/.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from models.job import Job
from models.types import JobStatus
from providers.media_base import BaseMediaProvider, MediaResponse
from providers.registry import ProviderRegistry
from worker.executor import execute_media_job


class _GoodMediaProvider(BaseMediaProvider):
    name = "good"

    async def produce(self, *, prompt, inputs, output_dir, output_basename, params=None):
        # Pretend we wrote a file — execute_media_job doesn't check the FS,
        # just records what produce() returned.
        return MediaResponse(
            file_path=f"{output_dir}/{output_basename}.png",
            url_path=f"/output/{output_basename}.png",
            mime_type="image/png",
            width=1024, height=768,
            duration_s=None,
            latency_ms=4242,
            cost_usd=Decimal("0"),
            model_used="wander_scene_image",
            provider=self.name,
            extra={"seed": 42, "workflow_template": params.get("workflow_template")},
        )


class _ExplodingMediaProvider(BaseMediaProvider):
    name = "boom"

    async def produce(self, **_kwargs):
        raise RuntimeError("comfy ate it")


async def _seed_media_job(
    db, *, client_id, task_type, inputs=None
) -> Job:
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="public",
        prompt="a misty village at sunrise",
        inputs=inputs or {},
        status=JobStatus.PENDING.value,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_success_path_writes_media_url_and_metadata(
    db_session: AsyncSession, seeded_client
) -> None:
    job = await _seed_media_job(
        db_session, client_id=seeded_client[0].id, task_type="wander_scene_image"
    )

    reg = ProviderRegistry()
    reg.register_media(_GoodMediaProvider())

    out = await execute_media_job(
        job=job,
        media_provider_name="good",
        media_kind="image",
        workflow_template="wander_scene_image",
        providers=reg,
        output_dir="/tmp/media-test",
        session=db_session,
    )

    assert out.status == JobStatus.COMPLETE.value
    assert out.media_url == f"/output/{job.id}.png"
    assert out.latency_ms == 4242
    assert out.cost_usd == Decimal("0")
    assert out.model_used == "wander_scene_image"
    media_meta = (out.job_metadata or {}).get("media") or {}
    assert media_meta["mime_type"] == "image/png"
    assert media_meta["width"] == 1024
    assert media_meta["provider"] == "good"
    assert media_meta["extra"]["seed"] == 42
    assert media_meta["extra"]["workflow_template"] == "wander_scene_image"


async def test_provider_exception_marks_job_failed_with_error_text(
    db_session: AsyncSession, seeded_client
) -> None:
    """Provider failures must not crash the worker — execute_media_job
    catches them, marks the job failed, and records the exception text in
    Job.error so /ui/jobs/{id} shows what went wrong."""
    job = await _seed_media_job(
        db_session, client_id=seeded_client[0].id, task_type="wander_scene_video"
    )

    reg = ProviderRegistry()
    reg.register_media(_ExplodingMediaProvider())

    out = await execute_media_job(
        job=job,
        media_provider_name="boom",
        media_kind="video",
        workflow_template="wander_scene_video",
        providers=reg,
        output_dir="/tmp/media-test",
        session=db_session,
    )

    assert out.status == JobStatus.FAILED.value
    assert "comfy ate it" in out.error
    assert out.media_url is None  # never set on failure
    assert out.completed_at is not None  # closed out timing-wise


async def test_dispatch_passes_workflow_template_through_params(
    db_session: AsyncSession, seeded_client
) -> None:
    """The runner derives the workflow template from the rule's
    preferred_model and passes it through extra_params. execute_media_job
    must surface it in params so the provider can branch on which workflow
    JSON to load."""
    job = await _seed_media_job(
        db_session, client_id=seeded_client[0].id, task_type="wander_scene_image"
    )
    reg = ProviderRegistry()
    captured = {}

    class _Capturing(BaseMediaProvider):
        name = "capture"

        async def produce(self, **kwargs):
            captured.update(kwargs)
            return MediaResponse(
                file_path="/tmp/x", url_path="/output/x",
                mime_type="image/png", width=None, height=None, duration_s=None,
                latency_ms=0, cost_usd=Decimal("0"), model_used="x",
                provider=self.name, extra={},
            )

    reg.register_media(_Capturing())
    await execute_media_job(
        job=job,
        media_provider_name="capture",
        media_kind="image",
        workflow_template="my_custom_workflow",
        providers=reg,
        output_dir="/tmp/x",
        session=db_session,
        extra_params={"width": 512, "height": 512},
    )

    # The runner's extra_params land merged with workflow_template into the
    # provider's params bag.
    assert captured["params"]["workflow_template"] == "my_custom_workflow"
    assert captured["params"]["width"] == 512
    assert captured["params"]["height"] == 512
    # The job's `inputs` flow through unchanged.
    assert captured["inputs"] == {}


async def test_unknown_media_provider_raises(
    db_session: AsyncSession, seeded_client
) -> None:
    job = await _seed_media_job(
        db_session, client_id=seeded_client[0].id, task_type="wander_scene_image"
    )
    reg = ProviderRegistry()  # nothing registered
    with pytest.raises(KeyError, match="media provider not registered"):
        await execute_media_job(
            job=job,
            media_provider_name="nobody",
            media_kind="image",
            workflow_template="x",
            providers=reg,
            output_dir="/tmp/x",
            session=db_session,
        )


async def _seed_completed_media_job(db, *, client_id, task_type, media_url):
    """Seed a complete media job with a known media_url so the next job in
    the chain has something to reference via source_*_job_id."""
    job = Job(
        client_app_id=client_id,
        task_type=task_type,
        sensitivity="public",
        prompt="upstream",
        status=JobStatus.COMPLETE.value,
        media_url=media_url,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_chain_by_job_id_resolves_to_worker_local_path(
    db_session: AsyncSession, seeded_client
) -> None:
    """source_image_job_id → look up the upstream job → hand the provider
    a worker-local path (/app/output/...). This is the contract for media
    chaining post-#14; clients pass ids, never URLs."""
    upstream = await _seed_completed_media_job(
        db_session,
        client_id=seeded_client[0].id,
        task_type="wander_scene_image",
        media_url=f"/output/{__import__('uuid').uuid4()}.png",
    )
    downstream = await _seed_media_job(
        db_session,
        client_id=seeded_client[0].id,
        task_type="wander_scene_video",
        inputs={"source_image_job_id": str(upstream.id)},
    )

    captured = {}

    class _Capturing(BaseMediaProvider):
        name = "capture"

        async def produce(self, **kwargs):
            captured.update(kwargs)
            return MediaResponse(
                file_path="/tmp/x", url_path=f"/output/{downstream.id}.mp4",
                mime_type="video/mp4", width=None, height=None, duration_s=None,
                latency_ms=0, cost_usd=Decimal("0"), model_used="x",
                provider=self.name, extra={},
            )

    reg = ProviderRegistry()
    reg.register_media(_Capturing())
    out = await execute_media_job(
        job=downstream,
        media_provider_name="capture",
        media_kind="video",
        workflow_template="wander_scene_video",
        providers=reg,
        output_dir="/tmp/x",
        session=db_session,
    )

    assert out.status == JobStatus.COMPLETE.value
    # Resolver translated the id ref into a worker-local file path; the
    # original *_job_id key was stripped so providers see only the *_url
    # they already understand.
    expected_path = upstream.media_url.replace("/output/", "/app/output/", 1)
    assert captured["inputs"]["source_image_url"] == expected_path
    assert "source_image_job_id" not in captured["inputs"]


async def test_chain_by_job_id_unknown_id_marks_failed_with_clear_error(
    db_session: AsyncSession, seeded_client
) -> None:
    """A bad job_id ref must fail the downstream job loudly (not silently
    skip the input) so the operator can fix the client-side chain."""
    bogus_id = "00000000-0000-0000-0000-000000000000"
    downstream = await _seed_media_job(
        db_session,
        client_id=seeded_client[0].id,
        task_type="wander_scene_video",
        inputs={"source_image_job_id": bogus_id},
    )

    reg = ProviderRegistry()
    reg.register_media(_GoodMediaProvider())  # will never get called

    out = await execute_media_job(
        job=downstream,
        media_provider_name="good",
        media_kind="video",
        workflow_template="wander_scene_video",
        providers=reg,
        output_dir="/tmp/x",
        session=db_session,
    )

    assert out.status == JobStatus.FAILED.value
    assert "source_image_job_id" in out.error
    assert bogus_id in out.error
    assert "referenced job not found" in out.error


async def test_chain_by_job_id_upstream_has_no_media_url_marks_failed(
    db_session: AsyncSession, seeded_client
) -> None:
    """An upstream that hasn't completed (no media_url) shouldn't silently
    produce empty inputs — the resolver must fail-fast and report status."""
    upstream = await _seed_media_job(
        db_session,
        client_id=seeded_client[0].id,
        task_type="wander_scene_image",
    )
    # upstream.media_url is None (pending) — resolver must reject.
    downstream = await _seed_media_job(
        db_session,
        client_id=seeded_client[0].id,
        task_type="wander_scene_video",
        inputs={"source_image_job_id": str(upstream.id)},
    )

    reg = ProviderRegistry()
    reg.register_media(_GoodMediaProvider())
    out = await execute_media_job(
        job=downstream,
        media_provider_name="good",
        media_kind="video",
        workflow_template="wander_scene_video",
        providers=reg,
        output_dir="/tmp/x",
        session=db_session,
    )

    assert out.status == JobStatus.FAILED.value
    assert "no media_url" in out.error
    assert f"status={JobStatus.PENDING.value}" in out.error


async def test_chain_by_url_still_works_but_logs_deprecation(
    db_session: AsyncSession, seeded_client, caplog
) -> None:
    """One-release grace period: the legacy `source_*_url` form keeps
    working but emits a warning so callers migrate before the next bump."""
    import logging

    downstream = await _seed_media_job(
        db_session,
        client_id=seeded_client[0].id,
        task_type="wander_scene_video",
        inputs={"source_image_url": "/app/output/some-existing-file.png"},
    )

    captured = {}

    class _Capturing(BaseMediaProvider):
        name = "capture"

        async def produce(self, **kwargs):
            captured.update(kwargs)
            return MediaResponse(
                file_path="/tmp/x", url_path=f"/output/{downstream.id}.mp4",
                mime_type="video/mp4", width=None, height=None, duration_s=None,
                latency_ms=0, cost_usd=Decimal("0"), model_used="x",
                provider=self.name, extra={},
            )

    reg = ProviderRegistry()
    reg.register_media(_Capturing())
    with caplog.at_level(logging.WARNING, logger="worker.executor"):
        out = await execute_media_job(
            job=downstream,
            media_provider_name="capture",
            media_kind="video",
            workflow_template="wander_scene_video",
            providers=reg,
            output_dir="/tmp/x",
            session=db_session,
        )

    assert out.status == JobStatus.COMPLETE.value
    # URL passed through untouched
    assert captured["inputs"]["source_image_url"] == "/app/output/some-existing-file.png"
    assert any(
        "deprecated URL form" in rec.message and "source_image_url" in rec.message
        for rec in caplog.records
    )
