"""Scenario-facing image generation (#53) — the /tts of scene stills.

POST /image {prompt, style?} -> 202 {job_id, poll_url, expected_output_url}.
The job runs the scene_image routing rule's ComfyUI workflow on the
conduct-media queue; a registered style overrides the workflow template
and/or injection params. Output is a silent still in the shared output
dir — Wander polls, then hands the local path to Perform (same-host
handoff, Perform SPEC §4: audio and video stay separate).

GET /styles and /styles/registry mirror /voices and /voices/registry.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only, current_client_or_admin
from db.session import get_session
from media_styles import (
    UnknownStyle,
    resolve_style,
    visible_styles,
    workflow_template_installed,
)
from models.client import ClientApp
from models.job import Job
from models.style import StyleAlias
from models.types import JobStatus, Sensitivity
from observability.tracing import rq_trace_meta
from rate_limit import rate_limited_client
from worker.runner import run_job

log = logging.getLogger(__name__)

# The routing rule every /image job runs under. It must exist with
# media_kind=image; its preferred_model is the default workflow template
# used when no style is given.
IMAGE_TASK_TYPE = "scene_image"

image_router = APIRouter(prefix="/image", tags=["image"])
styles_router = APIRouter(prefix="/styles", tags=["image"])
styles_admin_router = APIRouter(
    prefix="/styles/registry", tags=["image"], dependencies=[Depends(admin_only)]
)


class ImageCreateIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    style: str | None = Field(default=None, max_length=100)


@image_router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_image(
    body: ImageCreateIn,
    client: Annotated[ClientApp, Depends(rate_limited_client)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JSONResponse:
    from worker.queue import (  # noqa: PLC0415
        DEFAULT_MEDIA_JOB_TIMEOUT_S,
        get_media_queue,
    )

    try:
        style = await resolve_style(
            session, requested=body.style, client_id=client.id
        )
    except UnknownStyle as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown style {e.requested!r}; known styles: {', '.join(e.known)}",
        ) from e

    metadata: dict = {}
    if style is not None:
        # The worker's media branch honors this over the rule's defaults —
        # the resolved values are stamped here so the job traces to the
        # exact workflow + params it ran, even if the alias is edited later.
        metadata["style"] = style.name
        metadata["style_resolved"] = {
            "workflow_template": style.workflow_template,
            "params": style.params or {},
        }

    job = Job(
        client_app_id=client.id,
        task_type=IMAGE_TASK_TYPE,
        sensitivity=Sensitivity.PUBLIC.value,  # local ComfyUI, no external send
        priority=5,
        prompt=body.prompt,
        system_prompt="",
        model_requested=style.workflow_template if style else None,
        status=JobStatus.PENDING.value,
        job_metadata=metadata,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    try:
        get_media_queue().enqueue(
            run_job,
            str(job.id),
            job_id=str(job.id),
            job_timeout=DEFAULT_MEDIA_JOB_TIMEOUT_S,
            meta=rq_trace_meta(),
        )
    except Exception as e:
        log.exception("failed to enqueue image job %s", job.id)
        job.status = JobStatus.FAILED.value
        job.error = f"enqueue failed: {e}"
        await session.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "queue backend unavailable"
        ) from e

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": str(job.id),
            "status": JobStatus.PENDING.value,
            "poll_url": f"/jobs/{job.id}",
            "expected_output_url": f"/output/{job.id}.png",
            "style": style.name if style else None,
        },
    )


class StyleOut(BaseModel):
    name: str
    scope: str  # 'shared' | 'client'
    installed: bool  # workflow template JSON present on disk


class StyleListOut(BaseModel):
    styles: list[StyleOut]


@styles_router.get("")
async def list_styles(
    principal: Annotated[ClientApp | None, Depends(current_client_or_admin)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StyleListOut:
    client_id = principal.id if principal is not None else None
    rows = await visible_styles(session, client_id)
    return StyleListOut(
        styles=[
            StyleOut(
                name=r.name,
                scope="client" if r.client_id is not None else "shared",
                installed=workflow_template_installed(r.workflow_template),
            )
            for r in rows
        ]
    )


class StyleAliasOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    client_id: UUID | None
    workflow_template: str
    params: dict
    notes: str
    updated_at: datetime
    is_archived: bool


class StyleAliasIn(BaseModel):
    workflow_template: str = Field(min_length=1, max_length=200)
    params: dict = Field(default_factory=dict)
    notes: str = ""
    client_id: UUID | None = None


class StyleRegistryOut(BaseModel):
    aliases: list[StyleAliasOut]


@styles_admin_router.get("")
async def list_registry(
    session: Annotated[AsyncSession, Depends(get_session)],
    include_archived: Annotated[bool, Query()] = False,
) -> StyleRegistryOut:
    stmt = select(StyleAlias).order_by(StyleAlias.name, StyleAlias.client_id)
    if not include_archived:
        stmt = stmt.where(StyleAlias.is_archived.is_(False))
    rows = (await session.scalars(stmt)).all()
    return StyleRegistryOut(aliases=[StyleAliasOut.model_validate(r) for r in rows])


async def _get_alias(
    session: AsyncSession, name: str, client_id: UUID | None
) -> StyleAlias | None:
    stmt = select(StyleAlias).where(StyleAlias.name == name)
    stmt = (
        stmt.where(StyleAlias.client_id.is_(None))
        if client_id is None
        else stmt.where(StyleAlias.client_id == client_id)
    )
    return await session.scalar(stmt)


@styles_admin_router.put("/{name}")
async def upsert_alias(
    name: str,
    body: StyleAliasIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StyleAliasOut:
    if not name or len(name) > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name must be 1-100 chars")
    if not workflow_template_installed(body.workflow_template):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"workflow template {body.workflow_template!r} not found in comfy_workflows/",
        )
    alias = await _get_alias(session, name, body.client_id)
    if alias is None:
        alias = StyleAlias(
            name=name,
            client_id=body.client_id,
            workflow_template=body.workflow_template,
            params=body.params,
            notes=body.notes,
        )
        session.add(alias)
    else:
        alias.workflow_template = body.workflow_template
        alias.params = body.params
        alias.notes = body.notes
        alias.is_archived = False  # PUT revives, same contract as /routing
    await session.commit()
    await session.refresh(alias)
    return StyleAliasOut.model_validate(alias)


@styles_admin_router.delete("/{name}")
async def archive_alias(
    name: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client_id: Annotated[UUID | None, Query()] = None,
) -> StyleAliasOut:
    alias = await _get_alias(session, name, client_id)
    if alias is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no style alias {name!r}")
    alias.is_archived = True
    await session.commit()
    await session.refresh(alias)
    return StyleAliasOut.model_validate(alias)
