"""Named-voice registry routes (#51).

Two surfaces, mirroring how tts.py splits routers by audience:

- GET /voices — client discovery. Any authenticated client (or admin) sees
  the merged view it can actually use: shared aliases with its own overrides
  applied. Wander builds cast pickers from this instead of hardcoding names.
- /voices/registry — admin CRUD over the raw rows, same shape as /routing:
  PUT upserts (and revives archived rows), DELETE archives.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only, current_client_or_admin
from config.settings import get_settings
from db.session import get_session
from models.client import ClientApp
from models.voice import VoiceAlias
from tts.voices import visible_voices

router = APIRouter(prefix="/voices", tags=["voices"])
admin_router = APIRouter(
    prefix="/voices/registry", tags=["voices"], dependencies=[Depends(admin_only)]
)


class VoiceOut(BaseModel):
    name: str
    engine: str
    scope: str  # 'shared' | 'client'
    installed: bool  # piper only: backing file present on disk


class VoiceListOut(BaseModel):
    voices: list[VoiceOut]


class VoiceAliasOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    client_id: UUID | None
    engine: str
    voice_file: str
    notes: str
    updated_at: datetime
    is_archived: bool


class VoiceAliasIn(BaseModel):
    voice_file: str = Field(min_length=1, max_length=200)
    engine: str = Field(default="piper", max_length=20)
    notes: str = ""
    client_id: UUID | None = None


class VoiceRegistryOut(BaseModel):
    aliases: list[VoiceAliasOut]


@router.get("")
async def list_voices(
    principal: Annotated[ClientApp | None, Depends(current_client_or_admin)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> VoiceListOut:
    """The voices the caller can use, by logical name. Admin (principal None)
    sees the shared set."""
    client_id = principal.id if principal is not None else None
    voices_dir = Path(get_settings().tts_voices_dir)
    rows = await visible_voices(session, client_id)
    return VoiceListOut(
        voices=[
            VoiceOut(
                name=r.name,
                engine=r.engine,
                scope="client" if r.client_id is not None else "shared",
                installed=r.engine != "piper"
                or (voices_dir / f"{r.voice_file}.onnx").is_file(),
            )
            for r in rows
        ]
    )


@admin_router.get("")
async def list_registry(
    session: Annotated[AsyncSession, Depends(get_session)],
    include_archived: Annotated[bool, Query()] = False,
) -> VoiceRegistryOut:
    stmt = select(VoiceAlias).order_by(VoiceAlias.name, VoiceAlias.client_id)
    if not include_archived:
        stmt = stmt.where(VoiceAlias.is_archived.is_(False))
    rows = (await session.scalars(stmt)).all()
    return VoiceRegistryOut(aliases=[VoiceAliasOut.model_validate(r) for r in rows])


async def _get_alias(
    session: AsyncSession, name: str, client_id: UUID | None
) -> VoiceAlias | None:
    stmt = select(VoiceAlias).where(VoiceAlias.name == name)
    stmt = (
        stmt.where(VoiceAlias.client_id.is_(None))
        if client_id is None
        else stmt.where(VoiceAlias.client_id == client_id)
    )
    return await session.scalar(stmt)


@admin_router.put("/{name}")
async def upsert_alias(
    name: str,
    body: VoiceAliasIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> VoiceAliasOut:
    if not name or len(name) > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "name must be 1-100 chars")
    # Piper voices must exist on disk at registration — failing loudly here is
    # the whole point of the registry (Perform SPEC §8: no silent fallback).
    if body.engine == "piper":
        voices_dir = Path(get_settings().tts_voices_dir)
        if not (voices_dir / f"{body.voice_file}.onnx").is_file():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"voice file {body.voice_file!r} not installed in {voices_dir}",
            )
    alias = await _get_alias(session, name, body.client_id)
    if alias is None:
        alias = VoiceAlias(
            name=name,
            client_id=body.client_id,
            engine=body.engine,
            voice_file=body.voice_file,
            notes=body.notes,
        )
        session.add(alias)
    else:
        alias.engine = body.engine
        alias.voice_file = body.voice_file
        alias.notes = body.notes
        alias.is_archived = False  # PUT means "make this current", as /routing does
    await session.commit()
    await session.refresh(alias)
    return VoiceAliasOut.model_validate(alias)


@admin_router.delete("/{name}")
async def archive_alias(
    name: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client_id: Annotated[UUID | None, Query()] = None,
) -> VoiceAliasOut:
    """Soft-delete. Idempotent, same contract as /routing DELETE."""
    alias = await _get_alias(session, name, client_id)
    if alias is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no voice alias {name!r}")
    alias.is_archived = True
    await session.commit()
    await session.refresh(alias)
    return VoiceAliasOut.model_validate(alias)
