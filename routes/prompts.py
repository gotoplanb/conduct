"""Admin CRUD for the prompts table + version history endpoint.

The CLI (`conduct prompts edit/list/get/history`) is the primary caller —
the routes intentionally mirror what `git` users expect (list, show, edit,
log). Each PUT both upserts the live row AND appends a row to
prompt_versions so jobs that ran before the edit can still be traced to
their exact content.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from config.settings import get_settings
from db.session import get_session
from models.client import ClientApp
from models.prompt import Prompt, PromptVersion

router = APIRouter(prefix="/prompts", tags=["prompts"], dependencies=[Depends(admin_only)])

_bearer = HTTPBearer(auto_error=False)


def _editor_fingerprint(
    credentials: HTTPAuthorizationCredentials | None, cookie_value: str | None
) -> str:
    """Short SHA prefix of whichever admin credential authenticated this
    request. Stored in updated_by / edited_by so an audit query can group
    edits by source without storing the key itself."""
    key = (credentials.credentials if credentials else None) or cookie_value or ""
    if not key:
        return ""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# --- response/request shapes ----


class PromptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    task_type: str
    client_name: str | None
    content: str
    updated_at: datetime
    updated_by: str


class PromptListItem(BaseModel):
    task_type: str
    client_name: str | None
    updated_at: datetime
    bytes: int


class PromptListOut(BaseModel):
    prompts: list[PromptListItem]


class PromptUpsertIn(BaseModel):
    content: str = Field(min_length=1)


class PromptVersionOut(BaseModel):
    id: int
    edited_at: datetime
    edited_by: str
    bytes: int


class HistoryOut(BaseModel):
    task_type: str
    client_name: str | None
    versions: list[PromptVersionOut]


# --- helpers ----


async def _resolve_client_id(session: AsyncSession, client_name: str | None):
    if not client_name:
        return None
    c = await session.scalar(select(ClientApp).where(ClientApp.name == client_name))
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown client {client_name!r}")
    return c.id


async def _client_id_to_name(session: AsyncSession, client_id) -> str | None:
    if client_id is None:
        return None
    c = await session.get(ClientApp, client_id)
    return c.name if c else None


# --- routes ----


@router.get("")
async def list_prompts(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> PromptListOut:
    rows = (await session.scalars(
        select(Prompt).order_by(Prompt.task_type, Prompt.client_id)
    )).all()
    out: list[PromptListItem] = []
    for r in rows:
        name = await _client_id_to_name(session, r.client_id)
        out.append(
            PromptListItem(
                task_type=r.task_type,
                client_name=name,
                updated_at=r.updated_at,
                bytes=len(r.content.encode("utf-8")),
            )
        )
    return PromptListOut(prompts=out)


@router.get("/{task_type}")
async def get_prompt(
    task_type: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client: Annotated[str | None, Query(alias="client")] = None,
) -> PromptOut:
    client_id = await _resolve_client_id(session, client)
    stmt = select(Prompt).where(Prompt.task_type == task_type)
    if client_id is None:
        stmt = stmt.where(Prompt.client_id.is_(None))
    else:
        stmt = stmt.where(Prompt.client_id == client_id)
    row = await session.scalar(stmt)
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no prompt for task_type={task_type!r} client={client!r}",
        )
    return PromptOut(
        task_type=row.task_type,
        client_name=client,
        content=row.content,
        updated_at=row.updated_at,
        updated_by=row.updated_by,
    )


@router.put("/{task_type}")
async def upsert_prompt(
    task_type: str,
    body: PromptUpsertIn,
    session: Annotated[AsyncSession, Depends(get_session)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
    conduct_admin: Annotated[str | None, Cookie()] = None,
    client: Annotated[str | None, Query(alias="client")] = None,
) -> PromptOut:
    if not task_type or len(task_type) > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "task_type must be 1-100 chars")
    client_id = await _resolve_client_id(session, client)
    fp = _editor_fingerprint(credentials, conduct_admin)

    stmt = select(Prompt).where(Prompt.task_type == task_type)
    if client_id is None:
        stmt = stmt.where(Prompt.client_id.is_(None))
    else:
        stmt = stmt.where(Prompt.client_id == client_id)
    row = await session.scalar(stmt)

    if row is None:
        row = Prompt(
            task_type=task_type,
            client_id=client_id,
            content=body.content,
            updated_by=fp,
        )
        session.add(row)
    else:
        row.content = body.content
        row.updated_by = fp

    session.add(
        PromptVersion(
            task_type=task_type,
            client_id=client_id,
            content=body.content,
            edited_by=fp,
        )
    )
    await session.commit()
    await session.refresh(row)
    return PromptOut(
        task_type=row.task_type,
        client_name=client,
        content=row.content,
        updated_at=row.updated_at,
        updated_by=row.updated_by,
    )


@router.get("/{task_type}/history")
async def prompt_history(
    task_type: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client: Annotated[str | None, Query(alias="client")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> HistoryOut:
    client_id = await _resolve_client_id(session, client)
    stmt = select(PromptVersion).where(PromptVersion.task_type == task_type)
    if client_id is None:
        stmt = stmt.where(PromptVersion.client_id.is_(None))
    else:
        stmt = stmt.where(PromptVersion.client_id == client_id)
    rows = (await session.scalars(stmt.order_by(PromptVersion.edited_at.desc()).limit(limit))).all()
    return HistoryOut(
        task_type=task_type,
        client_name=client,
        versions=[
            PromptVersionOut(
                id=r.id,
                edited_at=r.edited_at,
                edited_by=r.edited_by,
                bytes=len(r.content.encode("utf-8")),
            )
            for r in rows
        ],
    )


# Keep the settings import used elsewhere happy; suppresses ruff F401 in case
# we reference get_settings in future audit logic.
_ = get_settings
