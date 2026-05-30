"""Admin JSON API for managing MCP-connector OAuth clients.

Parallels the /ui/connectors UI flow but returns JSON so the `conduct
connectors` CLI (and any other operator tool) can drive the same operations
without going through HTML.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from models.client import ClientApp
from models.oauth import OAuthClient
from oauth_provider import hash_secret, new_client_id, new_client_secret

router = APIRouter(prefix="/connectors", tags=["connectors"], dependencies=[Depends(admin_only)])

DEFAULT_REDIRECT = "https://claude.ai/api/mcp/auth_callback"
_CONNECTOR_NOT_FOUND = "connector not found"


class ConnectorCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    client_app_id: UUID
    redirect_uris: list[str] = Field(default_factory=list)


class ConnectorCreateOut(BaseModel):
    id: UUID
    name: str
    client_id: str
    client_secret: str
    client_app_id: UUID
    redirect_uris: list[str]
    is_active: bool
    created_at: datetime


class ConnectorOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    client_id: str
    client_app_id: UUID
    redirect_uris: list[str]
    is_active: bool
    created_at: datetime


class ConnectorPatchIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    is_active: bool | None = None
    redirect_uris: list[str] | None = None


class RotateSecretOut(BaseModel):
    id: UUID
    name: str
    client_id: str
    client_secret: str


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_connector(
    body: ConnectorCreateIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ConnectorCreateOut:
    app = await session.get(ClientApp, body.client_app_id)
    if app is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "client_app not found")
    raw_secret = new_client_secret()
    cid = new_client_id()
    redirects = body.redirect_uris or [DEFAULT_REDIRECT]
    row = OAuthClient(
        client_id=cid,
        client_secret_hash=hash_secret(raw_secret),
        name=body.name,
        client_app_id=app.id,
        redirect_uris=redirects,
        created_by="api",
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "connector conflict") from e
    await session.refresh(row)
    return ConnectorCreateOut(
        id=row.id,
        name=row.name,
        client_id=row.client_id,
        client_secret=raw_secret,
        client_app_id=row.client_app_id,
        redirect_uris=row.redirect_uris,
        is_active=row.is_active,
        created_at=row.created_at,
    )


@router.get("", response_model=list[ConnectorOut])
async def list_connectors(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> list[OAuthClient]:
    rows = (
        await session.scalars(select(OAuthClient).order_by(OAuthClient.created_at))
    ).all()
    return list(rows)


@router.patch("/{connector_id}", response_model=ConnectorOut)
async def patch_connector(
    connector_id: UUID,
    body: ConnectorPatchIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> OAuthClient:
    row = await session.get(OAuthClient, connector_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CONNECTOR_NOT_FOUND)
    updates = body.model_dump(exclude_unset=True)
    for k, v in updates.items():
        setattr(row, k, v)
    await session.commit()
    await session.refresh(row)
    return row


@router.post("/{connector_id}/rotate-secret")
async def rotate_connector_secret(
    connector_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RotateSecretOut:
    row = await session.get(OAuthClient, connector_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, _CONNECTOR_NOT_FOUND)
    raw_secret = new_client_secret()
    row.client_secret_hash = hash_secret(raw_secret)
    await session.commit()
    await session.refresh(row)
    return RotateSecretOut(
        id=row.id, name=row.name, client_id=row.client_id, client_secret=raw_secret
    )
