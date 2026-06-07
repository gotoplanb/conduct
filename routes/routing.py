from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import admin_only
from db.session import get_session
from models.routing import RoutingRule
from models.types import MediaKind, Sensitivity

router = APIRouter(prefix="/routing", tags=["routing"], dependencies=[Depends(admin_only)])


class ShadowModelSpec(BaseModel):
    model: str = Field(min_length=1, max_length=100)
    rate: float = Field(ge=0.0, le=1.0)
    daily_cost_cap_usd: float | None = Field(default=None, ge=0.0)


class RoutingRuleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_type: str
    preferred_model: str
    fallback_model: str
    sensitivity: Sensitivity
    max_tokens: int
    notes: str
    eval_shadow_models: list[ShadowModelSpec] = Field(default_factory=list)
    updated_at: datetime
    is_archived: bool = False
    media_kind: MediaKind = MediaKind.TEXT


class RoutingRuleIn(BaseModel):
    preferred_model: str = Field(min_length=1, max_length=100)
    fallback_model: str = Field(min_length=1, max_length=100)
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    max_tokens: int = Field(default=1000, ge=1, le=200_000)
    notes: str = ""
    eval_shadow_models: list[ShadowModelSpec] = Field(default_factory=list)
    media_kind: MediaKind = MediaKind.TEXT


class RoutingListOut(BaseModel):
    rules: list[RoutingRuleOut]


@router.get("")
async def list_routing(
    session: Annotated[AsyncSession, Depends(get_session)],
    include_archived: Annotated[bool, Query()] = False,
) -> RoutingListOut:
    stmt = select(RoutingRule).order_by(RoutingRule.task_type)
    if not include_archived:
        stmt = stmt.where(RoutingRule.is_archived.is_(False))
    rows = (await session.scalars(stmt)).all()
    return RoutingListOut(rules=[RoutingRuleOut.model_validate(r) for r in rows])


@router.get("/{task_type}")
async def get_routing(
    task_type: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    include_archived: Annotated[bool, Query()] = False,
) -> RoutingRuleOut:
    rule = await session.get(RoutingRule, task_type)
    if rule is None or (rule.is_archived and not include_archived):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no routing rule for {task_type!r}"
        )
    return RoutingRuleOut.model_validate(rule)


@router.put("/{task_type}")
async def upsert_routing(
    task_type: str,
    body: RoutingRuleIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RoutingRuleOut:
    if not task_type or len(task_type) > 100:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "task_type must be 1-100 chars")
    shadow_specs = [s.model_dump() for s in body.eval_shadow_models]
    rule = await session.get(RoutingRule, task_type)
    if rule is None:
        rule = RoutingRule(
            task_type=task_type,
            preferred_model=body.preferred_model,
            fallback_model=body.fallback_model,
            sensitivity=body.sensitivity.value,
            max_tokens=body.max_tokens,
            notes=body.notes,
            eval_shadow_models=shadow_specs,
            media_kind=body.media_kind.value,
        )
        session.add(rule)
    else:
        rule.preferred_model = body.preferred_model
        rule.fallback_model = body.fallback_model
        rule.sensitivity = body.sensitivity.value
        rule.max_tokens = body.max_tokens
        rule.notes = body.notes
        rule.eval_shadow_models = shadow_specs
        rule.media_kind = body.media_kind.value
        # Revive an archived rule transparently — PUT means "make this the
        # current rule", so the operator shouldn't have to know the row is
        # archived to bring it back. Soft-delete is for cleanup hygiene,
        # not a workflow speed bump.
        rule.is_archived = False
    await session.commit()
    await session.refresh(rule)
    return RoutingRuleOut.model_validate(rule)


@router.delete("/{task_type}")
async def archive_routing(
    task_type: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> RoutingRuleOut:
    """Soft-delete a routing rule. The row stays in the table (so
    JobShadow.parent_job_id references and historical /ui/jobs pages keep
    working); is_archived=true makes it invisible to GET listings + the
    routing engine. Idempotent — DELETE on an already-archived rule still
    returns 200."""
    rule = await session.get(RoutingRule, task_type)
    if rule is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"no routing rule for {task_type!r}"
        )
    rule.is_archived = True
    await session.commit()
    await session.refresh(rule)
    return RoutingRuleOut.model_validate(rule)
