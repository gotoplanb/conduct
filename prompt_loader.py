"""Prompt resolution against the Postgres-backed `prompts` table.

A job picks a prompt via two lookups:
  1. Per-client override: (task_type=T, client_id=C)
  2. Shared default:      (task_type=T, client_id IS NULL)

On every save (via `PUT /prompts/...`), a row is appended to
`prompt_versions`. Each job records the version id it resolved to, so the
trace from job → exact prompt content survives even after subsequent edits.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.client import ClientApp
from models.prompt import Prompt, PromptVersion


class PromptNotFoundError(Exception):
    """No row found for the requested (task_type, client) combination."""


@dataclass(frozen=True)
class ResolvedPrompt:
    content: str
    version_id: int | None  # PromptVersion.id of the most recent save
    source: str  # human-readable: "client:<name>:<task>" or "shared:<task>"


async def resolve_prompt(
    session: AsyncSession, task_type: str, client_name: str | None = None
) -> ResolvedPrompt:
    """Find the prompt for this task_type, preferring a per-client override
    when one exists. Raises PromptNotFoundError if neither path matches."""

    # Archived prompts are treated as nonexistent — the route's DELETE flips
    # is_archived but leaves PromptVersion history intact. Filter at both
    # lookup sites so a soft-deleted prompt can't keep silently dispatching.

    # 1. Client override (if a client was named).
    if client_name:
        client = await session.scalar(select(ClientApp).where(ClientApp.name == client_name))
        if client is not None:
            row = await session.scalar(
                select(Prompt).where(
                    Prompt.task_type == task_type,
                    Prompt.client_id == client.id,
                    Prompt.is_archived.is_(False),
                )
            )
            if row is not None:
                version_id = await _latest_version_id(
                    session, task_type=task_type, client_id=client.id
                )
                return ResolvedPrompt(
                    content=row.content,
                    version_id=version_id,
                    source=f"client:{client_name}:{task_type}",
                )

    # 2. Shared default.
    row = await session.scalar(
        select(Prompt).where(
            Prompt.task_type == task_type,
            Prompt.client_id.is_(None),
            Prompt.is_archived.is_(False),
        )
    )
    if row is None:
        raise PromptNotFoundError(
            f"no prompt found for task_type={task_type!r} (client={client_name!r})"
        )
    version_id = await _latest_version_id(session, task_type=task_type, client_id=None)
    return ResolvedPrompt(
        content=row.content,
        version_id=version_id,
        source=f"shared:{task_type}",
    )


async def _latest_version_id(
    session: AsyncSession, *, task_type: str, client_id
) -> int | None:
    """Return the most recent PromptVersion.id matching the given key. May be
    None on freshly-imported prompts that never went through a save flow."""
    stmt = select(PromptVersion.id).where(PromptVersion.task_type == task_type)
    if client_id is None:
        stmt = stmt.where(PromptVersion.client_id.is_(None))
    else:
        stmt = stmt.where(PromptVersion.client_id == client_id)
    # id is the tie-breaker: edited_at is a per-row datetime.now() that can
    # collide when versions are written in the same tick, and ORDER BY a
    # non-unique column has no defined order on ties. The monotonic PK reflects
    # true insertion order, so "highest id" is unambiguously the most recent
    # (conduct#48).
    stmt = stmt.order_by(
        PromptVersion.edited_at.desc(), PromptVersion.id.desc()
    ).limit(1)
    return await session.scalar(stmt)
