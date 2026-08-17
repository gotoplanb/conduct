"""Logical style resolution for scene images (#53).

The image sibling of tts/voices.py. Clients submit /image with a logical
style name; resolution order is client override > shared > UnknownStyle
(the route turns that into a 400 listing known names). Unlike voices there
is no literal passthrough — workflow template names are an implementation
detail Wander must never carry (#50's boundary), so only registered names
resolve. A missing style falls back to the scene_image routing rule's
default template.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.style import StyleAlias

WORKFLOWS_DIR = Path(__file__).resolve().parent / "comfy_workflows"


class UnknownStyle(Exception):
    """Requested style name isn't registered for this client."""

    def __init__(self, requested: str, known: list[str]):
        self.requested = requested
        self.known = known
        super().__init__(f"unknown style {requested!r}; known: {', '.join(known) or '(none)'}")


async def all_live_styles(session: AsyncSession) -> list[StyleAlias]:
    stmt = select(StyleAlias).where(StyleAlias.is_archived.is_(False))
    return list((await session.scalars(stmt)).all())


async def visible_styles(
    session: AsyncSession, client_id: UUID | None
) -> list[StyleAlias]:
    """Merged discovery view: client overrides shadow same-named shared rows."""
    stmt = select(StyleAlias).where(StyleAlias.is_archived.is_(False))
    if client_id is None:
        stmt = stmt.where(StyleAlias.client_id.is_(None))
    else:
        stmt = stmt.where(
            (StyleAlias.client_id == client_id) | (StyleAlias.client_id.is_(None))
        )
    rows = await session.scalars(stmt)
    merged: dict[str, StyleAlias] = {}
    for row in sorted(rows, key=lambda r: r.client_id is not None):
        merged[row.name] = row  # client-scoped sorts last, so it wins
    return sorted(merged.values(), key=lambda r: r.name)


async def resolve_style(
    session: AsyncSession,
    *,
    requested: str | None,
    client_id: UUID | None,
) -> StyleAlias | None:
    """Resolve a submitted style name. None means "no style" — the job runs
    on the scene_image rule's default template with default params."""
    if requested is None or requested == "":
        return None
    styles = {s.name: s for s in await visible_styles(session, client_id)}
    style = styles.get(requested)
    if style is None:
        raise UnknownStyle(requested, sorted(styles))
    return style


def workflow_template_installed(template: str) -> bool:
    return (WORKFLOWS_DIR / f"{template}.json").is_file()
