"""Named-voice resolution for TTS (#51, decided in #50).

Clients submit `voice` as a logical name ("ops-manager"). Resolution order:

1. Client-scoped VoiceAlias override (client_id matches, not archived)
2. Shared VoiceAlias (client_id IS NULL, not archived)
3. Literal passthrough: the name matches an installed Piper voice file —
   kept so pre-registry callers (the audiobook pipeline) don't break
4. UnknownVoice — the route turns this into a 400 listing known names,
   so a typo fails loudly at submit rather than silently at synthesis
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.voice import VoiceAlias


class UnknownVoice(Exception):
    """Requested voice is neither a registered name nor an installed file."""

    def __init__(self, requested: str, known: list[str]):
        self.requested = requested
        self.known = known
        super().__init__(f"unknown voice {requested!r}; known: {', '.join(known) or '(none)'}")


def _installed_voice_files(voices_dir: Path) -> list[str]:
    """Stems of Piper voices present on disk (en_US-amy-medium.onnx -> stem)."""
    if not voices_dir.is_dir():
        return []
    return sorted(
        p.stem.removesuffix(".onnx") if p.stem.endswith(".onnx") else p.stem
        for p in voices_dir.glob("*.onnx")
    )


async def _live_aliases(
    session: AsyncSession, client_id: UUID | None
) -> list[VoiceAlias]:
    """All non-archived aliases visible to a client: its overrides + shared."""
    stmt = select(VoiceAlias).where(VoiceAlias.is_archived.is_(False))
    if client_id is None:
        stmt = stmt.where(VoiceAlias.client_id.is_(None))
    else:
        stmt = stmt.where(
            (VoiceAlias.client_id == client_id) | (VoiceAlias.client_id.is_(None))
        )
    return list((await session.scalars(stmt)).all())


async def all_live_aliases(session: AsyncSession) -> list[VoiceAlias]:
    """Every non-archived alias regardless of scope — startup validation
    must cover client overrides too, not just the shared set."""
    stmt = select(VoiceAlias).where(VoiceAlias.is_archived.is_(False))
    return list((await session.scalars(stmt)).all())


async def visible_voices(
    session: AsyncSession, client_id: UUID | None
) -> list[VoiceAlias]:
    """Merged discovery view: client overrides shadow same-named shared rows."""
    rows = await _live_aliases(session, client_id)
    merged: dict[str, VoiceAlias] = {}
    for row in sorted(rows, key=lambda r: r.client_id is not None):
        merged[row.name] = row  # client-scoped sorts last, so it wins
    return sorted(merged.values(), key=lambda r: r.name)


async def resolve_voice(
    session: AsyncSession,
    *,
    requested: str | None,
    client_id: UUID | None,
    default_voice: str,
    voices_dir: Path,
) -> tuple[str, str | None]:
    """Resolve a submitted voice to (concrete_voice_file, logical_name|None).

    logical_name is None for the default and for literal passthrough — the
    caller records it in job metadata only when a registry name was used.
    """
    if requested is None or requested == "":
        return default_voice, None

    aliases = {a.name: a for a in await visible_voices(session, client_id)}
    alias = aliases.get(requested)
    if alias is not None:
        return alias.voice_file, alias.name

    installed = _installed_voice_files(voices_dir)
    if requested in installed:
        return requested, None

    raise UnknownVoice(requested, sorted(set(aliases) | set(installed)))


def missing_voice_files(
    aliases: list[VoiceAlias], voices_dir: Path
) -> list[tuple[str, str]]:
    """(name, voice_file) pairs whose Piper files are absent from disk.

    Startup validation (#51): a registered voice with no backing file is a
    hard configuration error and must be loud — a silent fallback would make
    two participants sound identical with no explanation (Perform SPEC §8).
    Only the piper engine reads local files; other engines validate elsewhere.
    """
    missing = []
    for a in aliases:
        if a.engine != "piper":
            continue
        if not (voices_dir / f"{a.voice_file}.onnx").is_file():
            missing.append((a.name, a.voice_file))
    return missing
