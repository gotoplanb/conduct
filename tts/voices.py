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

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.client import ClientApp
from models.types import Sensitivity
from models.voice import VoiceAlias
from tts.elevenlabs_engine import is_valid_voice_id

PIPER = "piper"
ELEVENLABS = "elevenlabs"
ENGINES = frozenset({PIPER, ELEVENLABS})
# Engines that send the text off-box. Same egress gate as cloud LLMs.
CLOUD_ENGINES = frozenset({ELEVENLABS})


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


@dataclass(frozen=True)
class ResolvedVoice:
    voice: str  # piper: file stem; elevenlabs: voice id
    alias_name: str | None  # None for the default and literal passthrough
    engine: str


async def resolve_voice_full(
    session: AsyncSession,
    *,
    requested: str | None,
    client_id: UUID | None,
    default_voice: str,
    voices_dir: Path,
) -> ResolvedVoice:
    """Resolve a submitted voice to its concrete voice + engine.

    The default and literal passthrough are always piper — a cloud engine is
    only ever reached through an explicitly registered alias (#52 non-goal:
    ElevenLabs is never a default).
    """
    if requested is None or requested == "":
        return ResolvedVoice(default_voice, None, PIPER)

    aliases = {a.name: a for a in await visible_voices(session, client_id)}
    alias = aliases.get(requested)
    if alias is not None:
        return ResolvedVoice(alias.voice_file, alias.name, alias.engine)

    installed = _installed_voice_files(voices_dir)
    if requested in installed:
        return ResolvedVoice(requested, None, PIPER)

    raise UnknownVoice(requested, sorted(set(aliases) | set(installed)))


async def resolve_voice(
    session: AsyncSession,
    *,
    requested: str | None,
    client_id: UUID | None,
    default_voice: str,
    voices_dir: Path,
) -> tuple[str, str | None]:
    """Resolve a submitted voice to (concrete_voice, logical_name|None).

    Engine-less shorthand for `resolve_voice_full`.
    """
    r = await resolve_voice_full(
        session,
        requested=requested,
        client_id=client_id,
        default_voice=default_voice,
        voices_dir=voices_dir,
    )
    return r.voice, r.alias_name


def cloud_tts_block_reason(
    engine: str, sensitivity: Sensitivity, client: ClientApp
) -> str | None:
    """Why a TTS job may not use this engine, or None if it may (#52).

    Mirrors the cloud-LLM gate in routing/engine.py: public is fine,
    confidential never leaves the box, internal needs the client's
    allow_cloud_for_internal opt-in. A cloud engine also needs the client's
    own key — there is no global fallback, by design.
    """
    if engine not in CLOUD_ENGINES:
        return None
    if sensitivity == Sensitivity.CONFIDENTIAL:
        return f"{engine} voices send text off-box; not allowed for sensitivity=confidential"
    if sensitivity == Sensitivity.INTERNAL and not client.allow_cloud_for_internal:
        return (
            f"{engine} voices send text off-box; sensitivity=internal requires "
            "allow_cloud_for_internal on this client (or submit sensitivity=public)"
        )
    if engine == ELEVENLABS and client.elevenlabs_api_key_encrypted is None:
        return "no ElevenLabs API key configured for this client"
    return None


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


def malformed_elevenlabs_ids(aliases: list[VoiceAlias]) -> list[tuple[str, str]]:
    """(name, voice_file) pairs for elevenlabs aliases whose voice id is not
    well-formed — the cloud-engine counterpart of `missing_voice_files`."""
    return [
        (a.name, a.voice_file)
        for a in aliases
        if a.engine == ELEVENLABS and not is_valid_voice_id(a.voice_file)
    ]
