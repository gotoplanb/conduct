"""Named-voice registry (#51): resolution, discovery, admin CRUD, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.voice import VoiceAlias
from tts.voices import (
    UnknownVoice,
    all_live_aliases,
    missing_voice_files,
    resolve_voice,
    visible_voices,
)


def _fake_voices_dir(tmp_path: Path, *stems: str) -> Path:
    for stem in stems:
        (tmp_path / f"{stem}.onnx").write_bytes(b"onnx")
        (tmp_path / f"{stem}.onnx.json").write_text("{}")
    return tmp_path


async def _add_alias(db_session, name, voice_file, client_id=None, archived=False):
    alias = VoiceAlias(
        name=name, client_id=client_id, voice_file=voice_file, is_archived=archived
    )
    db_session.add(alias)
    await db_session.commit()
    return alias


async def test_resolve_default_when_unset(db_session, tmp_path) -> None:
    voice, alias = await resolve_voice(
        db_session,
        requested=None,
        client_id=None,
        default_voice="en_US-amy-medium",
        voices_dir=tmp_path,
    )
    assert (voice, alias) == ("en_US-amy-medium", None)


async def test_resolve_shared_alias(db_session, seeded_client, tmp_path) -> None:
    capp, _ = seeded_client
    await _add_alias(db_session, "t-narrator", "en_US-amy-medium")
    voice, alias = await resolve_voice(
        db_session,
        requested="t-narrator",
        client_id=capp.id,
        default_voice="x",
        voices_dir=tmp_path,
    )
    assert (voice, alias) == ("en_US-amy-medium", "t-narrator")


async def test_resolve_client_override_beats_shared(
    db_session, seeded_client, tmp_path
) -> None:
    capp, _ = seeded_client
    await _add_alias(db_session, "t-narrator", "shared-voice")
    await _add_alias(db_session, "t-narrator", "client-voice", client_id=capp.id)
    voice, _ = await resolve_voice(
        db_session,
        requested="t-narrator",
        client_id=capp.id,
        default_voice="x",
        voices_dir=tmp_path,
    )
    assert voice == "client-voice"
    # A different client (None here) still gets the shared mapping.
    voice, _ = await resolve_voice(
        db_session,
        requested="t-narrator",
        client_id=None,
        default_voice="x",
        voices_dir=tmp_path,
    )
    assert voice == "shared-voice"


async def test_resolve_literal_installed_passthrough(db_session, tmp_path) -> None:
    voices_dir = _fake_voices_dir(tmp_path, "en_US-amy-medium")
    voice, alias = await resolve_voice(
        db_session,
        requested="en_US-amy-medium",
        client_id=None,
        default_voice="x",
        voices_dir=voices_dir,
    )
    assert (voice, alias) == ("en_US-amy-medium", None)


async def test_resolve_unknown_raises_with_known_names(
    db_session, seeded_client, tmp_path
) -> None:
    capp, _ = seeded_client
    await _add_alias(db_session, "t-narrator", "en_US-amy-medium")
    voices_dir = _fake_voices_dir(tmp_path, "en_GB-alan-medium")
    with pytest.raises(UnknownVoice) as exc:
        await resolve_voice(
            db_session,
            requested="oops",
            client_id=capp.id,
            default_voice="x",
            voices_dir=voices_dir,
        )
    assert exc.value.requested == "oops"
    assert "t-narrator" in exc.value.known
    assert "en_GB-alan-medium" in exc.value.known


async def test_archived_alias_is_invisible(db_session, tmp_path) -> None:
    await _add_alias(db_session, "gone", "en_US-amy-medium", archived=True)
    with pytest.raises(UnknownVoice):
        await resolve_voice(
            db_session,
            requested="gone",
            client_id=None,
            default_voice="x",
            voices_dir=tmp_path,
        )
    names = [a.name for a in await visible_voices(db_session, None)]
    assert "gone" not in names
    assert "gone" not in [a.name for a in await all_live_aliases(db_session)]


def test_missing_voice_files_flags_piper_only(tmp_path) -> None:
    voices_dir = _fake_voices_dir(tmp_path, "present")
    aliases = [
        VoiceAlias(name="ok", voice_file="present", engine="piper"),
        VoiceAlias(name="broken", voice_file="absent", engine="piper"),
        VoiceAlias(name="cloud", voice_file="whatever", engine="elevenlabs"),
    ]
    assert missing_voice_files(aliases, voices_dir) == [("broken", "absent")]


# --- route-level tests -------------------------------------------------------


async def test_tts_submit_unknown_voice_400(client, seeded_client, fake_redis) -> None:
    _, key = seeded_client
    resp = await client.post(
        "/tts",
        json={"text": "hello", "voice": "not-a-voice"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 400
    assert "known voices" in resp.json()["detail"]


async def test_tts_submit_resolves_alias(
    client, db_session, seeded_client, fake_redis
) -> None:
    capp, key = seeded_client
    await _add_alias(db_session, "t-narrator", "en_US-amy-medium")
    resp = await client.post(
        "/tts",
        json={"text": "hello", "voice": "t-narrator"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert resp.status_code == 202
    assert resp.json()["voice"] == "en_US-amy-medium"


async def test_voices_discovery_merged_view(
    client, db_session, seeded_client
) -> None:
    capp, key = seeded_client
    await _add_alias(db_session, "t-narrator", "shared-voice")
    await _add_alias(db_session, "t-narrator", "client-voice", client_id=capp.id)
    await _add_alias(db_session, "t-ops-manager", "en_GB-alan-medium")
    resp = await client.get("/voices", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    voices = {v["name"]: v for v in resp.json()["voices"]}
    assert voices["t-narrator"]["scope"] == "client"
    assert voices["t-ops-manager"]["scope"] == "shared"


async def test_registry_put_rejects_uninstalled_piper_file(
    client, admin_headers
) -> None:
    resp = await client.put(
        "/voices/registry/ghost",
        json={"voice_file": "definitely-not-installed"},
        headers=admin_headers,
    )
    assert resp.status_code == 400
    assert "not installed" in resp.json()["detail"]


async def test_registry_put_get_archive_cycle(
    client, db_session, admin_headers, monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    import routes.voices as voices_route

    voices_dir = _fake_voices_dir(tmp_path, "en_US-amy-medium")
    real = voices_route.get_settings()
    monkeypatch.setattr(
        voices_route,
        "get_settings",
        lambda: SimpleNamespace(**{**real.__dict__, "tts_voices_dir": str(voices_dir)}),
    )

    put = await client.put(
        "/voices/registry/t-narrator",
        json={"voice_file": "en_US-amy-medium", "notes": "seeded by test"},
        headers=admin_headers,
    )
    assert put.status_code == 200
    assert put.json()["is_archived"] is False

    listing = await client.get("/voices/registry", headers=admin_headers)
    assert "t-narrator" in [a["name"] for a in listing.json()["aliases"]]

    dele = await client.delete("/voices/registry/t-narrator", headers=admin_headers)
    assert dele.status_code == 200
    assert dele.json()["is_archived"] is True

    # PUT revives, same contract as /routing.
    put2 = await client.put(
        "/voices/registry/t-narrator",
        json={"voice_file": "en_US-amy-medium"},
        headers=admin_headers,
    )
    assert put2.json()["is_archived"] is False
