"""ElevenLabs engine end-to-end (#52): submit gate, registry, keys, executor."""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import select

from config.settings import get_settings
from models.client import ClientAppUsage
from models.job import Job
from models.types import JobStatus
from models.voice import VoiceAlias
from secrets_box import encrypt
from tts.executor import execute_tts

_VOICE = "JBFqnCBsd6RMkjVDRZzb"
_TTS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{_VOICE}"


async def _add_el_alias(db_session, name="t-el-narrator", client_id=None) -> None:
    db_session.add(
        VoiceAlias(name=name, client_id=client_id, engine="elevenlabs", voice_file=_VOICE)
    )
    await db_session.commit()


async def _give_key(db_session, capp, *, cloud_internal=False) -> None:
    capp.elevenlabs_api_key_encrypted = encrypt("xi-test-key")
    capp.allow_cloud_for_internal = cloud_internal
    await db_session.commit()


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --- submit gate ---------------------------------------------------------------


async def test_public_submit_with_key_is_accepted(
    client, db_session, seeded_client, fake_redis, secrets_key
) -> None:
    capp, key = seeded_client
    await _add_el_alias(db_session)
    await _give_key(db_session, capp)
    r = await client.post(
        "/tts",
        json={"text": "hello", "voice": "t-el-narrator", "sensitivity": "public"},
        headers=_auth(key),
    )
    assert r.status_code == 202, r.text
    assert r.json()["engine"] == "elevenlabs"
    job = await db_session.get(Job, r.json()["job_id"])
    assert job.sensitivity == "public"
    assert job.model_requested == _VOICE
    assert job.job_metadata == {"tts_engine": "elevenlabs", "voice_alias": "t-el-narrator"}


@pytest.mark.parametrize(
    ("sensitivity", "cloud_internal", "fragment"),
    [
        ("confidential", True, "confidential"),
        ("internal", False, "allow_cloud_for_internal"),
    ],
)
async def test_submit_blocked_by_sensitivity(
    client, db_session, seeded_client, fake_redis, secrets_key,
    sensitivity, cloud_internal, fragment,
) -> None:
    capp, key = seeded_client
    await _add_el_alias(db_session)
    await _give_key(db_session, capp, cloud_internal=cloud_internal)
    body = {"text": "hello", "voice": "t-el-narrator", "sensitivity": sensitivity}
    r = await client.post("/tts", json=body, headers=_auth(key))
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


async def test_internal_default_allowed_when_client_opted_in(
    client, db_session, seeded_client, fake_redis, secrets_key
) -> None:
    capp, key = seeded_client
    await _add_el_alias(db_session)
    await _give_key(db_session, capp, cloud_internal=True)
    r = await client.post(
        "/tts", json={"text": "hello", "voice": "t-el-narrator"}, headers=_auth(key)
    )
    assert r.status_code == 202


async def test_submit_without_key_is_refused(
    client, db_session, seeded_client, fake_redis
) -> None:
    _, key = seeded_client
    await _add_el_alias(db_session)
    r = await client.post(
        "/tts",
        json={"text": "hello", "voice": "t-el-narrator", "sensitivity": "public"},
        headers=_auth(key),
    )
    assert r.status_code == 400
    assert "no ElevenLabs API key" in r.json()["detail"]


async def test_piper_default_unaffected_by_confidential(
    client, seeded_client, fake_redis
) -> None:
    _, key = seeded_client
    r = await client.post(
        "/tts", json={"text": "hello", "sensitivity": "confidential"}, headers=_auth(key)
    )
    assert r.status_code == 202
    assert r.json()["engine"] == "piper"


# --- registry + discovery ------------------------------------------------------


async def test_registry_accepts_elevenlabs_without_local_file(client, admin_headers) -> None:
    r = await client.put(
        "/voices/registry/t-el-cast",
        json={"engine": "elevenlabs", "voice_file": _VOICE},
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["engine"] == "elevenlabs"


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"engine": "elevenlabs", "voice_file": "en_US-amy-medium"}, "not an ElevenLabs voice id"),
        ({"engine": "polly", "voice_file": "Joanna"}, "unknown engine"),
    ],
)
async def test_registry_rejects_bad_engine_config(client, admin_headers, body, fragment) -> None:
    r = await client.put("/voices/registry/t-el-bad", json=body, headers=admin_headers)
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


async def test_discovery_installed_reflects_client_key(
    client, db_session, seeded_client, secrets_key
) -> None:
    capp, key = seeded_client
    await _add_el_alias(db_session)

    async def installed() -> bool:
        r = await client.get("/voices", headers=_auth(key))
        return {v["name"]: v for v in r.json()["voices"]}["t-el-narrator"]["installed"]

    assert await installed() is False
    await _give_key(db_session, capp)
    assert await installed() is True


# --- per-client key endpoints --------------------------------------------------


async def test_set_and_clear_elevenlabs_key(
    client, db_session, seeded_client, admin_headers, secrets_key
) -> None:
    capp, _ = seeded_client
    r = await client.put(
        f"/clients/{capp.id}/elevenlabs-key",
        json={"api_key": "  xi-secret  "},
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["elevenlabs_api_key_set_at"] is not None
    assert "xi-secret" not in r.text
    await db_session.refresh(capp)
    assert capp.elevenlabs_api_key_encrypted not in (None, "xi-secret")

    r = await client.delete(f"/clients/{capp.id}/elevenlabs-key", headers=admin_headers)
    assert r.status_code == 200
    await db_session.refresh(capp)
    assert capp.elevenlabs_api_key_encrypted is None


async def test_set_key_unknown_client_404(client, admin_headers) -> None:
    r = await client.put(
        "/clients/00000000-0000-0000-0000-000000000000/elevenlabs-key",
        json={"api_key": "x"},
        headers=admin_headers,
    )
    assert r.status_code == 404


# --- executor --------------------------------------------------------------------


@pytest.fixture
def tts_output(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


async def _el_job(db_session, capp, text="hello there") -> Job:
    job = Job(
        client_app_id=capp.id,
        task_type="tts",
        sensitivity="public",
        priority=5,
        prompt=text,
        system_prompt="",
        model_requested=_VOICE,
        status=JobStatus.PENDING.value,
        job_metadata={"tts_engine": "elevenlabs"},
    )
    db_session.add(job)
    await db_session.commit()
    return job


@respx.mock
async def test_executor_elevenlabs_completes_with_cost_and_usage(
    db_session, seeded_client, secrets_key, tts_output
) -> None:
    capp, _ = seeded_client
    await _give_key(db_session, capp)
    route = respx.post(_TTS_URL).mock(
        return_value=httpx.Response(200, content=b"ID3mp3", headers={"request-id": "r1"})
    )
    text = "x" * 1000
    job = await _el_job(db_session, capp, text=text)

    await execute_tts(job=job, client=capp, session=db_session)

    assert route.calls.last.request.headers["xi-api-key"] == "xi-test-key"
    assert job.status == JobStatus.COMPLETE.value, job.error
    assert job.response == f"/output/{job.id}.mp3"
    assert (tts_output / f"{job.id}.mp3").read_bytes() == b"ID3mp3"
    # eleven_multilingual_v2 at $80 / 1M chars → 1000 chars = $0.08
    assert job.cost_usd == Decimal("0.080000")
    meta = job.job_metadata["tts"]
    assert meta["engine"] == "elevenlabs"
    assert meta["request_id"] == "r1"
    assert meta["model_id"] == "eleven_multilingual_v2"
    usage = await db_session.scalar(
        select(ClientAppUsage).where(ClientAppUsage.client_app_id == capp.id)
    )
    assert usage is not None
    assert usage.cost_usd == Decimal("0.08")


async def test_executor_fails_cleanly_when_key_cleared_after_submit(
    db_session, seeded_client, tts_output
) -> None:
    capp, _ = seeded_client
    job = await _el_job(db_session, capp)
    await execute_tts(job=job, client=capp, session=db_session)
    assert job.status == JobStatus.FAILED.value
    assert "no ElevenLabs API key" in job.error


@respx.mock
async def test_executor_surfaces_quota_error(
    db_session, seeded_client, secrets_key, tts_output
) -> None:
    capp, _ = seeded_client
    await _give_key(db_session, capp)
    respx.post(_TTS_URL).mock(
        return_value=httpx.Response(402, json={"detail": {"status": "quota_exceeded"}})
    )
    job = await _el_job(db_session, capp)
    await execute_tts(job=job, client=capp, session=db_session)
    assert job.status == JobStatus.FAILED.value
    assert "quota_exceeded" in job.error
    assert job.cost_usd is None
