"""Routes under /models — list, load, unload (incl. resident protection)."""

from __future__ import annotations

import pytest


async def test_list_models(client, admin_headers) -> None:
    r = await client.get("/models", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "local" in body and "cloud" in body
    # The stub provider returns two locals
    names = [m["name"] for m in body["local"]]
    assert "llama3.3:70b" in names


async def test_load_model_calls_ollama(client, admin_headers) -> None:
    r = await client.post("/models/llama3.3:70b/load", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "loaded"


async def test_unload_model(client, admin_headers) -> None:
    r = await client.post("/models/llama3.3:70b/unload", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "unloaded"


async def test_unload_refuses_resident_model(
    client, admin_headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESIDENT_MODELS", "qwen2.5:7b")
    from config.settings import get_settings

    get_settings.cache_clear()

    r = await client.post("/models/qwen2.5:7b/unload", headers=admin_headers)
    assert r.status_code == 409


async def test_models_requires_admin(client) -> None:
    r = await client.get("/models")
    assert r.status_code == 403
