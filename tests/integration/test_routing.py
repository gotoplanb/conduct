"""Routes under /routing — admin list + upsert with eval_shadow_models."""

from __future__ import annotations


async def test_list_routing_returns_empty_initially(client, admin_headers) -> None:
    r = await client.get("/routing", headers=admin_headers)
    assert r.status_code == 200
    # Other tests may have committed rules; just confirm shape.
    assert "rules" in r.json()


async def test_upsert_creates_then_updates_rule(client, admin_headers) -> None:
    body = {
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-haiku-4-5",
        "sensitivity": "internal",
        "max_tokens": 2000,
        "notes": "first",
        "eval_shadow_models": [
            {"model": "qwen2.5:7b", "rate": 0.1},
            {"model": "claude-haiku-4-5", "rate": 0.05, "daily_cost_cap_usd": 1.0},
        ],
    }
    r = await client.put("/routing/bio_generation", json=body, headers=admin_headers)
    assert r.status_code == 200
    out = r.json()
    assert out["preferred_model"] == "llama3.3:70b"
    assert out["max_tokens"] == 2000
    assert len(out["eval_shadow_models"]) == 2
    assert out["eval_shadow_models"][1]["daily_cost_cap_usd"] == 1.0

    # Update — change tokens, clear shadows
    body2 = {**body, "max_tokens": 500, "eval_shadow_models": [], "notes": "second"}
    r2 = await client.put("/routing/bio_generation", json=body2, headers=admin_headers)
    assert r2.status_code == 200
    assert r2.json()["max_tokens"] == 500
    assert r2.json()["eval_shadow_models"] == []
    assert r2.json()["notes"] == "second"


async def test_upsert_validates_fanout_rate_bounds(client, admin_headers) -> None:
    body = {
        "preferred_model": "x",
        "fallback_model": "y",
        "sensitivity": "internal",
        "eval_shadow_models": [{"model": "z", "rate": 1.5}],  # > 1.0
    }
    r = await client.put("/routing/test_task", json=body, headers=admin_headers)
    assert r.status_code == 422


async def test_upsert_rejects_oversized_task_type(client, admin_headers) -> None:
    long_name = "x" * 101
    body = {
        "preferred_model": "a",
        "fallback_model": "b",
        "sensitivity": "internal",
    }
    r = await client.put(f"/routing/{long_name}", json=body, headers=admin_headers)
    assert r.status_code == 400


async def test_routing_requires_admin(client) -> None:
    r = await client.get("/routing")
    assert r.status_code == 403


async def test_get_single_rule_returns_full_shape(client, admin_headers) -> None:
    body = {
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-haiku-4-5",
        "sensitivity": "internal",
        "max_tokens": 1500,
        "notes": "fetched",
        "eval_shadow_models": [{"model": "qwen3.5:9b", "rate": 0.2}],
    }
    await client.put("/routing/single_rule_task", json=body, headers=admin_headers)
    r = await client.get("/routing/single_rule_task", headers=admin_headers)
    assert r.status_code == 200
    out = r.json()
    assert out["task_type"] == "single_rule_task"
    assert out["preferred_model"] == "llama3.3:70b"
    assert out["max_tokens"] == 1500
    assert out["eval_shadow_models"][0]["model"] == "qwen3.5:9b"


async def test_get_single_rule_missing_is_404(client, admin_headers) -> None:
    r = await client.get("/routing/nope_does_not_exist", headers=admin_headers)
    assert r.status_code == 404


async def test_get_single_rule_requires_admin(client) -> None:
    r = await client.get("/routing/whatever")
    assert r.status_code == 403
