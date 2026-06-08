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


async def test_sampling_defaults_to_balanced_and_round_trips(client, admin_headers) -> None:
    """A rule with no sampling field defaults to 'balanced'; an explicit
    profile round-trips through PUT → GET."""
    base = {
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-haiku-4-5",
        "sensitivity": "internal",
    }
    # Default when omitted.
    r = await client.put("/routing/sampling_default_task", json=base, headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["sampling"] == "balanced"

    # Explicit profile persists and is readable.
    r2 = await client.put(
        "/routing/sampling_det_task",
        json={**base, "sampling": "deterministic"},
        headers=admin_headers,
    )
    assert r2.status_code == 200
    assert r2.json()["sampling"] == "deterministic"
    g = await client.get("/routing/sampling_det_task", headers=admin_headers)
    assert g.json()["sampling"] == "deterministic"


async def test_sampling_rejects_unknown_profile(client, admin_headers) -> None:
    body = {
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-haiku-4-5",
        "sampling": "wildly-creative",
    }
    r = await client.put("/routing/bad_sampling_task", json=body, headers=admin_headers)
    assert r.status_code == 422


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


# --- soft-delete (archive) endpoints + filtering ---


async def _put(client, admin_headers, task_type, **overrides):
    body = {
        "preferred_model": "llama3.3:70b",
        "fallback_model": "claude-haiku-4-5",
        "sensitivity": "internal",
        "max_tokens": 500,
        "notes": "",
        "eval_shadow_models": [],
        **overrides,
    }
    r = await client.put(f"/routing/{task_type}", json=body, headers=admin_headers)
    assert r.status_code == 200, r.text
    return r.json()


async def test_delete_archives_routing_rule(client, admin_headers) -> None:
    await _put(client, admin_headers, "to_archive")
    r = await client.delete("/routing/to_archive", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["is_archived"] is True


async def test_delete_unknown_routing_rule_is_404(client, admin_headers) -> None:
    r = await client.delete("/routing/never_existed", headers=admin_headers)
    assert r.status_code == 404


async def test_delete_routing_rule_is_idempotent(client, admin_headers) -> None:
    await _put(client, admin_headers, "to_archive_twice")
    await client.delete("/routing/to_archive_twice", headers=admin_headers)
    r = await client.delete("/routing/to_archive_twice", headers=admin_headers)
    # Still archived, still 200 — idempotent.
    assert r.status_code == 200
    assert r.json()["is_archived"] is True


async def test_list_routing_hides_archived_by_default(client, admin_headers) -> None:
    await _put(client, admin_headers, "visible_rule")
    await _put(client, admin_headers, "hidden_rule")
    await client.delete("/routing/hidden_rule", headers=admin_headers)

    r = await client.get("/routing", headers=admin_headers)
    visible_names = {row["task_type"] for row in r.json()["rules"]}
    assert "visible_rule" in visible_names
    assert "hidden_rule" not in visible_names


async def test_list_routing_include_archived(client, admin_headers) -> None:
    await _put(client, admin_headers, "exposed_rule")
    await client.delete("/routing/exposed_rule", headers=admin_headers)
    r = await client.get(
        "/routing?include_archived=true", headers=admin_headers
    )
    names = {row["task_type"]: row for row in r.json()["rules"]}
    assert names["exposed_rule"]["is_archived"] is True


async def test_get_archived_routing_rule_is_404_by_default(
    client, admin_headers
) -> None:
    await _put(client, admin_headers, "ghost_rule")
    await client.delete("/routing/ghost_rule", headers=admin_headers)

    r1 = await client.get("/routing/ghost_rule", headers=admin_headers)
    assert r1.status_code == 404

    r2 = await client.get(
        "/routing/ghost_rule?include_archived=true", headers=admin_headers
    )
    assert r2.status_code == 200
    assert r2.json()["is_archived"] is True


async def test_put_revives_archived_routing_rule(client, admin_headers) -> None:
    """PUT means 'this is the current rule' — operators shouldn't have to
    know about archives. The upsert revives transparently."""
    await _put(client, admin_headers, "revive_me")
    await client.delete("/routing/revive_me", headers=admin_headers)
    revived = await _put(
        client, admin_headers, "revive_me", preferred_model="llama3.3:70b"
    )
    assert revived["is_archived"] is False
    assert revived["preferred_model"] == "llama3.3:70b"
