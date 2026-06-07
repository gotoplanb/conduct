"""Routes under /prompts — admin CRUD + version history.

Coverage:
  - 401 without admin auth
  - PUT creates a shared prompt + writes a PromptVersion row
  - PUT a second time edits the live row + appends another version row
  - Per-client PUT lives independently of the shared row
  - GET returns the right row (and 404 when neither exists)
  - GET /history returns rows newest-first
  - Edge cases: empty content rejected, oversized task_type rejected,
    unknown client name → 404
"""

from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.fixture
def task() -> str:
    """Per-test task_type so tests don't collide with the dev DB."""
    return f"route_prompt_{uuid4().hex[:8]}"


async def test_requires_admin(client, task) -> None:
    r = await client.get("/prompts")
    assert r.status_code in (401, 403)
    r2 = await client.put(f"/prompts/{task}", json={"content": "x"})
    assert r2.status_code in (401, 403)


async def test_put_creates_shared_prompt(client, admin_headers, task) -> None:
    r = await client.put(
        f"/prompts/{task}", headers=admin_headers, json={"content": "hello world"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task_type"] == task
    assert body["client_name"] is None
    assert body["content"] == "hello world"
    assert body["updated_by"]  # fingerprint is non-empty for an authed call


async def test_put_then_get_roundtrips(client, admin_headers, task) -> None:
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "v1"})
    r = await client.get(f"/prompts/{task}", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["content"] == "v1"


async def test_put_twice_appends_history(client, admin_headers, task) -> None:
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "v1"})
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "v2"})

    hist = await client.get(f"/prompts/{task}/history", headers=admin_headers)
    assert hist.status_code == 200
    versions = hist.json()["versions"]
    assert len(versions) >= 2
    # Newest first.
    assert versions[0]["edited_at"] >= versions[1]["edited_at"]

    # Live row reflects the latest content.
    cur = await client.get(f"/prompts/{task}", headers=admin_headers)
    assert cur.json()["content"] == "v2"


async def test_client_override_separate_from_shared(
    client, admin_headers, seeded_client, task
) -> None:
    c, _ = seeded_client
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "shared"})
    await client.put(
        f"/prompts/{task}",
        params={"client": c.name},
        headers=admin_headers,
        json={"content": "client-specific"},
    )

    shared = await client.get(f"/prompts/{task}", headers=admin_headers)
    overridden = await client.get(
        f"/prompts/{task}", params={"client": c.name}, headers=admin_headers
    )
    assert shared.json()["content"] == "shared"
    assert overridden.json()["content"] == "client-specific"
    assert overridden.json()["client_name"] == c.name


async def test_list_includes_new_prompt(client, admin_headers, task) -> None:
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "abc"})
    r = await client.get("/prompts", headers=admin_headers)
    assert r.status_code == 200
    task_types = {p["task_type"] for p in r.json()["prompts"]}
    assert task in task_types


async def test_get_missing_returns_404(client, admin_headers, task) -> None:
    r = await client.get(f"/prompts/{task}", headers=admin_headers)
    assert r.status_code == 404


async def test_history_missing_returns_empty(client, admin_headers, task) -> None:
    r = await client.get(f"/prompts/{task}/history", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["versions"] == []


async def test_put_rejects_empty_content(client, admin_headers, task) -> None:
    r = await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": ""})
    assert r.status_code == 422


async def test_put_rejects_oversized_task_type(client, admin_headers) -> None:
    long_name = "x" * 101
    r = await client.put(
        f"/prompts/{long_name}", headers=admin_headers, json={"content": "x"}
    )
    assert r.status_code == 400


async def test_put_unknown_client_returns_404(client, admin_headers, task) -> None:
    r = await client.put(
        f"/prompts/{task}",
        params={"client": "does-not-exist-xyz"},
        headers=admin_headers,
        json={"content": "x"},
    )
    assert r.status_code == 404


async def test_history_limit_caps_rows(client, admin_headers, task) -> None:
    for i in range(5):
        await client.put(
            f"/prompts/{task}", headers=admin_headers, json={"content": f"v{i}"}
        )
    r = await client.get(
        f"/prompts/{task}/history", headers=admin_headers, params={"limit": 3}
    )
    assert r.status_code == 200
    assert len(r.json()["versions"]) == 3




# --- soft-delete (archive) ---


async def test_delete_archives_shared_prompt(client, admin_headers, task) -> None:
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "live"})
    r = await client.delete(f"/prompts/{task}", headers=admin_headers)
    assert r.status_code == 200
    # GET 404s by default once archived
    g = await client.get(f"/prompts/{task}", headers=admin_headers)
    assert g.status_code == 404


async def test_delete_unknown_prompt_is_404(client, admin_headers, task) -> None:
    r = await client.delete(f"/prompts/{task}", headers=admin_headers)
    assert r.status_code == 404


async def test_list_prompts_hides_archived_by_default(
    client, admin_headers, task
) -> None:
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "x"})
    await client.delete(f"/prompts/{task}", headers=admin_headers)

    r = await client.get("/prompts", headers=admin_headers)
    visible = {p["task_type"] for p in r.json()["prompts"]}
    assert task not in visible

    r = await client.get("/prompts?include_archived=true", headers=admin_headers)
    visible = {p["task_type"] for p in r.json()["prompts"]}
    assert task in visible


async def test_put_revives_archived_prompt(client, admin_headers, task) -> None:
    await client.put(f"/prompts/{task}", headers=admin_headers, json={"content": "v1"})
    await client.delete(f"/prompts/{task}", headers=admin_headers)
    # Re-put as if the archive never happened
    r = await client.put(
        f"/prompts/{task}", headers=admin_headers, json={"content": "v2"}
    )
    assert r.status_code == 200
    # Now visible again
    g = await client.get(f"/prompts/{task}", headers=admin_headers)
    assert g.status_code == 200
    assert g.json()["content"] == "v2"


async def test_delete_client_scoped_prompt(
    client, admin_headers, task, seeded_client
) -> None:
    """Archiving a per-client override leaves the shared default intact."""
    row, _ = seeded_client
    await client.put(
        f"/prompts/{task}", headers=admin_headers, json={"content": "shared"}
    )
    await client.put(
        f"/prompts/{task}?client={row.name}",
        headers=admin_headers,
        json={"content": "client override"},
    )
    r = await client.delete(
        f"/prompts/{task}?client={row.name}", headers=admin_headers
    )
    assert r.status_code == 200
    # Shared still resolves
    g = await client.get(f"/prompts/{task}", headers=admin_headers)
    assert g.status_code == 200
    assert g.json()["content"] == "shared"
    # Client override is gone (the client lookup falls back to shared
    # because the override is archived; route returns shared content).
    g2 = await client.get(
        f"/prompts/{task}?client={row.name}", headers=admin_headers
    )
    # The route only fetches the row by (task, client_id), so an archived
    # client override means no row — 404. Resolver behavior is tested
    # separately.
    assert g2.status_code == 404
