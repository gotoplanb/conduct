"""Unit tests for the DB-backed prompt resolver.

The resolver picks a per-client override when one exists, otherwise the
shared row; missing entirely → PromptNotFoundError. The savepoint fixture
in conftest.py rolls everything back after each test.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from auth import generate_api_key, hash_api_key
from models.client import ClientApp
from models.prompt import Prompt, PromptVersion
from prompt_loader import PromptNotFoundError, resolve_prompt


@pytest.fixture
def unique_task_type() -> str:
    """Avoid collisions with rows seeded by dev-env runs against the same DB."""
    return f"unit_resolve_{uuid4().hex[:8]}"


async def _make_client(session: AsyncSession, name: str) -> ClientApp:
    raw_key = generate_api_key()
    c = ClientApp(
        name=name,
        api_key_hash=hash_api_key(raw_key),
        notes="resolver test",
        rate_limit_per_minute=None,
        allow_cloud_for_internal=False,
    )
    session.add(c)
    await session.commit()
    await session.refresh(c)
    return c


async def test_client_override_wins(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    client = await _make_client(db_session, f"resolver-alpha-{uuid4().hex[:6]}")
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=None, content="shared body")
    )
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=client.id, content="alpha override")
    )
    await db_session.commit()

    out = await resolve_prompt(db_session, unique_task_type, client_name=client.name)
    assert out.content == "alpha override"
    assert out.source == f"client:{client.name}:{unique_task_type}"


async def test_falls_back_to_shared_when_no_client_override(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    client = await _make_client(db_session, f"resolver-beta-{uuid4().hex[:6]}")
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=None, content="shared default")
    )
    await db_session.commit()

    out = await resolve_prompt(db_session, unique_task_type, client_name=client.name)
    assert out.content == "shared default"
    assert out.source == f"shared:{unique_task_type}"


async def test_no_client_returns_shared(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=None, content="shared body")
    )
    await db_session.commit()

    out = await resolve_prompt(db_session, unique_task_type)
    assert out.content == "shared body"
    assert out.source == f"shared:{unique_task_type}"


async def test_missing_prompt_raises(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    with pytest.raises(PromptNotFoundError):
        await resolve_prompt(db_session, unique_task_type)


async def test_unknown_client_name_falls_back_to_shared(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    """A client_name that doesn't exist in client_apps shouldn't fail —
    we want graceful degradation to the shared row so a typo doesn't 500."""
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=None, content="shared body")
    )
    await db_session.commit()

    out = await resolve_prompt(db_session, unique_task_type, client_name="does-not-exist")
    assert out.content == "shared body"
    assert out.source == f"shared:{unique_task_type}"


async def test_version_id_returns_most_recent(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    """The resolver should surface the latest PromptVersion.id for the
    chosen (task_type, client_id) tuple — that's what jobs persist into
    job_metadata so we can replay later."""
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=None, content="v2 content")
    )
    db_session.add(
        PromptVersion(task_type=unique_task_type, client_id=None, content="v1 content")
    )
    db_session.add(
        PromptVersion(task_type=unique_task_type, client_id=None, content="v2 content")
    )
    await db_session.commit()

    out = await resolve_prompt(db_session, unique_task_type)
    assert out.version_id is not None
    # Verify it points at the newest row.
    from sqlalchemy import select

    latest_content = await db_session.scalar(
        select(PromptVersion.content).where(PromptVersion.id == out.version_id)
    )
    assert latest_content == "v2 content"


async def test_version_id_is_none_when_no_history(
    db_session: AsyncSession, unique_task_type: str
) -> None:
    """A freshly-imported prompt with no version rows should resolve cleanly
    with version_id=None — the live row is the source of truth even when
    audit history hasn't been written yet."""
    db_session.add(
        Prompt(task_type=unique_task_type, client_id=None, content="seed only")
    )
    await db_session.commit()

    out = await resolve_prompt(db_session, unique_task_type)
    assert out.content == "seed only"
    assert out.version_id is None


@pytest.mark.asyncio
async def test_resolve_falls_through_archived_client_override(
    db_session: AsyncSession,
) -> None:
    """An archived per-client override behaves as if it didn't exist — the
    resolver falls through to the shared default. Without this, a soft-
    deleted prompt could keep silently dispatching."""
    task = f"resolve-{uuid4().hex[:6]}"

    client = ClientApp(
        name=f"c-{uuid4().hex[:6]}",
        api_key_hash=hash_api_key(generate_api_key()),
    )
    db_session.add(client)
    await db_session.flush()
    # Shared default + archived per-client override.
    db_session.add(Prompt(task_type=task, client_id=None, content="shared body"))
    db_session.add(
        Prompt(
            task_type=task,
            client_id=client.id,
            content="archived client body",
            is_archived=True,
        )
    )
    await db_session.commit()

    resolved = await resolve_prompt(db_session, task, client_name=client.name)
    assert resolved.content == "shared body"
    assert resolved.source == f"shared:{task}"


@pytest.mark.asyncio
async def test_resolve_raises_when_only_remaining_is_archived(
    db_session: AsyncSession,
) -> None:
    """If both the client override and the shared default are archived, the
    resolver raises — the dispatch shouldn't keep working off a soft-deleted
    row."""
    task = f"resolve-{uuid4().hex[:6]}"
    db_session.add(
        Prompt(task_type=task, client_id=None, content="x", is_archived=True)
    )
    await db_session.commit()

    with pytest.raises(PromptNotFoundError):
        await resolve_prompt(db_session, task)
