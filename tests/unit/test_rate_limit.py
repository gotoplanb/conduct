"""Rate limit tests against fakeredis (no external deps)."""

from __future__ import annotations

from uuid import uuid4

import fakeredis
import pytest
from fastapi import HTTPException

import rate_limit
import worker.queue
from models.client import ClientApp


@pytest.fixture(autouse=True)
def _redis_in_memory(monkeypatch):
    """Replace get_redis with a fake one for the duration of each test."""
    fake = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(worker.queue, "_redis", fake)
    monkeypatch.setattr(worker.queue, "get_redis", lambda: fake)
    monkeypatch.setattr(rate_limit, "get_redis", lambda: fake)
    yield fake


def _client(rate: int | None) -> ClientApp:
    c = ClientApp(
        id=uuid4(),
        name="tester",
        api_key_hash="x" * 64,
        is_active=True,
        rate_limit_per_minute=rate,
        allow_cloud_for_internal=False,
        notes="",
    )
    return c


@pytest.mark.asyncio
async def test_unlimited_client_passes_through() -> None:
    c = _client(None)
    out = rate_limit.rate_limited_client(c)
    assert out is c


@pytest.mark.asyncio
async def test_under_limit_passes() -> None:
    c = _client(3)
    for _ in range(3):
        rate_limit.rate_limited_client(c)


@pytest.mark.asyncio
async def test_over_limit_raises_429_with_retry_after() -> None:
    c = _client(2)
    rate_limit.rate_limited_client(c)
    rate_limit.rate_limited_client(c)
    with pytest.raises(HTTPException) as exc:
        rate_limit.rate_limited_client(c)
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers
    assert exc.value.headers["X-RateLimit-Limit"] == "2"
    assert exc.value.headers["X-RateLimit-Remaining"] == "0"


@pytest.mark.asyncio
async def test_separate_clients_have_separate_buckets() -> None:
    a = _client(1)
    b = _client(1)
    rate_limit.rate_limited_client(a)
    # b should still have its own budget intact
    rate_limit.rate_limited_client(b)
    with pytest.raises(HTTPException):
        rate_limit.rate_limited_client(a)
    with pytest.raises(HTTPException):
        rate_limit.rate_limited_client(b)
