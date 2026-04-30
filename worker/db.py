"""Worker-side DB session factory.

The worker process runs each RQ job inside its own `asyncio.run()`. Reusing the
API's pooled engine across loops causes "Event loop is closed" errors, so the
worker uses a NullPool — connections are opened/closed per job. With one worker
serving one job at a time, the overhead is negligible.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from config.settings import get_settings

_engine = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_worker_session_maker() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_maker
    if _session_maker is None:
        _engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
        _session_maker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    return _session_maker
