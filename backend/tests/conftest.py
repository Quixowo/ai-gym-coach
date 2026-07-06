"""Shared pytest fixtures.

Two things are set up here for the whole session:

1. **Schema is migrated once** — a session-scoped autouse fixture runs
   ``alembic upgrade head`` a single time, synchronously, before any test.
   Per LESSONS.md, alembic's async ``env.py`` calls ``asyncio.run()`` internally,
   so it must NOT run inside pytest-asyncio's event loop — hence this fixture is
   a plain (sync) function and every test can then assume the schema exists.

2. **DB access uses a NullPool engine** — ``TestClient`` drives each request on
   its own short-lived event loop, but the app's default pooled engine keeps
   connections bound to the *first* request's loop, so the *second* request in a
   test raises "Event loop is closed". Overriding ``get_db`` (and the engine the
   ``/health`` route touches) with a ``NullPool`` engine means no connection is
   retained across requests/loops. Production keeps its pooled engine untouched.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from alembic import command
from app.core.config import settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    # env.py injects sqlalchemy.url from settings, so no override needed here.
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return cfg


@pytest.fixture(scope="session", autouse=True)
def migrated_db() -> None:
    """Run ``alembic upgrade head`` exactly once before the suite (sync).

    Idempotent: a no-op if the DB is already at head. Kept synchronous so it does
    not collide with pytest-asyncio's running loop (see module docstring /
    LESSONS.md). Requires a reachable Postgres (docker compose up -d / CI service).
    """
    command.upgrade(_alembic_config(), "head")


# NullPool test engine — see module docstring for why pooling breaks TestClient.
_test_engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
_test_session_maker = async_sessionmaker(_test_engine, class_=AsyncSession, expire_on_commit=False)


async def _override_get_db() -> AsyncGenerator[AsyncSession]:
    async with _test_session_maker() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def client(migrated_db: None):
    """Session-scoped TestClient with the NullPool DB override applied.

    ``get_db`` is overridden app-wide, and the module-level engine used by the
    ``/health`` route is swapped to the NullPool engine for the duration of the
    session so multi-request tests don't hit the closed-loop error.
    """
    from fastapi.testclient import TestClient

    import app.db.session as db_session
    from app.deps import get_db
    from app.main import app

    app.dependency_overrides[get_db] = _override_get_db
    original_engine = db_session.engine
    db_session.engine = _test_engine

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.pop(get_db, None)
    db_session.engine = original_engine
    asyncio.run(_test_engine.dispose())
