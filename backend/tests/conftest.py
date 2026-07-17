"""Shared pytest fixtures.

What the eval suites do and do NOT verify (CLAUDE.md rule 10):
the model-behavior suites — ``test_tool_correctness``, ``test_groundedness``,
``test_red_flag_recall`` — replay Claude/Voyage responses recorded once and committed
under ``tests/fixtures/claude_responses/``. They verify that the *code* handles a
given model response correctly (the orchestrator calls the right tool with the right
args, the RAG pipeline stays grounded / refuses traps, the classifier parses verdicts
and hits its recall/false-positive targets). They do NOT re-verify that a live model
still makes the same decisions today — CI never calls a live API and needs no API
keys. Re-checking live behavior is a manual, periodic activity: if a Claude model
version changes materially, re-run ``python -m tests.fixtures.record_fixtures`` and
re-review the fixtures. The pure-code suites (progression math, fuzzy search) need no
fixtures and test real behavior every run.

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


@pytest.fixture(scope="session", autouse=True)
def seeded_exercises(migrated_db: None) -> None:
    """Seed the exercise catalog once so tests don't depend on manual seeding.

    Runs the idempotent seed set through the NullPool test engine (not the app's
    pooled ``async_session_maker``) via ``asyncio.run`` in its own loop. Using the
    pooled engine here would cache a connection bound to this throwaway loop; when
    the loop closes, the poisoned pooled connection later breaks the ``/health``
    route's ``SELECT 1`` with "Event loop is closed" (see LESSONS.md). NullPool
    keeps nothing across loops, so the seed is self-contained.
    """
    from sqlalchemy import func, select

    from app.models.exercise import Exercise
    from seed.seed_exercises import EXERCISES

    async def _seed() -> None:
        async with _test_session_maker() as session:
            existing = set((await session.execute(select(Exercise.name))).scalars().all())
            to_add = [
                Exercise(
                    name=name,
                    primary_muscle_group=muscle,
                    movement_pattern=pattern,
                    equipment=equipment,
                )
                for (name, muscle, pattern, equipment) in EXERCISES
                if name not in existing
            ]
            if to_add:
                session.add_all(to_add)
                await session.commit()
            total = (await session.execute(select(func.count(Exercise.id)))).scalar_one()
            assert total >= len(EXERCISES)

    asyncio.run(_seed())


async def _override_get_db() -> AsyncGenerator[AsyncSession]:
    async with _test_session_maker() as session:
        yield session


# Public handle to the NullPool session maker for tests that drive the services /
# agent layer directly (progression math, tool handlers) rather than via HTTP. Using
# this — never the app's pooled ``async_session_maker`` — keeps DB setup on the same
# NullPool engine, avoiding the closed-loop family of bugs in LESSONS.md.
test_session_maker = _test_session_maker


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
