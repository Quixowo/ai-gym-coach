"""Migration integration test.

The actual ``alembic upgrade head`` is run once by the session-scoped
``migrated_db`` autouse fixture in ``conftest.py`` (see that module for why it
must run synchronously, outside pytest-asyncio's loop — LESSONS.md). This test
therefore only *asserts* the resulting schema: all seven public tables exist,
the ``vector`` extension is installed, and both special indexes are present.

The verification queries use their own isolated ``asyncio.run`` so they don't
collide with the alembic run or with pytest-asyncio.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings

EXPECTED_TABLES = {
    "users",
    "exercises",
    "programs",
    "program_exercises",
    "workout_sessions",
    "set_entries",
    "knowledge_chunks",
}

EXPECTED_INDEXES = {
    "ix_knowledge_chunks_embedding",
    "ix_set_entries_user_exercise_created",
}


async def _collect_schema() -> tuple[set[str], int | None, set[str]]:
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            tables = set(
                (
                    await conn.execute(
                        text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = 'public'"
                        )
                    )
                )
                .scalars()
                .all()
            )
            has_vector = (
                await conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            ).scalar_one_or_none()
            indexes = set(
                (
                    await conn.execute(
                        text("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
                    )
                )
                .scalars()
                .all()
            )
        return tables, has_vector, indexes
    finally:
        await engine.dispose()


def test_migration_head_builds_schema() -> None:
    """Schema was built by the ``migrated_db`` fixture; assert its shape."""
    tables, has_vector, indexes = asyncio.run(_collect_schema())

    assert EXPECTED_TABLES <= tables, f"missing tables: {EXPECTED_TABLES - tables}"
    assert has_vector == 1, "vector extension not installed"
    assert EXPECTED_INDEXES <= indexes, f"missing indexes: {EXPECTED_INDEXES - indexes}"
