from __future__ import annotations

import asyncio
from logging.config import fileConfig

import pgvector.sqlalchemy
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.core.config import settings

# Import models so that Base.metadata is fully populated for autogenerate.
from app.models import Base  # noqa: F401  (side-effect import: registers all models)

# Alembic Config object, provides access to values in alembic.ini.
config = context.config

# Inject the DB URL from app settings rather than hardcoding it in alembic.ini —
# single source of truth, matches the app's async (postgresql+asyncpg) driver.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _register_pgvector(connection) -> None:
    """Teach SQLAlchemy's reflection about the ``vector`` type.

    Without this, autogenerate emits ``SAWarning: Did not recognize type
    'vector'`` and can't diff pgvector columns.
    """
    connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to a script, no DBAPI)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    _register_pgvector(connection)
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through a run_sync bridge."""
    engine = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
