from __future__ import annotations

import os

from alembic import context
from jfl_core.db.tables import metadata
from sqlalchemy import engine_from_config, pool

config = context.config
config.set_main_option(
    "sqlalchemy.url",
    os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl"),
)
target_metadata = metadata


def include_object(obj, name, type_, reflected, compare_to):
    # pgvector's hnsw index is created by hand in the initial migration.
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
