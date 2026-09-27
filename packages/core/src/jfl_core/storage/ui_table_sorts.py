"""Which column and direction a user last sorted each table by.

One row per (user, table), upserted on a header click and never on a plain
render -- a page render is one `SELECT` for the one table it shows. Same shape
as `jfl_core.storage.ui_sections`, and for the same reason: this holds nothing
more sensitive than a column preference, but a `WHERE user_id = ...` a
reviewer has to notice is not enforcement, so it gets the identical structural
tenancy as everything else.

`sort_key` and `direction` are read back exactly as stored, with no
interpretation here -- validating a saved key against a table's actual columns
(and silently falling back when it no longer matches one) is
`jfl_web.sorting.parse_sort`'s job, not this repository's. A screen that drops
a column must not need a migration just because someone's saved preference
named it.

See `docs/ui-sections.md`'s persistence section for the sibling table this one
follows the shape of, and CLAUDE.md's owner feedback: "The sorting of the
applications isn't persistent, in fact it needs to be persistent on any
tables, etc."
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from jfl_core.db.tables import ui_table_sorts as table
from jfl_core.storage.tenancy import TenantScopedRepository


@dataclasses.dataclass(frozen=True, slots=True)
class TableSortState:
    """One user's saved sort for one table."""

    table_key: str
    sort_key: str
    direction: str
    updated_at: dt.datetime


class PostgresUiTableSortRepository(TenantScopedRepository):
    """One user's saved table sorts, and no one else's."""

    def get_sort(self, table_key: str) -> TableSortState | None:
        row = self._conn.execute(
            select(
                table.c.table_key, table.c.sort_key, table.c.direction, table.c.updated_at
            ).where(table.c.user_id == self._user_id, table.c.table_key == table_key)
        ).first()
        if row is None:
            return None
        return TableSortState(
            table_key=row.table_key,
            sort_key=row.sort_key,
            direction=row.direction,
            updated_at=row.updated_at,
        )

    def save_sort(self, table_key: str, sort_key: str, direction: str) -> None:
        """Upsert the one row for `table_key`. Called only from a header click
        (or a page normalising `?sort=` off the query string) -- never from a
        plain render, so visiting a page never writes.
        """
        statement = insert(table).values(
            id=uuid.uuid4(),
            user_id=self._user_id,
            table_key=table_key,
            sort_key=sort_key,
            direction=direction,
        )
        self._conn.execute(
            statement.on_conflict_do_update(
                index_elements=[table.c.user_id, table.c.table_key],
                set_={
                    "sort_key": statement.excluded.sort_key,
                    "direction": statement.excluded.direction,
                    "updated_at": dt.datetime.now(dt.UTC),
                },
            )
        )


__all__ = ["PostgresUiTableSortRepository", "TableSortState"]
