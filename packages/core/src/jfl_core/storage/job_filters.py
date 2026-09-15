"""The saved job filter and per-board exceptions, tenancy-scoped.

Stored as typed, matched elsewhere: `jfl_intake.filtering` is the pure matcher,
and nothing here normalises, splits or interprets text. A filter is a lens over
stored board data -- never a fetch parameter -- so nothing in this module reads
or writes a board's history.

An exception's `note` is the owner's own words and is written exactly as given.

No SQL above this layer, and no model call anywhere near it.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from jfl_core.db.tables import board_filter_exceptions as exceptions_table
from jfl_core.db.tables import job_filters as filters_table
from jfl_core.db.tables import watched_boards as boards_table
from jfl_core.models import BoardFilterException, JobFilter, Workplace, WorkplaceMode
from jfl_core.storage.tenancy import TenantScopedRepository

_FILTER_COLUMNS = (
    filters_table.c.workplace_mode,
    filters_table.c.workplaces,
    filters_table.c.title_includes,
    filters_table.c.title_excludes,
    filters_table.c.location,
    filters_table.c.updated_at,
)

_EXCEPTION_COLUMNS = (
    exceptions_table.c.id,
    exceptions_table.c.board_id,
    exceptions_table.c.workplaces,
    exceptions_table.c.location,
    exceptions_table.c.note,
    exceptions_table.c.created_at,
    exceptions_table.c.updated_at,
)

_WORKPLACE_ORDER: tuple[Workplace, ...] = ("remote", "hybrid", "onsite", "unknown")


def _ordered(workplaces: Collection[Workplace]) -> list[Workplace]:
    """Deduplicated, in the canonical order, so a set round-trips unchanged."""
    chosen = set(workplaces)
    return [w for w in _WORKPLACE_ORDER if w in chosen]


def _filter_from_row(row: Any) -> JobFilter:
    return JobFilter(
        workplace_mode=row.workplace_mode,
        workplaces=list(row.workplaces),
        title_includes=row.title_includes,
        title_excludes=row.title_excludes,
        location=row.location,
        updated_at=row.updated_at,
    )


def _exception_from_row(row: Any) -> BoardFilterException:
    return BoardFilterException(
        id=row.id,
        board_id=row.board_id,
        workplaces=list(row.workplaces),
        location=row.location,
        note=row.note,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresJobFilterRepository(TenantScopedRepository):
    """One user's saved job filter and board exceptions, and no one else's."""

    # -- the saved filter ----------------------------------------------------

    def get_filter(self) -> JobFilter:
        """The saved filter, or the empty one (matches everything) if none."""
        row = self._conn.execute(
            select(*_FILTER_COLUMNS).where(filters_table.c.user_id == self._user_id)
        ).first()
        return JobFilter() if row is None else _filter_from_row(row)

    def save_filter(
        self,
        *,
        workplaces: Collection[Workplace],
        title_includes: str,
        title_excludes: str,
        location: str,
        workplace_mode: WorkplaceMode = "custom",
    ) -> JobFilter:
        """Replace this user's filter. `workplaces` is stored whatever the mode --
        it is consulted only under `custom`, and keeping it means switching back
        restores what was ticked.
        """
        values = {
            "workplace_mode": workplace_mode,
            "workplaces": _ordered(workplaces),
            "title_includes": title_includes,
            "title_excludes": title_excludes,
            "location": location,
        }
        statement = pg_insert(filters_table).values(
            id=uuid.uuid4(), user_id=self._user_id, **values
        )
        row = self._conn.execute(
            statement.on_conflict_do_update(
                index_elements=["user_id"],
                set_={**values, "updated_at": func.now()},
            ).returning(*_FILTER_COLUMNS)
        ).one()
        return _filter_from_row(row)

    # -- board exceptions ----------------------------------------------------

    def list_exceptions(self, board_id: uuid.UUID | None = None) -> list[BoardFilterException]:
        """This user's exceptions, oldest first -- for one board, or all of them."""
        query = select(*_EXCEPTION_COLUMNS).where(exceptions_table.c.user_id == self._user_id)
        if board_id is not None:
            query = query.where(exceptions_table.c.board_id == board_id)
        rows = self._conn.execute(
            query.order_by(exceptions_table.c.created_at.asc(), exceptions_table.c.id.asc())
        ).all()
        return [_exception_from_row(r) for r in rows]

    def get_exception(self, exception_id: uuid.UUID) -> BoardFilterException | None:
        row = self._conn.execute(
            select(*_EXCEPTION_COLUMNS).where(
                exceptions_table.c.id == exception_id,
                exceptions_table.c.user_id == self._user_id,
            )
        ).first()
        return None if row is None else _exception_from_row(row)

    def add_exception(
        self,
        board_id: uuid.UUID,
        *,
        workplaces: Collection[Workplace],
        location: str,
        note: str,
    ) -> BoardFilterException | None:
        """None, writing nothing, if the board is not this user's. Ownership is
        checked here, in the caller's transaction, rather than trusted from the
        route -- the foreign key alone would happily accept another user's board.
        """
        owned = select(boards_table.c.id).where(
            boards_table.c.id == board_id, boards_table.c.user_id == self._user_id
        )
        if self._conn.execute(owned).first() is None:
            return None
        row = self._conn.execute(
            pg_insert(exceptions_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                board_id=board_id,
                workplaces=_ordered(workplaces),
                location=location,
                note=note,
            )
            .returning(*_EXCEPTION_COLUMNS)
        ).one()
        return _exception_from_row(row)

    def update_exception(
        self,
        exception_id: uuid.UUID,
        *,
        workplaces: Collection[Workplace],
        location: str,
        note: str,
    ) -> BoardFilterException | None:
        row = self._conn.execute(
            update(exceptions_table)
            .where(
                exceptions_table.c.id == exception_id,
                exceptions_table.c.user_id == self._user_id,
            )
            .values(workplaces=_ordered(workplaces), location=location, note=note)
            .returning(*_EXCEPTION_COLUMNS)
        ).first()
        return None if row is None else _exception_from_row(row)

    def remove_exception(self, exception_id: uuid.UUID) -> bool:
        row = self._conn.execute(
            delete(exceptions_table)
            .where(
                exceptions_table.c.id == exception_id,
                exceptions_table.c.user_id == self._user_id,
            )
            .returning(exceptions_table.c.id)
        ).first()
        return row is not None
