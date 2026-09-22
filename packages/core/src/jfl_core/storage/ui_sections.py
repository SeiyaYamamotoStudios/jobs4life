"""Which sections of a screen a user leaves open, and where that disagrees with
the default we chose for them.

One row per (user, section), upserted on toggle and never on render. A page
render is one `SELECT` of this user's rows; a toggle is one `INSERT ... ON
CONFLICT DO UPDATE`. Nothing here is on the path of anything that costs money,
and nothing here can change what a screen *says* -- only which parts of it start
folded.

The second column of interest is `against_default`. The owner's reason for
wanting this stored server-side rather than in `localStorage` was not sync, it
was measurement: *"we will have to track if people go against this."* A default
that every user immediately undoes should be answerable with a query rather than
noticed eventually, so every toggle records what the screen would have done
without it.

`default_open` is reported by the page that was on screen. It is the user's own
telemetry about their own account, so a tampered value corrupts nothing but
their own record of their own clicks -- worth stating, since this is the one
field here that is not derived server-side.

See `docs/ui-sections.md` for the component contract these rows feed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from jfl_core.db.tables import ui_section_states as table
from jfl_core.storage.tenancy import TenantScopedRepository


@dataclasses.dataclass(frozen=True)
class SectionState:
    """One user's standing choice about one section.

    `last_opened_at` is NULL until the first toggle. A caller must treat that as
    "no watermark", never as "the beginning of time": the difference is whether
    a section that has always been collapsed announces its whole contents as
    new the first time anybody looks at the page.
    """

    section_key: str
    is_open: bool
    default_open: bool
    toggles: int
    against_default: int
    last_opened_at: dt.datetime | None


class PostgresUiSectionRepository(TenantScopedRepository):
    """One user's section states, and no one else's."""

    def states(self) -> dict[str, SectionState]:
        """Every section this user has ever toggled, keyed by section id.

        One query per page: the screens ask for the whole map and look sections
        up in it, rather than issuing a read per panel. The row count is bounded
        by how many panels a person has ever clicked, so this stays small
        without needing a limit.
        """
        rows = self._conn.execute(
            select(
                table.c.section_key,
                table.c.is_open,
                table.c.default_open,
                table.c.toggles,
                table.c.against_default,
                table.c.last_opened_at,
            ).where(table.c.user_id == self._user_id)
        ).all()
        return {
            row.section_key: SectionState(
                section_key=row.section_key,
                is_open=row.is_open,
                default_open=row.default_open,
                toggles=row.toggles,
                against_default=row.against_default,
                last_opened_at=row.last_opened_at,
            )
            for row in rows
        }

    def record_toggle(self, section_key: str, *, is_open: bool, default_open: bool) -> None:
        """One write, whichever way the section was toggled.

        `last_opened_at` moves on **both** directions on purpose. Opening it
        means the user is looking at it now; closing it means they have just
        finished looking. Either way, everything currently inside has been seen,
        which is exactly what the change marker needs to know.

        `against_default` counts disagreements rather than recording the latest
        one, so a user who folds a panel away every single visit is
        distinguishable from one who did it once by accident.
        """
        against = 1 if is_open != default_open else 0
        statement = insert(table).values(
            id=uuid.uuid4(),
            user_id=self._user_id,
            section_key=section_key,
            is_open=is_open,
            default_open=default_open,
            toggles=1,
            against_default=against,
            last_opened_at=dt.datetime.now(dt.UTC),
        )
        self._conn.execute(
            statement.on_conflict_do_update(
                index_elements=[table.c.user_id, table.c.section_key],
                set_={
                    "is_open": statement.excluded.is_open,
                    "default_open": statement.excluded.default_open,
                    "toggles": table.c.toggles + 1,
                    "against_default": table.c.against_default + against,
                    "last_opened_at": statement.excluded.last_opened_at,
                    "updated_at": statement.excluded.last_opened_at,
                },
            )
        )


__all__ = ["PostgresUiSectionRepository", "SectionState"]
