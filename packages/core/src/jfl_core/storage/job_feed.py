"""The reader's side of the "what changed" feed, tenancy-scoped.

Events are not stored anywhere -- `PostgresBoardRepository.events_since` derives
them from the check history. What this repository keeps is only what the history
cannot know: when this user last looked, and which events they have been shown or
have dismissed. The rule that turns those into a page is `jfl_intake.feed` (pure).

**Marks are written only for events this user's own history produced.** A mark
names a job and a check by id, and the foreign keys alone would accept another
user's; so `record_seen` keeps only keys whose job and check belong to this user,
in the caller's transaction, rather than trusting the route.

No SQL above this layer, and no model call anywhere near it.

**`purge_stale_marks` is deliberately not a method on the repository above.**
It is global maintenance, not one user's data: the worker calls it once, across
every user, the same way `PostgresSessionRepository.purge_expired` purges every
user's expired sessions in one statement. `PostgresJobFeedRepository` cannot
express that -- it is constructed for exactly one `user_id` -- so this stays a
plain function taking a `Connection`, outside the tenancy scheme entirely (it is
not named `*Repository` and `test_tenancy_enforcement.py` does not walk it). Its
own `WHERE` never crosses a user boundary: see its docstring.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from jfl_core.db.tables import board_checks as checks_table
from jfl_core.db.tables import board_jobs as jobs_table
from jfl_core.db.tables import job_feed_marks as marks_table
from jfl_core.db.tables import job_feed_state as state_table
from jfl_core.models import BoardJobEvent, JobFeedMark
from jfl_core.storage.tenancy import TenantScopedRepository

# Rows per INSERT; psycopg caps one statement at 65,535 parameters.
_CHUNK = 1000

_MARK_COLUMNS = (
    marks_table.c.id,
    marks_table.c.job_id,
    marks_table.c.check_id,
    marks_table.c.kind,
    marks_table.c.event_at,
    marks_table.c.first_seen_at,
    marks_table.c.dismissed_at,
)


def _mark_from_row(row: Any) -> JobFeedMark:
    return JobFeedMark(
        id=row.id,
        job_id=row.job_id,
        check_id=row.check_id,
        kind=row.kind,
        event_at=row.event_at,
        first_seen_at=row.first_seen_at,
        dismissed_at=row.dismissed_at,
    )


class PostgresJobFeedRepository(TenantScopedRepository):
    """One user's last-looked time and event marks, and no one else's."""

    # -- last looked ---------------------------------------------------------

    def last_looked_at(self) -> dt.datetime | None:
        """None until the first view of the feed."""
        return self._conn.execute(
            select(state_table.c.last_looked_at).where(state_table.c.user_id == self._user_id)
        ).scalar_one_or_none()

    def set_last_looked_at(self, at: dt.datetime) -> None:
        """Move the last-looked time to `at`, never backwards: a slow request
        finishing after a newer one must not make already-seen events unseen.
        """
        statement = pg_insert(state_table).values(
            id=uuid.uuid4(), user_id=self._user_id, last_looked_at=at
        )
        self._conn.execute(
            statement.on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "last_looked_at": func.greatest(
                        func.coalesce(state_table.c.last_looked_at, at), at
                    ),
                    "updated_at": func.now(),
                },
            )
        )

    # -- marks ---------------------------------------------------------------

    def live_marks(self, *, now: dt.datetime, visible_for: dt.timedelta) -> list[JobFeedMark]:
        """Undismissed marks first seen within `visible_for` of `now`."""
        rows = self._conn.execute(
            select(*_MARK_COLUMNS).where(
                marks_table.c.user_id == self._user_id,
                marks_table.c.first_seen_at > now - visible_for,
                marks_table.c.dismissed_at.is_(None),
            )
        ).all()
        return [_mark_from_row(r) for r in rows]

    def marks_for_events_after(self, since: dt.datetime) -> list[JobFeedMark]:
        """Every mark, dismissed or not, for an event that happened after `since`
        -- the same window `events_since(since)` derives, so each derived event
        can be matched to its mark if it has one.
        """
        rows = self._conn.execute(
            select(*_MARK_COLUMNS).where(
                marks_table.c.user_id == self._user_id,
                marks_table.c.event_at > since,
            )
        ).all()
        return [_mark_from_row(r) for r in rows]

    def record_seen(
        self, events: Sequence[BoardJobEvent], *, now: dt.datetime
    ) -> list[JobFeedMark]:
        """Mark these events as first shown at `now`, and return their marks.

        Idempotent: an event that already has a mark keeps it unchanged -- its
        24 hours do not restart -- and that existing mark is what is returned.
        Events whose job or check is not this user's are skipped, writing nothing.
        """
        if not events:
            return []
        job_ids = {e.job.id for e in events}
        check_ids = {e.check_id for e in events}
        owned_jobs = set(
            self._conn.execute(
                select(jobs_table.c.id).where(
                    jobs_table.c.user_id == self._user_id, jobs_table.c.id.in_(job_ids)
                )
            ).scalars()
        )
        owned_checks = set(
            self._conn.execute(
                select(checks_table.c.id).where(
                    checks_table.c.user_id == self._user_id, checks_table.c.id.in_(check_ids)
                )
            ).scalars()
        )
        rows: dict[tuple[uuid.UUID, str, uuid.UUID], dict[str, Any]] = {}
        for e in events:
            if e.job.id not in owned_jobs or e.check_id not in owned_checks:
                continue
            rows[(e.job.id, e.kind, e.check_id)] = {
                "id": uuid.uuid4(),
                "user_id": self._user_id,
                "job_id": e.job.id,
                "check_id": e.check_id,
                "kind": e.kind,
                "event_at": e.at,
                "first_seen_at": now,
            }
        if not rows:
            return []
        values = list(rows.values())
        # Chunked for the reason `jfl_core.storage.boards` gives: a mass "gone"
        # on a large board, unfiltered, can be thousands of rows.
        for i in range(0, len(values), _CHUNK):
            self._conn.execute(
                pg_insert(marks_table)
                .values(values[i : i + _CHUNK])
                .on_conflict_do_nothing(index_elements=["user_id", "job_id", "kind", "check_id"])
            )
        found = self._conn.execute(
            select(*_MARK_COLUMNS).where(
                marks_table.c.user_id == self._user_id,
                marks_table.c.job_id.in_({key[0] for key in rows}),
                marks_table.c.check_id.in_({key[2] for key in rows}),
            )
        ).all()
        return [
            mark
            for mark in (_mark_from_row(r) for r in found)
            if (mark.job_id, mark.kind, mark.check_id) in rows
        ]

    def dismiss(self, mark_id: uuid.UUID, *, now: dt.datetime) -> bool:
        """Dismiss one mark. False, writing nothing, if it is not this user's --
        the same answer whether it does not exist or is someone else's. Already
        dismissed is True and keeps its original time.
        """
        row = self._conn.execute(
            update(marks_table)
            .where(marks_table.c.id == mark_id, marks_table.c.user_id == self._user_id)
            .values(dismissed_at=func.coalesce(marks_table.c.dismissed_at, now))
            .returning(marks_table.c.id)
        ).first()
        return row is not None

    def dismiss_live(self, *, now: dt.datetime, visible_for: dt.timedelta) -> int:
        """Dismiss every mark still keeping an event on the page. An event that
        arrived after the page was rendered has no mark yet, so it is not
        dismissed unseen.
        """
        result = self._conn.execute(
            update(marks_table)
            .where(
                marks_table.c.user_id == self._user_id,
                marks_table.c.first_seen_at > now - visible_for,
                marks_table.c.dismissed_at.is_(None),
            )
            .values(dismissed_at=now)
        )
        return result.rowcount


def purge_stale_marks(conn: Connection, *, now: dt.datetime, visible_for: dt.timedelta) -> int:
    """Delete marks that can no longer affect anything `/changes` can show, for
    every user in one statement. Table only grows otherwise (`NEXT.md`).

    **What a mark is for**, restated because the safe condition falls straight
    out of it: `jfl_intake.feed.visible_events` looks a derived event up by
    `(job_id, kind, check_id)`. No mark at all means "never shown" -- shown as
    new if `event.at > effective_last_looked_at(last_looked_at)`. A mark that
    exists but is not live (dismissed, or `first_seen_at` more than
    `visible_for` ago -- `mark_is_live`) means "already shown and done with" --
    suppressed regardless of that comparison. **A dead mark is a tombstone, and
    deleting one turns "already shown" back into "never shown".** That is the
    resurfacing bug this function must never cause: the same old event
    reappearing as news.

    **Why deleting a dead mark is actually safe.** `record_seen` and
    `set_last_looked_at` are only ever called together, with the same `now`, in
    the one transaction `GET /changes` runs (`jfl_web.routes.changes` -- the only
    production call site of either). A mark's `event_at` is always `<=` the
    `first_seen_at` it is given (an event has already happened before it can be
    shown), and `set_last_looked_at` only ever moves `last_looked_at` forward
    (`GREATEST`). So from the moment any mark for a user is created, that user's
    `last_looked_at` is `>= that mark's event_at`, **forever after** -- and
    `visible_events`'s "no mark -> is it new" check compares straight against
    `last_looked_at`, never against the widened `since` `derive_since` computes
    for the database query. A dead mark's event can therefore never again pass
    `event.at > last_looked_at`, tombstone or none: deleting it is inert.

    **The extra guard below is a deliberate margin, not load off that proof.**
    It refuses to delete a dead mark while another *live* mark for the same user
    has an `event_at` at or before it -- the shared-check case, where one event
    is dismissed and a sibling from the very same check is not. Nothing here
    depends on it for correctness against the code as it stands today (see
    `test_a_dead_mark_purge_correctly_keeps_would_resurface_its_event_if_deleted`,
    which forces the deletion this guard refuses and confirms the event still
    does not resurface). It is kept because it is checkable from this table
    alone, without leaning on `job_feed_state` staying in lock-step forever, and
    because `derive_since` widening `since` for exactly this pairing is the one
    place a future change to `visible_events` -- accepting the widened `since`
    instead of recomputing it -- would make that lean matter. Delaying deletion
    of a mark until its live sibling also dies costs nothing: the row is deleted
    on the very next run where it is safe by both measures.

    Not tenant-scoped (see the module docstring): the correlated subquery
    below still never compares one user's marks against another's --
    `other.c.user_id == marks_table.c.user_id` pins it to the row being
    considered -- so the single statement is exactly the union of what a
    per-user version would have deleted.
    """
    live_after = now - visible_for
    other = marks_table.alias("other_mark")
    blocked_by_a_live_mark = (
        select(1)
        .where(
            other.c.user_id == marks_table.c.user_id,
            other.c.dismissed_at.is_(None),
            other.c.first_seen_at > live_after,
            other.c.event_at <= marks_table.c.event_at,
        )
        .exists()
    )
    is_dead = or_(
        marks_table.c.dismissed_at.is_not(None), marks_table.c.first_seen_at <= live_after
    )
    result = conn.execute(delete(marks_table).where(is_dead, ~blocked_by_a_live_mark))
    return result.rowcount
