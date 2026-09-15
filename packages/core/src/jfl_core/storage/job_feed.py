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
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

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
