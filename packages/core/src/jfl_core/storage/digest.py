"""The emailed digest's storage, tenancy-scoped (PLAN.md C9).

Two classes, split the way `jfl_core.storage.boards` and `tasks` split theirs.

`PostgresDigestRepository` is a `TenantScopedRepository`: one user's opt-in, cursor
and send log. It touches `digest_settings` and `email_sends` and **nothing else it
could be confused with** -- in particular not `job_feed_state` or `job_feed_marks`.
That separation is the point of the feature's fourth rule: the email is independent
of the feed, so sending one must not make `/changes` believe it has been read, and
reading `/changes` must not make the next email skip what it showed. Nothing here
imports `jfl_core.storage.job_feed`, and a test holds the feed tables unchanged
across a send.

`PostgresDigestScheduler` is deliberately **not** tenancy-scoped and not named
`*Repository`, for the reason `PostgresBoardScheduler` gives: the recurring
scheduling pass is a daemon with no user in context, and has to find which users'
digests are due, across tenants. It returns user ids and nothing else -- no
address, no board, no event -- and whatever touches content is done by
constructing a repository with an id it returned.

**Advancing the cursor and recording the send are one call** (`complete_run`),
made by the handler in one transaction after the mail server has accepted the
message; see `jfl_worker.handlers.digest` for the duplicate window that leaves.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Collection
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from jfl_core.db.tables import digest_settings as settings_table
from jfl_core.db.tables import email_sends as sends_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import DigestState, EmailSend, EmailSendKind
from jfl_core.storage.tenancy import TenantScopedRepository

_SEND_COLUMNS = (
    sends_table.c.id,
    sends_table.c.kind,
    sends_table.c.sent_at,
    sends_table.c.jobs_listed,
    sends_table.c.events_found,
    sends_table.c.boards_failed,
)


def _send_from_row(row: Any) -> EmailSend:
    return EmailSend(
        id=row.id,
        kind=row.kind,
        sent_at=row.sent_at,
        jobs_listed=row.jobs_listed,
        events_found=row.events_found,
        boards_failed=row.boards_failed,
    )


class PostgresDigestRepository(TenantScopedRepository):
    """One user's digest opt-in, cursor and send log, and no one else's."""

    # -- state ---------------------------------------------------------------

    def get_state(self) -> DigestState | None:
        """None until the user has ever opted in."""
        row = self._conn.execute(
            select(
                settings_table.c.enabled,
                settings_table.c.opted_in_at,
                settings_table.c.notified_up_to,
                settings_table.c.last_evaluated_at,
            ).where(settings_table.c.user_id == self._user_id)
        ).first()
        if row is None:
            return None
        return DigestState(
            enabled=row.enabled,
            opted_in_at=row.opted_in_at,
            notified_up_to=row.notified_up_to,
            last_evaluated_at=row.last_evaluated_at,
        )

    def enable(self, *, now: dt.datetime) -> bool:
        """Opt in. The cursor and the evaluated-at time both start at `now`, so
        the first digest covers the time since opting in and never the history.
        Returns False, changing nothing, if already opted in -- a double click
        or a retried request must not move a cursor that has moved on since.
        """
        statement = pg_insert(settings_table).values(
            id=uuid.uuid4(),
            user_id=self._user_id,
            enabled=True,
            opted_in_at=now,
            notified_up_to=now,
            last_evaluated_at=now,
        )
        result = self._conn.execute(
            statement.on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "enabled": True,
                    "opted_in_at": now,
                    "notified_up_to": now,
                    "last_evaluated_at": now,
                    "updated_at": func.now(),
                },
                where=settings_table.c.enabled.is_(False),
            ).returning(settings_table.c.id)
        )
        # RETURNING rather than `rowcount`: on an INSERT ... ON CONFLICT the
        # driver does not report a usable rowcount, and a row comes back only
        # when the insert or the update actually happened.
        return result.first() is not None

    def disable(self) -> bool:
        """Opt out. True if it was on. Leaves the cursor alone; `enable` resets
        it, so switching back on never mails the weeks it was off.
        """
        result = self._conn.execute(
            update(settings_table)
            .where(settings_table.c.user_id == self._user_id, settings_table.c.enabled.is_(True))
            .values(enabled=False, updated_at=func.now())
        )
        return result.rowcount > 0

    def complete_run(
        self,
        *,
        evaluated_at: dt.datetime,
        notified_up_to: dt.datetime,
        sent: EmailSend | None,
    ) -> bool:
        """Finish one digest run: move the cursor to `notified_up_to`, mark the
        run evaluated at `evaluated_at`, and -- if an email was sent -- log it.
        All in the caller's transaction.

        The cursor only moves forward (`GREATEST`), like the feed's last-looked
        time, so a slow run finishing late cannot re-expose events a later one
        already covered. Returns False, writing nothing, if the user is not
        opted in: they unsubscribed while the task ran, and the send record for
        a mail they asked not to get is still true -- so it IS logged -- but the
        cursor of a disabled digest is not advanced behind their back.
        """
        if sent is not None:
            self._conn.execute(
                pg_insert(sends_table).values(
                    id=sent.id,
                    user_id=self._user_id,
                    kind=sent.kind,
                    sent_at=sent.sent_at,
                    jobs_listed=sent.jobs_listed,
                    events_found=sent.events_found,
                    boards_failed=sent.boards_failed,
                )
            )
        result = self._conn.execute(
            update(settings_table)
            .where(settings_table.c.user_id == self._user_id, settings_table.c.enabled.is_(True))
            .values(
                notified_up_to=func.greatest(
                    func.coalesce(settings_table.c.notified_up_to, notified_up_to),
                    notified_up_to,
                ),
                last_evaluated_at=func.greatest(
                    func.coalesce(settings_table.c.last_evaluated_at, evaluated_at),
                    evaluated_at,
                ),
                updated_at=func.now(),
            )
        )
        return result.rowcount > 0

    # -- the send log --------------------------------------------------------

    def record_send(self, send: EmailSend) -> None:
        """Log an email that does not touch the cursor -- the test email."""
        self._conn.execute(
            pg_insert(sends_table).values(
                id=send.id,
                user_id=self._user_id,
                kind=send.kind,
                sent_at=send.sent_at,
                jobs_listed=send.jobs_listed,
                events_found=send.events_found,
                boards_failed=send.boards_failed,
            )
        )

    def last_send(self, kinds: tuple[EmailSendKind, ...] | None = None) -> EmailSend | None:
        """The most recent email of any (or the given) kind."""
        query = (
            select(*_SEND_COLUMNS)
            .where(sends_table.c.user_id == self._user_id)
            .order_by(sends_table.c.sent_at.desc())
            .limit(1)
        )
        if kinds is not None:
            query = query.where(sends_table.c.kind.in_(kinds))
        row = self._conn.execute(query).first()
        return None if row is None else _send_from_row(row)

    # -- the one user-table read this feature needs -----------------------------

    def recipient_email(self) -> str | None:
        """The address on this user's own account row (Google-verified, refreshed
        at every login). The only column of `users` read here, and only for the
        user this repository is bound to.
        """
        value = self._conn.execute(
            select(users_table.c.email).where(
                users_table.c.id == self._user_id, users_table.c.is_active.is_(True)
            )
        ).scalar_one_or_none()
        return value or None


class PostgresDigestScheduler:
    """The scheduling pass's view of who is due a digest, across tenants. See the
    module docstring for why that is safe and why it is not a repository.

    Takes a `Connection` and never opens or commits a transaction.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def due_users(
        self,
        *,
        evaluated_before: dt.datetime,
        retry_not_before: dt.datetime,
        task_kind: str,
        limit: int,
        only_users: Collection[uuid.UUID] | None = None,
    ) -> list[uuid.UUID]:
        """Opted-in, active users with an address whose last completed run is
        older than `evaluated_before` (the most recent digest hour that has
        passed), and who have had no `task_kind` task created since
        `retry_not_before`.

        `only_users` confines the pass to those users (tests pass the ones they
        created, so a pass under test cannot pick up someone else's opt-in);
        production passes None.

        The second condition is a back-off, not a correctness rule: a user whose
        sends keep failing (a wrong SMTP password) would otherwise be re-enqueued
        by every pass, each with its own retries. One try an hour is enough to
        notice a fix and cheap enough to leave running.
        """
        recent_task = (
            select(tasks_table.c.id)
            .where(
                tasks_table.c.user_id == settings_table.c.user_id,
                tasks_table.c.kind == task_kind,
                tasks_table.c.created_at > retry_not_before,
            )
            .exists()
        )
        confine = [] if only_users is None else [settings_table.c.user_id.in_(list(only_users))]
        rows = self._conn.execute(
            select(settings_table.c.user_id)
            .select_from(
                settings_table.join(users_table, users_table.c.id == settings_table.c.user_id)
            )
            .where(
                *confine,
                settings_table.c.enabled.is_(True),
                users_table.c.is_active.is_(True),
                users_table.c.email.is_not(None),
                or_(
                    settings_table.c.last_evaluated_at.is_(None),
                    settings_table.c.last_evaluated_at < evaluated_before,
                ),
                ~recent_task,
            )
            .order_by(settings_table.c.last_evaluated_at.asc().nulls_first())
            .limit(limit)
        ).all()
        return [r.user_id for r in rows]


__all__ = ["PostgresDigestRepository", "PostgresDigestScheduler"]
