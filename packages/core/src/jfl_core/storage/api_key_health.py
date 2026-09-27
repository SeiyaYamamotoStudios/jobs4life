"""Whether one user's Anthropic account is accepting model calls.

Tenancy-scoped like every repository that touches a user's data: constructed
with the user, no per-call override. The worker writes it (a blocked call, a
successful one); the web reads it for the banner and resets it when the user
saves a new key.

Stores a category and two timestamps. Never an error message -- see
`jfl_core.model_api` for why SDK text stays out of the database.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import case, select, update
from sqlalchemy.dialects.postgresql import insert

from jfl_core.db.tables import api_key_health as health_table
from jfl_core.model_api import AccountBlock
from jfl_core.models import ApiKeyHealth
from jfl_core.storage.tenancy import TenantScopedRepository


class PostgresApiKeyHealthRepository(TenantScopedRepository):
    """One row per user, keyed by `user_id`. No row means `ok`: nothing has
    ever told us otherwise."""

    def get(self) -> ApiKeyHealth | None:
        row = self._conn.execute(
            select(health_table.c.status, health_table.c.since, health_table.c.checked_at).where(
                health_table.c.user_id == self._user_id
            )
        ).first()
        if row is None:
            return None
        return ApiKeyHealth(status=row.status, since=row.since, checked_at=row.checked_at)

    def mark_blocked(self, block: AccountBlock, *, now: dt.datetime) -> None:
        """Record that a model call was refused for an account-level reason.

        `since` moves only when the status changes, so "credits ran out three
        hours ago" stays three hours rather than resetting on every probe.
        """
        stmt = insert(health_table).values(
            user_id=self._user_id, status=block, since=now, checked_at=now
        )
        self._conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[health_table.c.user_id],
                set_={
                    "status": block,
                    "checked_at": now,
                    "since": _since_if_changed(stmt.excluded.status, now),
                },
            )
        )

    def mark_ok(self, *, now: dt.datetime) -> bool:
        """Clear a blocked state after a model call succeeded. One UPDATE that
        matches nothing in the normal case -- no row, or a row already `ok` --
        so calling it after every successful call costs an index probe.

        Returns whether anything changed.
        """
        result = self._conn.execute(
            update(health_table)
            .where(health_table.c.user_id == self._user_id, health_table.c.status != "ok")
            .values(status="ok", since=now, checked_at=now)
        )
        return result.rowcount > 0


def _since_if_changed(new_status: object, now: dt.datetime) -> object:
    """`since` for an upsert: unchanged when the status is the same, `now` when
    it moved."""
    return case(
        (health_table.c.status == new_status, health_table.c.since),
        else_=now,
    )
