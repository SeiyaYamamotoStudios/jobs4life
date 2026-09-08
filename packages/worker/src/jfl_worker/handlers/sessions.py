"""Session maintenance: delete session rows whose expiry has passed.

The first real handler, chosen because it needs no model call at all -- the
queue can be proved end to end, in production, for zero API spend.

It is housekeeping and not a security boundary: `PostgresSessionRepository.lookup`
already refuses an expired row, so an unpurged session authenticates nobody. What
this stops is the table growing forever, and what it removes is a hashed token
that has already stopped working.

Idempotent by construction -- a DELETE with a WHERE that matches nothing the
second time -- which is what at-least-once delivery requires of every handler.

`PostgresSessionRepository` is a `PreAuthRepository`, so this is not a tenancy
hole: expired sessions are global maintenance and belong to no one user. The
task row itself is owned by the seeded local user because `tasks.user_id` is NOT
NULL and there is no unowned path into that table.
"""

from __future__ import annotations

from collections.abc import Mapping

from jfl_core.storage.accounts import PostgresSessionRepository

from jfl_worker.registry import TaskContext

KIND = "purge_expired_sessions"


def purge_expired_sessions(ctx: TaskContext) -> Mapping[str, object]:
    with ctx.engine.begin() as conn:
        deleted = PostgresSessionRepository(conn).purge_expired(now=ctx.now)
    return {"sessions_deleted": deleted}
