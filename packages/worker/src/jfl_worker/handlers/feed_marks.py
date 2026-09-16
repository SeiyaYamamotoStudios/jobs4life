"""Feed-mark maintenance: delete `job_feed_marks` rows that can no longer affect
what `/changes` shows anyone.

Modelled exactly on `jfl_worker.handlers.sessions.purge_expired_sessions` -- the
same reasons apply verbatim. It needs no model call, so `JFL_DISABLE_MODEL_CALLS`
must not hold it: the kill switch is for spending a user's key, and this spends
none. It is housekeeping, not a correctness boundary -- `jfl_intake.feed` already
treats a dismissed or aged-out mark as "already shown, stop showing it" without
caring whether the row survives; what this stops is the table growing forever.

Idempotent by construction: a DELETE whose WHERE matches nothing the second
time, which is what at-least-once delivery requires of every handler.

The predicate itself -- which marks are provably safe to delete without ever
resurfacing an old event as news -- lives in
`jfl_core.storage.job_feed.purge_stale_marks`, next to the marks it deletes and
the feed rule it must not contradict.
"""

from __future__ import annotations

from collections.abc import Mapping

from jfl_core.storage.job_feed import purge_stale_marks
from jfl_intake.feed import VISIBLE_FOR

from jfl_worker.registry import TaskContext

KIND = "purge_stale_feed_marks"


def purge_stale_feed_marks(ctx: TaskContext) -> Mapping[str, object]:
    with ctx.engine.begin() as conn:
        deleted = purge_stale_marks(conn, now=ctx.now, visible_for=VISIBLE_FOR)
    return {"feed_marks_deleted": deleted}
