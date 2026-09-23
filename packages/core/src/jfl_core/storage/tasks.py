"""The background job queue: slice B1's storage layer.

Two classes here, and the split between them is the whole design.

`PostgresTaskRepository` is tenancy-scoped in the usual way (constructed with a
user, no per-call override) and is what a *request* touches: enqueue a task,
list mine, read one of mine.

`PostgresTaskQueue` is deliberately **not** tenancy-scoped, and is deliberately
not named `*Repository` so that `tests/test_tenancy_enforcement.py` neither
catches it by accident nor waves it through by accident. It is the worker's
side of the queue: it claims across every tenant, because a worker is a daemon
with no user in context. The argument that this is safe is the same one
`PreAuthRepository` makes for session lookup -- it is the step that *establishes*
the tenant rather than one that runs inside a tenant. A claimed `Task` carries
its `user_id`, and a handler that wants to touch anything must construct a
tenancy-scoped repository with it. What `PostgresTaskQueue` itself may do is
narrow on purpose: move rows between the four queue states. It joins nothing,
reads no user content, and returns no user row.

(Not a `PreAuthRepository` subclass either: that base's docstring restricts it
to identity resolution and session lookup, and a task payload is closer to user
content than to an identity. Widening that exemption to cover a worker would
quietly cost more than naming this what it is.)

**Delivery is at-least-once, not exactly-once.** There is no way to make it
otherwise: between "the handler finished" and "the row says succeeded" there is
a gap, and a `kill -9` or a lost VPS can land in it. So:

  * `attempts` increments at CLAIM time, not at failure time. A task that
    crashes the worker outright still burns an attempt, which is what stops a
    poison pill looping forever;
  * a row left in `running` by a dead worker is reclaimed once its `started_at`
    is older than a visibility timeout the caller supplies;
  * a task that runs out of attempts lands in `failed` with `last_error` set.
    It never disappears and it never retries again;
  * handlers must therefore be safe to run twice.

The transaction boundary belongs to the caller here, same as every other
repository in this package -- but for `claim` that is not a style choice.
`SELECT ... FOR UPDATE SKIP LOCKED` only holds its locks inside a transaction,
and the worker must COMMIT the claim before running the handler: holding the
claim open for the two minutes a gate call takes would block the reclaim sweep,
hide the row's state from every other reader, and turn one long task into one
long-held connection.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Connection

from jfl_core.db.tables import tasks as tasks_table
from jfl_core.models import ReclaimResult, Task, TaskStatus
from jfl_core.storage.tenancy import TenantScopedRepository

# Three attempts: enough that a transient 529 from the model API or a Postgres
# restart does not lose the work, few enough that a genuinely broken task
# reaches a human today rather than in a week.
DEFAULT_MAX_ATTEMPTS = 3

# Statuses a task can still be waiting in. Used for "is one of these already
# queued?" -- see `enqueue_unique`.
UNFINISHED_STATUSES: tuple[TaskStatus, ...] = ("pending", "running")

_TASK_COLUMNS = (
    tasks_table.c.id,
    tasks_table.c.user_id,
    tasks_table.c.kind,
    tasks_table.c.payload,
    tasks_table.c.status,
    tasks_table.c.attempts,
    tasks_table.c.max_attempts,
    tasks_table.c.last_error,
    tasks_table.c.scheduled_at,
    tasks_table.c.started_at,
    tasks_table.c.finished_at,
    tasks_table.c.created_at,
    tasks_table.c.updated_at,
)

# Postgres rejects a NUL byte in text, and a truncated traceback is far more
# useful than a row that would not save. Both are applied to `last_error` only;
# nothing here ever writes a payload into an error.
_MAX_ERROR_CHARS = 4000


def _clean_error(error: str) -> str:
    text = error.replace("\x00", "")
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    return text[: _MAX_ERROR_CHARS - 15] + "... [truncated]"


def _task_from_row(row: Any) -> Task:
    return Task(
        id=row.id,
        user_id=row.user_id,
        kind=row.kind,
        payload=row.payload or {},
        status=row.status,
        attempts=row.attempts,
        max_attempts=row.max_attempts,
        last_error=row.last_error,
        scheduled_at=row.scheduled_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class TaskNotFoundError(RuntimeError):
    """No task with this id, or not one this caller may act on.

    Same message either way, for the same reason
    `ApplicationNotFoundError` gives: telling a caller which ids exist is itself
    a leak.
    """

    def __init__(self, task_id: uuid.UUID) -> None:
        super().__init__(f"no task {task_id}")


class PostgresTaskRepository(TenantScopedRepository):
    """Enqueue and inspect background work for exactly one user."""

    def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        scheduled_at: dt.datetime | None = None,
    ) -> Task:
        """Queue a task. `scheduled_at` in the future delays its first attempt.

        `payload` is arguments only -- ids, flags. Never a secret: it is read
        back by the worker and shown in admin queries, and a credential belongs
        in `user_credentials`, encrypted.
        """
        values: dict[str, Any] = {
            "id": uuid.uuid4(),
            "user_id": self._user_id,
            "kind": kind,
            "payload": payload or {},
            "status": "pending",
            "max_attempts": max_attempts,
        }
        if scheduled_at is not None:
            values["scheduled_at"] = scheduled_at
        row = self._conn.execute(
            insert(tasks_table).values(**values).returning(*_TASK_COLUMNS)
        ).one()
        return _task_from_row(row)

    def enqueue_unique(
        self,
        *,
        kind: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        scheduled_at: dt.datetime | None = None,
    ) -> Task | None:
        """Queue a task only if one of this kind is not already pending or
        running for this user. Returns None if it was skipped.

        For recurring maintenance, where a backlog is worse than a skipped
        tick: an hourly session purge that has not run for six hours wants one
        purge, not six.

        Deliberately a read then a write rather than one clever statement.
        Under READ COMMITTED two callers racing here can both see nothing and
        both insert, so this is a duplicate *suppressor*, not a lock -- which is
        all it needs to be, because the handlers it guards are idempotent and
        at-least-once delivery already means they must be.
        """
        existing = self._conn.execute(
            select(tasks_table.c.id)
            .where(
                tasks_table.c.user_id == self._user_id,
                tasks_table.c.kind == kind,
                tasks_table.c.status.in_(UNFINISHED_STATUSES),
            )
            .limit(1)
        ).first()
        if existing is not None:
            return None
        return self.enqueue(
            kind=kind, payload=payload, max_attempts=max_attempts, scheduled_at=scheduled_at
        )

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """Newest first -- "what is happening to my application right now"."""
        query = (
            select(*_TASK_COLUMNS)
            .where(tasks_table.c.user_id == self._user_id)
            .order_by(tasks_table.c.created_at.desc())
            .limit(limit)
        )
        if status is not None:
            query = query.where(tasks_table.c.status == status)
        if kind is not None:
            query = query.where(tasks_table.c.kind == kind)
        return [_task_from_row(row) for row in self._conn.execute(query).all()]

    def get_task(self, task_id: uuid.UUID) -> Task | None:
        row = self._conn.execute(
            select(*_TASK_COLUMNS).where(
                tasks_table.c.id == task_id,
                tasks_table.c.user_id == self._user_id,
            )
        ).first()
        return None if row is None else _task_from_row(row)

    def follow_up(self, task_id: uuid.UUID) -> Task | None:
        """The task queued as the next step after `task_id`, if one was.

        A chained task carries `"after": "<the task before it>"` in its
        payload -- see `jfl_worker.chain`. This is how the worker avoids
        queueing the next step twice when a task is redelivered, and how the
        drafting screen follows one button press through its steps.
        """
        row = self._conn.execute(
            select(*_TASK_COLUMNS)
            .where(
                tasks_table.c.user_id == self._user_id,
                tasks_table.c.payload["after"].astext == str(task_id),
            )
            .order_by(tasks_table.c.created_at)
            .limit(1)
        ).first()
        return None if row is None else _task_from_row(row)


class PostgresTaskQueue:
    """The worker's side of the queue. Cross-tenant by necessity; see the module
    docstring for why that is safe and why it is not a repository.

    Takes a `Connection` and never opens or commits a transaction: the caller
    owns that boundary, and for `claim` it must, because SKIP LOCKED's locks
    live and die with the transaction.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def claim(
        self,
        *,
        kinds: Sequence[str],
        now: dt.datetime,
        limit: int = 1,
    ) -> list[Task]:
        """Take up to `limit` due tasks, marking them `running`.

        `SELECT ... FOR UPDATE SKIP LOCKED` is the whole point: two workers
        polling at the same instant take *different* rows, and a row another
        worker is already holding is stepped over rather than waited on -- so a
        two-minute gate call never blocks a one-second task behind it.

        Only `kinds` this worker actually has a handler for are claimed. That
        makes an unrecognised kind harmless: it stays `pending` until a worker
        that knows it is deployed, instead of being claimed and failed by an
        older container mid-rollout.
        """
        if not kinds:
            return []

        locked = (
            self._conn.execute(
                select(tasks_table.c.id)
                .where(
                    tasks_table.c.status == "pending",
                    tasks_table.c.kind.in_(list(kinds)),
                    tasks_table.c.scheduled_at <= now,
                )
                .order_by(tasks_table.c.scheduled_at.asc(), tasks_table.c.created_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )
        if not locked:
            return []

        rows = self._conn.execute(
            update(tasks_table)
            .where(tasks_table.c.id.in_(list(locked)))
            .values(
                status="running",
                # At claim time, not at failure time -- a worker that dies mid-task
                # has still used an attempt. See the module docstring.
                attempts=tasks_table.c.attempts + 1,
                started_at=now,
                finished_at=None,
            )
            .returning(*_TASK_COLUMNS)
        ).all()
        tasks = [_task_from_row(row) for row in rows]
        # UPDATE ... RETURNING makes no ordering promise; restore the queue order
        # the SELECT chose so a batch is handled oldest-first.
        tasks.sort(key=lambda t: (t.scheduled_at, t.created_at))
        return tasks

    def mark_succeeded(self, task_id: uuid.UUID, *, now: dt.datetime) -> Task:
        """`last_error` is left alone on purpose: on a task that failed twice and
        then worked, it is the record of what went wrong.
        """
        return self._transition(
            task_id,
            status="succeeded",
            finished_at=now,
        )

    def mark_failed(
        self,
        task_id: uuid.UUID,
        *,
        now: dt.datetime,
        error: str,
        retry_at: dt.datetime,
    ) -> Task:
        """Record a failed attempt, and either schedule the retry or give up.

        Giving up is explicit and visible: `failed`, `finished_at` set,
        `last_error` holding the reason. A task never vanishes and never loops.
        `retry_at` is the caller's backoff -- policy lives in the worker, not in
        SQL.
        """
        row = self._conn.execute(
            select(tasks_table.c.attempts, tasks_table.c.max_attempts).where(
                tasks_table.c.id == task_id
            )
        ).first()
        if row is None:
            raise TaskNotFoundError(task_id)

        exhausted = row.attempts >= row.max_attempts
        if exhausted:
            return self._transition(
                task_id,
                status="failed",
                finished_at=now,
                last_error=_clean_error(error),
            )
        return self._transition(
            task_id,
            status="pending",
            scheduled_at=retry_at,
            started_at=None,
            finished_at=None,
            last_error=_clean_error(error),
        )

    def fail_permanently(self, task_id: uuid.UUID, *, now: dt.datetime, error: str) -> Task:
        """Give up now, with attempts left on the clock.

        For a failure a retry cannot fix: the user has no API key stored, the
        payload names an application that is not theirs, the model refused. The
        backoff ladder exists to ride out a transient 529 or a Postgres restart;
        spending three attempts and twenty minutes to rediscover a missing key
        is noise in the log and a worse answer on the screen, and where the
        handler did reach the model it would be paying twice more to be told the
        same thing.

        `attempts` is left exactly as it is: it records what happened, and this
        is a decision about the failure's kind, not its count.
        """
        return self._transition(
            task_id,
            status="failed",
            finished_at=now,
            last_error=_clean_error(error),
        )

    def release(
        self,
        task_id: uuid.UUID,
        *,
        retry_at: dt.datetime,
        note: str | None = None,
    ) -> Task:
        """Put a claimed task back, WITHOUT spending the attempt.

        This is the kill switch's path (and shutdown's). Refusing to run a task
        because an operator disabled model calls is not the task failing, and
        charging it an attempt would quietly fail every queued task while the
        switch was on -- turning an incident lever into data loss.
        """
        return self._transition(
            task_id,
            status="pending",
            attempts=tasks_table.c.attempts - 1,
            scheduled_at=retry_at,
            started_at=None,
            finished_at=None,
            last_error=note,
        )

    def reclaim_stale(
        self,
        *,
        now: dt.datetime,
        cutoff: dt.datetime,
        retry_at: dt.datetime,
        limit: int = 100,
    ) -> ReclaimResult:
        """Rescue rows a dead worker left in `running`.

        This is at-least-once delivery made concrete. A worker that is SIGKILLed
        mid-task (Docker does exactly that, `stop_grace_period` seconds after
        SIGTERM) commits nothing, so the row sits `running` with nobody running
        it. Anything whose `started_at` is older than `cutoff` -- the visibility
        timeout, chosen by the caller to be comfortably longer than the slowest
        task -- goes back to `pending`, or to `failed` if its attempts are gone.

        A live worker on a genuinely slow task will be reclaimed if it exceeds
        the timeout, and its work will then run twice. That is the trade the
        whole design is built on, and the reason handlers must be idempotent.
        """
        stale = (
            self._conn.execute(
                select(tasks_table.c.id)
                .where(
                    tasks_table.c.status == "running",
                    tasks_table.c.started_at <= cutoff,
                )
                .order_by(tasks_table.c.started_at.asc())
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )
        if not stale:
            return ReclaimResult()

        stale_ids = list(stale)
        message = f"reclaimed: still 'running' at {now.isoformat()}, worker presumed dead"

        failed = (
            self._conn.execute(
                update(tasks_table)
                .where(
                    tasks_table.c.id.in_(stale_ids),
                    tasks_table.c.attempts >= tasks_table.c.max_attempts,
                )
                .values(status="failed", finished_at=now, last_error=_clean_error(message))
                .returning(tasks_table.c.id)
            )
            .scalars()
            .all()
        )
        requeued = (
            self._conn.execute(
                update(tasks_table)
                .where(
                    tasks_table.c.id.in_(stale_ids),
                    tasks_table.c.attempts < tasks_table.c.max_attempts,
                )
                .values(
                    status="pending",
                    scheduled_at=retry_at,
                    started_at=None,
                    finished_at=None,
                    last_error=_clean_error(message),
                )
                .returning(tasks_table.c.id)
            )
            .scalars()
            .all()
        )
        return ReclaimResult(requeued=list(requeued), failed=list(failed))

    def _transition(self, task_id: uuid.UUID, **values: Any) -> Task:
        """One UPDATE, one row, or `TaskNotFoundError`.

        `updated_at` is absent from every caller's `values` on purpose: the
        column's `onupdate=func.now()` in tables.py fills it in for any UPDATE
        built from this table that does not set it, so it cannot be forgotten.
        """
        row = self._conn.execute(
            update(tasks_table)
            .where(tasks_table.c.id == task_id)
            .values(**values)
            .returning(*_TASK_COLUMNS)
        ).first()
        if row is None:
            raise TaskNotFoundError(task_id)
        return _task_from_row(row)
