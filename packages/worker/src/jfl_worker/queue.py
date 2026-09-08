"""What the worker needs from storage, and how it gets it from Postgres.

Two protocols and two factories. The protocols exist so the loop in `runner.py`
can be exercised without a database -- the queue's SQL is tested against real
Postgres in `tests/`, and the loop's decisions (retry, give up, refuse) are
tested against a fake, where a "worker died mid-task" case is one line rather
than a container kill.

Each scope is one transaction, opened per call and committed on exit. That is
deliberate and load-bearing: **the claim must be committed before the handler
runs.** Holding the claim's transaction open for the two minutes a gate call
takes would keep the row's state invisible to everything else, block the
reclaim sweep behind a lock, and pin a connection for the duration.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from jfl_core.models import ReclaimResult, Task
from jfl_core.storage.tasks import PostgresTaskQueue, PostgresTaskRepository
from sqlalchemy.engine import Engine


class TaskQueue(Protocol):
    """The worker's side of the queue. Mirrors `PostgresTaskQueue`."""

    def claim(self, *, kinds: Sequence[str], now: dt.datetime, limit: int = 1) -> list[Task]: ...

    def mark_succeeded(self, task_id: uuid.UUID, *, now: dt.datetime) -> Task: ...

    def mark_failed(
        self, task_id: uuid.UUID, *, now: dt.datetime, error: str, retry_at: dt.datetime
    ) -> Task: ...

    def release(
        self, task_id: uuid.UUID, *, retry_at: dt.datetime, note: str | None = None
    ) -> Task: ...

    def reclaim_stale(
        self,
        *,
        now: dt.datetime,
        cutoff: dt.datetime,
        retry_at: dt.datetime,
        limit: int = 100,
    ) -> ReclaimResult: ...


class TaskEnqueuer(Protocol):
    """The one write the worker makes as a tenant: its own recurring work.

    Narrower than `PostgresTaskRepository` on purpose -- the worker enqueues
    maintenance and reads nothing back.
    """

    def enqueue_unique(
        self,
        *,
        kind: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int = ...,
        scheduled_at: dt.datetime | None = None,
    ) -> Task | None: ...


QueueScope = Callable[[], AbstractContextManager[TaskQueue]]
EnqueuerScope = Callable[[], AbstractContextManager[TaskEnqueuer]]


def postgres_queue_scope(engine: Engine) -> QueueScope:
    @contextmanager
    def scope() -> Iterator[TaskQueue]:
        with engine.begin() as conn:
            yield PostgresTaskQueue(conn)

    return scope


def postgres_enqueuer_scope(engine: Engine, user_id: uuid.UUID) -> EnqueuerScope:
    """Bound to one user at construction, like every tenancy-scoped repository.

    The worker passes its `system_user_id`; there is no per-call override, here
    or anywhere else.
    """

    @contextmanager
    def scope() -> Iterator[TaskEnqueuer]:
        with engine.begin() as conn:
            yield PostgresTaskRepository(conn, user_id)

    return scope
