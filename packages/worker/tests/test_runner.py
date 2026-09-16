"""The loop's decisions, against an in-memory queue.

The SQL is tested against real Postgres in `tests/test_tasks_queue_integration.py`
and `tests/test_worker_integration.py` -- SKIP LOCKED cannot be proved with a
fake. What is proved here is what the *loop* decides: when it retries, when it
gives up, when it refuses, and what it asks the queue for. `FakeQueue` mirrors
the Postgres semantics that matter to those decisions, above all that `attempts`
increments at CLAIM time.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from jfl_core.models import ReclaimResult, Task
from jfl_worker.log import configure_logging
from jfl_worker.queue import TaskEnqueuer, TaskQueue
from jfl_worker.registry import HandlerRegistry, PermanentTaskError, TaskContext
from jfl_worker.runner import (
    BOARD_SCHEDULE_KIND,
    PURGE_FEED_MARKS_KIND,
    PURGE_SESSIONS_KIND,
    Worker,
)
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

NOW = dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.UTC)


def make_task(
    *,
    kind: str = "thing",
    status: str = "pending",
    attempts: int = 0,
    max_attempts: int = 3,
    scheduled_at: dt.datetime = NOW,
) -> Task:
    return Task(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        kind=kind,
        payload={"secret-looking": "value"},
        status=status,  # type: ignore[arg-type]
        attempts=attempts,
        max_attempts=max_attempts,
        scheduled_at=scheduled_at,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeQueue:
    """In-memory, single-threaded, and faithful on the points that matter."""

    def __init__(self) -> None:
        self.tasks: dict[uuid.UUID, Task] = {}
        self.claim_calls: list[tuple[str, ...]] = []
        self.reclaim_calls: list[dt.datetime] = []

    def add(self, task: Task) -> Task:
        self.tasks[task.id] = task
        return task

    def claim(self, *, kinds: Sequence[str], now: dt.datetime, limit: int = 1) -> list[Task]:
        self.claim_calls.append(tuple(kinds))
        due = sorted(
            (
                t
                for t in self.tasks.values()
                if t.status == "pending" and t.kind in kinds and t.scheduled_at <= now
            ),
            key=lambda t: (t.scheduled_at, t.created_at),
        )
        claimed = []
        for task in due[:limit]:
            updated = task.model_copy(
                update={"status": "running", "attempts": task.attempts + 1, "started_at": now}
            )
            self.tasks[task.id] = updated
            claimed.append(updated)
        return claimed

    def mark_succeeded(self, task_id: uuid.UUID, *, now: dt.datetime) -> Task:
        return self._update(task_id, status="succeeded", finished_at=now)

    def mark_failed(
        self, task_id: uuid.UUID, *, now: dt.datetime, error: str, retry_at: dt.datetime
    ) -> Task:
        task = self.tasks[task_id]
        if task.attempts >= task.max_attempts:
            return self._update(task_id, status="failed", finished_at=now, last_error=error)
        return self._update(
            task_id, status="pending", scheduled_at=retry_at, started_at=None, last_error=error
        )

    def fail_permanently(self, task_id: uuid.UUID, *, now: dt.datetime, error: str) -> Task:
        """Terminal immediately, with attempts left on the clock -- exactly what
        Postgres does, and the point of the distinction being tested.
        """
        return self._update(task_id, status="failed", finished_at=now, last_error=error)

    def release(
        self, task_id: uuid.UUID, *, retry_at: dt.datetime, note: str | None = None
    ) -> Task:
        task = self.tasks[task_id]
        return self._update(
            task_id,
            status="pending",
            attempts=task.attempts - 1,
            scheduled_at=retry_at,
            started_at=None,
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
        self.reclaim_calls.append(cutoff)
        return ReclaimResult()

    def _update(self, task_id: uuid.UUID, **values: Any) -> Task:
        updated = self.tasks[task_id].model_copy(update=values)
        self.tasks[task_id] = updated
        return updated


class FakeEnqueuer:
    def __init__(self, queue: FakeQueue) -> None:
        self._queue = queue
        self.calls: list[str] = []

    def enqueue_unique(
        self,
        *,
        kind: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int = 3,
        scheduled_at: dt.datetime | None = None,
    ) -> Task | None:
        self.calls.append(kind)
        unfinished = any(
            t.kind == kind and t.status in ("pending", "running")
            for t in self._queue.tasks.values()
        )
        if unfinished:
            return None
        return self._queue.add(
            make_task(kind=kind, scheduled_at=scheduled_at or NOW, max_attempts=max_attempts)
        )


@pytest.fixture
def engine() -> Engine:
    """Never connected to. `create_engine` is lazy, and no handler in this file
    touches the database -- the guard is that a test that tried would fail on
    connection refused rather than quietly reaching a real one.
    """
    return create_engine("postgresql+psycopg://nobody@127.0.0.1:1/none")


def build_worker(
    engine: Engine,
    registry: HandlerRegistry,
    queue: FakeQueue,
    *,
    env: Mapping[str, str] | None = None,
    settings: WorkerSettings | None = None,
    stream: io.StringIO | None = None,
    clock: dt.datetime = NOW,
) -> tuple[Worker, FakeEnqueuer]:
    enqueuer = FakeEnqueuer(queue)

    @contextmanager
    def queue_scope() -> Iterator[TaskQueue]:
        yield queue

    @contextmanager
    def enqueuer_scope() -> Iterator[TaskEnqueuer]:
        yield enqueuer

    worker = Worker(
        registry=registry,
        settings=settings or WorkerSettings(database_url="x", poll_interval=0.01),
        queue_scope=queue_scope,
        enqueuer_scope=enqueuer_scope,
        engine=engine,
        clock=lambda: clock,
        env={} if env is None else env,
        logger=configure_logging(stream=stream or io.StringIO()),
    )
    return worker, enqueuer


def test_a_successful_task_is_marked_succeeded(engine: Engine) -> None:
    queue = FakeQueue()
    task = queue.add(make_task())
    calls: list[uuid.UUID] = []

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        calls.append(ctx.task.id)
        return {"rows": 3}

    registry = HandlerRegistry()
    registry.register("thing", handler, calls_model=False)
    worker, _ = build_worker(engine, registry, queue)

    assert worker.run_once() == 1
    assert calls == [task.id]
    assert queue.tasks[task.id].status == "succeeded"
    assert queue.tasks[task.id].attempts == 1


def test_a_permanent_failure_is_terminal_on_the_first_attempt(engine: Engine) -> None:
    """`PermanentTaskError` skips the ladder entirely.

    The case that motivated it: a user with no API key stored. Nothing changes
    between attempts, so three of them would buy twenty minutes of a spinner on
    a screen that should already be saying "add your API key" -- and, for a
    handler that had reached the model, two more charges to be told the same.
    """
    queue = FakeQueue()
    task = queue.add(make_task(max_attempts=3))

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        raise PermanentTaskError("no Anthropic API key stored for this user")

    registry = HandlerRegistry()
    registry.register("thing", handler, calls_model=True)
    stream = io.StringIO()
    worker, _ = build_worker(engine, registry, queue, stream=stream)

    worker.run_once()

    failed = queue.tasks[task.id]
    assert failed.status == "failed"
    assert failed.finished_at == NOW
    # One attempt spent, two left unused: the count records what happened, and
    # giving up is a judgement about the failure's kind, not its number.
    assert failed.attempts == 1
    assert "no Anthropic API key stored" in (failed.last_error or "")

    # And it stays failed: nothing re-claims it, whatever the clock says.
    later = NOW + dt.timedelta(hours=1)
    worker, _ = build_worker(engine, registry, queue, clock=later)
    assert worker.run_once() == 0

    line = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert line["event"] == "task.failed"
    assert line["permanent"] is True


def test_an_ordinary_failure_is_still_retried(engine: Engine) -> None:
    """The distinction is only worth having if the other branch survives."""
    queue = FakeQueue()
    task = queue.add(make_task(max_attempts=3))

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        raise RuntimeError("a 529 from the model API, say")

    registry = HandlerRegistry()
    registry.register("thing", handler, calls_model=True)
    worker, _ = build_worker(engine, registry, queue)

    worker.run_once()

    retrying = queue.tasks[task.id]
    assert retrying.status == "pending"
    assert retrying.scheduled_at > NOW


def test_a_failing_task_retries_with_backoff_then_lands_failed(engine: Engine) -> None:
    """The brief's rule, end to end: attempts cap, `last_error` set, no loop."""
    queue = FakeQueue()
    task = queue.add(make_task(max_attempts=2))

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        raise RuntimeError("handler exploded")

    registry = HandlerRegistry()
    registry.register("thing", handler, calls_model=False)
    settings = WorkerSettings(database_url="x", retry_base=30.0)
    worker, _ = build_worker(engine, registry, queue, settings=settings)

    worker.run_once()
    first = queue.tasks[task.id]
    assert first.status == "pending"
    assert first.attempts == 1
    assert first.scheduled_at == NOW + dt.timedelta(seconds=30)  # backoff, not immediate
    assert "handler exploded" in (first.last_error or "")

    # The retry is not due yet, so the loop finds nothing -- that IS the backoff.
    assert worker.run_once() == 0

    # Wind the clock past the backoff.
    later = NOW + dt.timedelta(minutes=5)
    worker, _ = build_worker(engine, registry, queue, settings=settings, clock=later)
    worker.run_once()
    second = queue.tasks[task.id]
    assert second.status == "failed"
    assert second.attempts == 2 == second.max_attempts
    assert "RuntimeError: handler exploded" in (second.last_error or "")

    # Exhausted: never claimed again, however long anyone waits.
    much_later = NOW + dt.timedelta(days=7)
    worker, _ = build_worker(engine, registry, queue, settings=settings, clock=much_later)
    assert worker.run_once() == 0
    assert queue.tasks[task.id].status == "failed"


def test_the_kill_switch_leaves_a_model_task_pending_and_uncharged(engine: Engine) -> None:
    queue = FakeQueue()
    task = queue.add(make_task(kind="draft_cv"))
    called: list[uuid.UUID] = []

    def handler(ctx: TaskContext) -> None:
        called.append(ctx.task.id)  # pragma: no cover - must never run

    registry = HandlerRegistry()
    registry.register("draft_cv", handler, calls_model=True)
    worker, _ = build_worker(engine, registry, queue, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    assert worker.run_once() == 0
    assert called == []
    assert queue.tasks[task.id].status == "pending"
    assert queue.tasks[task.id].attempts == 0
    # The kind was never even asked for: with nothing runnable, the loop does
    # not reach the queue at all.
    assert queue.claim_calls == []


def test_the_kill_switch_is_re_checked_at_dispatch(engine: Engine) -> None:
    """Thrown between claim and call -- the case the kind filter cannot catch,
    and the reason the brief says "dispatch time, not startup".
    """
    queue = FakeQueue()
    task = queue.add(make_task(kind="draft_cv"))
    called: list[uuid.UUID] = []

    def handler(ctx: TaskContext) -> None:
        called.append(ctx.task.id)  # pragma: no cover - must never run

    registry = HandlerRegistry()
    registry.register("draft_cv", handler, calls_model=True)

    env: dict[str, str] = {}
    worker, _ = build_worker(engine, registry, queue, env=env)

    # A queue whose claim flips the switch mid-flight: the worker has the row,
    # the operator has just pulled the lever.
    real_claim = queue.claim

    def claim_then_pull_the_lever(**kwargs: Any) -> list[Task]:
        claimed = real_claim(**kwargs)
        env["JFL_DISABLE_MODEL_CALLS"] = "1"
        return claimed

    queue.claim = claim_then_pull_the_lever  # type: ignore[method-assign]

    assert worker.run_once() == 1  # one task was claimed
    assert called == []  # and refused rather than run
    released = queue.tasks[task.id]
    assert released.status == "pending"
    assert released.attempts == 0  # refusing is not failing: the attempt is given back
    assert "JFL_DISABLE_MODEL_CALLS" in (released.last_error or "")


def test_a_kind_with_no_handler_is_never_asked_for(engine: Engine) -> None:
    queue = FakeQueue()
    task = queue.add(make_task(kind="from_a_newer_deploy"))
    registry = HandlerRegistry()
    registry.register("thing", lambda ctx: None, calls_model=False)
    worker, _ = build_worker(engine, registry, queue)

    assert worker.run_once() == 0
    assert queue.tasks[task.id].status == "pending"
    assert queue.tasks[task.id].attempts == 0


def test_an_unregistered_kind_in_flight_is_released_not_failed(engine: Engine) -> None:
    """Belt and braces for the rollout case: a task claimed by a worker that
    then finds no handler for it. It goes back with its attempts intact rather
    than burning one on a kind it was never going to be able to run.
    """
    queue = FakeQueue()
    task = queue.add(make_task(kind="thing"))
    worker, _ = build_worker(engine, HandlerRegistry(), queue)

    # Claimed by hand: `run_once` would never claim a kind it has no handler for,
    # which is the whole point -- this is the narrow window where it can happen.
    claimed = queue.claim(kinds=["thing"], now=NOW, limit=1)
    worker._dispatch(claimed[0])

    released = queue.tasks[task.id]
    assert released.status == "pending"
    assert released.attempts == 0
    assert "no handler registered" in (released.last_error or "")


def test_the_session_purge_is_enqueued_on_a_ticker_not_every_poll(engine: Engine) -> None:
    """The scheduling decision, pinned: the worker enqueues the purge on an
    in-process ticker, so it runs through claim/dispatch/retry like any other
    task -- but once an hour, not once every two seconds.
    """
    queue = FakeQueue()
    registry = HandlerRegistry()
    registry.register(PURGE_SESSIONS_KIND, lambda ctx: None, calls_model=False)
    settings = WorkerSettings(database_url="x", purge_interval=3600.0)
    worker, enqueuer = build_worker(engine, registry, queue, settings=settings)

    def purges(calls: list[str]) -> list[str]:
        # The loop has a second recurring kind (the watched-board scheduling
        # pass) on its own ticker; this test is about the purge's.
        return [kind for kind in calls if kind == PURGE_SESSIONS_KIND]

    assert worker.run_once() == 1  # enqueued, then claimed and run
    assert purges(enqueuer.calls) == [PURGE_SESSIONS_KIND]

    worker.run_once()  # same instant: the ticker is not due again
    assert purges(enqueuer.calls) == [PURGE_SESSIONS_KIND]

    two_hours_on, later_enqueuer = build_worker(
        engine, registry, queue, settings=settings, clock=NOW + dt.timedelta(hours=2)
    )
    two_hours_on.run_once()
    assert purges(later_enqueuer.calls) == [PURGE_SESSIONS_KIND]


def test_the_feed_mark_purge_is_enqueued_on_its_own_ticker(engine: Engine) -> None:
    """Same shape as the session purge above: its own ticker, its own interval,
    through `enqueue_unique` so a worker down for a day restarts into one purge.
    """
    queue = FakeQueue()
    registry = HandlerRegistry()
    registry.register(PURGE_FEED_MARKS_KIND, lambda ctx: None, calls_model=False)
    settings = WorkerSettings(database_url="x", feed_mark_purge_interval=3600.0)
    worker, enqueuer = build_worker(engine, registry, queue, settings=settings)

    def ticks() -> int:
        return enqueuer.calls.count(PURGE_FEED_MARKS_KIND)

    worker._run_maintenance(NOW)
    assert ticks() == 1
    worker._run_maintenance(NOW + dt.timedelta(minutes=30))
    assert ticks() == 1  # not due yet
    worker._run_maintenance(NOW + dt.timedelta(hours=1))
    assert ticks() == 2

    scheduled = [t for t in queue.tasks.values() if t.kind == PURGE_FEED_MARKS_KIND]
    assert len(scheduled) == 1  # only one queued at a time


def test_the_board_scheduling_pass_is_enqueued_on_its_own_ticker(engine: Engine) -> None:
    """Every fifteen minutes by default, through `enqueue_unique` like the purge:
    a worker that was down for a day restarts into one scheduling pass.
    """
    queue = FakeQueue()
    registry = HandlerRegistry()
    registry.register(BOARD_SCHEDULE_KIND, lambda ctx: None, calls_model=False)
    settings = WorkerSettings(database_url="x", board_schedule_interval=900.0)
    worker, enqueuer = build_worker(engine, registry, queue, settings=settings)

    def ticks() -> int:
        return enqueuer.calls.count(BOARD_SCHEDULE_KIND)

    worker._run_maintenance(NOW)
    assert ticks() == 1
    worker._run_maintenance(NOW + dt.timedelta(minutes=10))
    assert ticks() == 1  # not due yet
    worker._run_maintenance(NOW + dt.timedelta(minutes=15))
    assert ticks() == 2

    # Only one is ever queued at a time: the second tick found the first pending.
    scheduled = [t for t in queue.tasks.values() if t.kind == BOARD_SCHEDULE_KIND]
    assert len(scheduled) == 1


def test_reclaim_uses_the_visibility_timeout_as_its_cutoff(engine: Engine) -> None:
    queue = FakeQueue()
    registry = HandlerRegistry()
    registry.register("thing", lambda ctx: None, calls_model=False)
    settings = WorkerSettings(database_url="x", visibility_timeout=900.0)
    worker, _ = build_worker(engine, registry, queue, settings=settings)

    worker.run_once()
    assert queue.reclaim_calls == [NOW - dt.timedelta(seconds=900)]


def test_run_forever_stops_after_finishing_the_task_in_hand(engine: Engine) -> None:
    """SIGTERM arrives mid-task: the handler completes, the row is recorded, and
    only then does the process exit.
    """
    queue = FakeQueue()
    task = queue.add(make_task())
    registry = HandlerRegistry()

    holder: dict[str, Worker] = {}

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        holder["worker"].request_stop()  # the SIGTERM
        return {"finished": True}

    registry.register("thing", handler, calls_model=False)
    worker, _ = build_worker(engine, registry, queue)
    holder["worker"] = worker

    worker.run_forever()

    assert queue.tasks[task.id].status == "succeeded"
    assert worker.stopping


def test_a_queue_error_does_not_kill_the_loop(engine: Engine) -> None:
    queue = FakeQueue()
    registry = HandlerRegistry()
    registry.register("thing", lambda ctx: None, calls_model=False)
    stream = io.StringIO()
    settings = WorkerSettings(database_url="x", error_backoff=0.01)
    worker, _ = build_worker(engine, registry, queue, settings=settings, stream=stream)

    calls = {"n": 0}
    real_claim = queue.claim

    def flaky_claim(**kwargs: Any) -> list[Task]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("postgres went away")
        worker.request_stop()
        return real_claim(**kwargs)

    queue.claim = flaky_claim  # type: ignore[method-assign]
    worker.run_forever()

    events = [json.loads(line)["event"] for line in stream.getvalue().strip().splitlines()]
    assert "worker.poll_failed" in events
    assert events[-1] == "worker.stopped"
    assert calls["n"] == 2  # it kept going


def test_a_failure_line_never_carries_the_payload(engine: Engine) -> None:
    queue = FakeQueue()
    queue.add(make_task())
    registry = HandlerRegistry()

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        raise RuntimeError("boom")

    registry.register("thing", handler, calls_model=False)
    stream = io.StringIO()
    worker, _ = build_worker(engine, registry, queue, stream=stream)
    worker.run_once()

    output = stream.getvalue()
    assert "secret-looking" not in output
    assert "task.retrying" in output
