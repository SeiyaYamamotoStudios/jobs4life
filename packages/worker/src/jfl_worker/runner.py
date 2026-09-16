"""The poll loop.

Shape of one iteration:

  1. maintenance, if due -- reclaim rows a dead worker left `running`, and
     enqueue the three recurring tasks: the session purge, the feed-mark purge,
     and the watched-board scheduling pass;
  2. claim up to `batch_size` due tasks of the kinds this worker can run;
  3. dispatch each one, recording success or failure;
  4. if nothing was claimed, sleep for `poll_interval`.

Three properties worth stating plainly, because they are what makes this
correct rather than merely working:

**Delivery is at-least-once.** The claim is committed before the handler runs,
so a worker killed mid-task leaves the row `running` with nobody running it.
That is not a bug to be designed away -- it is the only honest answer available,
because "the handler finished" and "the row says succeeded" are two events with
a gap between them. `reclaim_stale` closes it after the visibility timeout, and
handlers must be safe to run twice.

**Shutdown is graceful, up to a point.** SIGTERM sets a flag; the loop finishes
the task in hand and exits before claiming another. Docker sends SIGTERM and
then SIGKILL `stop_grace_period` later, so a task longer than that grace is
killed anyway -- which is the at-least-once case above, and why the compose
service sets a grace period longer than the longest expected task.

**The kill switch is checked twice.** Once when choosing which kinds to ask for
(so a disabled model task is never claimed and stays `pending`), and again
immediately before dispatch (so a switch thrown mid-flight releases the task
instead of spending on it). The second check is the one the brief demands:
at dispatch, not at startup.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping

from jfl_core.models import Task
from jfl_intake.scheduling import SCHEDULE_BOARD_CHECKS_KIND
from sqlalchemy.engine import Engine

from jfl_worker.log import LOGGER_NAME, log_event
from jfl_worker.queue import EnqueuerScope, QueueScope
from jfl_worker.registry import HandlerRegistry, PermanentTaskError, TaskContext
from jfl_worker.settings import WorkerSettings, model_calls_disabled

PURGE_SESSIONS_KIND = "purge_expired_sessions"
PURGE_FEED_MARKS_KIND = "purge_stale_feed_marks"
BOARD_SCHEDULE_KIND = SCHEDULE_BOARD_CHECKS_KIND


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


class Worker:
    """One process, one loop, tasks run serially.

    Collaborators are injected rather than constructed: `main.py` wires the
    Postgres ones, tests wire fakes. `env` is the mapping the kill switch reads;
    None means the real environment.
    """

    def __init__(
        self,
        *,
        registry: HandlerRegistry,
        settings: WorkerSettings,
        queue_scope: QueueScope,
        enqueuer_scope: EnqueuerScope,
        engine: Engine,
        clock: Callable[[], dt.datetime] = _utcnow,
        env: Mapping[str, str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings
        self._queue_scope = queue_scope
        self._enqueuer_scope = enqueuer_scope
        self._engine = engine
        self._clock = clock
        self._env = env
        self._log = logger or logging.getLogger(LOGGER_NAME)
        # An Event, not a bool: `wait()` is the sleep, so SIGTERM during an idle
        # poll wakes the loop immediately instead of after `poll_interval`.
        self._stop = threading.Event()
        # None means "due now". On the first iteration after a restart the
        # worker therefore sweeps for orphans and enqueues a purge straight
        # away, which is exactly the moment both are most likely to be needed.
        self._next_reclaim_at: dt.datetime | None = None
        self._next_purge_at: dt.datetime | None = None
        self._next_feed_mark_purge_at: dt.datetime | None = None
        self._next_board_schedule_at: dt.datetime | None = None

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self) -> None:
        """Signal-handler safe: sets a flag and returns."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run_forever(self) -> None:
        log_event(
            self._log,
            logging.INFO,
            "worker.started",
            kinds=list(self._registry.kinds()),
            poll_interval=self._settings.poll_interval,
            visibility_timeout=self._settings.visibility_timeout,
            model_calls_disabled=model_calls_disabled(self._env),
        )
        while not self._stop.is_set():
            try:
                handled = self.run_once()
            except Exception:
                # A poll failure -- Postgres restarting, most likely -- must not
                # end the process: the container would restart into the same
                # outage, and the backoff below is gentler than Docker's.
                log_event(self._log, logging.ERROR, "worker.poll_failed", exc_info=True)
                self._stop.wait(self._settings.error_backoff)
                continue
            if handled == 0:
                self._stop.wait(self._settings.poll_interval)
        log_event(self._log, logging.INFO, "worker.stopped")

    # -- one iteration -----------------------------------------------------

    def run_once(self) -> int:
        """Do at most one batch of work. Returns how many tasks were dispatched."""
        now = self._clock()
        self._run_maintenance(now)

        kinds = self._registry.runnable_kinds(allow_model_calls=not model_calls_disabled(self._env))
        if not kinds:
            return 0

        with self._queue_scope() as queue:
            claimed = queue.claim(kinds=kinds, now=now, limit=self._settings.batch_size)

        # Claimed tasks are dispatched even if a stop arrived in between: the
        # rows are already `running` with an attempt spent, and finishing them
        # is cheaper than leaving them for the reclaim sweep. With
        # `batch_size == 1` that is one task, which is what "finish the current
        # task, then exit" means.
        for task in claimed:
            self._dispatch(task)
        return len(claimed)

    def _run_maintenance(self, now: dt.datetime) -> None:
        if self._next_reclaim_at is None or now >= self._next_reclaim_at:
            self._reclaim(now)
            self._next_reclaim_at = now + dt.timedelta(seconds=self._settings.reclaim_interval)
        if self._next_purge_at is None or now >= self._next_purge_at:
            self._enqueue_purge(now)
            self._next_purge_at = now + dt.timedelta(seconds=self._settings.purge_interval)
        if self._next_feed_mark_purge_at is None or now >= self._next_feed_mark_purge_at:
            self._enqueue_recurring(PURGE_FEED_MARKS_KIND, now)
            self._next_feed_mark_purge_at = now + dt.timedelta(
                seconds=self._settings.feed_mark_purge_interval
            )
        if self._next_board_schedule_at is None or now >= self._next_board_schedule_at:
            self._enqueue_recurring(BOARD_SCHEDULE_KIND, now)
            self._next_board_schedule_at = now + dt.timedelta(
                seconds=self._settings.board_schedule_interval
            )

    def _reclaim(self, now: dt.datetime) -> None:
        cutoff = now - dt.timedelta(seconds=self._settings.visibility_timeout)
        with self._queue_scope() as queue:
            result = queue.reclaim_stale(now=now, cutoff=cutoff, retry_at=now)
        if not result:
            return
        # WARNING, not INFO: every row here is a task that was running when
        # something killed its worker. One is a deploy; a steady trickle is a
        # handler crashing the process.
        log_event(
            self._log,
            logging.WARNING,
            "queue.reclaimed_stale",
            requeued=len(result.requeued),
            failed=len(result.failed),
            visibility_timeout=self._settings.visibility_timeout,
        )

    def _enqueue_purge(self, now: dt.datetime) -> None:
        """Enqueue the recurring session purge as a normal task.

        Deliberately enqueued rather than run inline on the timer. Inline would
        be three lines shorter and would sit outside every guarantee the queue
        provides: no attempt counter, no `last_error`, no retry, no row anyone
        can query afterwards to see whether it ran. Going through the queue also
        means this slice exercises claim, dispatch and completion on real work
        from the day it lands, with no model call and no spend.

        `enqueue_unique` so a worker that was down for six hours enqueues one
        purge on restart, not a backlog of six.
        """
        self._enqueue_recurring(PURGE_SESSIONS_KIND, now)

    def _enqueue_recurring(self, kind: str, now: dt.datetime) -> None:
        """One recurring task, through the queue, at most one queued at a time.

        Shared by the session purge (above), the feed-mark purge, and the
        watched-board scheduling pass, which enqueues each due board's
        `check_board` itself -- so a worker down for a day restarts into one
        scheduling pass, not ninety-six.
        """
        # `scheduled_at=now`, not the server's `now()` default: the loop has
        # already fixed `now` for this iteration, and a row scheduled a
        # millisecond later than that would not be due until the next poll.
        with self._enqueuer_scope() as enqueuer:
            task = enqueuer.enqueue_unique(kind=kind, scheduled_at=now)
        if task is not None:
            log_event(
                self._log, logging.INFO, "queue.enqueued", task_id=str(task.id), kind=task.kind
            )

    # -- dispatch ----------------------------------------------------------

    def _dispatch(self, task: Task) -> None:
        now = self._clock()
        spec = self._registry.get(task.kind)
        if spec is None:
            # Unreachable through `claim`, which only asks for registered kinds.
            # Reachable if a handler is unregistered while a task is in flight,
            # so it releases rather than failing: the task keeps its attempts
            # and waits for a worker that knows it.
            self._release(task, now, note=f"no handler registered for kind {task.kind!r}")
            return

        if spec.calls_model and model_calls_disabled(self._env):
            # The dispatch-time check. The kind filter in `run_once` normally
            # stops this task being claimed at all; this catches the switch
            # being thrown in the moment between claim and call, which is
            # exactly when an incident is unfolding.
            self._release(task, now, note="refused: JFL_DISABLE_MODEL_CALLS is set")
            log_event(
                self._log,
                logging.WARNING,
                "task.refused_model_call",
                task_id=str(task.id),
                kind=task.kind,
                user_id=str(task.user_id),
                attempts=task.attempts,
            )
            return

        log_event(
            self._log,
            logging.INFO,
            "task.started",
            task_id=str(task.id),
            kind=task.kind,
            user_id=str(task.user_id),
            attempt=task.attempts,
            max_attempts=task.max_attempts,
        )
        started = time.monotonic()
        try:
            result = spec.handler(TaskContext(task=task, engine=self._engine, now=now))
        except Exception as exc:
            self._fail(task, exc, elapsed_ms=self._elapsed_ms(started))
            return

        finished = self._clock()
        with self._queue_scope() as queue:
            queue.mark_succeeded(task.id, now=finished)
        log_event(
            self._log,
            logging.INFO,
            "task.succeeded",
            task_id=str(task.id),
            kind=task.kind,
            user_id=str(task.user_id),
            attempt=task.attempts,
            duration_ms=self._elapsed_ms(started),
            **dict(result or {}),
        )

    def _fail(self, task: Task, exc: Exception, *, elapsed_ms: int) -> None:
        now = self._clock()
        # Type and message only. The traceback goes to the log line, not to the
        # database row, and neither carries the payload.
        error = f"{type(exc).__name__}: {exc}"
        # A handler that raises `PermanentTaskError` has said the retry ladder
        # cannot help -- no key stored, no such application, a refusal. Retrying
        # would spend attempts, and for anything that reached the model, money,
        # to be told the same thing.
        permanent = isinstance(exc, PermanentTaskError)
        retry_at = now + self._settings.retry_delay(task.attempts)
        with self._queue_scope() as queue:
            updated = (
                queue.fail_permanently(task.id, now=now, error=error)
                if permanent
                else queue.mark_failed(task.id, now=now, error=error, retry_at=retry_at)
            )

        exhausted = updated.status == "failed"
        log_event(
            self._log,
            # A retry is expected operations; running out of attempts is a task
            # that now needs a person, so it is the louder line.
            logging.ERROR if exhausted else logging.WARNING,
            "task.failed" if exhausted else "task.retrying",
            task_id=str(task.id),
            kind=task.kind,
            user_id=str(task.user_id),
            attempt=task.attempts,
            max_attempts=task.max_attempts,
            duration_ms=elapsed_ms,
            error=error,
            permanent=permanent,
            retry_at=None if exhausted else retry_at.isoformat(),
            exc_info=True,
        )

    def _release(self, task: Task, now: dt.datetime, *, note: str) -> None:
        """Put a task back without spending an attempt. See
        `PostgresTaskQueue.release` for why refusing is not failing.
        """
        retry_at = now + dt.timedelta(seconds=self._settings.kill_switch_retry_delay)
        with self._queue_scope() as queue:
            queue.release(task.id, retry_at=retry_at, note=note)

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    # -- introspection, for tests and for main() ---------------------------

    @property
    def system_user_id(self) -> uuid.UUID:
        return self._settings.system_user_id
