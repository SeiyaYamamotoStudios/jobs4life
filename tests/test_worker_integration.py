"""The worker, end to end, against real Postgres. Needs `docker compose up -d`
and `alembic upgrade head`.

This is the proof that the queue works as a whole and not just as a set of
statements: the worker enqueues its own recurring task, claims it, runs the real
`purge_expired_sessions` handler against real session rows, and records the
outcome. It costs nothing to run -- the shipped handler makes no model call, and
there is no Anthropic client anywhere in `jfl_worker` to construct.

The worker's `system_user_id` is pointed at a throwaway user rather than the
seeded local one, so everything this file creates is removed by deleting that
user (tasks and sessions both cascade).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pytest
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import sessions as sessions_table
from jfl_core.db.tables import users
from jfl_core.storage.accounts import PostgresSessionRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import PURGE_EXPIRED_SESSIONS, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.registry import HandlerRegistry, TaskContext
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")


@contextmanager
def _never_fetch_a_board() -> Iterator[Transport]:
    """`check_board`'s transport in this file. Nothing here watches a job board,
    so being asked to open one means a board check from elsewhere was claimed --
    fail loudly rather than fetch it. (The root conftest refuses the socket too.)
    """
    raise AssertionError("a worker test tried to fetch a job board")
    yield  # pragma: no cover


@pytest.fixture(scope="module")
def engine() -> Engine:
    return create_engine(DATABASE_URL)


@pytest.fixture
def user(engine: Engine) -> Iterator[uuid.UUID]:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    try:
        yield uid
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users).where(users.c.id == uid))


def _build_worker(
    engine: Engine,
    user_id: uuid.UUID,
    *,
    registry: HandlerRegistry | None = None,
    env: Mapping[str, str] | None = None,
    stream: io.StringIO | None = None,
) -> Worker:
    settings = WorkerSettings(
        database_url=DATABASE_URL, system_user_id=user_id, master_key=MasterKey.generate()
    )
    return Worker(
        # The real registry, minus its ability to reach a job board: `check_board`
        # gets a transport that refuses, and the scheduling pass only sees this
        # test's user -- so a database left dirty by a crashed run cannot turn
        # this loop into a live request.
        registry=registry
        or build_registry(settings, board_transport=_never_fetch_a_board, board_owners={user_id}),
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, user_id),
        engine=engine,
        env={} if env is None else env,
        logger=configure_logging(stream=stream or io.StringIO()),
    )


def _make_session(engine: Engine, user_id: uuid.UUID, *, expires_at: dt.datetime) -> str:
    token_hash = uuid.uuid4().hex + uuid.uuid4().hex  # 64 chars, like a sha256
    with engine.begin() as conn:
        PostgresSessionRepository(conn).create(
            user_id=user_id,
            token_hash=token_hash,
            csrf_token=uuid.uuid4().hex,
            expires_at=expires_at,
            user_agent="pytest",
        )
    return token_hash


def _session_exists(engine: Engine, token_hash: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            select(sessions_table.c.id).where(sessions_table.c.token_hash == token_hash)
        ).first()
    return row is not None


def test_the_worker_enqueues_claims_and_runs_the_session_purge(
    engine: Engine, user: uuid.UUID
) -> None:
    """One `run_once`, and the whole path is exercised: ticker -> enqueue ->
    claim -> dispatch -> handler -> succeeded. No model call, no spend.
    """
    now = dt.datetime.now(tz=dt.UTC)
    stale = _make_session(engine, user, expires_at=now - dt.timedelta(hours=1))
    live = _make_session(engine, user, expires_at=now + dt.timedelta(hours=1))

    stream = io.StringIO()
    worker = _build_worker(engine, user, stream=stream)

    assert worker.run_once() == 1

    assert not _session_exists(engine, stale)
    assert _session_exists(engine, live)

    with engine.connect() as conn:
        tasks = PostgresTaskRepository(conn, user).list_tasks(kind=PURGE_EXPIRED_SESSIONS)
    assert len(tasks) == 1
    assert tasks[0].status == "succeeded"
    assert tasks[0].attempts == 1
    assert tasks[0].finished_at is not None
    assert tasks[0].last_error is None

    events = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    succeeded = [e for e in events if e["event"] == "task.succeeded"]
    assert len(succeeded) == 1
    assert succeeded[0]["kind"] == PURGE_EXPIRED_SESSIONS
    assert succeeded[0]["sessions_deleted"] >= 1
    assert "duration_ms" in succeeded[0]


def test_the_purge_is_not_re_enqueued_while_one_is_still_queued(
    engine: Engine, user: uuid.UUID
) -> None:
    """A restart loop, or a second worker, must not build a backlog of purges.

    Both workers here carry an empty registry, so nothing claims the task and it
    stays `pending` across both ticks -- which is exactly the state
    `enqueue_unique` exists to notice.
    """
    first = _build_worker(engine, user, registry=HandlerRegistry())
    second = _build_worker(engine, user, registry=HandlerRegistry())

    first.run_once()
    second.run_once()

    with engine.connect() as conn:
        tasks = PostgresTaskRepository(conn, user).list_tasks(kind=PURGE_EXPIRED_SESSIONS)
    assert len(tasks) == 1
    assert tasks[0].status == "pending"


def test_the_kill_switch_leaves_a_model_task_pending_in_the_database(
    engine: Engine, user: uuid.UUID
) -> None:
    """Requirement 10, against real rows: with `JFL_DISABLE_MODEL_CALLS` set, a
    handler marked as calling a model is never run and its task stays `pending`
    with its attempts unspent -- so nothing is lost when the switch comes off.
    """
    ran: list[uuid.UUID] = []

    def pretend_model_call(ctx: TaskContext) -> Mapping[str, object]:
        ran.append(ctx.task.id)  # pragma: no cover - must never run
        return {}

    registry = HandlerRegistry()
    registry.register("pretend_model_call", pretend_model_call, calls_model=True)

    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(kind="pretend_model_call")

    switched_off = _build_worker(
        engine, user, registry=registry, env={"JFL_DISABLE_MODEL_CALLS": "1"}
    )
    assert switched_off.run_once() == 0
    assert ran == []

    with engine.connect() as conn:
        stored = PostgresTaskRepository(conn, user).get_task(task.id)
    assert stored is not None
    assert stored.status == "pending"
    assert stored.attempts == 0
    assert stored.started_at is None

    # And with the switch off it runs, so the task really was only deferred.
    switched_on = _build_worker(engine, user, registry=registry, env={})
    assert switched_on.run_once() == 1
    assert ran == [task.id]

    with engine.connect() as conn:
        stored = PostgresTaskRepository(conn, user).get_task(task.id)
    assert stored is not None and stored.status == "succeeded"


def test_a_handler_that_raises_lands_the_task_in_failed_with_its_error(
    engine: Engine, user: uuid.UUID
) -> None:
    """The retry ladder, against the database rather than a fake: two attempts,
    then `failed` with `last_error`, and never claimed again.
    """

    def explode(ctx: TaskContext) -> Mapping[str, object]:
        raise RuntimeError("handler exploded")

    registry = HandlerRegistry()
    registry.register("explodes", explode, calls_model=False)

    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(kind="explodes", max_attempts=2)

    worker = _build_worker(engine, user, registry=registry)
    assert worker.run_once() == 1

    with engine.connect() as conn:
        after_first = PostgresTaskRepository(conn, user).get_task(task.id)
    assert after_first is not None
    assert after_first.status == "pending"
    assert after_first.attempts == 1
    assert "handler exploded" in (after_first.last_error or "")
    # Parked by the backoff: the next poll finds nothing due.
    assert after_first.scheduled_at > dt.datetime.now(tz=dt.UTC)
    assert worker.run_once() == 0

    # Bring the retry forward rather than waiting 30 real seconds for it.
    _fast_forward(engine, task.id)

    assert worker.run_once() == 1
    with engine.connect() as conn:
        final = PostgresTaskRepository(conn, user).get_task(task.id)
    assert final is not None
    assert final.status == "failed"
    assert final.attempts == 2
    assert "RuntimeError: handler exploded" in (final.last_error or "")
    assert final.finished_at is not None

    assert worker.run_once() == 0  # exhausted: never picked up again


def _fast_forward(engine: Engine, task_id: uuid.UUID) -> None:
    """Make a backed-off task due now, so the test does not sleep for the real
    backoff. Written as raw SQL against the one column, deliberately: no
    production code path may move `scheduled_at` backwards.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "update tasks set scheduled_at = now() - interval '1 second' where id = %(id)s",
            {"id": str(task_id)},
        )
