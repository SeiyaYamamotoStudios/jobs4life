"""The queue's SQL, against real Postgres. Needs `docker compose up -d` and
`alembic upgrade head`.

Most tests here use the transaction-rollback fixture the other integration
tests use. The concurrency tests deliberately do not: `FOR UPDATE SKIP LOCKED`
is invisible inside one transaction, and proving that two workers never take the
same row needs two connections and rows both of them can see. Those tests commit
and clean up after themselves by deleting the user, which cascades to the tasks.

No Anthropic call is possible anywhere in this file -- there is no client here to
construct.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users
from jfl_core.storage.tasks import PostgresTaskQueue, PostgresTaskRepository, TaskNotFoundError
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.integration

KIND = "test_kind"
NOW = dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def engine() -> Engine:
    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine: Engine):
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()  # every test leaves the database as it found it


def _make_user(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


@pytest.fixture
def alice(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


@pytest.fixture
def bob(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


@pytest.fixture
def repo(conn: Connection, alice: uuid.UUID) -> PostgresTaskRepository:
    return PostgresTaskRepository(conn, alice)


@pytest.fixture
def queue(conn: Connection) -> PostgresTaskQueue:
    return PostgresTaskQueue(conn)


@pytest.fixture
def committed_user(engine: Engine):
    """A user that really exists, for the tests that need two connections.

    Deleting the user cascades to their tasks, so cleanup is one statement and
    cannot leave orphans behind if an assertion fails.
    """
    uid = uuid.uuid4()
    with engine.begin() as c:
        c.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    try:
        yield uid
    finally:
        with engine.begin() as c:
            c.execute(delete(users).where(users.c.id == uid))


# -- enqueue and read back -------------------------------------------------


def test_enqueue_defaults(repo: PostgresTaskRepository, alice: uuid.UUID) -> None:
    task = repo.enqueue(kind=KIND, payload={"job_id": "abc"})
    assert task.user_id == alice
    assert task.status == "pending"
    assert task.attempts == 0
    assert task.max_attempts == 3
    assert task.payload == {"job_id": "abc"}
    assert task.started_at is None and task.finished_at is None


def test_one_users_tasks_are_invisible_to_another(
    conn: Connection, repo: PostgresTaskRepository, bob: uuid.UUID
) -> None:
    task = repo.enqueue(kind=KIND)
    bobs = PostgresTaskRepository(conn, bob)
    assert bobs.get_task(task.id) is None
    assert bobs.list_tasks() == []


def test_enqueue_unique_skips_a_duplicate_but_not_a_finished_one(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    first = repo.enqueue_unique(kind=KIND, scheduled_at=NOW)
    assert first is not None
    assert repo.enqueue_unique(kind=KIND, scheduled_at=NOW) is None  # still pending

    claimed = queue.claim(kinds=[KIND], now=NOW)
    assert len(claimed) == 1
    assert repo.enqueue_unique(kind=KIND, scheduled_at=NOW) is None  # running: still a duplicate

    queue.mark_succeeded(claimed[0].id, now=NOW)
    assert repo.enqueue_unique(kind=KIND, scheduled_at=NOW) is not None  # finished: one is due


# -- claiming --------------------------------------------------------------


def test_claim_marks_running_and_spends_an_attempt(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    task = repo.enqueue(kind=KIND, scheduled_at=NOW)
    claimed = queue.claim(kinds=[KIND], now=NOW, limit=5)
    assert [t.id for t in claimed] == [task.id]
    assert claimed[0].status == "running"
    # Incremented at CLAIM, not at failure: a worker that dies mid-task has
    # still used an attempt.
    assert claimed[0].attempts == 1
    assert claimed[0].started_at is not None


def test_claim_ignores_kinds_this_worker_cannot_run(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    task = repo.enqueue(kind="from_a_newer_deploy", scheduled_at=NOW)
    assert queue.claim(kinds=[KIND], now=NOW) == []
    assert queue.claim(kinds=[], now=NOW) == []
    still = repo.get_task(task.id)
    assert still is not None and still.status == "pending" and still.attempts == 0


def test_claim_respects_scheduled_at(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    """This is what makes backoff real: a retry parked in the future is not due."""
    repo.enqueue(kind=KIND, scheduled_at=NOW + dt.timedelta(minutes=5))
    assert queue.claim(kinds=[KIND], now=NOW) == []
    assert len(queue.claim(kinds=[KIND], now=NOW + dt.timedelta(minutes=6))) == 1


def test_claim_takes_the_oldest_first(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    later = repo.enqueue(kind=KIND, scheduled_at=NOW + dt.timedelta(minutes=1))
    earlier = repo.enqueue(kind=KIND, scheduled_at=NOW)
    claimed = queue.claim(kinds=[KIND], now=NOW + dt.timedelta(minutes=2), limit=2)
    assert [t.id for t in claimed] == [earlier.id, later.id]


# -- SKIP LOCKED, with two real connections --------------------------------


def test_two_connections_claiming_at_once_get_different_rows(
    engine: Engine, committed_user: uuid.UUID
) -> None:
    """The property the whole queue rests on.

    Two workers polling at the same instant must never take the same row. This
    cannot be shown with a mock and cannot be shown inside one transaction, so:
    two committed rows, two connections, two open transactions, two claims.
    """
    with engine.begin() as c:
        repo = PostgresTaskRepository(c, committed_user)
        first = repo.enqueue(kind=KIND, scheduled_at=NOW)
        second = repo.enqueue(kind=KIND, scheduled_at=NOW + dt.timedelta(seconds=1))

    with engine.connect() as c1, engine.connect() as c2, c1.begin(), c2.begin():
        claimed_by_one = PostgresTaskQueue(c1).claim(kinds=[KIND], now=NOW + dt.timedelta(1))
        claimed_by_two = PostgresTaskQueue(c2).claim(kinds=[KIND], now=NOW + dt.timedelta(1))

    assert len(claimed_by_one) == 1 and len(claimed_by_two) == 1
    assert claimed_by_one[0].id != claimed_by_two[0].id
    assert {claimed_by_one[0].id, claimed_by_two[0].id} == {first.id, second.id}


def test_a_second_claimer_skips_rather_than_waits(
    engine: Engine, committed_user: uuid.UUID
) -> None:
    """With ONE pending row held by another transaction, the second claimer gets
    nothing back *immediately*. Without SKIP LOCKED it would block until the
    first transaction ended -- which is how a two-minute gate call would come to
    hold up every quick task behind it.

    `statement_timeout` is the assertion: if this ever regresses to a plain
    `FOR UPDATE`, the test fails in two seconds instead of hanging the suite.
    """
    with engine.begin() as c:
        PostgresTaskRepository(c, committed_user).enqueue(kind=KIND, scheduled_at=NOW)

    with engine.connect() as c1, engine.connect() as c2:
        tx1 = c1.begin()
        holder = PostgresTaskQueue(c1).claim(kinds=[KIND], now=NOW + dt.timedelta(1))
        assert len(holder) == 1  # c1 holds the only row, and has not committed

        with c2.begin():
            c2.exec_driver_sql("set local statement_timeout = '2s'")
            try:
                second = PostgresTaskQueue(c2).claim(kinds=[KIND], now=NOW + dt.timedelta(1))
            except DBAPIError as exc:  # pragma: no cover - only on a regression
                pytest.fail(f"the second claim blocked instead of skipping: {exc}")
        assert second == []

        tx1.rollback()


# -- failure, retry, and giving up ----------------------------------------


def test_a_failed_attempt_is_rescheduled_with_the_error_recorded(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    task = repo.enqueue(kind=KIND, max_attempts=3, scheduled_at=NOW)
    claimed = queue.claim(kinds=[KIND], now=NOW)[0]
    assert claimed.id == task.id
    retry_at = NOW + dt.timedelta(seconds=30)

    updated = queue.mark_failed(claimed.id, now=NOW, error="RuntimeError: boom", retry_at=retry_at)
    assert updated.status == "pending"
    assert updated.last_error == "RuntimeError: boom"
    assert updated.scheduled_at == retry_at
    assert updated.started_at is None
    assert updated.finished_at is None
    assert queue.claim(kinds=[KIND], now=NOW) == []  # not due yet
    assert len(queue.claim(kinds=[KIND], now=retry_at)) == 1


def test_an_exhausted_task_ends_failed_and_is_never_claimed_again(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    """The brief's rule: never silently disappears, never loops."""
    task = repo.enqueue(kind=KIND, max_attempts=2, scheduled_at=NOW)

    first = queue.claim(kinds=[KIND], now=NOW)[0]
    assert first.attempts == 1
    queue.mark_failed(first.id, now=NOW, error="first failure", retry_at=NOW)

    second = queue.claim(kinds=[KIND], now=NOW)[0]
    assert second.attempts == 2
    final = queue.mark_failed(second.id, now=NOW, error="ValueError: last failure", retry_at=NOW)

    assert final.status == "failed"
    assert final.last_error == "ValueError: last failure"
    assert final.finished_at is not None
    # Not retried, however long anyone waits, and still there to be read.
    assert queue.claim(kinds=[KIND], now=NOW + dt.timedelta(days=30)) == []
    stored = repo.get_task(task.id)
    assert stored is not None and stored.status == "failed"


def test_mark_failed_on_an_unknown_task_raises(queue: PostgresTaskQueue) -> None:
    with pytest.raises(TaskNotFoundError):
        queue.mark_failed(uuid.uuid4(), now=NOW, error="x", retry_at=NOW)


def test_success_keeps_the_error_from_the_attempt_that_failed(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    repo.enqueue(kind=KIND, scheduled_at=NOW)
    first = queue.claim(kinds=[KIND], now=NOW)[0]
    queue.mark_failed(first.id, now=NOW, error="transient 529", retry_at=NOW)
    second = queue.claim(kinds=[KIND], now=NOW)[0]

    done = queue.mark_succeeded(second.id, now=NOW)
    assert done.status == "succeeded"
    assert done.finished_at is not None
    assert done.last_error == "transient 529"  # the story of the retry, kept


# -- release, the kill switch's path --------------------------------------


def test_release_gives_the_attempt_back(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    repo.enqueue(kind=KIND, scheduled_at=NOW)
    claimed = queue.claim(kinds=[KIND], now=NOW)[0]
    assert claimed.attempts == 1

    released = queue.release(claimed.id, retry_at=NOW, note="refused: switch is set")
    assert released.status == "pending"
    assert released.attempts == 0  # refusing is not failing
    assert released.started_at is None
    assert released.last_error == "refused: switch is set"


# -- reclaiming what a dead worker left behind ----------------------------


def test_a_stale_running_row_is_reclaimed_after_the_visibility_timeout(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    """At-least-once delivery, made concrete: the worker was killed, the row
    still says `running`, and nobody is running it.
    """
    task = repo.enqueue(kind=KIND, max_attempts=3, scheduled_at=NOW)
    queue.claim(kinds=[KIND], now=NOW)  # started_at = NOW; then the worker dies

    fifteen_minutes = dt.timedelta(minutes=15)
    too_early = queue.reclaim_stale(
        now=NOW + dt.timedelta(minutes=5),
        cutoff=NOW + dt.timedelta(minutes=5) - fifteen_minutes,
        retry_at=NOW,
    )
    assert not too_early  # a live worker on a slow task is left alone

    later = NOW + dt.timedelta(minutes=20)
    result = queue.reclaim_stale(now=later, cutoff=later - fifteen_minutes, retry_at=later)
    assert result.requeued == [task.id]
    assert result.failed == []

    reclaimed = repo.get_task(task.id)
    assert reclaimed is not None
    assert reclaimed.status == "pending"
    assert reclaimed.attempts == 1  # the dead attempt is still spent
    assert "worker presumed dead" in (reclaimed.last_error or "")
    assert len(queue.claim(kinds=[KIND], now=later)) == 1  # and it runs again


def test_a_stale_row_with_no_attempts_left_is_failed_not_requeued(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    """The poison-pill case: a task that kills its worker every time must stop,
    which is only possible because attempts are spent at claim time.
    """
    task = repo.enqueue(kind=KIND, max_attempts=1, scheduled_at=NOW)
    queue.claim(kinds=[KIND], now=NOW)

    later = NOW + dt.timedelta(hours=1)
    result = queue.reclaim_stale(now=later, cutoff=later - dt.timedelta(minutes=15), retry_at=later)
    assert result.failed == [task.id]
    assert result.requeued == []

    dead = repo.get_task(task.id)
    assert dead is not None and dead.status == "failed"
    assert dead.finished_at is not None
    assert "worker presumed dead" in (dead.last_error or "")


def test_reclaim_leaves_finished_rows_alone(
    repo: PostgresTaskRepository, queue: PostgresTaskQueue
) -> None:
    repo.enqueue(kind=KIND, scheduled_at=NOW)
    claimed = queue.claim(kinds=[KIND], now=NOW)[0]
    queue.mark_succeeded(claimed.id, now=NOW)

    later = NOW + dt.timedelta(days=1)
    assert not queue.reclaim_stale(now=later, cutoff=later, retry_at=later)


# -- schema-level guarantees ----------------------------------------------


def test_the_status_check_constraint_rejects_an_invented_status(
    conn: Connection, alice: uuid.UUID
) -> None:
    with pytest.raises(DBAPIError):
        conn.execute(
            insert(tasks_table).values(
                id=uuid.uuid4(), user_id=alice, kind=KIND, status="in_progress"
            )
        )


def test_deleting_a_user_takes_their_tasks_with_them(
    conn: Connection, repo: PostgresTaskRepository, alice: uuid.UUID
) -> None:
    task = repo.enqueue(kind=KIND)
    conn.execute(delete(users).where(users.c.id == alice))
    assert conn.execute(select(tasks_table.c.id).where(tasks_table.c.id == task.id)).first() is None
