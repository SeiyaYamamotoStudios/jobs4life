"""`fetch_job_description`, end to end against real Postgres: slice C7's
"Track as application" button, the worker half.

Needs `docker compose up -d` and `alembic upgrade head`. Follows the pattern of
`tests/test_extraction_integration.py` and `tests/test_boards_worker_integration.py`
-- a throwaway user per test (cascades everything away on delete), the real
worker loop, and `jfl_intake.descriptions.fetch_description` monkeypatched in
`jfl_worker.handlers.description` so no test depends on a real per-platform
adapter existing yet.

**No network and no model call anywhere in this file.** `fetch_description` is
always a fake; the root `conftest.py` socket guard would refuse a real one
anyway. `extract_job_ad`, the model-calling task this handler enqueues on
success, is never registered in this file's test worker -- these tests prove
it was queued, never that it ran, so nothing here needs a real or fake
Anthropic client either.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import jfl_worker.handlers.description as description_module
import pytest
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import jobs as jobs_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users
from jfl_core.models import CheckPlan, ObservedJob, Task
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.http import Transport
from jfl_intake.normalise import fingerprint
from jfl_worker.handlers import EXTRACT_JOB_AD, FETCH_JOB_DESCRIPTION
from jfl_worker.handlers.description import KIND as FETCH_JOB_DESCRIPTION_KIND
from jfl_worker.handlers.description import build_fetch_job_description
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.registry import HandlerRegistry, PermanentTaskError, TaskContext
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

DAY0 = dt.datetime(2026, 9, 1, 6, 0, tzinfo=dt.UTC)


@contextmanager
def _dummy_transport() -> Iterator[Transport]:
    """`description_transport`: the handler always opens one before calling
    `fetch_description`, same as `check_board` does before its own fetch, even
    though every `fetch_description` in this file is a fake that ignores it.
    """
    yield object()  # type: ignore[misc]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    created = create_engine(DATABASE_URL)
    yield created
    created.dispose()


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


@pytest.fixture
def master_key() -> MasterKey:
    return MasterKey.generate()


@pytest.fixture
def log_stream() -> io.StringIO:
    return io.StringIO()


def store_key(engine: Engine, owner: uuid.UUID, master_key: MasterKey) -> None:
    """A key on file: without one the fetched ad is attached but no read is
    queued (nothing could run it), which is its own test below.
    """
    key = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, owner).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=owner, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def add_board_job(engine: Engine, owner: uuid.UUID, *, label: str = "Acme") -> Any:
    """A real, checked-in board job -- add a board and run one baseline check
    against it, same path `check_board` takes.
    """
    with engine.begin() as conn:
        boards = PostgresBoardRepository(conn, owner)
        board = boards.add_board(
            platform="greenhouse",
            board_url="https://boards.greenhouse.io/acme",
            board_key={"token": "acme"},
            label=label,
        )
        observed = ObservedJob(
            external_id="1",
            title="Senior Platform Engineer",
            location="London",
            url="https://job-boards.greenhouse.io/acme/jobs/1",
            fingerprint=fingerprint("Senior Platform Engineer", "London"),
        )
        result = FetchResult(status="complete", jobs=(observed,), expected_total=1)
        state = boards.lock_check_state(
            board.id, observed_external_ids=["1"], closed_since=DAY0 - REPOST_WINDOW
        )
        assert state is not None
        plan: CheckPlan = plan_check(state, result, observed_at=DAY0)
        boards.apply_check_plan(plan, started_at=DAY0 - dt.timedelta(minutes=1), finished_at=DAY0)
        (job,) = boards.list_jobs(board.id)
    return job


def add_tracked_application(engine: Engine, owner: uuid.UUID, job: Any) -> uuid.UUID:
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, owner).create_application(
            title=job.title,
            source="Watched board",
            extraction_status="pending",
            board_job_id=job.id,
        )
    return application.id


def enqueue(
    engine: Engine, owner: uuid.UUID, application_id: uuid.UUID, *, max_attempts: int = 3
) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, owner).enqueue(
            kind=FETCH_JOB_DESCRIPTION_KIND,
            payload={"application_id": str(application_id)},
            max_attempts=max_attempts,
        )
    return task.id


def build_worker(
    engine: Engine, owner: uuid.UUID, master_key: MasterKey, stream: io.StringIO
) -> Worker:
    """Only `fetch_job_description` is registered -- deliberately, not the full
    production registry. The worker's maintenance ticker enqueues its own
    recurring tasks (session purge, board scheduling) on every fresh instance's
    first poll regardless of what is registered; with the full registry those
    become claimable once this test's own task stops being due, which is
    exactly the kind of cross-talk that would make "exhausted: never claimed
    again" flicker for a reason that has nothing to do with this handler.
    `extract_job_ad` is deliberately unregistered too -- these tests prove it
    stays queued, never that it runs, so nothing here needs a real or fake
    Anthropic client.
    """
    settings = WorkerSettings(
        database_url=DATABASE_URL, system_user_id=owner, master_key=master_key
    )
    registry = HandlerRegistry()
    registry.register(
        FETCH_JOB_DESCRIPTION,
        build_fetch_job_description(transport_factory=_dummy_transport),
        calls_model=False,
    )
    return Worker(
        registry=registry,
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, owner),
        engine=engine,
        env={},
        logger=configure_logging(stream=stream),
    )


def task_row(engine: Engine, task_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return conn.execute(select(tasks_table).where(tasks_table.c.id == task_id)).one()


def application_row(engine: Engine, application_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return conn.execute(
            select(applications_table).where(applications_table.c.id == application_id)
        ).one()


def fast_forward(engine: Engine, task_id: uuid.UUID) -> None:
    """Same helper as `test_worker_integration.py`'s: make a backed-off task
    due now, raw SQL against the one column, so the test does not sleep for a
    real backoff.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "update tasks set scheduled_at = now() - interval '1 second' where id = %(id)s",
            {"id": str(task_id)},
        )


class FakeDescriptions:
    """Stands in for `jfl_intake.descriptions.fetch_description`. Records every
    call so a test can assert it was asked for the right job, and can be told
    what to answer.
    """

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, platform: Any, board_key: Any, external_id: str, url: str | None, transport: Any
    ) -> Any:
        self.calls.append((platform, external_id))
        return self.result


def install_fake_description(monkeypatch: pytest.MonkeyPatch, result: Any) -> FakeDescriptions:
    fake = FakeDescriptions(result)
    monkeypatch.setattr(description_module, "fetch_description", fake)
    return fake


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_successful_fetch_attaches_the_ad_and_queues_extraction_held_by_the_switch(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jfl_intake.descriptions import DescriptionResult

    store_key(engine, user, master_key)
    job = add_board_job(engine, user)
    application_id = add_tracked_application(engine, user, job)
    task_id = enqueue(engine, user, application_id)

    install_fake_description(
        monkeypatch,
        DescriptionResult(text="Own the deployment pipeline.", error_code=None, requests=2),
    )
    worker = build_worker(engine, user, master_key, log_stream)
    assert worker.run_once() == 1

    assert task_row(engine, task_id).status == "succeeded"

    row = application_row(engine, application_id)
    assert row.job_id is not None
    assert row.extraction_status == "pending"
    assert row.extraction_error_code is None

    with engine.begin() as conn:
        ad_text = conn.execute(
            select(jobs_table.c.raw_text).where(jobs_table.c.id == row.job_id)
        ).scalar_one()
    assert "Senior Platform Engineer" in ad_text
    assert "Acme" in ad_text  # the board's label
    assert "Own the deployment pipeline." in ad_text
    assert "https://job-boards.greenhouse.io/acme/jobs/1" in ad_text

    # extract_job_ad was enqueued in the same transaction, and stays `pending`
    # -- the kill switch holds it, exactly as it would a hand-pasted ad.
    with engine.connect() as conn:
        (extract_task,) = PostgresTaskRepository(conn, user).list_tasks(kind=EXTRACT_JOB_AD)
    assert extract_task.status == "pending"
    assert extract_task.attempts == 0
    assert extract_task.payload == {"application_id": str(application_id)}


# --------------------------------------------------------------------------
# Transient failure: retried, and every attempt records its own failure
# --------------------------------------------------------------------------


def test_a_transient_failure_is_retried_and_records_description_unavailable(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jfl_intake.descriptions import DescriptionResult

    job = add_board_job(engine, user)
    application_id = add_tracked_application(engine, user, job)
    task_id = enqueue(engine, user, application_id)

    install_fake_description(monkeypatch, DescriptionResult(text=None, error_code="unreachable"))
    worker = build_worker(engine, user, master_key, log_stream)
    assert worker.run_once() == 1

    task = task_row(engine, task_id)
    assert task.status == "pending"  # the backoff ladder rides it out
    assert task.attempts == 1
    assert "unreachable" in (task.last_error or "")

    row = application_row(engine, application_id)
    assert row.job_id is None  # nothing was attached
    # Retrying, not failed: a retry is queued, so the panel says so rather
    # than offering a paste box the next attempt may make unnecessary.
    assert row.extraction_status == "pending"
    assert row.extraction_error_code == "description_unavailable"


def test_repeated_transient_failures_exhaust_and_leave_the_application_failed(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requirement this exists to prove: once the retry ladder gives up,
    the application must not be left `pending` forever with nobody retrying it
    -- it must already be sitting in `failed` / `description_unavailable`,
    because every attempt recorded its own failure rather than only the last.
    """
    from jfl_intake.descriptions import DescriptionResult

    job = add_board_job(engine, user)
    application_id = add_tracked_application(engine, user, job)
    task_id = enqueue(engine, user, application_id, max_attempts=2)

    install_fake_description(monkeypatch, DescriptionResult(text=None, error_code="unreachable"))
    worker = build_worker(engine, user, master_key, log_stream)

    assert worker.run_once() == 1
    assert task_row(engine, task_id).status == "pending"
    assert (
        application_row(engine, application_id).extraction_error_code == "description_unavailable"
    )

    fast_forward(engine, task_id)
    assert worker.run_once() == 1

    final = task_row(engine, task_id)
    assert final.status == "failed"
    assert final.attempts == 2
    assert final.finished_at is not None

    row = application_row(engine, application_id)
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "description_unavailable"

    # Exhausted: never picked up again.
    assert worker.run_once() == 0


# --------------------------------------------------------------------------
# Permanent failure: fails immediately, no retry spent chasing it
# --------------------------------------------------------------------------


def test_a_permanent_failure_fails_the_task_and_the_application_at_once(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jfl_intake.descriptions import DescriptionResult

    job = add_board_job(engine, user)
    application_id = add_tracked_application(engine, user, job)
    task_id = enqueue(engine, user, application_id)

    install_fake_description(
        monkeypatch, DescriptionResult(text=None, error_code="unsupported_platform")
    )
    worker = build_worker(engine, user, master_key, log_stream)
    assert worker.run_once() == 1

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "unsupported_platform" in (task.last_error or "")

    row = application_row(engine, application_id)
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "description_unavailable"


# --------------------------------------------------------------------------
# Idempotency: at-least-once delivery must not fetch or enqueue twice
# --------------------------------------------------------------------------


def test_a_redelivered_task_does_not_fetch_or_enqueue_a_second_time(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jfl_intake.descriptions import DescriptionResult

    store_key(engine, user, master_key)
    job = add_board_job(engine, user)
    application_id = add_tracked_application(engine, user, job)

    fake = install_fake_description(
        monkeypatch, DescriptionResult(text="Some description.", error_code=None)
    )
    handler = build_fetch_job_description(transport_factory=_dummy_transport)
    now = dt.datetime.now(tz=dt.UTC)
    task = Task(
        id=uuid.uuid4(),
        user_id=user,
        kind=FETCH_JOB_DESCRIPTION,
        payload={"application_id": str(application_id)},
        status="running",
        attempts=1,
        max_attempts=3,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )
    ctx = TaskContext(task=task, engine=engine, now=now)

    first = handler(ctx)
    assert first is not None and "skipped" not in first

    again = handler(ctx)  # at-least-once: the same task, run twice
    assert again == {"application_id": str(application_id), "skipped": "ad already attached"}

    assert len(fake.calls) == 1  # fetched once, not twice
    with engine.connect() as conn:
        extract_tasks = PostgresTaskRepository(conn, user).list_tasks(kind=EXTRACT_JOB_AD)
    assert len(extract_tasks) == 1  # enqueued once, not twice


# --------------------------------------------------------------------------
# Defensive: nothing to fetch from
# --------------------------------------------------------------------------


def test_no_board_job_id_fails_permanently(engine: Engine, user: uuid.UUID) -> None:
    """The route only ever enqueues this kind for applications it creates with
    a `board_job_id` -- this is the defensive branch for anything else that
    somehow ends up here.
    """
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user).create_application(
            title="No board job", extraction_status="pending"
        )

    handler = build_fetch_job_description(transport_factory=_dummy_transport)
    now = dt.datetime.now(tz=dt.UTC)
    task = Task(
        id=uuid.uuid4(),
        user_id=user,
        kind=FETCH_JOB_DESCRIPTION,
        payload={"application_id": str(application.id)},
        status="running",
        attempts=1,
        max_attempts=3,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )
    ctx = TaskContext(task=task, engine=engine, now=now)

    with pytest.raises(PermanentTaskError):
        handler(ctx)

    row = application_row(engine, application.id)
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "description_unavailable"


def test_with_no_key_the_ad_is_attached_and_nothing_is_queued(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fetch is free, so it runs; the read and the score chained from it
    are model calls, so with no key neither is queued and the application says
    why. The ad is kept, so adding a key and pressing "Read the job ad" works
    without fetching again.
    """
    from jfl_intake.descriptions import DescriptionResult

    job = add_board_job(engine, user)
    application_id = add_tracked_application(engine, user, job)
    enqueue(engine, user, application_id)
    install_fake_description(
        monkeypatch, DescriptionResult(text="We need Python.", error_code=None, requests=1)
    )
    build_worker(engine, user, master_key, log_stream).run_once()

    row = application_row(engine, application_id)
    assert row.job_id is not None
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "no_api_key"
    with engine.begin() as conn:
        queued = conn.execute(
            select(tasks_table).where(
                tasks_table.c.user_id == user, tasks_table.c.kind == EXTRACT_JOB_AD
            )
        ).all()
    assert queued == []
