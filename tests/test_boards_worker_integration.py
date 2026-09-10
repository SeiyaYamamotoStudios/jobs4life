"""The watched-board handlers, end to end, against real Postgres.

Needs `docker compose up -d` and `alembic upgrade head`. Handlers open their own
transactions, so these tests commit, and clean up by deleting the throwaway users
they create -- which cascades to boards, checks, jobs, intervals and tasks.

**No network, structurally.** Every `check_board` here is given a scripted
transport that fails loudly if asked for more responses than were scripted;
every scheduling pass is confined to the users this test created, so a database
left dirty by a crashed run cannot hand it someone else's board; and underneath
both, the root `conftest.py` refuses any socket that is not loopback. No model
call is possible either: nothing on this path imports a client.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import pytest
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import board_job_presence, board_jobs, users, watched_boards
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.models import BoardJob, BoardPlatform, Task
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import HttpResponse, Transport
from jfl_intake.scheduling import (
    CHECK_BOARD_KIND,
    ENQUEUE_SPACING,
    enqueue_board_check,
    next_check_at,
)
from jfl_worker.handlers.boards import (
    SCHEDULE_KIND,
    build_check_board,
    build_schedule_board_checks,
)
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.registry import Handler, HandlerRegistry, TaskContext
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, func, insert, select, update
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
API = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"
ADOBE = {"tenant": "adobe", "wd": "wd5", "site": "external_experienced"}
WORKDAY_API = "https://adobe.wd5.myworkdayjobs.com/wday/cxs/adobe/external_experienced/jobs"


@pytest.fixture(scope="module")
def engine() -> Engine:
    return create_engine(DATABASE_URL)


@contextmanager
def _throwaway_user(engine: Engine) -> Iterator[uuid.UUID]:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    try:
        yield uid
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users).where(users.c.id == uid))


@pytest.fixture
def user(engine: Engine) -> Iterator[uuid.UUID]:
    with _throwaway_user(engine) as uid:
        yield uid


@pytest.fixture
def other_user(engine: Engine) -> Iterator[uuid.UUID]:
    with _throwaway_user(engine) as uid:
        yield uid


class ScriptedTransport:
    """Hands out scripted responses in order, and refuses to improvise."""

    def __init__(self) -> None:
        self.script: list[HttpResponse | Exception] = []
        self.urls: list[str] = []

    def get_json(self, url: str) -> HttpResponse:
        return self._next(url)

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        assert body["limit"] == 20
        return self._next(url)

    def _next(self, url: str) -> HttpResponse:
        self.urls.append(url)
        if not self.script:
            raise AssertionError("a check asked for a response nobody scripted")
        response = self.script.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def factory(self) -> Callable[[], Any]:
        @contextmanager
        def open_transport() -> Iterator[Transport]:
            yield self

        return open_transport


def greenhouse(*jobs: tuple[int, str]) -> HttpResponse:
    return HttpResponse(
        200,
        {
            "jobs": [
                {
                    "id": job_id,
                    "title": title,
                    "location": {"name": "London"},
                    "absolute_url": f"https://job-boards.greenhouse.io/acme/jobs/{job_id}",
                    "requisition_id": f"REQ-{job_id}",
                }
                for job_id, title in jobs
            ],
            "meta": {"total": len(jobs)},
        },
    )


def workday_posting(posting_id: str, requisition: str, title: str) -> dict[str, Any]:
    slug = "-".join(title.replace(",", "").split())
    return {
        "title": title,
        "externalPath": f"/job/San-Jose/{slug}_{posting_id}",
        "locationsText": "San Jose",
        "postedOn": "Posted Today",
        "bulletFields": [requisition],
    }


def workday_page(*postings: dict[str, Any]) -> HttpResponse:
    return HttpResponse(
        200,
        {
            "total": len(postings),
            "jobPostings": list(postings),
            "facets": [],
            "userAuthenticated": False,
        },
    )


def add_board(
    engine: Engine,
    owner: uuid.UUID,
    board_key: dict[str, str] | None = None,
    *,
    platform: BoardPlatform = "greenhouse",
) -> uuid.UUID:
    with engine.begin() as conn:
        return (
            PostgresBoardRepository(conn, owner)
            .add_board(
                platform=platform,
                board_url=f"https://example.test/{platform}",
                board_key=board_key or {"token": "acme"},
            )
            .id
        )


def check_now(engine: Engine, owner: uuid.UUID, board_id: uuid.UUID) -> Task | None:
    with engine.begin() as conn:
        return enqueue_board_check(
            PostgresBoardRepository(conn, owner), PostgresTaskRepository(conn, owner), board_id
        )


def build_worker(engine: Engine, owner: uuid.UUID, registry: HandlerRegistry) -> Worker:
    settings = WorkerSettings(
        database_url=DATABASE_URL, system_user_id=owner, master_key=MasterKey.generate()
    )
    return Worker(
        registry=registry,
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, owner),
        engine=engine,
        env={},
        logger=configure_logging(stream=io.StringIO()),
    )


def registry_with(**handlers: Handler) -> HandlerRegistry:
    registry = HandlerRegistry()
    for kind, handler in handlers.items():
        registry.register(kind, handler, calls_model=False)
    return registry


def only_check_board(transport: ScriptedTransport) -> HandlerRegistry:
    return registry_with(
        **{CHECK_BOARD_KIND: build_check_board(transport_factory=transport.factory())}
    )


def get_task(engine: Engine, owner: uuid.UUID, task_id: uuid.UUID) -> Task:
    with engine.connect() as conn:
        task = PostgresTaskRepository(conn, owner).get_task(task_id)
    assert task is not None
    return task


def board_state(engine: Engine, owner: uuid.UUID, board_id: uuid.UUID) -> Any:
    with engine.connect() as conn:
        repo = PostgresBoardRepository(conn, owner)
        board = repo.get_board(board_id)
        checks = repo.list_checks(board_id)
        open_jobs = {j.external_id for j in repo.list_jobs(board_id, open_only=True)}
        events = repo.events_for_check(checks[0].id) if checks else []
    return board, checks, open_jobs, {(e.kind, e.job.external_id) for e in events}


def jobs_by_external_id(
    engine: Engine, owner: uuid.UUID, board_id: uuid.UUID
) -> dict[str, BoardJob]:
    with engine.connect() as conn:
        return {j.external_id: j for j in PostgresBoardRepository(conn, owner).list_jobs(board_id)}


def test_check_board_takes_a_baseline_then_records_what_changed(
    engine: Engine, user: uuid.UUID
) -> None:
    transport = ScriptedTransport()
    worker = build_worker(engine, user, only_check_board(transport))
    board_id = add_board(engine, user)

    transport.script.append(greenhouse((1, "Engineer"), (2, "Designer"), (3, "Manager")))
    first = check_now(engine, user, board_id)
    assert first is not None
    assert worker.run_once() == 1
    assert get_task(engine, user, first.id).status == "succeeded"

    board, checks, open_jobs, events = board_state(engine, user, board_id)
    assert board.baseline_check_id == checks[0].id and checks[0].is_baseline
    assert open_jobs == {"1", "2", "3"} and events == set()
    assert transport.urls == [API]
    # Greenhouse's requisition round-trips into storage beside the public id.
    stored = {
        ext: job.requisition_id for ext, job in jobs_by_external_id(engine, user, board_id).items()
    }
    assert stored == {"1": "REQ-1", "2": "REQ-2", "3": "REQ-3"}

    transport.script.append(greenhouse((1, "Engineer"), (2, "Designer"), (4, "Researcher")))
    second = check_now(engine, user, board_id)
    assert second is not None
    assert worker.run_once() == 1

    board, checks, open_jobs, events = board_state(engine, user, board_id)
    assert checks[0].status == "complete" and not checks[0].is_baseline
    assert open_jobs == {"1", "2", "4"}
    assert events == {("new", "4"), ("gone", "3")}


def test_a_workday_role_relisted_under_a_new_posting_id_is_reposted(
    engine: Engine, user: uuid.UUID
) -> None:
    """Adapter, engine and storage together: `R171808-1` vanishes in the check
    where `R171808-2` -- same requisition, title and location -- appears. It is
    recorded as a repost of the old posting: not `returned`, and not one job that
    stayed up continuously, which is what keying by requisition would have said.
    """
    transport = ScriptedTransport()
    worker = build_worker(engine, user, only_check_board(transport))
    board_id = add_board(engine, user, ADOBE, platform="workday")
    title = "Senior Manager, Digital Monetization Growth"

    transport.script.append(
        workday_page(
            workday_posting("R171808-1", "R171808", title),
            workday_posting("R170001", "R170001", "Designer"),
        )
    )
    check_now(engine, user, board_id)
    assert worker.run_once() == 1

    transport.script.append(
        workday_page(
            workday_posting("R171808-2", "R171808", title),
            workday_posting("R170001", "R170001", "Designer"),
        )
    )
    check_now(engine, user, board_id)
    assert worker.run_once() == 1

    _, checks, open_jobs, events = board_state(engine, user, board_id)
    assert [c.status for c in checks] == ["complete", "complete"]
    assert events == {("reposted", "R171808-2"), ("gone", "R171808-1")}
    assert open_jobs == {"R171808-2", "R170001"}

    jobs = jobs_by_external_id(engine, user, board_id)
    assert jobs["R171808-2"].reposted_from_job_id == jobs["R171808-1"].id
    assert jobs["R171808-1"].requisition_id == jobs["R171808-2"].requisition_id == "R171808"
    with engine.connect() as conn:
        (interval,) = PostgresBoardRepository(conn, user).list_presence(jobs["R171808-1"].id)
    assert interval.closed_at is not None  # it left; it did not stay up
    assert transport.urls == [WORKDAY_API, WORKDAY_API]


def test_an_unreachable_board_is_recorded_retried_and_changes_nothing(
    engine: Engine, user: uuid.UUID
) -> None:
    transport = ScriptedTransport()
    worker = build_worker(engine, user, only_check_board(transport))
    board_id = add_board(engine, user)
    transport.script.append(greenhouse((1, "Engineer"), (2, "Designer")))
    check_now(engine, user, board_id)
    worker.run_once()

    transport.script.append(HttpResponse(503, None))
    task = check_now(engine, user, board_id)
    assert task is not None
    assert worker.run_once() == 1

    retrying = get_task(engine, user, task.id)
    assert retrying.status == "pending"  # the queue's backoff rides it out
    assert retrying.attempts == 1
    assert retrying.scheduled_at > retrying.updated_at - dt.timedelta(seconds=1)
    assert retrying.last_error == "BoardUnreachableError: board check unreachable: server_error"

    board, checks, open_jobs, events = board_state(engine, user, board_id)
    assert checks[0].status == "unreachable" and checks[0].error_code == "server_error"
    assert board.consecutive_failures == 1
    assert open_jobs == {"1", "2"}  # a failed fetch is not evidence anything vanished
    assert events == set()


def test_a_board_that_is_not_found_fails_permanently_and_changes_nothing(
    engine: Engine, user: uuid.UUID
) -> None:
    transport = ScriptedTransport()
    worker = build_worker(engine, user, only_check_board(transport))
    board_id = add_board(engine, user)
    transport.script.append(greenhouse((1, "Engineer")))
    check_now(engine, user, board_id)
    worker.run_once()

    transport.script.append(HttpResponse(404, None))
    task = check_now(engine, user, board_id)
    assert task is not None
    worker.run_once()

    failed = get_task(engine, user, task.id)
    assert failed.status == "failed" and failed.attempts == 1
    assert failed.last_error == "PermanentTaskError: board check failed: not_found"
    _, checks, open_jobs, _ = board_state(engine, user, board_id)
    assert checks[0].status == "failed" and checks[0].error_code == "not_found"
    assert open_jobs == {"1"}


def test_a_board_key_no_adapter_can_use_fails_permanently_without_a_request(
    engine: Engine, user: uuid.UUID
) -> None:
    transport = ScriptedTransport()  # empty: any request fails the test
    worker = build_worker(engine, user, only_check_board(transport))
    board_id = add_board(engine, user, board_key={"token": "../../admin"})
    task = check_now(engine, user, board_id)
    assert task is not None

    worker.run_once()

    assert transport.urls == []
    assert get_task(engine, user, task.id).status == "failed"
    _, checks, _, _ = board_state(engine, user, board_id)
    assert checks[0].status == "failed" and checks[0].error_code == "unsupported_board"


def test_a_redelivered_check_opens_no_duplicate_intervals(engine: Engine, user: uuid.UUID) -> None:
    transport = ScriptedTransport()
    handler = build_check_board(transport_factory=transport.factory())
    board_id = add_board(engine, user)
    now = dt.datetime.now(tz=dt.UTC)
    task = Task(
        id=uuid.uuid4(),
        user_id=user,
        kind=CHECK_BOARD_KIND,
        payload={"board_id": str(board_id)},
        status="running",
        attempts=1,
        max_attempts=3,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )
    ctx = TaskContext(task=task, engine=engine, now=now)

    transport.script.extend([greenhouse((1, "A"), (2, "B"))] * 3)
    handler(ctx)  # baseline
    first = handler(ctx)
    again = handler(ctx)  # at-least-once: the same task, run twice
    assert first is not None and again is not None
    assert first["status"] == again["status"] == "complete"
    assert again["new"] == again["gone"] == again["returned"] == 0

    with engine.connect() as conn:
        counts = (
            conn.execute(
                select(func.count())
                .select_from(board_job_presence.join(board_jobs))
                .where(board_jobs.c.board_id == board_id)
                .group_by(board_jobs.c.id)
            )
            .scalars()
            .all()
        )
    assert sorted(counts) == [1, 1]


def test_check_now_does_not_queue_a_second_check_of_the_same_board(
    engine: Engine, user: uuid.UUID
) -> None:
    board_id = add_board(engine, user)
    assert check_now(engine, user, board_id) is not None
    assert check_now(engine, user, board_id) is None
    with engine.connect() as conn:
        queued = PostgresTaskRepository(conn, user).list_tasks(kind=CHECK_BOARD_KIND)
    assert len(queued) == 1


def test_the_scheduling_pass_enqueues_each_due_board_for_its_owner_staggered(
    engine: Engine, user: uuid.UUID, other_user: uuid.UUID
) -> None:
    mine = add_board(engine, user)
    theirs = add_board(engine, other_user)
    now = dt.datetime.now(tz=dt.UTC) + dt.timedelta(seconds=5)
    system_task = Task(
        id=uuid.uuid4(),
        user_id=user,
        kind=SCHEDULE_KIND,
        payload={},
        status="running",
        attempts=1,
        max_attempts=3,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )
    # Confined to this test's two users: the real cross-tenant claim, minus any
    # board a crashed earlier run left behind.
    schedule = build_schedule_board_checks(only_owners={user, other_user})

    outcome = schedule(TaskContext(task=system_task, engine=engine, now=now))
    assert outcome is not None
    assert outcome["boards_due"] == outcome["checks_enqueued"] == 2

    scheduled: dict[uuid.UUID, dt.datetime] = {}
    for owner, board_id in ((user, mine), (other_user, theirs)):
        with engine.connect() as conn:
            (task,) = PostgresTaskRepository(conn, owner).list_tasks(kind=CHECK_BOARD_KIND)
            next_at = conn.execute(
                select(watched_boards.c.next_check_at).where(watched_boards.c.id == board_id)
            ).scalar_one()
        assert task.user_id == owner  # the board's owner, not the system user
        assert task.payload == {"board_id": str(board_id)}
        assert next_at == next_check_at(board_id, after=now)
        scheduled[board_id] = task.scheduled_at
    assert abs(scheduled[mine] - scheduled[theirs]) == ENQUEUE_SPACING

    # Nothing is due now, so a second pass enqueues nothing.
    idle = schedule(TaskContext(task=system_task, engine=engine, now=now))
    assert idle is not None and idle["boards_due"] == 0
    # And a board that comes due again while its check is still queued is moved
    # on without a duplicate.
    with engine.begin() as conn:
        conn.execute(
            update(watched_boards).where(watched_boards.c.id == mine).values(next_check_at=now)
        )
    again = schedule(TaskContext(task=system_task, engine=engine, now=now))
    assert again is not None and again["checks_already_queued"] == 1
    with engine.connect() as conn:
        rows = conn.execute(
            select(func.count())
            .select_from(tasks_table)
            .where(
                tasks_table.c.user_id.in_([user, other_user]),
                tasks_table.c.kind == CHECK_BOARD_KIND,
            )
        ).scalar_one()
    assert rows == 2


def test_a_scoped_scheduling_pass_leaves_other_users_boards_alone(
    engine: Engine, user: uuid.UUID, other_user: uuid.UUID
) -> None:
    add_board(engine, user)
    theirs = add_board(engine, other_user)
    now = dt.datetime.now(tz=dt.UTC) + dt.timedelta(seconds=5)
    system_task = Task(
        id=uuid.uuid4(),
        user_id=user,
        kind=SCHEDULE_KIND,
        payload={},
        status="running",
        attempts=1,
        max_attempts=3,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )

    outcome = build_schedule_board_checks(only_owners={user})(
        TaskContext(task=system_task, engine=engine, now=now)
    )

    assert outcome is not None and outcome["boards_due"] == 1
    with engine.connect() as conn:
        assert PostgresTaskRepository(conn, other_user).list_tasks(kind=CHECK_BOARD_KIND) == []
        untouched = conn.execute(
            select(watched_boards.c.next_check_at).where(watched_boards.c.id == theirs)
        ).scalar_one()
    assert untouched <= now  # still due: this pass never claimed it


def test_the_loop_schedules_and_baselines_a_newly_watched_board(
    engine: Engine, user: uuid.UUID
) -> None:
    """Ticker -> scheduling pass -> check_board -> baseline, with nobody pressing
    anything: a board is watched, and the worker takes it from there.
    """
    transport = ScriptedTransport()
    transport.script.append(greenhouse((10, "Engineer"), (11, "Designer")))
    registry = registry_with(
        **{
            SCHEDULE_KIND: build_schedule_board_checks(only_owners={user}),
            CHECK_BOARD_KIND: build_check_board(transport_factory=transport.factory()),
        }
    )
    worker = build_worker(engine, user, registry)
    board_id = add_board(engine, user)

    assert worker.run_once() == 1  # the ticker's scheduling pass
    assert worker.run_once() == 1  # the check it enqueued

    board, checks, open_jobs, _ = board_state(engine, user, board_id)
    assert len(checks) == 1 and checks[0].is_baseline
    assert board.baseline_check_id == checks[0].id
    assert open_jobs == {"10", "11"}
