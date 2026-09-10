"""Watched boards against real Postgres: the history the owner relies on.

Needs `docker compose up -d` and `alembic upgrade head`. Follows the
transaction-rollback fixture pattern: every test leaves the database as it found
it. No network and no model call anywhere in this file -- results are built by
hand as `FetchResult`s, and the pure engine plans them, exactly as the worker's
`check_board` handler does.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import (
    board_checks,
    board_job_presence,
    board_jobs,
    tasks,
    users,
    watched_boards,
)
from jfl_core.models import BoardCheck, BoardJobEvent, CheckPlan, ObservedJob
from jfl_core.storage.boards import (
    BoardNotFoundError,
    PostgresBoardRepository,
    PostgresBoardScheduler,
)
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_intake.scheduling import CHECK_BOARD_KIND
from sqlalchemy import create_engine, delete, func, insert, select, update
from sqlalchemy.engine import Connection, Engine

pytestmark = pytest.mark.integration

DAY0 = dt.datetime(2026, 9, 1, 6, 0, tzinfo=dt.UTC)


def day(n: float) -> dt.datetime:
    return DAY0 + dt.timedelta(days=n)


@pytest.fixture(scope="module")
def engine() -> Engine:
    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()


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
def repo(conn: Connection, alice: uuid.UUID) -> PostgresBoardRepository:
    return PostgresBoardRepository(conn, alice)


def _board(repo: PostgresBoardRepository, token: str = "acme") -> uuid.UUID:
    return repo.add_board(
        platform="greenhouse",
        board_url=f"https://boards.greenhouse.io/{token}",
        board_key={"token": token},
        label=token.title(),
    ).id


def obs(
    ext: str,
    title: str | None = None,
    location: str = "London",
    *,
    requisition: str | None = None,
) -> ObservedJob:
    name = title or f"Role {ext}"
    return ObservedJob(
        external_id=ext,
        requisition_id=requisition,
        title=name,
        location=location,
        url=f"https://job-boards.greenhouse.io/acme/jobs/{ext}",
        fingerprint=fingerprint(name, location),
    )


def complete(*jobs: ObservedJob) -> FetchResult:
    return FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))


def run_check(
    repo: PostgresBoardRepository, board_id: uuid.UUID, result: FetchResult, at: dt.datetime
) -> tuple[CheckPlan, BoardCheck]:
    """What `check_board` does after its fetch: lock, plan, apply."""
    state = repo.lock_check_state(
        board_id,
        observed_external_ids=[j.external_id for j in result.jobs],
        closed_since=at - REPOST_WINDOW,
    )
    assert state is not None
    plan = plan_check(state, result, observed_at=at)
    check = repo.apply_check_plan(plan, started_at=at - dt.timedelta(minutes=1), finished_at=at)
    return plan, check


def by_kind(events: list[BoardJobEvent]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for event in events:
        grouped.setdefault(event.kind, set()).add(event.job.external_id)
    return grouped


def open_ids(repo: PostgresBoardRepository, board_id: uuid.UUID) -> set[str]:
    return {j.external_id for j in repo.list_jobs(board_id, open_only=True)}


def open_interval_counts(conn: Connection, board_id: uuid.UUID) -> list[int]:
    rows = conn.execute(
        select(func.count())
        .select_from(board_job_presence.join(board_jobs))
        .where(board_jobs.c.board_id == board_id, board_job_presence.c.closed_check_id.is_(None))
        .group_by(board_job_presence.c.job_id)
    ).scalars()
    return list(rows)


# -- boards --------------------------------------------------------------------


def test_watching_a_board_twice_is_one_watch(repo: PostgresBoardRepository) -> None:
    first = repo.add_board(
        platform="workday",
        board_url="https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
        board_key={"tenant": "nvidia", "wd": "wd5", "site": "NVIDIAExternalCareerSite"},
    )
    again = repo.add_board(
        platform="workday",
        board_url="https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite",
        board_key={"site": "NVIDIAExternalCareerSite", "tenant": "nvidia", "wd": "wd5"},
    )
    assert again.id == first.id
    assert [b.id for b in repo.list_boards()] == [first.id]
    assert first.board_key == {"tenant": "nvidia", "wd": "wd5", "site": "NVIDIAExternalCareerSite"}
    assert first.baseline_check_id is None and first.consecutive_failures == 0


# -- the rule ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("unreachable", "timeout"),
        ("failed", "not_found"),
        ("incomplete", "count_mismatch"),
        ("truncated", "listing_ceiling"),
    ],
)
def test_a_check_that_is_not_complete_closes_no_interval(
    repo: PostgresBoardRepository, conn: Connection, status: str, code: str
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a"), obs("b"), obs("c")), day(0))

    partial = FetchResult(
        status=status,  # type: ignore[arg-type]
        jobs=(obs("a"), obs("never-seen")),
        expected_total=3,
        error_code=code,  # type: ignore[arg-type]
    )
    _, check = run_check(repo, board_id, partial, day(1))

    assert check.status == status and check.error_code == code
    assert check.jobs_seen == 2 and not check.is_baseline
    assert open_ids(repo, board_id) == {"a", "b", "c"}  # nothing closed
    assert {j.external_id for j in repo.list_jobs(board_id)} == {"a", "b", "c"}  # nothing added
    assert repo.events_for_check(check.id) == []
    assert repo.events_since(day(0.5)) == []
    assert open_interval_counts(conn, board_id) == [1, 1, 1]

    board = repo.get_board(board_id)
    assert board is not None
    assert board.consecutive_failures == 1
    assert board.last_check_id == check.id


def test_failures_accumulate_and_a_complete_check_resets_them(
    repo: PostgresBoardRepository,
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a")), day(0))
    for n in range(3):
        run_check(
            repo, board_id, FetchResult(status="unreachable", error_code="timeout"), day(n + 1)
        )
    board = repo.get_board(board_id)
    assert board is not None and board.consecutive_failures == 3

    run_check(repo, board_id, complete(obs("a")), day(5))
    board = repo.get_board(board_id)
    assert board is not None and board.consecutive_failures == 0
    assert [c.status for c in repo.list_checks(board_id)] == [
        "complete",
        "unreachable",
        "unreachable",
        "unreachable",
        "complete",
    ]


# -- baseline and events ------------------------------------------------------------


def test_the_baseline_marks_nothing_new(repo: PostgresBoardRepository) -> None:
    board_id = _board(repo)
    _, check = run_check(repo, board_id, complete(obs("a"), obs("b"), obs("c")), day(0))

    assert check.status == "complete" and check.is_baseline
    board = repo.get_board(board_id)
    assert board is not None and board.baseline_check_id == check.id
    assert open_ids(repo, board_id) == {"a", "b", "c"}
    assert repo.events_for_check(check.id) == []
    assert repo.events_since(day(-1)) == []  # open when watching started, never "new"


def test_a_later_complete_check_records_new_gone_and_returned(
    repo: PostgresBoardRepository,
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a"), obs("b"), obs("c")), day(0))

    _, second = run_check(repo, board_id, complete(obs("a"), obs("b"), obs("d")), day(1))
    assert by_kind(repo.events_for_check(second.id)) == {"new": {"d"}, "gone": {"c"}}

    _, third = run_check(repo, board_id, complete(obs("a"), obs("b"), obs("c"), obs("d")), day(2))
    assert by_kind(repo.events_for_check(third.id)) == {"returned": {"c"}}
    assert open_ids(repo, board_id) == {"a", "b", "c", "d"}

    c = next(j for j in repo.list_jobs(board_id) if j.external_id == "c")
    intervals = repo.list_presence(c.id)
    assert [(i.opened_at, i.closed_at) for i in intervals] == [(day(0), day(1)), (day(2), None)]
    assert c.first_seen_at == day(0) and c.last_seen_at == day(2)

    since = repo.events_since(day(0.5), board_id=board_id)
    assert [(e.kind, e.job.external_id, e.at) for e in since] == [
        ("gone", "c", day(1)),
        ("new", "d", day(1)),
        ("returned", "c", day(2)),
    ]


def test_a_sighting_refreshes_what_the_board_now_says(repo: PostgresBoardRepository) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a", title="Engineer")), day(0))
    run_check(
        repo, board_id, complete(obs("a", title="Senior Engineer", location="Remote")), day(1)
    )
    (job,) = repo.list_jobs(board_id)
    assert (job.title, job.location) == ("Senior Engineer", "Remote")
    assert job.fingerprint == fingerprint("Senior Engineer", "Remote")


def test_the_requisition_is_stored_beside_the_posting_and_refreshed(
    repo: PostgresBoardRepository,
) -> None:
    """Stored only: identity is the posting id, and no rule reads the requisition."""
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("R1-1", requisition="R1"), obs("x")), day(0))
    stored = {j.external_id: j.requisition_id for j in repo.list_jobs(board_id)}
    assert stored == {"R1-1": "R1", "x": None}

    run_check(
        repo,
        board_id,
        complete(obs("R1-1", requisition="R1b"), obs("x", requisition="X9")),
        day(1),
    )
    stored = {j.external_id: j.requisition_id for j in repo.list_jobs(board_id)}
    assert stored == {"R1-1": "R1b", "x": "X9"}


# -- drop guard ------------------------------------------------------------------------


def test_the_drop_guard_holds_a_collapse_and_closes_nothing(repo: PostgresBoardRepository) -> None:
    board_id = _board(repo)
    everyone = [obs(f"j{i}") for i in range(20)]
    run_check(repo, board_id, complete(*everyone), day(0))
    assert repo.accept_drop(board_id) is False  # nothing is held yet

    _, held = run_check(repo, board_id, complete(*everyone[:3]), day(1))
    assert held.status == "held" and held.error_code == "drop_guard"
    assert len(open_ids(repo, board_id)) == 20
    assert repo.events_for_check(held.id) == []
    board = repo.get_board(board_id)
    assert board is not None
    assert board.held_check_id == held.id
    assert board.consecutive_failures == 0  # the fetch worked; this is a flag

    # Still collapsed the next day: still held.
    _, again = run_check(repo, board_id, complete(*everyone[:3]), day(2))
    assert again.status == "held"

    # A person says it is real; the next complete check is applied.
    assert repo.accept_drop(board_id) is True
    _, applied = run_check(repo, board_id, complete(*everyone[:3]), day(3))
    assert applied.status == "complete"
    assert len(by_kind(repo.events_for_check(applied.id))["gone"]) == 17
    board = repo.get_board(board_id)
    assert board is not None and board.held_check_id is None and board.drop_accepted is False


# -- reposted --------------------------------------------------------------------------


def test_reposted_is_detected_within_the_window_and_is_not_returned(
    repo: PostgresBoardRepository,
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("x", title="Staff Engineer"), obs("keep")), day(0))
    _, closed = run_check(repo, board_id, complete(obs("keep")), day(1))
    assert by_kind(repo.events_for_check(closed.id)) == {"gone": {"x"}}

    _, reposted = run_check(
        repo, board_id, complete(obs("keep"), obs("y", title="Staff Engineer")), day(31)
    )
    (event,) = repo.events_for_check(reposted.id)
    x = next(j for j in repo.list_jobs(board_id) if j.external_id == "x")
    assert event.kind == "reposted"
    assert event.job.external_id == "y"
    assert event.job.reposted_from_job_id == x.id

    # The old id itself coming back is `returned` -- and not a second repost.
    _, back = run_check(
        repo,
        board_id,
        complete(obs("keep"), obs("y", title="Staff Engineer"), obs("x", title="Staff Engineer")),
        day(32),
    )
    assert by_kind(repo.events_for_check(back.id)) == {"returned": {"x"}}


def test_a_matching_role_closed_outside_the_window_is_just_new(
    repo: PostgresBoardRepository,
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("z", title="Staff Engineer"), obs("keep")), day(0))
    run_check(repo, board_id, complete(obs("keep")), day(1))
    _, later = run_check(
        repo, board_id, complete(obs("keep"), obs("w", title="Staff Engineer")), day(62)
    )
    assert by_kind(repo.events_for_check(later.id)) == {"new": {"w"}}


# -- idempotency ---------------------------------------------------------------------


def test_running_the_same_check_twice_opens_no_duplicate_intervals(
    repo: PostgresBoardRepository, conn: Connection
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a"), obs("b"), obs("c")), day(0))
    result = complete(obs("a"), obs("c"), obs("d"))

    _, first = run_check(repo, board_id, result, day(1))
    _, second = run_check(repo, board_id, result, day(1))  # a redelivered task

    assert by_kind(repo.events_for_check(first.id)) == {"new": {"d"}, "gone": {"b"}}
    assert repo.events_for_check(second.id) == []
    assert sorted(open_interval_counts(conn, board_id)) == [1, 1, 1]
    assert len(repo.list_jobs(board_id)) == 4


def test_a_stale_plan_applied_twice_still_cannot_duplicate_anything(
    repo: PostgresBoardRepository, conn: Connection
) -> None:
    """Belt and braces under the lock: the schema itself refuses a second open
    interval and a second job row, so even a plan applied twice without
    re-planning is harmless.
    """
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a"), obs("b")), day(0))
    result = complete(obs("a"), obs("c"))
    state = repo.lock_check_state(board_id, observed_external_ids=["a", "c"], closed_since=day(-60))
    assert state is not None
    plan = plan_check(state, result, observed_at=day(1))

    repo.apply_check_plan(plan, started_at=day(1), finished_at=day(1))
    repo.apply_check_plan(plan, started_at=day(1), finished_at=day(1))

    assert sorted(open_interval_counts(conn, board_id)) == [1, 1]
    assert len(repo.list_jobs(board_id)) == 3
    assert open_ids(repo, board_id) == {"a", "c"}


# -- tenancy ----------------------------------------------------------------------------


def test_two_users_cannot_see_each_others_boards_checks_or_jobs(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    alices = PostgresBoardRepository(conn, alice)
    bobs = PostgresBoardRepository(conn, bob)
    board_id = _board(alices)
    plan, check = run_check(alices, board_id, complete(obs("a"), obs("b")), day(0))
    job = alices.list_jobs(board_id)[0]

    assert bobs.get_board(board_id) is None
    assert bobs.list_boards() == []
    assert bobs.list_checks(board_id) == []
    assert bobs.list_jobs(board_id) == []
    assert bobs.list_presence(job.id) == []
    assert bobs.events_since(day(-1)) == []
    assert bobs.events_for_check(check.id) == []
    assert (
        bobs.lock_check_state(board_id, observed_external_ids=["a"], closed_since=day(-60)) is None
    )
    assert bobs.accept_drop(board_id) is False
    assert bobs.remove_board(board_id) is False
    with pytest.raises(BoardNotFoundError):
        bobs.apply_check_plan(plan, started_at=day(1), finished_at=day(1))

    # Watches are per user: the same board is a separate watch with its own history.
    bobs_board = _board(bobs)
    assert bobs_board != board_id
    assert [b.id for b in alices.list_boards()] == [board_id]
    assert alices.get_board(board_id) is not None


def test_a_check_task_is_matched_to_its_board_and_its_owner(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_a, board_b = _board(repo, "acme"), _board(repo, "globex")
    task = PostgresTaskRepository(conn, alice).enqueue(
        kind=CHECK_BOARD_KIND, payload={"board_id": str(board_a)}
    )

    assert repo.check_task_queued(board_a, kind=CHECK_BOARD_KIND) is True
    assert repo.check_task_queued(board_b, kind=CHECK_BOARD_KIND) is False
    assert (
        PostgresBoardRepository(conn, bob).check_task_queued(board_a, kind=CHECK_BOARD_KIND)
        is False
    )

    conn.execute(update(tasks).where(tasks.c.id == task.id).values(status="succeeded"))
    assert repo.check_task_queued(board_a, kind=CHECK_BOARD_KIND) is False


# -- deletion through the foreign-key cycle ---------------------------------------------


def test_removing_a_board_discards_its_history(
    repo: PostgresBoardRepository, conn: Connection
) -> None:
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a"), obs("b")), day(0))
    run_check(repo, board_id, complete(obs("a")), day(1))
    run_check(repo, board_id, FetchResult(status="failed", error_code="not_found"), day(2))

    assert repo.remove_board(board_id) is True
    assert repo.get_board(board_id) is None
    for table in (board_checks, board_jobs):
        assert (
            conn.execute(
                select(func.count()).select_from(table).where(table.c.board_id == board_id)
            ).scalar_one()
            == 0
        )


def test_deleting_a_user_cascades_through_every_board_table(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo)
    run_check(repo, board_id, complete(obs("a"), obs("b")), day(0))
    run_check(repo, board_id, complete(obs("a")), day(1))

    conn.execute(delete(users).where(users.c.id == alice))

    for table in (watched_boards, board_checks, board_jobs, board_job_presence):
        count = conn.execute(
            select(func.count()).select_from(table).where(table.c.user_id == alice)
        ).scalar_one()
        assert count == 0, table.name


# -- the cross-tenant scheduler ------------------------------------------------------------


def test_the_scheduler_claims_due_boards_across_tenants_and_moves_them_on(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    carol = _make_user(conn)
    alices = _board(PostgresBoardRepository(conn, alice))
    bobs = _board(PostgresBoardRepository(conn, bob))
    carols = _board(PostgresBoardRepository(conn, carol))
    scheduler = PostgresBoardScheduler(conn)
    soon = dt.datetime.now(tz=dt.UTC) + dt.timedelta(minutes=1)

    # Scoped to this test's users, so boards left by any other run never enter
    # the claim -- and carol's, though due, is outside the scope.
    ours = {alice, bob}
    due = {d.board_id: d.user_id for d in scheduler.claim_due(now=soon, limit=10, only_owners=ours)}
    assert due == {alices: alice, bobs: bob}

    scheduler.reschedule(alices, next_check_at=soon + dt.timedelta(days=1))
    again = {d.board_id for d in scheduler.claim_due(now=soon, limit=10, only_owners=ours)}
    assert again == {bobs}

    assert scheduler.claim_due(now=soon, limit=10, only_owners=set()) == []
    only_carol = scheduler.claim_due(now=soon, limit=10, only_owners={carol})
    assert [d.board_id for d in only_carol] == [carols]
