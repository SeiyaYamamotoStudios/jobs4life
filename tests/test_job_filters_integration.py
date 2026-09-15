"""Workplace, locations, the saved filter and board exceptions against real Postgres.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures, as in `test_boards_repo_integration.py`: every test leaves the database
as it found it. No network and no model call -- results are built by hand as
`FetchResult`s and planned by the pure engine, exactly as the worker does.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import board_filter_exceptions, board_jobs, job_filters, users
from jfl_core.models import ObservedJob, Workplace
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from sqlalchemy import create_engine, insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

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


def _board(repo: PostgresBoardRepository, token: str, platform: str = "greenhouse") -> uuid.UUID:
    return repo.add_board(
        platform=platform,  # type: ignore[arg-type]
        board_url=f"https://boards.greenhouse.io/{token}",
        board_key={"token": token},
        label=token.title(),
    ).id


def obs(
    ext: str,
    title: str | None = None,
    location: str = "London, UK",
    *,
    workplace: Workplace = "unknown",
    label: str | None = None,
    locations: tuple[str, ...] | None = None,
) -> ObservedJob:
    name = title or f"Role {ext}"
    return ObservedJob(
        external_id=ext,
        title=name,
        location=location,
        url=f"https://job-boards.greenhouse.io/acme/jobs/{ext}",
        fingerprint=fingerprint(name, location),
        workplace=workplace,
        workplace_label=label,
        locations=(location,) if locations is None else locations,
    )


def run_check(
    repo: PostgresBoardRepository, board_id: uuid.UUID, jobs: list[ObservedJob], at: dt.datetime
) -> None:
    result = FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))
    state = repo.lock_check_state(
        board_id,
        observed_external_ids=[j.external_id for j in jobs],
        closed_since=at - REPOST_WINDOW,
    )
    assert state is not None
    plan = plan_check(state, result, observed_at=at)
    repo.apply_check_plan(plan, started_at=at - dt.timedelta(minutes=1), finished_at=at)


# -- workplace and locations persist and refresh ------------------------------------


def test_workplace_and_locations_persist_and_refresh_without_touching_identity(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo, "acme")
    first = obs(
        "1",
        "Engineering Manager",
        "London, UK; Remote-Friendly, United States",
        workplace="onsite",
        label="On-Site",
        locations=("London, UK", "Remote-Friendly, United States"),
    )
    run_check(repo, board_id, [first], day(0))
    (job,) = repo.list_jobs(board_id)
    assert (job.workplace, job.workplace_label, job.locations) == (
        "onsite",
        "On-Site",
        ["London, UK", "Remote-Friendly, United States"],
    )
    original_fingerprint = job.fingerprint

    # The employer changes the workplace and adds a place, but not the primary
    # location string -- so the fingerprint, and the job's identity, stay put.
    again = obs(
        "1",
        "Engineering Manager",
        "London, UK; Remote-Friendly, United States",
        workplace="hybrid",
        label="Hybrid (Travel-Required)",
        locations=("London, UK", "Remote-Friendly, United States", "Dublin, IE"),
    )
    run_check(repo, board_id, [again], day(1))
    (refreshed,) = repo.list_jobs(board_id)
    assert refreshed.id == job.id
    assert refreshed.external_id == "1"
    assert refreshed.fingerprint == original_fingerprint
    assert refreshed.workplace == "hybrid"
    assert refreshed.workplace_label == "Hybrid (Travel-Required)"
    assert refreshed.locations == ["London, UK", "Remote-Friendly, United States", "Dublin, IE"]


def test_the_database_refuses_a_workplace_outside_the_set(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo, "acme")
    run_check(repo, board_id, [obs("1")], day(0))
    with pytest.raises(IntegrityError), conn.begin_nested():
        conn.execute(board_jobs.update().values(workplace="office"))


def test_a_row_written_before_the_columns_reads_unknown_and_empty(
    conn: Connection, alice: uuid.UUID
) -> None:
    """The server defaults stand in for rows from before the migration."""
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo, "acme")
    run_check(repo, board_id, [obs("1")], day(0))
    conn.execute(
        board_jobs.update().values(workplace="unknown", workplace_label=None, locations=[])
    )
    (job,) = repo.list_jobs(board_id)
    assert job.workplace == "unknown" and job.locations == []


# -- open jobs across boards ----------------------------------------------------------


def test_list_open_jobs_spans_boards_newest_first_and_skips_closed_and_other_users(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    a = _board(repo, "acme")
    b = _board(repo, "globex")
    run_check(repo, a, [obs("a1"), obs("a2")], day(0))
    run_check(repo, b, [obs("b1")], day(1))
    run_check(repo, a, [obs("a1"), obs("a3")], day(2))  # a2 gone, a3 new

    bobs = PostgresBoardRepository(conn, bob)
    run_check(bobs, _board(bobs, "acme"), [obs("bob1")], day(3))

    got = [(j.external_id, j.board_id) for j in repo.list_open_jobs()]
    assert got == [("a3", a), ("b1", b), ("a1", a)]
    assert [j.external_id for j in bobs.list_open_jobs()] == ["bob1"]


def test_a_board_with_no_complete_check_contributes_no_open_jobs(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo, "acme")
    result = FetchResult(status="unreachable", error_code="timeout")
    state = repo.lock_check_state(board_id, observed_external_ids=[], closed_since=day(-60))
    assert state is not None
    repo.apply_check_plan(
        plan_check(state, result, observed_at=day(0)), started_at=day(0), finished_at=day(0)
    )
    assert repo.list_open_jobs() == []


# -- the per-board include-unstated setting ---------------------------------------------


def test_include_unstated_defaults_to_none_and_can_be_set_and_reset(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo, "acme")
    board = repo.get_board(board_id)
    assert board is not None and board.include_unstated_workplace is None

    assert repo.set_include_unstated_workplace(board_id, False)
    board = repo.get_board(board_id)
    assert board is not None and board.include_unstated_workplace is False

    assert not PostgresBoardRepository(conn, bob).set_include_unstated_workplace(board_id, True)
    board = repo.get_board(board_id)
    assert board is not None and board.include_unstated_workplace is False

    assert repo.set_include_unstated_workplace(board_id, None)
    board = repo.get_board(board_id)
    assert board is not None and board.include_unstated_workplace is None


def test_hybrid_too_heavy_defaults_off_and_only_the_owner_can_set_it(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    repo = PostgresBoardRepository(conn, alice)
    board_id = _board(repo, "anthropic")
    board = repo.get_board(board_id)
    assert board is not None and board.hybrid_too_heavy is False

    assert repo.set_hybrid_too_heavy(board_id, True)
    board = repo.get_board(board_id)
    assert board is not None and board.hybrid_too_heavy is True
    assert [b.hybrid_too_heavy for b in repo.list_boards()] == [True]

    assert not PostgresBoardRepository(conn, bob).set_hybrid_too_heavy(board_id, False)
    board = repo.get_board(board_id)
    assert board is not None and board.hybrid_too_heavy is True

    assert repo.set_hybrid_too_heavy(board_id, False)
    board = repo.get_board(board_id)
    assert board is not None and board.hybrid_too_heavy is False
    assert not repo.set_hybrid_too_heavy(uuid.uuid4(), True)


# -- the saved filter ---------------------------------------------------------------------


def test_the_saved_filter_round_trips_one_per_user_and_is_private(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    filters = PostgresJobFilterRepository(conn, alice)
    empty = filters.get_filter()
    assert empty.workplaces == [] and empty.title_includes == "" and empty.location == ""

    filters.save_filter(
        workplaces=["hybrid", "remote", "remote"],
        title_includes="engineering manager, head of engineering",
        title_excludes="sales",
        location="",
    )
    saved = filters.save_filter(
        workplaces=["unknown", "remote"],
        title_includes="  Engineering Manager ",
        title_excludes="",
        location="london",
    )
    assert saved.workplaces == ["remote", "unknown"]
    assert saved.title_includes == "  Engineering Manager "  # stored as typed
    assert filters.get_filter() == saved

    assert PostgresJobFilterRepository(conn, bob).get_filter().title_includes == ""


def test_the_workplace_mode_round_trips_and_keeps_the_custom_ticks(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    filters = PostgresJobFilterRepository(conn, alice)
    assert filters.get_filter().workplace_mode == "custom"

    for mode in ("remote_only", "remote_friendly", "custom"):
        saved = filters.save_filter(
            workplace_mode=mode,  # type: ignore[arg-type]
            workplaces=["hybrid", "remote"],
            title_includes="engineering manager",
            title_excludes="",
            location="",
        )
        assert saved.workplace_mode == mode
        # The ticks are kept under a preset, so Custom comes back as it was.
        assert saved.workplaces == ["remote", "hybrid"]
        assert filters.get_filter() == saved

    filters.save_filter(
        workplace_mode="remote_friendly",
        workplaces=[],
        title_includes="",
        title_excludes="",
        location="",
    )
    assert PostgresJobFilterRepository(conn, bob).get_filter().workplace_mode == "custom"


def test_a_filter_saved_before_the_presets_reads_as_custom(
    conn: Connection, alice: uuid.UUID
) -> None:
    """The server default stands in for rows written before the column existed."""
    conn.execute(insert(job_filters).values(id=uuid.uuid4(), user_id=alice, workplaces=["remote"]))
    saved = PostgresJobFilterRepository(conn, alice).get_filter()
    assert saved.workplace_mode == "custom" and saved.workplaces == ["remote"]


def test_the_database_refuses_a_workplace_mode_outside_the_set(
    conn: Connection, alice: uuid.UUID
) -> None:
    with pytest.raises(IntegrityError), conn.begin_nested():
        conn.execute(
            insert(job_filters).values(
                id=uuid.uuid4(), user_id=alice, workplace_mode="mostly_remote"
            )
        )


# -- board exceptions ----------------------------------------------------------------------


def test_exceptions_add_edit_remove_and_keep_the_note_verbatim(
    conn: Connection, alice: uuid.UUID
) -> None:
    boards = PostgresBoardRepository(conn, alice)
    filters = PostgresJobFilterRepository(conn, alice)
    a = _board(boards, "anthropic")
    b = _board(boards, "globex")
    note = "Accepts ~25% in office — 1 day a week in London  "
    added = filters.add_exception(a, workplaces=["onsite", "hybrid"], location="london", note=note)
    assert added is not None
    assert added.note == note and added.workplaces == ["hybrid", "onsite"]
    other = filters.add_exception(b, workplaces=[], location="", note="")
    assert other is not None

    assert [e.id for e in filters.list_exceptions(a)] == [added.id]
    assert {e.id for e in filters.list_exceptions()} == {added.id, other.id}

    edited = filters.update_exception(
        added.id, workplaces=["onsite"], location="london, dublin", note="1 day a week"
    )
    assert (
        edited is not None and edited.location == "london, dublin" and edited.note == "1 day a week"
    )

    assert filters.remove_exception(added.id)
    assert not filters.remove_exception(added.id)
    assert filters.list_exceptions(a) == []

    boards.remove_board(b)  # cascades
    assert filters.list_exceptions() == []


def test_a_user_cannot_see_add_to_edit_or_remove_another_users_exceptions(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    alice_boards = PostgresBoardRepository(conn, alice)
    alice_filters = PostgresJobFilterRepository(conn, alice)
    board_id = _board(alice_boards, "anthropic")
    mine = alice_filters.add_exception(
        board_id, workplaces=["onsite"], location="london", note="mine"
    )
    assert mine is not None

    bob_filters = PostgresJobFilterRepository(conn, bob)
    assert bob_filters.list_exceptions() == []
    assert bob_filters.list_exceptions(board_id) == []
    assert bob_filters.get_exception(mine.id) is None
    assert bob_filters.add_exception(board_id, workplaces=[], location="", note="x") is None
    assert bob_filters.update_exception(mine.id, workplaces=[], location="", note="x") is None
    assert not bob_filters.remove_exception(mine.id)

    assert alice_filters.get_exception(mine.id) == mine


def test_the_database_refuses_workplaces_outside_the_set_on_both_filter_tables(
    conn: Connection, alice: uuid.UUID
) -> None:
    """The repository drops unknown values before writing; the CHECK is the
    second line, for anything that writes around it.
    """
    boards = PostgresBoardRepository(conn, alice)
    board_id = _board(boards, "acme")
    with pytest.raises(IntegrityError), conn.begin_nested():
        conn.execute(
            insert(board_filter_exceptions).values(
                id=uuid.uuid4(), user_id=alice, board_id=board_id, workplaces=["office"]
            )
        )
    with pytest.raises(IntegrityError), conn.begin_nested():
        conn.execute(
            insert(job_filters).values(id=uuid.uuid4(), user_id=alice, workplaces=["anywhere"])
        )
