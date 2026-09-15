"""Slice C7's repository additions, against real Postgres: `board_job_id` on
applications, `PostgresBoardRepository.get_job`, and the three
`PostgresApplicationRepository` methods "Track as application" needs --
`find_live_by_board_job`, `tracked_board_jobs`, `attach_job_ad`.

Needs `docker compose up -d` and `alembic upgrade head`. Follows the
transaction-rollback fixture pattern used throughout this directory: every
test leaves the database as it found it. No network and no model call
anywhere in this file.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import users
from jfl_core.models import BoardJob, CheckPlan, ObservedJob
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from sqlalchemy import create_engine, insert
from sqlalchemy.engine import Connection, Engine

pytestmark = pytest.mark.integration

DAY0 = dt.datetime(2026, 9, 1, 6, 0, tzinfo=dt.UTC)


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
def boards(conn: Connection, alice: uuid.UUID) -> PostgresBoardRepository:
    return PostgresBoardRepository(conn, alice)


@pytest.fixture
def apps(conn: Connection, alice: uuid.UUID) -> PostgresApplicationRepository:
    return PostgresApplicationRepository(conn, alice)


def _obs(ext: str, title: str = "Senior Engineer", location: str = "London") -> ObservedJob:
    return ObservedJob(
        external_id=ext,
        title=title,
        location=location,
        url=f"https://job-boards.greenhouse.io/acme/jobs/{ext}",
        fingerprint=fingerprint(title, location),
    )


def _board_job(boards: PostgresBoardRepository, *, label: str = "Acme") -> BoardJob:
    """A real, checked-in board job: add a board and run one baseline check
    against it, the same path `check_board` takes. Simpler than hand-writing
    every FK this table needs (board, check, presence interval).
    """
    board = boards.add_board(
        platform="greenhouse",
        board_url="https://boards.greenhouse.io/acme",
        board_key={"token": "acme"},
        label=label,
    )
    result = FetchResult(status="complete", jobs=(_obs("1"),), expected_total=1)
    state = boards.lock_check_state(
        board.id, observed_external_ids=["1"], closed_since=DAY0 - REPOST_WINDOW
    )
    assert state is not None
    plan: CheckPlan = plan_check(state, result, observed_at=DAY0)
    boards.apply_check_plan(plan, started_at=DAY0 - dt.timedelta(minutes=1), finished_at=DAY0)
    (job,) = boards.list_jobs(board.id)
    return job


# --- PostgresBoardRepository.get_job ----------------------------------------


def test_get_job_returns_the_job(boards: PostgresBoardRepository) -> None:
    job = _board_job(boards)
    fetched = boards.get_job(job.id)
    assert fetched is not None and fetched.id == job.id and fetched.title == "Senior Engineer"


def test_get_job_is_none_for_an_unknown_id(boards: PostgresBoardRepository) -> None:
    assert boards.get_job(uuid.uuid4()) is None


def test_get_job_is_none_for_another_users_job(
    conn: Connection, boards: PostgresBoardRepository, bob: uuid.UUID
) -> None:
    job = _board_job(boards)
    assert PostgresBoardRepository(conn, bob).get_job(job.id) is None


# --- create_application(board_job_id=...) -----------------------------------


def test_create_application_records_the_board_job_id(
    boards: PostgresBoardRepository, apps: PostgresApplicationRepository
) -> None:
    job = _board_job(boards)
    application = apps.create_application(
        title=job.title, source="Watched board", board_job_id=job.id
    )
    assert application.board_job_id == job.id
    # Not provisional: the employer's own words, not a derived placeholder --
    # see `jfl_web.routes.tracking`.
    assert application.title_is_provisional is False


# --- find_live_by_board_job --------------------------------------------------


def test_find_live_by_board_job_finds_a_live_application(
    boards: PostgresBoardRepository, apps: PostgresApplicationRepository
) -> None:
    job = _board_job(boards)
    application = apps.create_application(title=job.title, board_job_id=job.id)
    assert apps.find_live_by_board_job(job.id) == application.id


def test_find_live_by_board_job_is_none_for_an_untracked_job(
    boards: PostgresBoardRepository, apps: PostgresApplicationRepository
) -> None:
    job = _board_job(boards)
    assert apps.find_live_by_board_job(job.id) is None


def test_find_live_by_board_job_ignores_an_archived_application(
    boards: PostgresBoardRepository, apps: PostgresApplicationRepository
) -> None:
    """Archiving takes it off the owner's lists on purpose -- pressing "Track
    as application" again should offer a fresh start, not resurrect it.
    """
    job = _board_job(boards)
    application = apps.create_application(title=job.title, board_job_id=job.id)
    apps.archive(application.id)
    assert apps.find_live_by_board_job(job.id) is None


def test_find_live_by_board_job_is_tenant_scoped(
    conn: Connection, boards: PostgresBoardRepository, apps: PostgresApplicationRepository
) -> None:
    job = _board_job(boards)
    apps.create_application(title=job.title, board_job_id=job.id)
    other_uid = uuid.uuid4()
    conn.execute(insert(users).values(id=other_uid, email=f"{other_uid}@test.invalid"))
    others = PostgresApplicationRepository(conn, other_uid)
    assert others.find_live_by_board_job(job.id) is None


# --- tracked_board_jobs -------------------------------------------------------


def test_tracked_board_jobs_maps_only_live_tracked_jobs(
    boards: PostgresBoardRepository, apps: PostgresApplicationRepository
) -> None:
    tracked_job = _board_job(boards, label="Tracked Co")
    untracked_job = _board_job(boards, label="Untracked Co")
    archived_job = _board_job(boards, label="Archived Co")

    tracked_app = apps.create_application(title=tracked_job.title, board_job_id=tracked_job.id)
    archived_app = apps.create_application(title=archived_job.title, board_job_id=archived_job.id)
    apps.archive(archived_app.id)

    result = apps.tracked_board_jobs([tracked_job.id, untracked_job.id, archived_job.id])
    assert result == {tracked_job.id: tracked_app.id}


def test_tracked_board_jobs_of_an_empty_collection_makes_no_query_and_returns_empty(
    apps: PostgresApplicationRepository,
) -> None:
    assert apps.tracked_board_jobs([]) == {}


# --- attach_job_ad ------------------------------------------------------------


def test_attach_job_ad_sets_job_id_and_marks_extraction_pending(
    apps: PostgresApplicationRepository,
) -> None:
    application = apps.create_application(title="Some Role", extraction_status="failed")
    apps.fail_extraction(application.id, "description_unavailable")

    ok = apps.attach_job_ad(application.id, "Some Role\n\nWe are hiring.")
    assert ok is True

    detail = apps.get_application(application.id)
    assert detail is not None
    assert detail.application.job_id is not None
    assert detail.application.extraction_status == "pending"
    assert detail.application.extraction_error_code is None


def test_attach_job_ad_reuses_the_same_job_row_as_a_matching_paste(
    apps: PostgresApplicationRepository,
) -> None:
    """`_store_raw_job`'s derivation is deterministic on (user, text) -- pasting
    the same text through the ordinary create flow resolves to the same `jobs`
    row this attaches, which is the guarantee the design calls for.
    """
    text = "Staff Engineer at Acme. Remote."
    pasted = apps.create_application(title="Staff Engineer", raw_job_text=text)

    tracked = apps.create_application(title="Staff Engineer")
    apps.attach_job_ad(tracked.id, text)

    detail = apps.get_application(tracked.id)
    assert detail is not None
    assert detail.application.job_id == pasted.job_id


def test_attach_job_ad_returns_false_for_an_unknown_application(
    apps: PostgresApplicationRepository,
) -> None:
    assert apps.attach_job_ad(uuid.uuid4(), "some text") is False


def test_attach_job_ad_is_tenant_scoped(
    conn: Connection, apps: PostgresApplicationRepository, bob: uuid.UUID
) -> None:
    application = apps.create_application(title="Alice's role")
    bobs_view = PostgresApplicationRepository(conn, bob)
    assert bobs_view.attach_job_ad(application.id, "text") is False
