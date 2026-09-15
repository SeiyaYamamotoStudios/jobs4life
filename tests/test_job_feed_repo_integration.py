"""The feed's reader-side storage against real Postgres: last-looked times and marks.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures, as in `test_boards_repo_integration.py`; board history is written the
way the worker writes it -- lock, pure plan, apply -- from hand-built
`FetchResult`s, so nothing leaves the machine.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import job_feed_marks, users
from jfl_core.models import BoardJobEvent, ObservedJob
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.job_feed import PostgresJobFeedRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.feed import VISIBLE_FOR
from jfl_intake.normalise import fingerprint
from sqlalchemy import create_engine, func, insert, select
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


def obs(ext: str) -> ObservedJob:
    title = f"Role {ext}"
    return ObservedJob(
        external_id=ext,
        title=title,
        location="London",
        url=f"https://example.invalid/jobs/{ext}",
        fingerprint=fingerprint(title, "London"),
    )


def run_check(
    repo: PostgresBoardRepository, board_id: uuid.UUID, ids: list[str], at: dt.datetime
) -> None:
    jobs = [obs(i) for i in ids]
    state = repo.lock_check_state(
        board_id, observed_external_ids=ids, closed_since=at - REPOST_WINDOW
    )
    assert state is not None
    result = FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))
    repo.apply_check_plan(plan_check(state, result, observed_at=at), started_at=at, finished_at=at)


def board_with_events(conn: Connection, user_id: uuid.UUID) -> list[BoardJobEvent]:
    """Baseline a, b on day 0; day 1 sees b, c -- so `a` is gone and `c` is new."""
    boards = PostgresBoardRepository(conn, user_id)
    board = boards.add_board(
        platform="greenhouse",
        board_url=f"https://boards.greenhouse.io/{user_id.hex[:8]}",
        board_key={"token": user_id.hex[:8]},
    )
    run_check(boards, board.id, ["a", "b"], day(0))
    run_check(boards, board.id, ["b", "c"], day(1))
    events = boards.events_since(day(-1))
    assert {(e.kind, e.job.external_id) for e in events} == {("gone", "a"), ("new", "c")}
    return events


# -- last looked --------------------------------------------------------------------


def test_last_looked_starts_empty_and_only_moves_forward(
    conn: Connection, alice: uuid.UUID
) -> None:
    feed = PostgresJobFeedRepository(conn, alice)
    assert feed.last_looked_at() is None
    feed.set_last_looked_at(day(2))
    assert feed.last_looked_at() == day(2)
    feed.set_last_looked_at(day(1))  # a slower request finishing late
    assert feed.last_looked_at() == day(2)
    feed.set_last_looked_at(day(3))
    assert feed.last_looked_at() == day(3)


# -- marks --------------------------------------------------------------------------


def test_record_seen_is_idempotent_and_keeps_the_first_seen_time(
    conn: Connection, alice: uuid.UUID
) -> None:
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)

    first = feed.record_seen(events, now=day(1.5))
    assert {(m.kind, m.first_seen_at, m.event_at) for m in first} == {
        ("gone", day(1.5), day(1)),
        ("new", day(1.5), day(1)),
    }
    again = feed.record_seen(events, now=day(1.9))
    assert {(m.id, m.first_seen_at) for m in again} == {(m.id, m.first_seen_at) for m in first}
    count = conn.execute(
        select(func.count()).select_from(job_feed_marks).where(job_feed_marks.c.user_id == alice)
    ).scalar_one()
    assert count == 2


def test_live_marks_and_marks_after_a_time(conn: Connection, alice: uuid.UUID) -> None:
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    feed.record_seen(events, now=day(1.5))

    assert len(feed.live_marks(now=day(2), visible_for=VISIBLE_FOR)) == 2
    assert feed.live_marks(now=day(2.5), visible_for=VISIBLE_FOR) == []  # exactly 24h: gone
    assert len(feed.marks_for_events_after(day(0.5))) == 2
    assert feed.marks_for_events_after(day(1)) == []  # exclusive, like events_since


def test_dismiss_one_and_dismiss_live(conn: Connection, alice: uuid.UUID) -> None:
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    gone, new = sorted(feed.record_seen(events, now=day(1.5)), key=lambda m: m.kind)

    assert feed.dismiss(gone.id, now=day(1.6))
    assert feed.dismiss(gone.id, now=day(1.7))  # already dismissed: still true...
    (kept,) = [m for m in feed.marks_for_events_after(day(0)) if m.id == gone.id]
    assert kept.dismissed_at == day(1.6)  # ...and keeps the original time
    assert [m.id for m in feed.live_marks(now=day(1.8), visible_for=VISIBLE_FOR)] == [new.id]

    assert feed.dismiss_live(now=day(1.8), visible_for=VISIBLE_FOR) == 1
    assert feed.live_marks(now=day(1.8), visible_for=VISIBLE_FOR) == []
    assert not feed.dismiss(uuid.uuid4(), now=day(1.8))


def test_marks_cascade_with_the_board(conn: Connection, alice: uuid.UUID) -> None:
    events = board_with_events(conn, alice)
    PostgresJobFeedRepository(conn, alice).record_seen(events, now=day(1.5))
    PostgresBoardRepository(conn, alice).remove_board(events[0].board_id)
    assert PostgresJobFeedRepository(conn, alice).marks_for_events_after(day(-1)) == []


# -- tenancy ------------------------------------------------------------------------


def test_another_user_cannot_see_mark_or_dismiss_this_users_events(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    events = board_with_events(conn, alice)
    alices = PostgresJobFeedRepository(conn, alice)
    bobs = PostgresJobFeedRepository(conn, bob)
    marks = alices.record_seen(events, now=day(1.5))
    alices.set_last_looked_at(day(1.5))

    assert bobs.last_looked_at() is None
    assert bobs.live_marks(now=day(2), visible_for=VISIBLE_FOR) == []
    assert bobs.marks_for_events_after(day(-1)) == []
    # Alice's job and check ids handed to Bob's repository write nothing.
    assert bobs.record_seen(events, now=day(1.6)) == []
    for m in marks:
        assert not bobs.dismiss(m.id, now=day(1.6))
    assert bobs.dismiss_live(now=day(1.6), visible_for=VISIBLE_FOR) == 0

    assert {m.dismissed_at for m in alices.live_marks(now=day(2), visible_for=VISIBLE_FOR)} == {
        None
    }
    rows = conn.execute(select(job_feed_marks.c.user_id)).scalars().all()
    assert bob not in rows
