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
from jfl_core.storage.job_feed import PostgresJobFeedRepository, purge_stale_marks
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.feed import VISIBLE_FOR, derive_since, visible_events
from jfl_intake.normalise import fingerprint
from sqlalchemy import create_engine, delete, func, insert, select
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


# -- purge --------------------------------------------------------------------------
#
# `board_with_events` always gives two marks sharing one `event_at` (day(1) --
# the one check that produced "gone a" and "new c" at once), which is exactly
# the case `purge_stale_marks` has to get right: two events from the same check
# can have independent lifetimes once one is dismissed and the other is not.


def test_purge_deletes_a_dead_mark_with_nothing_live_to_block_it(
    conn: Connection, alice: uuid.UUID
) -> None:
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    feed.record_seen(events, now=day(1.5))
    feed.set_last_looked_at(day(1.5))
    feed.dismiss_live(now=day(1.5), visible_for=VISIBLE_FOR)  # both dead, none live

    assert purge_stale_marks(conn, now=day(1.6), visible_for=VISIBLE_FOR) == 2
    assert feed.marks_for_events_after(day(0)) == []


def test_purge_keeps_a_mark_still_inside_its_visibility_window(
    conn: Connection, alice: uuid.UUID
) -> None:
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    feed.record_seen(events, now=day(1.5))

    # Within 24h and undismissed: still live, so still on the page.
    assert purge_stale_marks(conn, now=day(1.6), visible_for=VISIBLE_FOR) == 0
    assert len(feed.marks_for_events_after(day(0))) == 2


def test_purge_refuses_a_dead_mark_that_a_live_mark_still_needs(
    conn: Connection, alice: uuid.UUID
) -> None:
    """The case the predicate exists for: `gone` and `new` share `event_at`
    (one check produced both). Dismissing `gone` alone must not make it
    purgeable while `new`, sharing that exact event time, is still live --
    `derive_since` can widen `since` back to just before `new`'s event, which
    sits at or before `gone`'s. See `purge_stale_marks`'s docstring.
    """
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    gone_mark, new_mark = sorted(feed.record_seen(events, now=day(1.5)), key=lambda m: m.kind)
    assert gone_mark.kind == "gone" and new_mark.kind == "new"
    assert gone_mark.event_at == new_mark.event_at  # the shared-check case
    feed.set_last_looked_at(day(1.5))
    feed.dismiss(gone_mark.id, now=day(1.6))  # dead; `new_mark` stays live

    assert purge_stale_marks(conn, now=day(1.6), visible_for=VISIBLE_FOR) == 0
    kept_ids = {m.id for m in feed.marks_for_events_after(day(0))}
    assert kept_ids == {gone_mark.id, new_mark.id}

    # Once the blocking live mark also dies (dismissed here), both are fair
    # game -- the guard is gated on liveness, not on being dismissed first.
    feed.dismiss(new_mark.id, now=day(1.7))
    assert purge_stale_marks(conn, now=day(1.7), visible_for=VISIBLE_FOR) == 2
    assert feed.marks_for_events_after(day(0)) == []


def test_a_dead_mark_purge_correctly_keeps_would_resurface_its_event_if_deleted(
    conn: Connection, alice: uuid.UUID
) -> None:
    """The guard's own docstring names this test: forcing the deletion
    `purge_stale_marks` refuses does **not** resurface the old event, because
    `visible_events` judges "is this new" against `last_looked_at` directly,
    never against the widened `since` `derive_since` builds for the database
    query -- and `last_looked_at` already sits at or after this event by the
    time the mark existed to be dismissed at all. Two independent things are
    true at once here: the guard is conservative (it refuses a deletion that
    is not actually dangerous against today's code), and the underlying
    "since you last looked" comparison is where the real safety lives. Forcing
    the deletion anyway is how the second claim gets checked without relying
    on trusting the first.
    """
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    gone_mark, new_mark = sorted(feed.record_seen(events, now=day(1.5)), key=lambda m: m.kind)
    feed.set_last_looked_at(day(1.5))
    feed.dismiss(gone_mark.id, now=day(1.6))

    assert purge_stale_marks(conn, now=day(1.9), visible_for=VISIBLE_FOR) == 0  # refused, correctly

    # Force exactly what the guard refused, to check the deeper claim directly.
    conn.execute(delete(job_feed_marks).where(job_feed_marks.c.id == gone_mark.id))
    later = day(1.9)  # `new_mark` is still live (first_seen_at=day(1.5), <24h)
    boards = PostgresBoardRepository(conn, alice)
    previous_look = feed.last_looked_at()
    # The widened `since`: pulled back to just before `new_mark`'s event, which
    # is the same instant as the deleted mark's -- so the database query below
    # re-fetches the "gone" event too. If anything were going to resurface it,
    # it would be here.
    since = derive_since(
        previous_look, feed.live_marks(now=later, visible_for=VISIBLE_FOR), now=later
    )
    assert since < gone_mark.event_at  # confirms the widening really did reach back this far
    items = visible_events(
        boards.events_since(since),
        feed.marks_for_events_after(since),
        last_looked_at=previous_look,
        now=later,
    )
    resurfaced = {(i.event.kind, i.event.job.external_id) for i in items if i.mark is None}
    assert resurfaced == set()  # not resurfaced, even with the tombstone gone
    assert {i.event.kind for i in items} == {"new"}  # only the still-live sibling shows
    # `new_mark` itself was never touched by any of this.
    assert new_mark.id in {m.id for m in feed.marks_for_events_after(day(0))}


def test_purge_does_not_cross_a_tenancy_boundary(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    """Alice's dead marks must be purgeable even though Bob has a live mark at
    the very same `event_at` -- the two users' histories happen to share a
    timestamp only because both boards were built by `board_with_events` the
    same way. The correlated subquery must not let Bob's liveness protect
    Alice's rows, or one user's activity would keep another user's table
    growing. (Both of Alice's own marks are dismissed first, so nothing of
    *hers* is left live to trigger the same-user guard `test_purge_refuses_a_
    dead_mark_that_a_live_mark_still_needs` covers -- this test is only about
    the tenancy boundary.)
    """
    alice_events = board_with_events(conn, alice)
    bob_events = board_with_events(conn, bob)
    alice_feed = PostgresJobFeedRepository(conn, alice)
    bob_feed = PostgresJobFeedRepository(conn, bob)

    alice_marks = alice_feed.record_seen(alice_events, now=day(1.5))
    (alice_gone,) = [m for m in alice_marks if m.kind == "gone"]
    bob_marks = bob_feed.record_seen(bob_events, now=day(1.5))
    # Same instant, different user -- both boards were built the same way.
    assert any(m.event_at == alice_gone.event_at for m in bob_marks)

    alice_feed.set_last_looked_at(day(1.5))
    alice_feed.dismiss_live(now=day(1.6), visible_for=VISIBLE_FOR)  # both of Alice's marks, dead
    # Bob's marks are untouched: still live, still sharing `alice_gone`'s instant.

    assert purge_stale_marks(conn, now=day(1.6), visible_for=VISIBLE_FOR) == 2
    assert alice_feed.marks_for_events_after(day(0)) == []
    assert len(bob_feed.marks_for_events_after(day(0))) == 2  # Bob's are untouched


def test_purge_is_idempotent(conn: Connection, alice: uuid.UUID) -> None:
    events = board_with_events(conn, alice)
    feed = PostgresJobFeedRepository(conn, alice)
    feed.record_seen(events, now=day(1.5))
    feed.set_last_looked_at(day(1.5))
    feed.dismiss_live(now=day(1.5), visible_for=VISIBLE_FOR)

    assert purge_stale_marks(conn, now=day(1.6), visible_for=VISIBLE_FOR) == 2
    assert purge_stale_marks(conn, now=day(1.6), visible_for=VISIBLE_FOR) == 0
    assert purge_stale_marks(conn, now=day(5), visible_for=VISIBLE_FOR) == 0
