"""The feed's visibility rule (`jfl_intake.feed`). No database, no network, no clock."""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import BoardJob, BoardJobEvent, BoardJobEventKind, JobFeedMark
from jfl_intake.feed import (
    FIRST_VISIT_LOOKBACK,
    VISIBLE_FOR,
    derive_since,
    visible_events,
)

USER = uuid.uuid4()
BOARD = uuid.uuid4()
NOW = dt.datetime(2026, 9, 15, 9, 0, tzinfo=dt.UTC)


def hours(n: float) -> dt.timedelta:
    return dt.timedelta(hours=n)


def event(
    title: str, at: dt.datetime, kind: BoardJobEventKind = "new", *, check: uuid.UUID | None = None
) -> BoardJobEvent:
    check_id = check or uuid.uuid4()
    return BoardJobEvent(
        kind=kind,
        board_id=BOARD,
        check_id=check_id,
        at=at,
        job=BoardJob(
            id=uuid.uuid4(),
            user_id=USER,
            board_id=BOARD,
            external_id=title,
            title=title,
            fingerprint=title,
            first_seen_check_id=check_id,
            first_seen_at=at,
            last_seen_at=at,
        ),
    )


def mark(
    e: BoardJobEvent, first_seen_at: dt.datetime, dismissed_at: dt.datetime | None = None
) -> JobFeedMark:
    return JobFeedMark(
        id=uuid.uuid4(),
        job_id=e.job.id,
        check_id=e.check_id,
        kind=e.kind,
        event_at=e.at,
        first_seen_at=first_seen_at,
        dismissed_at=dismissed_at,
    )


def shown(
    events: list[BoardJobEvent],
    marks: list[JobFeedMark],
    *,
    last_looked_at: dt.datetime | None,
    now: dt.datetime = NOW,
) -> list[str]:
    items = visible_events(events, marks, last_looked_at=last_looked_at, now=now)
    return [item.event.job.title for item in items]


# -- unseen -------------------------------------------------------------------------


def test_an_event_after_the_last_look_is_unseen_and_shown_without_a_mark() -> None:
    e = event("fresh", NOW - hours(2))
    (item,) = visible_events([e], [], last_looked_at=NOW - hours(3), now=NOW)
    assert item.event is e and item.mark is None


def test_an_unmarked_event_at_or_before_the_last_look_is_not_shown() -> None:
    looked = NOW - hours(3)
    before = event("before", looked - hours(1))
    exactly = event("exactly", looked)
    assert shown([before, exactly], [], last_looked_at=looked) == []


def test_events_between_two_looks_show_on_the_second() -> None:
    first_look = NOW - hours(30)
    between = event("between", NOW - hours(10))
    older = event("older", first_look - hours(1))
    assert shown([between, older], [], last_looked_at=first_look) == ["between"]


# -- seen ---------------------------------------------------------------------------


def test_a_seen_event_stays_for_24_hours_from_first_seen_not_from_when_it_happened() -> None:
    # Happened three days ago, first seen 23 hours ago: still visible, though the
    # user has looked since (the last look is after it).
    e = event("old news", NOW - dt.timedelta(days=3))
    m = mark(e, first_seen_at=NOW - hours(23))
    assert shown([e], [m], last_looked_at=NOW - hours(1)) == ["old news"]


def test_the_24_hour_boundary_is_exclusive() -> None:
    e = event("boundary", NOW - hours(30))
    just_inside = mark(e, first_seen_at=NOW - VISIBLE_FOR + dt.timedelta(microseconds=1))
    exactly = mark(e, first_seen_at=NOW - VISIBLE_FOR)
    assert shown([e], [just_inside], last_looked_at=NOW - hours(1)) == ["boundary"]
    assert shown([e], [exactly], last_looked_at=NOW - hours(1)) == []


def test_a_dismissed_event_is_gone_even_within_24_hours() -> None:
    e = event("dismissed", NOW - hours(5))
    m = mark(e, first_seen_at=NOW - hours(4), dismissed_at=NOW - hours(3))
    assert shown([e], [m], last_looked_at=NOW - hours(4)) == []


def test_a_mark_decides_even_for_an_event_dated_after_the_last_look() -> None:
    """A dismissal is final whatever the clocks did."""
    e = event("future-dated", NOW - hours(1))
    dismissed = mark(e, first_seen_at=NOW - hours(2), dismissed_at=NOW - hours(2))
    expired = mark(e, first_seen_at=NOW - hours(25))
    assert shown([e], [dismissed], last_looked_at=NOW - hours(2)) == []
    assert shown([e], [expired], last_looked_at=NOW - hours(2)) == []


def test_a_mark_for_a_different_kind_or_check_of_the_same_job_does_not_apply() -> None:
    gone = event("flapping", NOW - hours(10), "gone")
    returned = BoardJobEvent(
        kind="returned", board_id=BOARD, check_id=uuid.uuid4(), at=NOW - hours(2), job=gone.job
    )
    dismissed_gone = mark(gone, first_seen_at=NOW - hours(9), dismissed_at=NOW - hours(9))
    items = visible_events(
        [gone, returned], [dismissed_gone], last_looked_at=NOW - hours(9), now=NOW
    )
    assert [(i.event.kind, i.mark) for i in items] == [("returned", None)]


# -- first visit --------------------------------------------------------------------


def test_a_first_visit_looks_back_seven_days() -> None:
    inside = event("inside", NOW - FIRST_VISIT_LOOKBACK + hours(1))
    outside = event("outside", NOW - FIRST_VISIT_LOOKBACK - hours(1))
    assert shown([inside, outside], [], last_looked_at=None) == ["inside"]


# -- ordering -----------------------------------------------------------------------


def test_newest_first_and_within_one_check_gone_before_new() -> None:
    check = uuid.uuid4()
    at = NOW - hours(1)
    items = [
        event("older", NOW - hours(5)),
        event("new one", at, "new", check=check),
        event("gone one", at, "gone", check=check),
    ]
    assert shown(items, [], last_looked_at=NOW - hours(6)) == ["gone one", "new one", "older"]


# -- how far back to derive ---------------------------------------------------------


def test_derive_since_is_the_last_look_when_no_live_mark_is_older() -> None:
    looked = NOW - hours(3)
    e = event("recent", NOW - hours(1))
    assert derive_since(looked, [mark(e, NOW - hours(1))], now=NOW) == looked
    assert derive_since(None, [], now=NOW) == NOW - FIRST_VISIT_LOOKBACK


def test_derive_since_reaches_back_to_the_oldest_live_mark_and_ignores_dead_ones() -> None:
    looked = NOW - hours(3)
    live = event("live", NOW - dt.timedelta(days=2))
    expired = event("expired", NOW - dt.timedelta(days=5))
    dismissed = event("dismissed", NOW - dt.timedelta(days=4))
    marks = [
        mark(live, NOW - hours(20)),
        mark(expired, NOW - hours(30)),
        mark(dismissed, NOW - hours(2), dismissed_at=NOW - hours(1)),
    ]
    since = derive_since(looked, marks, now=NOW)
    assert since < live.at
    assert since == live.at - dt.timedelta(microseconds=1)
