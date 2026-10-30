"""The digest decision (`jfl_intake.digest.plan_digest`): what goes in, which boards
are named as unchecked, and whether anything is sent. Pure; no database, no mail.
"""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import (
    BoardCheck,
    BoardCheckErrorCode,
    BoardCheckStatus,
    BoardFilterException,
    BoardJob,
    BoardJobEvent,
    BoardJobEventKind,
    JobFilter,
    WatchedBoard,
    Workplace,
)
from jfl_intake.digest import (
    HEARTBEAT_EVERY,
    MAX_JOBS_LISTED,
    DigestPlan,
    plan_digest,
)

USER = uuid.uuid4()
NOW = dt.datetime(2026, 10, 30, 8, 0, tzinfo=dt.UTC)
SINCE = NOW - dt.timedelta(days=1)
UNTIL = NOW - dt.timedelta(minutes=5)
REMOTE_ONLY = JobFilter(workplace_mode="remote_only")
ANY = JobFilter(workplace_mode="custom", workplaces=["remote", "hybrid", "onsite", "unknown"])


def make_board(label: str | None = "Acme", platform: str = "greenhouse") -> WatchedBoard:
    return WatchedBoard(
        id=uuid.uuid4(),
        user_id=USER,
        platform=platform,  # type: ignore[arg-type]
        board_url=f"https://example.invalid/{label}",
        board_key={"token": str(label)},
        label=label,
        created_at=SINCE,
        next_check_at=NOW,
    )


def make_event(
    board: WatchedBoard,
    title: str,
    *,
    kind: BoardJobEventKind = "new",
    at: dt.datetime | None = None,
    workplace: Workplace = "remote",
) -> BoardJobEvent:
    when = at or NOW - dt.timedelta(hours=3)
    check = uuid.uuid4()
    return BoardJobEvent(
        kind=kind,
        board_id=board.id,
        check_id=check,
        at=when,
        job=BoardJob(
            id=uuid.uuid4(),
            user_id=USER,
            board_id=board.id,
            external_id=title,
            title=title,
            fingerprint=title,
            workplace=workplace,
            first_seen_check_id=check,
            first_seen_at=when,
            last_seen_at=when,
        ),
    )


def make_check(
    board: WatchedBoard,
    status: BoardCheckStatus = "complete",
    *,
    at: dt.datetime | None = None,
    code: BoardCheckErrorCode | None = None,
) -> BoardCheck:
    when = at or NOW - dt.timedelta(hours=2)
    return BoardCheck(
        id=uuid.uuid4(),
        user_id=USER,
        board_id=board.id,
        started_at=when,
        finished_at=when,
        status=status,
        jobs_seen=5,
        error_code=code,
    )


def plan(
    boards: list[WatchedBoard],
    events: list[BoardJobEvent] | None = None,
    checks: list[BoardCheck] | None = None,
    *,
    saved: JobFilter = ANY,
    exceptions: list[BoardFilterException] | None = None,
    last_send_at: dt.datetime | None = NOW - dt.timedelta(days=1),
) -> DigestPlan:
    return plan_digest(
        events=events or [],
        checks=checks or [],
        boards=boards,
        saved_filter=saved,
        exceptions=exceptions or [],
        since=SINCE,
        until=UNTIL,
        last_send_at=last_send_at,
    )


# -- what goes in ---------------------------------------------------------------


def test_only_new_and_returned_events_are_listed() -> None:
    board = make_board()
    events = [
        make_event(board, "New one", kind="new"),
        make_event(board, "Back again", kind="returned"),
        make_event(board, "Gone one", kind="gone"),
        make_event(board, "Reposted one", kind="reposted"),
    ]
    result = plan([board], events)
    assert result.kind == "digest"
    assert {j.event.job.title for j in result.jobs} == {"New one", "Back again"}
    assert result.events_found == 2
    assert result.returned_matching == 1


def test_events_outside_the_window_are_ignored() -> None:
    board = make_board()
    events = [
        make_event(board, "Too old", at=SINCE),  # exclusive lower bound
        make_event(board, "Too new", at=UNTIL + dt.timedelta(seconds=1)),
        make_event(board, "Just inside", at=UNTIL),
    ]
    result = plan([board], events)
    assert [j.event.job.title for j in result.jobs] == ["Just inside"]


def test_the_saved_filter_applies_and_the_counts_say_so() -> None:
    board = make_board()
    events = [
        make_event(board, "Remote role", workplace="remote"),
        make_event(board, "Office role", workplace="onsite"),
    ]
    result = plan([board], events, saved=REMOTE_ONLY)
    assert [j.event.job.title for j in result.jobs] == ["Remote role"]
    # "N matching your filter of M": both numbers are carried, never N alone.
    assert (result.jobs_matching, result.events_found) == (1, 2)


def test_a_job_whose_workplace_is_unstated_stays_hidden_and_is_counted() -> None:
    # Lever states workplace, so an unstated one is held back (Greenhouse and
    # Workday, which cannot state it, include such jobs by default).
    board = make_board(platform="lever")
    events = [make_event(board, "Mystery role", workplace="unknown")]
    result = plan([board], events, saved=REMOTE_ONLY)
    assert result.jobs == ()
    assert result.hidden_unstated == 1


def test_a_per_board_exception_lets_a_job_through_as_on_the_pages() -> None:
    board = make_board()
    exception = BoardFilterException(
        id=uuid.uuid4(),
        board_id=board.id,
        workplaces=["onsite"],
        note="close to home",
        created_at=NOW,
        updated_at=NOW,
    )
    events = [make_event(board, "Local office role", workplace="onsite")]
    without = plan([board], events, saved=REMOTE_ONLY)
    with_exception = plan([board], events, saved=REMOTE_ONLY, exceptions=[exception])
    assert without.jobs == ()
    assert [j.event.job.title for j in with_exception.jobs] == ["Local office role"]
    assert with_exception.jobs[0].match.reason == "exception"


def test_newest_first_and_capped_with_the_overflow_counted() -> None:
    board = make_board()
    events = [
        make_event(board, f"Role {i:03d}", at=UNTIL - dt.timedelta(minutes=i))
        for i in range(MAX_JOBS_LISTED + 7)
    ]
    result = plan([board], events)
    assert len(result.jobs) == MAX_JOBS_LISTED
    assert result.overflow == 7
    assert result.jobs_matching == MAX_JOBS_LISTED + 7
    assert result.jobs[0].event.job.title == "Role 000"


# -- honesty about failures -----------------------------------------------------


def test_a_board_whose_latest_check_failed_is_named_even_with_no_events() -> None:
    board = make_board("Broken Co")
    result = plan([board], checks=[make_check(board, "unreachable", code="timeout")])
    assert result.kind == "digest"
    assert [f.board.label for f in result.failed] == ["Broken Co"]
    assert result.failed[0].check.error_code == "timeout"


def test_every_non_complete_status_is_named_including_held() -> None:
    boards = [make_board(f"B{i}") for i in range(5)]
    statuses: list[BoardCheckStatus] = ["incomplete", "truncated", "unreachable", "failed", "held"]
    checks = [make_check(b, s, code="timeout") for b, s in zip(boards, statuses, strict=True)]
    result = plan(boards, checks=checks)
    assert len(result.failed) == 5


def test_a_board_that_recovered_later_in_the_window_is_not_named() -> None:
    """A complete check re-derives the whole board, so nothing was missed."""
    board = make_board()
    checks = [
        make_check(board, "failed", at=NOW - dt.timedelta(hours=9), code="timeout"),
        make_check(board, "complete", at=NOW - dt.timedelta(hours=1)),
    ]
    assert plan([board], checks=checks).failed == ()


def test_a_failure_after_a_success_is_named() -> None:
    board = make_board()
    checks = [
        make_check(board, "complete", at=NOW - dt.timedelta(hours=9)),
        make_check(board, "failed", at=NOW - dt.timedelta(hours=1), code="not_found"),
    ]
    assert len(plan([board], checks=checks).failed) == 1


def test_checks_outside_the_window_are_not_failures() -> None:
    board = make_board()
    old = make_check(board, "failed", at=SINCE - dt.timedelta(hours=1), code="timeout")
    assert plan([board], checks=[old]).failed == ()


def test_failed_boards_do_not_hide_matching_jobs_and_both_are_reported() -> None:
    good, bad = make_board("Good"), make_board("Bad")
    result = plan(
        [good, bad],
        [make_event(good, "A role")],
        [make_check(good), make_check(bad, "truncated", code="listing_ceiling")],
    )
    assert len(result.jobs) == 1 and len(result.failed) == 1


# -- when to send ---------------------------------------------------------------


def test_never_an_email_for_a_user_with_no_boards() -> None:
    result = plan([], last_send_at=None)
    assert result.kind is None


def test_a_quiet_day_sends_nothing_when_something_went_recently() -> None:
    board = make_board()
    result = plan([board], checks=[make_check(board)], last_send_at=NOW - dt.timedelta(days=2))
    assert result.kind is None


def test_a_weekly_heartbeat_when_nothing_has_been_sent_for_a_week() -> None:
    board = make_board()
    result = plan(
        [board],
        checks=[make_check(board)],
        last_send_at=NOW - HEARTBEAT_EVERY - dt.timedelta(hours=1),
    )
    assert result.kind == "heartbeat"
    assert result.boards_watched == 1


def test_heartbeat_is_weekly_not_eight_daily_despite_the_run_drifting_minutes() -> None:
    board = make_board()
    last = UNTIL - HEARTBEAT_EVERY + dt.timedelta(minutes=20)
    assert plan([board], last_send_at=last).kind == "heartbeat"


def test_a_heartbeat_when_nothing_was_ever_sent() -> None:
    board = make_board()
    assert plan([board], last_send_at=None).kind == "heartbeat"


def test_events_beat_the_heartbeat() -> None:
    board = make_board()
    result = plan([board], [make_event(board, "A role")], last_send_at=None)
    assert result.kind == "digest"
