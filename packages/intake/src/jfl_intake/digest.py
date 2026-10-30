"""What the emailed digest says, and whether it is sent at all. Pure: no database,
no network, no clock, no SMTP.

The owner is putting his search on ice and wants the daily board checks to come
to him (PLAN.md C9). Everything that is a *decision* lives here, as one function
over plain inputs, so the rules can be tested without a mail server and read in
one place; `jfl_worker.handlers.digest` only gathers the inputs and acts on the
answer.

**What goes in.** Events of kind `new` and `returned` that happened in the
window `(since, until]`, run through exactly the lens `/changes` and `/jobs` use
(`jfl_intake.boardfilter.filter_open_jobs`: the saved filter plus per-board
exceptions, `show_hidden_unstated=False`). `gone` and `reposted` stay in the feed
only -- a job going away is not something to wake an inbox for, and a repost is
the same role the owner has already seen. The number hidden by the filter is
carried, so the email says "N matching your filter of M", never N alone.

**Events from checks that were not complete are never here**: the repository's
`events_since` already excludes them (rule C3), so nothing below re-derives it.

**Honesty about failures -- which boards are named.** A quiet inbox must never
imply "nothing changed" when the truth is "we could not look". A board is named
when the **latest** check of it that finished inside the window was not
`complete`: statuses `incomplete`, `truncated`, `unreachable`, `failed`, and
`held`.

  * *`held` counts.* A held check fetched the board fine, but the drop guard
    refused to apply it (the open count fell by half or more -- far likelier a
    broken adapter than an employer closing the roles). A held check changes no
    job's state, so new jobs on that board are not recorded either; "nothing new"
    would be false. It is named, in the words the Boards page uses for it.
  * *Latest, not any.* A board that failed at 07:00 and was checked complete at
    19:00 the same day lost nothing: a complete check re-derives the whole board,
    so anything posted in between is in that check's events. Naming it would be a
    warning about a problem that fixed itself. A board whose last check of the
    window failed is named every day it keeps failing, which is the point.

**When to send.** A digest if there are matching events or named boards. Else a
**weekly heartbeat** -- "still watching N boards, nothing new" -- if no email of
any kind went to this user in the last `HEARTBEAT_EVERY`; without it, an owner on
a quiet board list cannot tell "nothing happened" from "the digest broke". Never
an email to a user who watches no boards. Otherwise nothing, and the caller still
advances the cursor.

**Bounded.** At most `MAX_JOBS_LISTED` jobs are listed (an unfiltered Workday
board can post hundreds in a day); the rest are counted and the email says so and
links to `/changes`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jfl_core.models import (
    BoardCheck,
    BoardFilterException,
    BoardJob,
    BoardJobEvent,
    BoardJobEventKind,
    JobFilter,
    WatchedBoard,
)

from jfl_intake.boardfilter import filter_open_jobs
from jfl_intake.filtering import Match

# Task kinds, here so the web layer (which must not import `jfl_worker`) and the
# worker's registration share one spelling, as `jfl_intake.scheduling` does for
# board checks.
SCHEDULE_DIGESTS_KIND = "schedule_digests"
SEND_DIGEST_KIND = "send_digest"
SEND_TEST_EMAIL_KIND = "send_test_email"

DIGEST_EVENT_KINDS: frozenset[BoardJobEventKind] = frozenset({"new", "returned"})
HEARTBEAT_EVERY = dt.timedelta(days=7)
# The run that sends is a few minutes after the hour and the next one a few
# minutes after that hour a week later, so a strict seven days would slip a day
# every time (the second run lands minutes short). An hour of slack keeps it weekly.
HEARTBEAT_SLACK = dt.timedelta(hours=1)
MAX_JOBS_LISTED = 40
# A check stamps its events with the time it finished and commits a moment later.
# The window therefore ends this far before the run, so an event whose commit lands
# just after the run started is picked up by the next one instead of being skipped
# for good -- the cursor only ever moves to the end of the window, never past it.
SETTLE = dt.timedelta(minutes=5)

DigestKind = Literal["digest", "heartbeat"]

_KIND_ORDER: dict[BoardJobEventKind, int] = {"new": 0, "returned": 1, "reposted": 2, "gone": 3}


@dataclass(frozen=True, slots=True)
class DigestJob:
    event: BoardJobEvent
    match: Match[BoardJob]
    board: WatchedBoard | None


@dataclass(frozen=True, slots=True)
class FailedBoard:
    board: WatchedBoard
    check: BoardCheck


@dataclass(frozen=True, slots=True)
class DigestPlan:
    # None: send nothing. The caller still advances the cursor.
    kind: DigestKind | None
    jobs: tuple[DigestJob, ...] = ()
    # New/returned events in the window before the filter: the "M" in "N of M".
    events_found: int = 0
    # After the filter, before the cap: the "N".
    jobs_matching: int = 0
    # Of `jobs_matching`, how many are `returned` rather than `new`.
    returned_matching: int = 0
    # Matching events held back only because the job's workplace is not stated.
    hidden_unstated: int = 0
    failed: tuple[FailedBoard, ...] = ()
    boards_watched: int = 0

    @property
    def overflow(self) -> int:
        """Matching jobs not listed because of `MAX_JOBS_LISTED`."""
        return self.jobs_matching - len(self.jobs)


def unrecovered_failures(
    boards: Sequence[WatchedBoard],
    checks: Sequence[BoardCheck],
    *,
    since: dt.datetime,
    until: dt.datetime,
) -> list[FailedBoard]:
    """Boards whose latest check finishing in `(since, until]` was not complete.
    Ordered by label so the email reads the same from one day to the next.
    """
    latest: dict[object, BoardCheck] = {}
    for check in checks:
        if not since < check.finished_at <= until:
            continue
        current = latest.get(check.board_id)
        if current is None or check.finished_at >= current.finished_at:
            latest[check.board_id] = check
    failed = [
        FailedBoard(board=board, check=latest[board.id])
        for board in boards
        if board.id in latest and latest[board.id].status != "complete"
    ]
    return sorted(failed, key=lambda f: (f.board.label or f.board.board_url).casefold())


def plan_digest(
    *,
    events: Sequence[BoardJobEvent],
    checks: Sequence[BoardCheck],
    boards: Sequence[WatchedBoard],
    saved_filter: JobFilter,
    exceptions: Sequence[BoardFilterException],
    since: dt.datetime,
    until: dt.datetime,
    last_send_at: dt.datetime | None,
) -> DigestPlan:
    """Decide what the digest for `(since, until]` contains and whether it is sent.

    `last_send_at` is the time of the most recent email of ANY kind to this user
    (digest, heartbeat or test), None if there never was one.
    """
    if not boards:
        return DigestPlan(kind=None)

    in_window = [e for e in events if e.kind in DIGEST_EVENT_KINDS and since < e.at <= until]
    result = filter_open_jobs(
        [e.job for e in in_window],
        saved_filter,
        boards,
        exceptions,
        show_hidden_unstated=False,
    )
    match_by_job = {m.job.id: m for m in result.matches}
    board_by_id = {b.id: b for b in boards}
    shown = [e for e in in_window if e.job.id in match_by_job]
    shown.sort(key=lambda e: (-e.at.timestamp(), _KIND_ORDER[e.kind], e.job.title.casefold()))
    listed = tuple(
        DigestJob(event=e, match=match_by_job[e.job.id], board=board_by_id.get(e.job.board_id))
        for e in shown[:MAX_JOBS_LISTED]
    )
    failed = tuple(unrecovered_failures(boards, checks, since=since, until=until))

    def plan(kind: DigestKind | None) -> DigestPlan:
        return DigestPlan(
            kind=kind,
            jobs=listed,
            events_found=len(in_window),
            jobs_matching=len(shown),
            returned_matching=sum(1 for e in shown if e.kind == "returned"),
            hidden_unstated=result.hidden_unstated,
            failed=failed,
            boards_watched=len(boards),
        )

    if shown or failed:
        return plan("digest")
    if last_send_at is None or until - last_send_at >= HEARTBEAT_EVERY - HEARTBEAT_SLACK:
        return plan("heartbeat")
    return plan(None)
