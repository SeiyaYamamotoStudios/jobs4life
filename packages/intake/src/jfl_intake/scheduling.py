"""When boards are checked, and the one enqueue the web layer calls.

Two task kinds, defined here rather than in the worker so that the web layer --
which must not import `jfl_worker` -- and the worker's registration share one
spelling of each:

  * `check_board` -- check one board. Payload: `{"board_id": ...}`, ids only;
  * `schedule_board_checks` -- recurring, enqueued by the worker's ticker the
    way the session purge is, and it enqueues `check_board` for every board due.

**Staggering.** Every board has a fixed daily slot: its id's first eight bytes
modulo 86,400 seconds past midnight UTC. Ids are random, so slots spread
uniformly over the day and two hundred boards never fire in the same minute --
which matters less for load than for fairness, because the worker is serial and
claims in `scheduled_at` order: a burst of board checks would sit in front of a
user's job-ad extraction. The next check is the first slot at least
`MIN_GAP` (12 hours) after the last one was scheduled, so a board is checked
about once a day at about the same time, and a board whose first check ran
off-slot settles onto its slot the next day.

After worker downtime every overdue board is due at once. Those are enqueued
`ENQUEUE_SPACING` (30 seconds) apart, which keeps the queue interleavable with
other users' work while the backlog drains.

**No backlog of duplicates.** The ticker uses the queue's `enqueue_unique`, so a
worker down for a week enqueues one scheduling pass on restart, not 672. A board
is claimed by that pass and moved to its next slot in the same transaction, so
it is due at most once per slot however late the pass runs. And `check_board`
itself is deduplicated per board: no second check is queued while one is pending
or running, whether the first came from the schedule or from "check now".
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from jfl_core.models import Task, WatchedBoard

CHECK_BOARD_KIND = "check_board"
SCHEDULE_BOARD_CHECKS_KIND = "schedule_board_checks"

CHECK_INTERVAL = dt.timedelta(days=1)
MIN_GAP = dt.timedelta(hours=12)
ENQUEUE_SPACING = dt.timedelta(seconds=30)
# Boards claimed per scheduling pass. The pass runs every 15 minutes, so this
# drains ~19k boards a day -- far past v1 -- while keeping one pass's
# transaction short.
SCHEDULE_BATCH = 200

_SECONDS_PER_DAY = 86_400


def daily_slot(board_id: uuid.UUID) -> dt.timedelta:
    """Seconds past midnight UTC at which this board is checked. Stable for the
    life of the board, so its check time does not drift.
    """
    return dt.timedelta(seconds=int.from_bytes(board_id.bytes[:8], "big") % _SECONDS_PER_DAY)


def next_check_at(board_id: uuid.UUID, *, after: dt.datetime) -> dt.datetime:
    """The first daily slot for this board at least `MIN_GAP` after `after`.
    Always in `[after + 12h, after + 36h)`.
    """
    earliest = (after + MIN_GAP).astimezone(dt.UTC)
    midnight = dt.datetime.combine(earliest.date(), dt.time(0), tzinfo=dt.UTC)
    candidate = midnight + daily_slot(board_id)
    if candidate < earliest:
        candidate += CHECK_INTERVAL
    return candidate


class BoardCheckSource(Protocol):
    """The two reads "check now" needs. `PostgresBoardRepository` satisfies it."""

    @property
    def user_id(self) -> uuid.UUID: ...

    def get_board(self, board_id: uuid.UUID) -> WatchedBoard | None: ...

    def check_task_queued(self, board_id: uuid.UUID, *, kind: str) -> bool: ...


class TaskSink(Protocol):
    """`PostgresTaskRepository` satisfies it."""

    @property
    def user_id(self) -> uuid.UUID: ...

    def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int = ...,
        scheduled_at: dt.datetime | None = None,
    ) -> Task: ...


def enqueue_board_check(
    boards: BoardCheckSource,
    tasks: TaskSink,
    board_id: uuid.UUID,
    *,
    scheduled_at: dt.datetime | None = None,
) -> Task | None:
    """Queue a check of one board. "Check now" is this with no `scheduled_at`.

    Returns None, writing nothing, when the board is not this user's (or does
    not exist -- the same answer, deliberately) or a check of it is already
    pending or running.

    Takes two repositories already bound to a user and no user id: the caller
    cannot name the wrong tenant, and two repositories bound to different users
    are refused outright rather than trusted.
    """
    if boards.user_id != tasks.user_id:
        raise ValueError("board and task repositories are bound to different users")
    if boards.get_board(board_id) is None:
        return None
    if boards.check_task_queued(board_id, kind=CHECK_BOARD_KIND):
        return None
    return tasks.enqueue(
        kind=CHECK_BOARD_KIND, payload={"board_id": str(board_id)}, scheduled_at=scheduled_at
    )


def enqueue_all_board_checks(
    boards: BoardCheckSource,
    tasks: TaskSink,
    board_ids: Sequence[uuid.UUID],
    *,
    now: dt.datetime,
    spacing: dt.timedelta = ENQUEUE_SPACING,
) -> tuple[int, int]:
    """The web layer's "check all boards" button: `enqueue_board_check` for
    every id in `board_ids`, one call each, no second check mechanism.

    Staggered exactly the way a restart's backlog already is -- see the
    module docstring's "No backlog of duplicates" and `_schedule_board_checks`
    in the worker, which this mirrors. A single worker process runs one task
    at a time, so staggering buys nothing in wall-clock throughput; what it
    buys is fairness, because the claim order is `scheduled_at asc`: without
    it, N boards all due "now" would sort ahead of anything else queued in
    the meantime, and a person with fifty boards would make their own job-ad
    extraction -- or another user's unrelated work -- wait behind the whole
    batch. `spacing` counts only checks this call actually queues, so a board
    skipped as already-queued costs no slot: ten boards where three are
    already checking still land thirty seconds apart, not forty-five.

    Returns `(queued, skipped)`.
    """
    queued = 0
    skipped = 0
    for board_id in board_ids:
        task = enqueue_board_check(boards, tasks, board_id, scheduled_at=now + spacing * queued)
        if task is None:
            skipped += 1
        else:
            queued += 1
    return queued, skipped
