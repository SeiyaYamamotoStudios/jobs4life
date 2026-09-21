"""Staggered daily slots, and the "check now" enqueue, against fakes."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pytest
from jfl_core.models import Task, WatchedBoard
from jfl_intake.scheduling import (
    CHECK_BOARD_KIND,
    ENQUEUE_SPACING,
    MIN_GAP,
    daily_slot,
    enqueue_all_board_checks,
    enqueue_board_check,
    next_check_at,
)

NOW = dt.datetime(2026, 9, 10, 9, 30, tzinfo=dt.UTC)


def test_slots_are_stable_and_spread_across_the_day() -> None:
    board = uuid.uuid4()
    assert daily_slot(board) == daily_slot(board)
    assert dt.timedelta(0) <= daily_slot(board) < dt.timedelta(days=1)

    hours = {int(daily_slot(uuid.uuid4()).total_seconds() // 3600) for _ in range(500)}
    assert len(hours) >= 20  # 500 random ids leave almost no hour of the day empty


@pytest.mark.parametrize("seed", range(20))
def test_the_next_check_is_on_the_boards_slot_at_least_twelve_hours_on(seed: int) -> None:
    board = uuid.UUID(int=seed * 7_919_000_003_000_001)
    after = NOW + dt.timedelta(minutes=seed * 37)
    upcoming = next_check_at(board, after=after)

    assert after + MIN_GAP <= upcoming < after + MIN_GAP + dt.timedelta(days=1)
    midnight = dt.datetime.combine(upcoming.date(), dt.time(0), tzinfo=dt.UTC)
    assert upcoming - midnight == daily_slot(board)
    # And once on its slot, a board stays a day apart.
    assert next_check_at(board, after=upcoming) == upcoming + dt.timedelta(days=1)


class FakeBoards:
    def __init__(self, user_id: uuid.UUID, boards: set[uuid.UUID], queued: set[uuid.UUID]) -> None:
        self._user_id = user_id
        self._boards = boards
        self._queued = queued

    @property
    def user_id(self) -> uuid.UUID:
        return self._user_id

    def get_board(self, board_id: uuid.UUID) -> WatchedBoard | None:
        if board_id not in self._boards:
            return None
        return WatchedBoard(
            id=board_id,
            user_id=self._user_id,
            platform="greenhouse",
            board_url="https://boards.greenhouse.io/x",
            board_key={"token": "x"},
            created_at=NOW,
            next_check_at=NOW,
        )

    def check_task_queued(self, board_id: uuid.UUID, *, kind: str) -> bool:
        assert kind == CHECK_BOARD_KIND
        return board_id in self._queued


class FakeTasks:
    def __init__(self, user_id: uuid.UUID) -> None:
        self._user_id = user_id
        self.enqueued: list[dict[str, Any]] = []

    @property
    def user_id(self) -> uuid.UUID:
        return self._user_id

    def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any] | None = None,
        max_attempts: int = 3,
        scheduled_at: dt.datetime | None = None,
    ) -> Task:
        self.enqueued.append({"kind": kind, "payload": payload, "scheduled_at": scheduled_at})
        return Task(
            id=uuid.uuid4(),
            user_id=self._user_id,
            kind=kind,
            payload=payload or {},
            status="pending",
            attempts=0,
            max_attempts=max_attempts,
            scheduled_at=scheduled_at or NOW,
            created_at=NOW,
            updated_at=NOW,
        )


def test_check_now_enqueues_a_task_carrying_only_the_board_id() -> None:
    user, board = uuid.uuid4(), uuid.uuid4()
    tasks = FakeTasks(user)
    task = enqueue_board_check(FakeBoards(user, {board}, set()), tasks, board)
    assert task is not None
    assert tasks.enqueued == [
        {"kind": CHECK_BOARD_KIND, "payload": {"board_id": str(board)}, "scheduled_at": None}
    ]


def test_check_now_is_a_no_op_while_a_check_is_already_queued() -> None:
    user, board = uuid.uuid4(), uuid.uuid4()
    tasks = FakeTasks(user)
    assert enqueue_board_check(FakeBoards(user, {board}, {board}), tasks, board) is None
    assert tasks.enqueued == []


def test_check_now_on_a_board_that_is_not_this_users_writes_nothing() -> None:
    user = uuid.uuid4()
    tasks = FakeTasks(user)
    assert enqueue_board_check(FakeBoards(user, set(), set()), tasks, uuid.uuid4()) is None
    assert tasks.enqueued == []


def test_repositories_bound_to_different_users_are_refused() -> None:
    board = uuid.uuid4()
    with pytest.raises(ValueError, match="different users"):
        enqueue_board_check(
            FakeBoards(uuid.uuid4(), {board}, set()), FakeTasks(uuid.uuid4()), board
        )


# --------------------------------------------------------------------------
# enqueue_all_board_checks -- "check all boards now"
# --------------------------------------------------------------------------


def test_check_all_queues_one_per_board() -> None:
    user = uuid.uuid4()
    board_ids = [uuid.uuid4() for _ in range(3)]
    tasks = FakeTasks(user)
    queued, skipped = enqueue_all_board_checks(
        FakeBoards(user, set(board_ids), set()), tasks, board_ids, now=NOW
    )
    assert (queued, skipped) == (3, 0)
    assert [e["payload"] for e in tasks.enqueued] == [{"board_id": str(b)} for b in board_ids]


def test_check_all_skips_boards_with_a_check_already_queued() -> None:
    user = uuid.uuid4()
    board_ids = [uuid.uuid4() for _ in range(3)]
    already_queued = {board_ids[1]}
    tasks = FakeTasks(user)
    queued, skipped = enqueue_all_board_checks(
        FakeBoards(user, set(board_ids), already_queued), tasks, board_ids, now=NOW
    )
    assert (queued, skipped) == (2, 1)
    queued_ids = {e["payload"]["board_id"] for e in tasks.enqueued}
    assert queued_ids == {str(board_ids[0]), str(board_ids[2])}


def test_check_all_staggers_only_the_boards_actually_queued() -> None:
    """A skipped board costs no slot: three boards where the middle one is
    already queued still land `ENQUEUE_SPACING` apart, not with a gap left
    for the one that was skipped.
    """
    user = uuid.uuid4()
    board_ids = [uuid.uuid4() for _ in range(3)]
    already_queued = {board_ids[1]}
    tasks = FakeTasks(user)
    enqueue_all_board_checks(
        FakeBoards(user, set(board_ids), already_queued), tasks, board_ids, now=NOW
    )
    scheduled = [e["scheduled_at"] for e in tasks.enqueued]
    assert scheduled == [NOW, NOW + ENQUEUE_SPACING]


def test_check_all_with_no_boards_queues_nothing() -> None:
    user = uuid.uuid4()
    tasks = FakeTasks(user)
    queued, skipped = enqueue_all_board_checks(FakeBoards(user, set(), set()), tasks, [], now=NOW)
    assert (queued, skipped) == (0, 0)
    assert tasks.enqueued == []
