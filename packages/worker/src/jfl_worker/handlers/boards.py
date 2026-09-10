"""`check_board` and `schedule_board_checks`: watched job boards, in the background.

**Both are registered with `calls_model=False`, and that is true.** Nothing on
either path imports `anthropic`: the adapters are HTTP clients for public ATS
APIs and the check engine is pure rules. `JFL_DISABLE_MODEL_CALLS` therefore
does not stop board checks, which is correct -- they cost no user money.

`check_board` (payload: `{"board_id": ...}`)
--------------------------------------------

Three steps, and the transaction boundaries are the design:

  1. read the board (short transaction);
  2. fetch it through the adapter -- no transaction held, because a Workday
     board is a hundred-odd requests and a couple of minutes;
  3. in ONE transaction: lock the board row, read its history, run the pure
     `plan_check`, apply the plan. Two checks of one board serialise here, and
     the second plans against what the first wrote.

Every check is recorded, whatever its outcome, before the task decides how to
end. Then:

  * `unreachable` (timeout, connection error, 429, 5xx) raises an ordinary
    exception, so the queue's retry and backoff ride it out. Each attempt is a
    real check and is recorded as one, with `consecutive_failures` counting them;
  * `failed` (404, other 4xx, malformed response, or a stored board key no
    adapter can use) raises `PermanentTaskError`: the same request will get the
    same answer, so retries would only add rows;
  * `incomplete`, `truncated`, `held` and `complete` return normally. The first
    three are recorded outcomes that a retry minutes later would not change, and
    the board is checked again at its next daily slot.

At-least-once delivery is safe: a redelivered task runs another check, which
plans against the already-applied history and opens no duplicate interval.

`schedule_board_checks` (no payload)
------------------------------------

Enqueued by the worker's ticker exactly as the session purge is, with the
queue's `enqueue_unique`, so a stopped worker does not come back to a backlog of
scheduling passes. It claims every board whose `next_check_at` has passed --
across tenants, via `PostgresBoardScheduler`, which returns ids and owners only
-- moves each to its next daily slot, and enqueues a `check_board` for it with a
repository bound to that board's owner. Claim, reschedule and enqueue share one
transaction, so a pass that dies part-way leaves the boards due rather than
rescheduled-but-unchecked.

The task row for the pass itself belongs to the worker's system user, like the
purge's: `tasks.user_id` is NOT NULL, and the pass is nobody's in particular.
The `check_board` tasks it enqueues each belong to the board's owner.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager

from jfl_core.storage.boards import PostgresBoardRepository, PostgresBoardScheduler
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.adapters import (
    AdapterRegistry,
    InvalidBoardKeyError,
    UnsupportedPlatformError,
    default_registry,
)
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.http import PoliteTransport, RequestBudget, Transport, httpx_transport
from jfl_intake.scheduling import (
    CHECK_BOARD_KIND,
    ENQUEUE_SPACING,
    SCHEDULE_BATCH,
    SCHEDULE_BOARD_CHECKS_KIND,
    enqueue_board_check,
    next_check_at,
)

from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = CHECK_BOARD_KIND
SCHEDULE_KIND = SCHEDULE_BOARD_CHECKS_KIND

TransportFactory = Callable[[], AbstractContextManager[Transport]]


class BoardUnreachableError(RuntimeError):
    """A check that could not reach the board. Retryable. Built from a literal
    and a closed-set code, never from an exception or a response body.
    """


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


def polite_httpx_transport(budget: RequestBudget | None = None) -> TransportFactory:
    """The production transport: one httpx client per check, wrapped in the
    politeness limits of `RequestBudget`.
    """
    chosen = budget or RequestBudget()

    @contextmanager
    def factory() -> Iterator[Transport]:
        with httpx_transport() as inner:
            yield PoliteTransport(inner, chosen)

    return factory


def _board_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("board_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no board_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise PermanentTaskError("payload board_id is not a uuid") from None


def build_check_board(
    *,
    adapters: AdapterRegistry | None = None,
    transport_factory: TransportFactory | None = None,
    clock: Callable[[], dt.datetime] = _utcnow,
) -> Handler:
    """Bind the handler to its adapters and transport. Tests pass a fake
    transport factory; nothing else differs between a test and production.
    """
    registry = adapters or default_registry()
    factory = transport_factory or polite_httpx_transport()

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _check_board(ctx, registry, factory, clock)

    return handler


def _check_board(
    ctx: TaskContext,
    adapters: AdapterRegistry,
    transport_factory: TransportFactory,
    clock: Callable[[], dt.datetime],
) -> Mapping[str, object]:
    board_id = _board_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        board = PostgresBoardRepository(conn, ctx.user_id).get_board(board_id)
    if board is None:
        # Removed since the task was queued, or never this user's. Nothing to
        # check and nothing a retry could find.
        return {"board_id": str(board_id), "skipped": "no such board"}

    started_at = ctx.now
    try:
        adapter = adapters.get(board.platform)
        adapter.validate_key(board.board_key)
    except (UnsupportedPlatformError, InvalidBoardKeyError):
        result = FetchResult(status="failed", error_code="unsupported_board")
    else:
        with transport_factory() as transport:
            result = adapter.fetch(board.board_key, transport)
    finished_at = clock()

    with ctx.engine.begin() as conn:
        repo = PostgresBoardRepository(conn, ctx.user_id)
        state = repo.lock_check_state(
            board_id,
            observed_external_ids=[job.external_id for job in result.jobs],
            closed_since=finished_at - REPOST_WINDOW,
        )
        if state is None:
            return {"board_id": str(board_id), "skipped": "board removed during check"}
        plan = plan_check(state, result, observed_at=finished_at)
        check = repo.apply_check_plan(plan, started_at=started_at, finished_at=finished_at)

    if check.status == "unreachable":
        raise BoardUnreachableError(f"board check unreachable: {check.error_code}")
    if check.status == "failed":
        raise PermanentTaskError(f"board check failed: {check.error_code}")

    return {
        "board_id": str(board_id),
        "platform": board.platform,
        "check_id": str(check.id),
        "status": check.status,
        "error_code": check.error_code,
        "jobs_seen": check.jobs_seen,
        "expected_total": check.expected_total,
        "requests": result.requests,
        **plan.summary(),
    }


def build_schedule_board_checks(
    *,
    batch: int = SCHEDULE_BATCH,
    spacing: dt.timedelta = ENQUEUE_SPACING,
    only_owners: Collection[uuid.UUID] | None = None,
) -> Handler:
    """`only_owners=None` -- what production registers -- schedules every
    tenant's boards. A collection narrows the pass to those users' boards; tests
    pass the users they created, so a pass under test cannot pick up boards some
    other run left behind.
    """
    owners = None if only_owners is None else frozenset(only_owners)

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _schedule_board_checks(ctx, batch=batch, spacing=spacing, only_owners=owners)

    return handler


def _schedule_board_checks(
    ctx: TaskContext,
    *,
    batch: int,
    spacing: dt.timedelta,
    only_owners: frozenset[uuid.UUID] | None,
) -> Mapping[str, object]:
    enqueued = 0
    already_queued = 0
    with ctx.engine.begin() as conn:
        scheduler = PostgresBoardScheduler(conn)
        due = scheduler.claim_due(now=ctx.now, limit=batch, only_owners=only_owners)
        for board in due:
            scheduler.reschedule(
                board.board_id, next_check_at=next_check_at(board.board_id, after=ctx.now)
            )
            # Scoped to the board's owner, whose task this is.
            task = enqueue_board_check(
                PostgresBoardRepository(conn, board.user_id),
                PostgresTaskRepository(conn, board.user_id),
                board.board_id,
                scheduled_at=ctx.now + spacing * enqueued,
            )
            if task is None:
                already_queued += 1
            else:
                enqueued += 1
    return {
        "boards_due": len(due),
        "checks_enqueued": enqueued,
        "checks_already_queued": already_queued,
    }
