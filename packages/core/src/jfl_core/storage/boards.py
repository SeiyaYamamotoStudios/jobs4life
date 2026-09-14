"""Watched job boards: the check history, tenancy-scoped.

Two classes, split the same way `jfl_core.storage.tasks` splits the queue.

`PostgresBoardRepository` is a `TenantScopedRepository`: constructed with the
user whose boards it may touch, with no per-call override. It holds everything
a request or a `check_board` handler does -- add a board, read its history, and
apply a check.

`PostgresBoardScheduler` is deliberately **not** tenancy-scoped and deliberately
not named `*Repository`, for the reason `PostgresTaskQueue` gives: the daily
scheduling pass is a daemon with no user in context, and it has to find every
board that is due, across tenants. What it may do is narrow on purpose -- read a
board's id and owner, and move `next_check_at`. It reads no board URL, no job,
no check, and returns ids only; anything that touches a board's content is done
by constructing a `PostgresBoardRepository` with the owner it returned.

**Applying a check is two calls in one transaction**, and the order matters:
`lock_check_state` takes a row lock on the board (`SELECT ... FOR UPDATE`) and
reads the history the diff needs; the caller runs the pure
`jfl_intake.engine.plan_check` against it; `apply_check_plan` writes the result.
Two checks of one board -- a redelivered task, or "check now" racing the
schedule -- therefore serialise on the lock, and the second plans against what
the first wrote. The network fetch happens before any of this and holds no
transaction.

**Idempotency is structural as well.** A plan applied twice cannot open a
second interval for a job (the partial unique index on open intervals, written
with ON CONFLICT DO NOTHING), cannot insert a job twice (unique on board and
external id), and cannot record a second baseline (a partial unique index).
Closing only ever touches intervals that are still open.

No SQL above this layer, and no model call anywhere near it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from sqlalchemy import Text, any_, bindparam, delete, or_, select, update
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

from jfl_core.db.tables import board_checks as checks_table
from jfl_core.db.tables import board_job_presence as presence_table
from jfl_core.db.tables import board_jobs as jobs_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import watched_boards as boards_table
from jfl_core.models import (
    BoardCheck,
    BoardCheckState,
    BoardJob,
    BoardJobEvent,
    BoardJobPresence,
    BoardPlatform,
    CheckPlan,
    DueBoard,
    KnownBoardJob,
    PlannedSighting,
    WatchedBoard,
)
from jfl_core.storage.tasks import UNFINISHED_STATUSES
from jfl_core.storage.tenancy import TenantScopedRepository

# Rows per INSERT. A baseline of a large Workday board is several thousand jobs,
# and psycopg caps one statement at 65,535 parameters.
_CHUNK = 1000

_BOARD_COLUMNS = (
    boards_table.c.id,
    boards_table.c.user_id,
    boards_table.c.platform,
    boards_table.c.board_url,
    boards_table.c.board_key,
    boards_table.c.label,
    boards_table.c.created_at,
    boards_table.c.next_check_at,
    boards_table.c.last_check_id,
    boards_table.c.consecutive_failures,
    boards_table.c.baseline_check_id,
    boards_table.c.held_check_id,
    boards_table.c.drop_accepted,
    boards_table.c.include_unstated_workplace,
)

_CHECK_COLUMNS = (
    checks_table.c.id,
    checks_table.c.user_id,
    checks_table.c.board_id,
    checks_table.c.started_at,
    checks_table.c.finished_at,
    checks_table.c.status,
    checks_table.c.jobs_seen,
    checks_table.c.expected_total,
    checks_table.c.error_code,
    checks_table.c.is_baseline,
)

_JOB_COLUMNS = (
    jobs_table.c.id,
    jobs_table.c.user_id,
    jobs_table.c.board_id,
    jobs_table.c.external_id,
    jobs_table.c.requisition_id,
    jobs_table.c.title,
    jobs_table.c.location,
    jobs_table.c.url,
    jobs_table.c.fingerprint,
    jobs_table.c.workplace,
    jobs_table.c.workplace_label,
    jobs_table.c.locations,
    jobs_table.c.first_seen_check_id,
    jobs_table.c.first_seen_at,
    jobs_table.c.last_seen_at,
    jobs_table.c.reposted_from_job_id,
)

_PRESENCE_COLUMNS = (
    presence_table.c.id,
    presence_table.c.user_id,
    presence_table.c.job_id,
    presence_table.c.opened_check_id,
    presence_table.c.opened_at,
    presence_table.c.closed_check_id,
    presence_table.c.closed_at,
)

_EVENT_ORDER = {"gone": 0, "returned": 1, "reposted": 2, "new": 3}


class BoardNotFoundError(RuntimeError):
    """No board with this id belongs to this user -- the same message whether it
    does not exist or is someone else's, for the reason `ApplicationNotFoundError`
    gives.
    """

    def __init__(self, board_id: uuid.UUID) -> None:
        super().__init__(f"no board {board_id} for this user")


def _board_from_row(row: Any) -> WatchedBoard:
    return WatchedBoard(
        id=row.id,
        user_id=row.user_id,
        platform=row.platform,
        board_url=row.board_url,
        board_key=dict(row.board_key),
        label=row.label,
        created_at=row.created_at,
        next_check_at=row.next_check_at,
        last_check_id=row.last_check_id,
        consecutive_failures=row.consecutive_failures,
        baseline_check_id=row.baseline_check_id,
        held_check_id=row.held_check_id,
        drop_accepted=row.drop_accepted,
        include_unstated_workplace=row.include_unstated_workplace,
    )


def _check_from_row(row: Any) -> BoardCheck:
    return BoardCheck(
        id=row.id,
        user_id=row.user_id,
        board_id=row.board_id,
        started_at=row.started_at,
        finished_at=row.finished_at,
        status=row.status,
        jobs_seen=row.jobs_seen,
        expected_total=row.expected_total,
        error_code=row.error_code,
        is_baseline=row.is_baseline,
    )


def _job_from_row(row: Any) -> BoardJob:
    return BoardJob(
        id=row.id,
        user_id=row.user_id,
        board_id=row.board_id,
        external_id=row.external_id,
        requisition_id=row.requisition_id,
        title=row.title,
        location=row.location,
        url=row.url,
        fingerprint=row.fingerprint,
        workplace=row.workplace,
        workplace_label=row.workplace_label,
        locations=list(row.locations),
        first_seen_check_id=row.first_seen_check_id,
        first_seen_at=row.first_seen_at,
        last_seen_at=row.last_seen_at,
        reposted_from_job_id=row.reposted_from_job_id,
    )


def _presence_from_row(row: Any) -> BoardJobPresence:
    return BoardJobPresence(
        id=row.id,
        user_id=row.user_id,
        job_id=row.job_id,
        opened_check_id=row.opened_check_id,
        opened_at=row.opened_at,
        closed_check_id=row.closed_check_id,
        closed_at=row.closed_at,
    )


def _chunks[T](items: Sequence[T]) -> list[Sequence[T]]:
    return [items[i : i + _CHUNK] for i in range(0, len(items), _CHUNK)]


class PostgresBoardRepository(TenantScopedRepository):
    """Watched boards and their check history, for exactly one user."""

    # -- boards --------------------------------------------------------------

    def add_board(
        self,
        *,
        platform: BoardPlatform,
        board_url: str,
        board_key: Mapping[str, str],
        label: str | None = None,
    ) -> WatchedBoard:
        """Watch a board. Idempotent: watching one this user already watches
        returns the existing row unchanged, so a double-submitted form cannot
        create two histories of one board.

        `next_check_at` defaults to now, so the scheduling pass takes the
        baseline within one tick without anyone pressing anything.
        """
        key = dict(board_key)
        row = self._conn.execute(
            pg_insert(boards_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                platform=platform,
                board_url=board_url,
                board_key=key,
                label=label,
            )
            .on_conflict_do_nothing(index_elements=["user_id", "platform", "board_key"])
            .returning(*_BOARD_COLUMNS)
        ).first()
        if row is None:
            row = self._conn.execute(
                select(*_BOARD_COLUMNS).where(
                    boards_table.c.user_id == self._user_id,
                    boards_table.c.platform == platform,
                    boards_table.c.board_key == key,
                )
            ).one()
        return _board_from_row(row)

    def list_boards(self) -> list[WatchedBoard]:
        rows = self._conn.execute(
            select(*_BOARD_COLUMNS)
            .where(boards_table.c.user_id == self._user_id)
            .order_by(boards_table.c.created_at.asc(), boards_table.c.id.asc())
        ).all()
        return [_board_from_row(r) for r in rows]

    def get_board(self, board_id: uuid.UUID) -> WatchedBoard | None:
        row = self._conn.execute(
            select(*_BOARD_COLUMNS).where(
                boards_table.c.id == board_id, boards_table.c.user_id == self._user_id
            )
        ).first()
        return None if row is None else _board_from_row(row)

    def remove_board(self, board_id: uuid.UUID) -> bool:
        """Stop watching, and discard the board's history with it (cascade)."""
        result = self._conn.execute(
            delete(boards_table)
            .where(boards_table.c.id == board_id, boards_table.c.user_id == self._user_id)
            .returning(boards_table.c.id)
        ).first()
        return result is not None

    def accept_drop(self, board_id: uuid.UUID) -> bool:
        """A person says the collapse the drop guard held is real. The next
        complete check is applied whatever its count. False, writing nothing, if
        the board is not held.
        """
        row = self._conn.execute(
            update(boards_table)
            .where(
                boards_table.c.id == board_id,
                boards_table.c.user_id == self._user_id,
                boards_table.c.held_check_id.is_not(None),
            )
            .values(drop_accepted=True)
            .returning(boards_table.c.id)
        ).first()
        return row is not None

    def set_include_unstated_workplace(self, board_id: uuid.UUID, value: bool | None) -> bool:
        """Whether the saved job filter lets this board's jobs with no stated
        workplace through. None returns the board to its platform default. False,
        writing nothing, if the board is not this user's.
        """
        row = self._conn.execute(
            update(boards_table)
            .where(boards_table.c.id == board_id, boards_table.c.user_id == self._user_id)
            .values(include_unstated_workplace=value)
            .returning(boards_table.c.id)
        ).first()
        return row is not None

    def check_task_queued(self, board_id: uuid.UUID, *, kind: str) -> bool:
        """Is a check of this board already pending or running for this user?

        A read of `tasks` by payload, kept here rather than widening
        `PostgresTaskRepository.enqueue_unique` -- whose uniqueness is per kind
        -- because per-board uniqueness is this domain's rule, not the queue's.
        """
        row = self._conn.execute(
            select(tasks_table.c.id)
            .where(
                tasks_table.c.user_id == self._user_id,
                tasks_table.c.kind == kind,
                tasks_table.c.status.in_(UNFINISHED_STATUSES),
                tasks_table.c.payload["board_id"].astext == str(board_id),
            )
            .limit(1)
        ).first()
        return row is not None

    # -- applying a check ----------------------------------------------------

    def lock_check_state(
        self,
        board_id: uuid.UUID,
        *,
        observed_external_ids: Sequence[str],
        closed_since: dt.datetime,
    ) -> BoardCheckState | None:
        """Lock the board row and read what the diff needs. None if the board is
        not this user's (or was deleted while its check was in flight).

        Must be followed by `apply_check_plan` in the SAME transaction; the lock
        is what makes the plan current when it is written.

        Loads not every job ever seen but the three sets a diff can touch: jobs
        with an open interval (candidates for "gone"), jobs whose external id
        was just observed (candidates for "returned"), and jobs whose last
        interval closed since `closed_since` (repost sources). All three are
        served by indexes on `board_job_presence`.
        """
        board = self._conn.execute(
            select(
                boards_table.c.id,
                boards_table.c.baseline_check_id,
                boards_table.c.held_check_id,
                boards_table.c.drop_accepted,
            )
            .where(boards_table.c.id == board_id, boards_table.c.user_id == self._user_id)
            .with_for_update()
        ).first()
        if board is None:
            return None

        p = presence_table
        j = jobs_table
        successor = jobs_table.alias("successor")
        is_open = select(p.c.id).where(p.c.job_id == j.c.id, p.c.closed_check_id.is_(None)).exists()
        last_closed = (
            select(p.c.closed_at)
            .where(p.c.job_id == j.c.id)
            .order_by(p.c.closed_at.desc().nulls_last())
            .limit(1)
            .scalar_subquery()
        )
        has_successor = (
            select(successor.c.id).where(successor.c.reposted_from_job_id == j.c.id).exists()
        )

        conditions = [is_open, last_closed >= closed_since]
        ids = list(observed_external_ids)
        if ids:
            conditions.append(
                j.c.external_id == any_(bindparam("observed_ids", value=ids, type_=ARRAY(Text)))
            )

        rows = self._conn.execute(
            select(
                j.c.id,
                j.c.external_id,
                j.c.fingerprint,
                is_open.label("is_open"),
                last_closed.label("last_closed_at"),
                has_successor.label("has_successor"),
            ).where(j.c.board_id == board_id, j.c.user_id == self._user_id, or_(*conditions))
        ).all()

        return BoardCheckState(
            board_id=board.id,
            baseline_check_id=board.baseline_check_id,
            held_check_id=board.held_check_id,
            drop_accepted=board.drop_accepted,
            known_jobs=[
                KnownBoardJob(
                    job_id=r.id,
                    external_id=r.external_id,
                    fingerprint=r.fingerprint,
                    is_open=bool(r.is_open),
                    last_closed_at=r.last_closed_at,
                    has_repost_successor=bool(r.has_successor),
                )
                for r in rows
            ],
        )

    def apply_check_plan(
        self, plan: CheckPlan, *, started_at: dt.datetime, finished_at: dt.datetime
    ) -> BoardCheck:
        """Record the check, and -- only if it is complete -- apply its diff.

        Every timestamp a job gets from this check is `finished_at`: the moment
        the fetch that saw (or did not see) it ended.
        """
        board = self._conn.execute(
            select(boards_table.c.id).where(
                boards_table.c.id == plan.board_id, boards_table.c.user_id == self._user_id
            )
        ).first()
        if board is None:
            raise BoardNotFoundError(plan.board_id)

        check_id = uuid.uuid4()
        check_row = self._conn.execute(
            pg_insert(checks_table)
            .values(
                id=check_id,
                user_id=self._user_id,
                board_id=plan.board_id,
                started_at=started_at,
                finished_at=finished_at,
                status=plan.status,
                jobs_seen=plan.jobs_seen,
                expected_total=plan.expected_total,
                error_code=plan.error_code,
                is_baseline=plan.is_baseline,
            )
            .returning(*_CHECK_COLUMNS)
        ).one()

        board_values: dict[str, Any] = {"last_check_id": check_id}
        if plan.changes_job_state:
            self._apply_jobs(plan, check_id, finished_at)
            board_values.update(consecutive_failures=0, held_check_id=None, drop_accepted=False)
            if plan.is_baseline:
                board_values["baseline_check_id"] = check_id
        elif plan.status == "held":
            # The fetch worked, so this is not a failure; it is a flag.
            board_values["held_check_id"] = check_id
        else:
            board_values["consecutive_failures"] = boards_table.c.consecutive_failures + 1

        self._conn.execute(
            update(boards_table)
            .where(boards_table.c.id == plan.board_id, boards_table.c.user_id == self._user_id)
            .values(**board_values)
        )
        return _check_from_row(check_row)

    def _apply_jobs(self, plan: CheckPlan, check_id: uuid.UUID, at: dt.datetime) -> None:
        # 1. New jobs. ON CONFLICT DO NOTHING, so a plan replayed against a board
        #    that already has the job cannot duplicate it; such a job is treated
        #    as a sighting below instead.
        inserted: dict[str, uuid.UUID] = {}
        rows = [
            {
                "id": uuid.uuid4(),
                "user_id": self._user_id,
                "board_id": plan.board_id,
                "external_id": n.job.external_id,
                "requisition_id": n.job.requisition_id,
                "title": n.job.title,
                "location": n.job.location,
                "url": n.job.url,
                "fingerprint": n.job.fingerprint,
                "workplace": n.job.workplace,
                "workplace_label": n.job.workplace_label,
                "locations": list(n.job.locations),
                "first_seen_check_id": check_id,
                "first_seen_at": at,
                "last_seen_at": at,
                "reposted_from_job_id": n.reposted_from_job_id,
            }
            for n in plan.new_jobs
        ]
        for chunk in _chunks(rows):
            result = self._conn.execute(
                pg_insert(jobs_table)
                .values(list(chunk))
                .on_conflict_do_nothing(index_elements=["board_id", "external_id"])
                .returning(jobs_table.c.id, jobs_table.c.external_id)
            ).all()
            inserted.update({r.external_id: r.id for r in result})

        conflicted = [n.job for n in plan.new_jobs if n.job.external_id not in inserted]
        replayed: list[PlannedSighting] = []
        if conflicted:
            existing = self._conn.execute(
                select(jobs_table.c.id, jobs_table.c.external_id).where(
                    jobs_table.c.board_id == plan.board_id,
                    jobs_table.c.user_id == self._user_id,
                    jobs_table.c.external_id
                    == any_(
                        bindparam(
                            "conflicted_ids",
                            value=[job.external_id for job in conflicted],
                            type_=ARRAY(Text),
                        )
                    ),
                )
            ).all()
            by_ext = {r.external_id: r.id for r in existing}
            replayed = [
                PlannedSighting(job_id=by_ext[job.external_id], job=job)
                for job in conflicted
                if job.external_id in by_ext
            ]

        # 2. Refresh what the board now says about every job seen again.
        sightings = [*plan.still_open, *plan.returned, *replayed]
        if sightings:
            self._conn.execute(
                update(jobs_table)
                .where(
                    jobs_table.c.id == bindparam("b_id"),
                    jobs_table.c.board_id == plan.board_id,
                    jobs_table.c.user_id == self._user_id,
                )
                .values(
                    title=bindparam("b_title"),
                    location=bindparam("b_location"),
                    url=bindparam("b_url"),
                    fingerprint=bindparam("b_fingerprint"),
                    requisition_id=bindparam("b_requisition_id"),
                    # Descriptive, like the requisition: refreshed, never identity.
                    workplace=bindparam("b_workplace"),
                    workplace_label=bindparam("b_workplace_label"),
                    locations=bindparam("b_locations", type_=ARRAY(Text)),
                    last_seen_at=at,
                ),
                [
                    {
                        "b_id": s.job_id,
                        "b_title": s.job.title,
                        "b_location": s.job.location,
                        "b_url": s.job.url,
                        "b_fingerprint": s.job.fingerprint,
                        "b_requisition_id": s.job.requisition_id,
                        "b_workplace": s.job.workplace,
                        "b_workplace_label": s.job.workplace_label,
                        "b_locations": list(s.job.locations),
                    }
                    for s in sightings
                ],
            )

        # 3. Open an interval for every new, returned or replayed job. The partial
        #    unique index allows one open interval per job, so a job already open
        #    is skipped rather than given a second.
        to_open = [
            *inserted.values(),
            *(s.job_id for s in plan.returned),
            *(s.job_id for s in replayed),
        ]
        presence_rows = [
            {
                "id": uuid.uuid4(),
                "user_id": self._user_id,
                "job_id": job_id,
                "opened_check_id": check_id,
                "opened_at": at,
            }
            for job_id in to_open
        ]
        for presence_chunk in _chunks(presence_rows):
            self._conn.execute(
                pg_insert(presence_table)
                .values(list(presence_chunk))
                .on_conflict_do_nothing(
                    index_elements=["job_id"],
                    index_where=presence_table.c.closed_check_id.is_(None),
                )
            )

        # 4. Close what is gone. Only intervals still open, and only for jobs on
        #    this board, so a replay cannot close something twice or elsewhere.
        if plan.gone_job_ids:
            board_job_ids = select(jobs_table.c.id).where(
                jobs_table.c.board_id == plan.board_id, jobs_table.c.user_id == self._user_id
            )
            self._conn.execute(
                update(presence_table)
                .where(
                    presence_table.c.job_id
                    == any_(
                        bindparam(
                            "gone_ids",
                            value=list(plan.gone_job_ids),
                            type_=ARRAY(UUID(as_uuid=True)),
                        )
                    ),
                    presence_table.c.job_id.in_(board_job_ids),
                    presence_table.c.user_id == self._user_id,
                    presence_table.c.closed_check_id.is_(None),
                )
                .values(closed_check_id=check_id, closed_at=at)
            )

    # -- reading history -----------------------------------------------------

    def list_checks(self, board_id: uuid.UUID, *, limit: int = 50) -> list[BoardCheck]:
        """Newest first."""
        rows = self._conn.execute(
            select(*_CHECK_COLUMNS)
            .where(checks_table.c.board_id == board_id, checks_table.c.user_id == self._user_id)
            .order_by(checks_table.c.started_at.desc(), checks_table.c.finished_at.desc())
            .limit(limit)
        ).all()
        return [_check_from_row(r) for r in rows]

    def list_jobs(self, board_id: uuid.UUID, *, open_only: bool = False) -> list[BoardJob]:
        query = (
            select(*_JOB_COLUMNS)
            .where(jobs_table.c.board_id == board_id, jobs_table.c.user_id == self._user_id)
            .order_by(jobs_table.c.first_seen_at.asc(), jobs_table.c.external_id.asc())
        )
        if open_only:
            query = query.where(
                select(presence_table.c.id)
                .where(
                    presence_table.c.job_id == jobs_table.c.id,
                    presence_table.c.closed_check_id.is_(None),
                )
                .exists()
            )
        return [_job_from_row(r) for r in self._conn.execute(query).all()]

    def list_open_jobs(self) -> list[BoardJob]:
        """Every currently open job across all of this user's boards, newest
        first by when we first saw it. A job is open only through an interval a
        complete check opened, so a board with no complete check contributes
        nothing here -- callers name such boards rather than let them vanish.
        """
        query = (
            select(*_JOB_COLUMNS)
            .where(
                jobs_table.c.user_id == self._user_id,
                select(presence_table.c.id)
                .where(
                    presence_table.c.job_id == jobs_table.c.id,
                    presence_table.c.user_id == self._user_id,
                    presence_table.c.closed_check_id.is_(None),
                )
                .exists(),
            )
            .order_by(
                jobs_table.c.first_seen_at.desc(),
                jobs_table.c.board_id.asc(),
                jobs_table.c.external_id.asc(),
            )
        )
        return [_job_from_row(r) for r in self._conn.execute(query).all()]

    def list_presence(self, job_id: uuid.UUID) -> list[BoardJobPresence]:
        """A job's intervals, oldest first."""
        rows = self._conn.execute(
            select(*_PRESENCE_COLUMNS)
            .where(presence_table.c.job_id == job_id, presence_table.c.user_id == self._user_id)
            .order_by(presence_table.c.opened_at.asc())
        ).all()
        return [_presence_from_row(r) for r in rows]

    def events_for_check(self, check_id: uuid.UUID) -> list[BoardJobEvent]:
        """What one check changed. Empty for a baseline and for every check
        that was not complete -- which is the rule, visible from the outside.
        """
        return self._events(
            new=jobs_table.c.first_seen_check_id == check_id,
            opened=presence_table.c.opened_check_id == check_id,
            closed=presence_table.c.closed_check_id == check_id,
        )

    def events_since(
        self, since: dt.datetime, *, board_id: uuid.UUID | None = None
    ) -> list[BoardJobEvent]:
        """What changed after `since`, across this user's boards or one of them."""
        new: Any = jobs_table.c.first_seen_at > since
        opened: Any = presence_table.c.opened_at > since
        closed: Any = presence_table.c.closed_at > since
        if board_id is not None:
            new = new & (jobs_table.c.board_id == board_id)
            opened = opened & (jobs_table.c.board_id == board_id)
            closed = closed & (jobs_table.c.board_id == board_id)
        return self._events(new=new, opened=opened, closed=closed)

    def _events(self, *, new: Any, opened: Any, closed: Any) -> list[BoardJobEvent]:
        events: list[BoardJobEvent] = []

        new_rows = self._conn.execute(
            select(*_JOB_COLUMNS)
            .select_from(
                jobs_table.join(checks_table, checks_table.c.id == jobs_table.c.first_seen_check_id)
            )
            .where(
                jobs_table.c.user_id == self._user_id,
                checks_table.c.is_baseline.is_(False),
                new,
            )
        ).all()
        for r in new_rows:
            job = _job_from_row(r)
            events.append(
                BoardJobEvent(
                    kind="reposted" if job.reposted_from_job_id is not None else "new",
                    board_id=job.board_id,
                    check_id=job.first_seen_check_id,
                    at=job.first_seen_at,
                    job=job,
                )
            )

        joined = presence_table.join(jobs_table, jobs_table.c.id == presence_table.c.job_id)
        returned_rows = self._conn.execute(
            select(*_JOB_COLUMNS, presence_table.c.opened_check_id, presence_table.c.opened_at)
            .select_from(joined)
            .where(
                presence_table.c.user_id == self._user_id,
                presence_table.c.opened_check_id != jobs_table.c.first_seen_check_id,
                opened,
            )
        ).all()
        for r in returned_rows:
            job = _job_from_row(r)
            events.append(
                BoardJobEvent(
                    kind="returned",
                    board_id=job.board_id,
                    check_id=r.opened_check_id,
                    at=r.opened_at,
                    job=job,
                )
            )

        gone_rows = self._conn.execute(
            select(*_JOB_COLUMNS, presence_table.c.closed_check_id, presence_table.c.closed_at)
            .select_from(joined)
            .where(
                presence_table.c.user_id == self._user_id,
                presence_table.c.closed_check_id.is_not(None),
                closed,
            )
        ).all()
        for r in gone_rows:
            job = _job_from_row(r)
            events.append(
                BoardJobEvent(
                    kind="gone",
                    board_id=job.board_id,
                    check_id=r.closed_check_id,
                    at=r.closed_at,
                    job=job,
                )
            )

        events.sort(key=lambda e: (e.at, _EVENT_ORDER[e.kind], e.job.external_id))
        return events


class PostgresBoardScheduler:
    """The scheduling pass's view of every board, across tenants. See the module
    docstring for why that is safe and why it is not a repository.

    Takes a `Connection` and never opens or commits a transaction. `claim_due`
    uses SKIP LOCKED, so it must run inside the caller's transaction, and the
    boards it returns stay locked -- against a second scheduling pass -- until
    that transaction ends.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def claim_due(
        self,
        *,
        now: dt.datetime,
        limit: int,
        only_owners: Collection[uuid.UUID] | None = None,
    ) -> list[DueBoard]:
        """Boards whose `next_check_at` has passed, locked for this transaction.

        `only_owners=None` is production: the pass is for every tenant. A
        collection narrows it to those users' boards (an empty one claims
        nothing). Tests pass the users they created, so a scheduling pass under
        test can never pick up -- and enqueue a live check for -- a board some
        other run left behind.
        """
        query = select(boards_table.c.id, boards_table.c.user_id).where(
            boards_table.c.next_check_at <= now
        )
        if only_owners is not None:
            query = query.where(boards_table.c.user_id.in_(list(only_owners)))
        rows = self._conn.execute(
            query.order_by(boards_table.c.next_check_at.asc(), boards_table.c.id.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
        return [DueBoard(board_id=r.id, user_id=r.user_id) for r in rows]

    def reschedule(self, board_id: uuid.UUID, *, next_check_at: dt.datetime) -> None:
        self._conn.execute(
            update(boards_table)
            .where(boards_table.c.id == board_id)
            .values(next_check_at=next_check_at)
        )
