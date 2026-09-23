"""The two scores for an application -- PLAN.md slice B4, tenancy-scoped.

**Two axes, never composited.** There is no method here that combines
`could_get_score` and `want_it_score`, and there must never be one: CLAUDE.md's
standing decision is that a role the user would love and will not get, and one
they would dislike and would walk into, must never land on the same number.
The disagreement between the two is the signal.

**Append-only across runs.** `create_pending` inserts a new row every time the
user asks for a score, so a re-score never overwrites what the tool said last
time -- the page reads `latest` and the history stays. A row's own `status`
moves `pending` -> `done`/`failed` exactly once, which is that run's state
rather than a rewrite of an earlier score.

No model call anywhere in this module, and no SQL above it.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update

from jfl_core.db.tables import application_scores as table
from jfl_core.models import (
    ApplicationScore,
    ConstraintVerdict,
    HardGateBreach,
    NotStated,
    ObjectiveVerdict,
    ScoreErrorCode,
    ScoreLever,
)
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.application_id,
    table.c.status,
    table.c.error_code,
    table.c.could_get_score,
    table.c.could_get_assessment,
    table.c.want_it_score,
    table.c.want_it_assessment,
    table.c.constraint_verdicts,
    table.c.objective_verdicts,
    table.c.hard_gate_breaches,
    table.c.levers,
    table.c.not_stated,
    table.c.model,
    table.c.cost_usd,
    table.c.trace_id,
    table.c.created_at,
    table.c.updated_at,
)


def _from_row(row: Any) -> ApplicationScore:
    return ApplicationScore(
        id=row.id,
        application_id=row.application_id,
        status=row.status,
        error_code=row.error_code,
        could_get_score=row.could_get_score,
        could_get_assessment=row.could_get_assessment,
        want_it_score=row.want_it_score,
        want_it_assessment=row.want_it_assessment,
        constraint_verdicts=[
            ConstraintVerdict.model_validate(item) for item in (row.constraint_verdicts or [])
        ],
        objective_verdicts=[
            ObjectiveVerdict.model_validate(item) for item in (row.objective_verdicts or [])
        ],
        hard_gate_breaches=[
            HardGateBreach.model_validate(item) for item in (row.hard_gate_breaches or [])
        ],
        levers=[ScoreLever.model_validate(item) for item in (row.levers or [])],
        not_stated=[NotStated.model_validate(item) for item in (row.not_stated or [])],
        model=row.model,
        cost_usd=row.cost_usd,
        trace_id=row.trace_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresScoreRepository(TenantScopedRepository):
    """This user's application scores, and no one else's."""

    def get(self, score_id: uuid.UUID) -> ApplicationScore | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == score_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def latest(self, application_id: uuid.UUID) -> ApplicationScore | None:
        """The most recent run for this application, whatever its state -- what
        the detail page shows and what the htmx poll re-reads. A `pending` row
        is the latest precisely while the work is in flight, which is what makes
        the panel say "scoring..." without any client-side state.
        """
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.application_id == application_id, table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def history(self, application_id: uuid.UUID) -> list[ApplicationScore]:
        """Every run against this application, oldest first. Nothing here is
        ever deleted or overwritten, so this is the record of what the tool
        said and what it cost.
        """
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.application_id == application_id, table.c.user_id == self._user_id)
            .order_by(table.c.created_at.asc())
        ).all()
        return [_from_row(r) for r in rows]

    def create_pending(self, application_id: uuid.UUID) -> ApplicationScore:
        """A new `pending` run. Always inserts: scoring costs the user money
        and happens because they pressed something, so two presses are two
        runs, and the earlier one's result is kept rather than replaced.
        """
        row = self._conn.execute(
            table.insert()
            .values(id=uuid.uuid4(), user_id=self._user_id, application_id=application_id)
            .returning(*_COLUMNS)
        ).one()
        return _from_row(row)

    def mark_done(
        self,
        score_id: uuid.UUID,
        *,
        could_get_score: int,
        could_get_assessment: str,
        want_it_score: int | None,
        want_it_assessment: str,
        constraint_verdicts: list[ConstraintVerdict],
        objective_verdicts: list[ObjectiveVerdict],
        hard_gate_breaches: list[HardGateBreach],
        levers: list[ScoreLever],
        not_stated: list[NotStated],
        model: str,
        cost_usd: Decimal | None,
        trace_id: uuid.UUID,
    ) -> None:
        """Both axes land in one write, and neither is derivable from the
        other. Keyword-only on purpose: two integers in a row is exactly the
        call where the axes could be swapped silently.

        `want_it_score` may be None on a finished run: with no constraints and
        no objectives recorded there is nothing for the ad to be measured
        against, and storing 1 would be a claim where there is only a silence.
        The verdicts it was derived from land in the same write.
        """
        self._conn.execute(
            update(table)
            .where(table.c.id == score_id, table.c.user_id == self._user_id)
            .values(
                status="done",
                error_code=None,
                could_get_score=could_get_score,
                could_get_assessment=could_get_assessment,
                want_it_score=want_it_score,
                want_it_assessment=want_it_assessment,
                constraint_verdicts=[v.model_dump() for v in constraint_verdicts],
                objective_verdicts=[v.model_dump() for v in objective_verdicts],
                hard_gate_breaches=[b.model_dump() for b in hard_gate_breaches],
                levers=[lever.model_dump() for lever in levers],
                not_stated=[n.model_dump() for n in not_stated],
                model=model,
                cost_usd=cost_usd,
                trace_id=trace_id,
            )
        )

    def mark_failed(self, score_id: uuid.UUID, code: ScoreErrorCode) -> None:
        """A code from the closed set, never a message: the worker writes this
        while holding the user's decrypted API key.
        """
        self._conn.execute(
            update(table)
            .where(table.c.id == score_id, table.c.user_id == self._user_id)
            .values(status="failed", error_code=code)
        )

    def note_retry(self, score_id: uuid.UUID, code: ScoreErrorCode) -> None:
        """An attempt failed and the queue will try again: the row stays
        `pending`, and carries the code of the attempt that failed.

        `pending` with an `error_code` is how the panel tells "retrying" from
        "failed" without reading the task queue. Marking the row `failed` here
        -- what this used to do -- was wrong twice over: the page said "failed"
        while a retry was still queued, and the retry then found a row that was
        no longer `pending` and skipped itself, so it could never succeed.
        Guarded on `pending` so a late note can never reopen a finished run.
        """
        self._conn.execute(
            update(table)
            .where(
                table.c.id == score_id,
                table.c.user_id == self._user_id,
                table.c.status == "pending",
            )
            .values(error_code=code)
        )

    def latest_for_applications(
        self, application_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, tuple[ApplicationScore | None, ApplicationScore | None]]:
        """For each application: (the latest run of any state, the latest
        finished run). One query for the whole list screen.

        Two, because the list shows the last numbers the tool actually produced
        *and* whether a run is in flight -- a re-score in progress must not blank
        the numbers the previous run gave, nor hide that it is running.
        Applications with no run at all are simply absent from the result.
        """
        if not application_ids:
            return {}
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(
                table.c.user_id == self._user_id,
                table.c.application_id.in_(application_ids),
            )
            .order_by(table.c.application_id, table.c.created_at.desc())
        ).all()
        result: dict[uuid.UUID, tuple[ApplicationScore | None, ApplicationScore | None]] = {}
        for row in rows:
            score = _from_row(row)
            latest, latest_done = result.get(score.application_id, (None, None))
            if latest is None:
                latest = score
            if latest_done is None and score.status == "done":
                latest_done = score
            result[score.application_id] = (latest, latest_done)
        return result

    def has_any(self, application_id: uuid.UUID) -> bool:
        """Whether this application has ever had a scoring run, in any state --
        what keeps the automatic first score to exactly one.
        """
        return (
            self._conn.execute(
                select(table.c.id)
                .where(table.c.application_id == application_id, table.c.user_id == self._user_id)
                .limit(1)
            ).first()
            is not None
        )
