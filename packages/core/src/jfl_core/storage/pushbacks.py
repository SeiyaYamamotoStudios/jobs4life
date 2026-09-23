"""The pushback log, and the drift meter that reads it. Tenancy-scoped.

**The log is the store.** There is no stored preference weight for a pushback
to corrupt: a dimension's displacement is `sum(applied_delta)` over this user's
applied rows, so the profile cannot drift quietly -- drifting is a query, and
the drift meter is that query put on the screen. Corrections never write to
`profiles`, never write to `spans`, never write to `requirement_coverage` and
never write to `application_scores`; this module imports none of them, which is
what makes that a property of the code rather than a promise in a docstring.

**Append-only.** A row is inserted the moment the user submits, before anything
is read and before anything moves, because the design's rule is that a
pushback is recorded whether or not it changes anything. The only later writes
to a row are its reading (what kind of statement it is and which way it
pushes), the single application of its effect, and -- if the user says "Not
what I meant" -- a withdrawal mark. A withdrawn row keeps its delta and its
receipt; every sum below skips it. Changing your mind is a new pushback, never
an edit.

The arithmetic is `jfl_core.pushback` -- pure, and deliberately not here. This
module decides nothing; it reads what the record already holds, hands it to
`decide()`, and stores what came back.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select, update

from jfl_core.db.tables import score_overrides as overrides_table
from jfl_core.db.tables import score_pushbacks as table
from jfl_core.models import (
    ClassificationSource,
    Pushback,
    PushbackErrorCode,
    ScoreOverride,
)
from jfl_core.pushback import (
    Axis,
    Direction,
    DriftMeter,
    PushbackEffect,
    PushbackKind,
    decide,
    observations,
)
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.application_id,
    table.c.score_id,
    table.c.axis,
    table.c.dimension,
    table.c.target_dimension,
    table.c.shown_score,
    table.c.shown_explanation,
    table.c.user_text,
    table.c.asserted_direction,
    table.c.asserted_points,
    table.c.status,
    table.c.classification,
    table.c.classification_source,
    table.c.classification_note,
    table.c.new_information,
    table.c.error_code,
    table.c.trace_id,
    table.c.applied_delta,
    table.c.prior_observations,
    table.c.disposition,
    table.c.effect,
    table.c.evidence_question,
    table.c.resulting_span_id,
    table.c.created_at,
    table.c.updated_at,
    table.c.applied_at,
    table.c.withdrawn_at,
)

_OVERRIDE_COLUMNS = (
    overrides_table.c.id,
    overrides_table.c.application_id,
    overrides_table.c.axis,
    overrides_table.c.value,
    overrides_table.c.note,
    overrides_table.c.created_at,
)


def _from_row(row: Any) -> Pushback:
    return Pushback(
        id=row.id,
        application_id=row.application_id,
        score_id=row.score_id,
        axis=row.axis,
        dimension=row.dimension,
        target_dimension=row.target_dimension,
        shown_score=row.shown_score,
        shown_explanation=row.shown_explanation,
        user_text=row.user_text,
        asserted_direction=row.asserted_direction,
        asserted_points=float(row.asserted_points),
        status=row.status,
        classification=row.classification,
        classification_source=row.classification_source,
        classification_note=row.classification_note,
        new_information=row.new_information,
        error_code=row.error_code,
        trace_id=row.trace_id,
        applied_delta=None if row.applied_delta is None else float(row.applied_delta),
        prior_observations=row.prior_observations,
        disposition=row.disposition,
        effect=dict(row.effect or {}),
        evidence_question=row.evidence_question,
        resulting_span_id=row.resulting_span_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
        applied_at=row.applied_at,
        withdrawn_at=row.withdrawn_at,
    )


def normalise(text: str) -> str:
    """Fold a pushback's text for the exact-restatement check.

    Whitespace and case only. Nothing cleverer: a stemmer or a similarity
    threshold would be a heuristic estimating whether two sentences mean the
    same thing, and the 2026-09-01 deletion of the two heuristic gate rules is
    the standing lesson about what those are worth. This one cannot produce a
    false positive, which is the only kind of guarantee worth having here.
    """
    return " ".join(text.split()).casefold()


class PostgresPushbackRepository(TenantScopedRepository):
    """This user's pushbacks, and no one else's."""

    def record(
        self,
        *,
        application_id: uuid.UUID,
        score_id: uuid.UUID,
        axis: Axis,
        dimension: str,
        shown_score: int | None,
        shown_explanation: str,
        user_text: str,
        asserted_direction: Direction,
        asserted_points: float,
    ) -> Pushback:
        """Record the disagreement. Nothing is applied and nothing moves.

        The words are stored exactly as typed, with the number and sentence
        they were shown beside them. Always inserts: two identical pushbacks
        are two disagreements, and the second one counts as an observation even
        though it will contribute no delta.
        """
        row = self._conn.execute(
            table.insert()
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                score_id=score_id,
                axis=axis,
                dimension=dimension,
                shown_score=shown_score,
                shown_explanation=shown_explanation,
                user_text=user_text,
                asserted_direction=asserted_direction,
                asserted_points=Decimal(str(round(float(asserted_points), 2))),
                status="awaiting_classification",
            )
            .returning(*_COLUMNS)
        ).one()
        return _from_row(row)

    def get(self, pushback_id: uuid.UUID) -> Pushback | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == pushback_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def set_classification(
        self,
        pushback_id: uuid.UUID,
        *,
        classification: PushbackKind,
        new_information: bool,
        note: str = "",
        source: ClassificationSource = "model",
        trace_id: uuid.UUID | None = None,
    ) -> Pushback | None:
        """Put a proposed classification on the row, applying nothing.

        An applied pushback is never reclassified: the log is append-only and
        the receipt already told the user what happened.
        """
        self._conn.execute(
            update(table)
            .where(
                table.c.id == pushback_id,
                table.c.user_id == self._user_id,
                table.c.status != "applied",
            )
            .values(
                status="classified",
                classification=classification,
                classification_source=source,
                classification_note=note,
                new_information=new_information,
                error_code=None,
                trace_id=trace_id,
            )
        )
        return self.get(pushback_id)

    def set_reading(
        self,
        pushback_id: uuid.UUID,
        *,
        axis: Axis,
        direction: Direction,
        shown_score: int | None,
        shown_explanation: str,
    ) -> Pushback | None:
        """Say which score this was about and which way it pushes, before it
        is applied.

        The one-box form asks for words and nothing else, so these are not
        known when the row is inserted: the reading (a model's, or the user's
        pick from the plain list) supplies them. The dimension is the axis as a
        whole -- the box does not ask which constraint or claim, and guessing
        one would be a second judgement to get wrong. The stimulus is re-pointed
        at the number on that axis, which is the one the words were about.
        Never touches an applied row.
        """
        from jfl_core.pushback import COULD_GET_OVERALL, WANT_OVERALL

        self._conn.execute(
            update(table)
            .where(
                table.c.id == pushback_id,
                table.c.user_id == self._user_id,
                table.c.status != "applied",
            )
            .values(
                axis=axis,
                dimension=WANT_OVERALL if axis == "want" else COULD_GET_OVERALL,
                asserted_direction=direction,
                shown_score=shown_score,
                shown_explanation=shown_explanation,
            )
        )
        return self.get(pushback_id)

    def withdraw(self, pushback_id: uuid.UUID) -> Pushback | None:
        """ "Not what I meant": this correction stops counting.

        A mark, not a delete and not an edit. The row keeps what it did, so the
        log still shows that it happened and that it was undone; every sum that
        makes up the profile skips it from now on. Only an applied row can be
        withdrawn (a CHECK says so too), and only once.
        """
        self._conn.execute(
            update(table)
            .where(
                table.c.id == pushback_id,
                table.c.user_id == self._user_id,
                table.c.status == "applied",
                table.c.withdrawn_at.is_(None),
            )
            .values(withdrawn_at=func.now())
        )
        return self.get(pushback_id)

    def mark_classification_failed(
        self, pushback_id: uuid.UUID, code: PushbackErrorCode
    ) -> Pushback | None:
        """The cheap reading call failed. Not much of a failure.

        The row stays `awaiting_classification` with a code on it, nothing has
        moved, and the screen asks the user which of a short list of plain
        readings they meant. A model outage costs one click, never the loop.
        """
        self._conn.execute(
            update(table)
            .where(
                table.c.id == pushback_id,
                table.c.user_id == self._user_id,
                table.c.status != "applied",
            )
            .values(error_code=code)
        )
        return self.get(pushback_id)

    def apply(
        self,
        pushback_id: uuid.UUID,
        *,
        classification: PushbackKind,
        new_information: bool,
        evidence_question: str = "",
        submitted_applications: int = 0,
    ) -> Pushback | None:
        """Apply a reading of this pushback, once.

        The reading comes from the cheap model call, applied straight away, or
        from the user's own pick when they said "Not what I meant" or the call
        failed. Undo-after rather than confirm-before: what protects against a
        misreading is that the result is shown at once, in plain words, with a
        one-click way back -- and that the rule in `decide()` never lets any
        reading move "could I get this" upward.

        Two inputs come from the record rather than the request -- the prior
        observations on this dimension and its accumulated displacement -- so
        neither is something a caller can talk up.
        """
        existing = self.get(pushback_id)
        if existing is None or existing.status == "applied":
            return existing

        from jfl_core.pushback import target_dimension

        target = target_dimension(classification, existing.dimension)
        direction: Direction = "up" if existing.asserted_direction == "up" else "down"

        # An exact restatement of something already on the record for this
        # dimension and direction is a repetition whatever the model said and
        # whatever the user ticked. This is the half of "repetition is not
        # evidence" that no prompt can be argued out of.
        if self._restates(target, direction, existing.user_text, exclude=existing.id):
            new_information = False

        prior = self._prior_pushbacks(target)
        n = observations(prior_pushbacks=prior, submitted_applications=submitted_applications)
        effect = decide(
            kind=classification,
            dimension=existing.dimension,
            direction=direction,
            asserted=existing.asserted_points,
            prior_observations=n,
            displacement=self.displacement(target),
            new_information=new_information,
        )
        self._store(pushback_id, classification, new_information, effect, evidence_question)
        return self.get(pushback_id)

    def _store(
        self,
        pushback_id: uuid.UUID,
        classification: PushbackKind,
        new_information: bool,
        effect: PushbackEffect,
        evidence_question: str,
    ) -> None:
        self._conn.execute(
            update(table)
            .where(
                table.c.id == pushback_id,
                table.c.user_id == self._user_id,
                table.c.status != "applied",
            )
            .values(
                status="applied",
                classification=classification,
                new_information=new_information,
                target_dimension=effect.target,
                applied_delta=Decimal(str(round(effect.applied_delta, 3))),
                prior_observations=effect.prior_observations,
                disposition=effect.disposition,
                effect={
                    "target": effect.target,
                    "disposition": effect.disposition,
                    "applied_delta": round(effect.applied_delta, 3),
                    "asserted": round(effect.asserted, 3),
                    "after_shrinkage": round(effect.after_shrinkage, 3),
                    "prior_observations": effect.prior_observations,
                    "comparison_offered": effect.comparison_offered,
                    "repetition": effect.repetition,
                    "evidence_required": effect.evidence_required,
                },
                evidence_question=evidence_question if effect.evidence_required else "",
                applied_at=func.now(),
            )
        )

    def record_evidence(self, pushback_id: uuid.UUID, span_id: uuid.UUID) -> Pushback | None:
        """Note which corpus span answered this pushback's evidence question.

        The span is not created here and could not be: it comes back from the
        one existing write path from a user's own words into the corpus. All
        this does is record which one, so the screen can say the question is
        answered -- and the number still does not move until the job is scored
        again against the corpus that now holds the fact.
        """
        self._conn.execute(
            update(table)
            .where(table.c.id == pushback_id, table.c.user_id == self._user_id)
            .values(resulting_span_id=span_id)
        )
        return self.get(pushback_id)

    # -- reading -------------------------------------------------------------

    def for_application(self, application_id: uuid.UUID) -> list[Pushback]:
        """Every pushback against this application, oldest first."""
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(
                table.c.application_id == application_id,
                table.c.user_id == self._user_id,
            )
            .order_by(table.c.created_at.asc())
        ).all()
        return [_from_row(row) for row in rows]

    def recent(self, limit: int = 100) -> list[Pushback]:
        """The whole log, newest first. Every disagreement, in the user's words."""
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc())
            .limit(limit)
        ).all()
        return [_from_row(row) for row in rows]

    def applied_log(self) -> list[Pushback]:
        """Every correction that still counts, in the order it was applied.

        What the before -> after on a receipt is computed from: the profile
        just before a correction is the sum of everything applied ahead of it.
        """
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(
                table.c.user_id == self._user_id,
                table.c.status == "applied",
                table.c.withdrawn_at.is_(None),
            )
            .order_by(table.c.applied_at.asc(), table.c.created_at.asc())
        ).all()
        return [_from_row(row) for row in rows]

    def displacements(self) -> dict[str, float]:
        """Every dimension this user's corrections have moved, and by how much.

        Signed, summed over applied rows. This is the whole of "the profile"
        as far as pushback is concerned, and it is derived rather than stored
        precisely so that it cannot drift out of step with the log that
        justifies it.
        """
        rows = self._conn.execute(
            select(table.c.target_dimension, func.sum(table.c.applied_delta))
            .where(
                table.c.user_id == self._user_id,
                table.c.status == "applied",
                table.c.withdrawn_at.is_(None),
            )
            .group_by(table.c.target_dimension)
        ).all()
        return {row[0]: float(row[1] or 0) for row in rows if row[0]}

    def displacement(self, dimension: str) -> float:
        value = self._conn.execute(
            select(func.coalesce(func.sum(table.c.applied_delta), 0)).where(
                table.c.user_id == self._user_id,
                table.c.status == "applied",
                table.c.withdrawn_at.is_(None),
                table.c.target_dimension == dimension,
            )
        ).scalar_one()
        return float(value)

    def drift_meter(self) -> DriftMeter:
        """ "11 pushbacks, 10 upward, +3.1 net" -- the three numbers, counted.

        Over applied rows, so all three describe the same set. A pushback still
        being read has not done anything yet and is not counted as though it
        had; one the user withdrew as a misreading no longer counts either.
        """
        row = self._conn.execute(
            select(
                func.count(),
                func.count().filter(table.c.asserted_direction == "up"),
                func.count().filter(table.c.asserted_direction == "down"),
                func.coalesce(func.sum(table.c.applied_delta), 0),
                func.count().filter(table.c.disposition == "pending_evidence"),
            ).where(
                table.c.user_id == self._user_id,
                table.c.status == "applied",
                table.c.withdrawn_at.is_(None),
            )
        ).one()
        return DriftMeter(
            total=row[0],
            upward=row[1],
            downward=row[2],
            net=float(row[3]),
            pending_evidence=row[4],
        )

    def awaiting_evidence(self) -> list[Pushback]:
        """Capability claims the tool has not taken on trust, still unanswered.

        These are the open questions the loop is actually waiting on: a number
        that will move the moment a confirmed fact exists, and not before.
        """
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(
                table.c.user_id == self._user_id,
                table.c.disposition == "pending_evidence",
                table.c.resulting_span_id.is_(None),
                table.c.withdrawn_at.is_(None),
            )
            .order_by(table.c.created_at.asc())
        ).all()
        return [_from_row(row) for row in rows]

    def _prior_pushbacks(self, dimension: str) -> int:
        """How many applied pushbacks already touch this dimension -- `n`.

        Restatements are in the count. That is the point: they contribute no
        delta and they still make the next correction move less.
        """
        value = self._conn.execute(
            select(func.count()).where(
                table.c.user_id == self._user_id,
                table.c.status == "applied",
                table.c.withdrawn_at.is_(None),
                table.c.target_dimension == dimension,
            )
        ).scalar_one()
        return int(value)

    def _restates(
        self, dimension: str, direction: Direction, user_text: str, *, exclude: uuid.UUID
    ) -> bool:
        folded = normalise(user_text)
        if not folded:
            return True
        rows = self._conn.execute(
            select(table.c.user_text).where(
                table.c.user_id == self._user_id,
                table.c.target_dimension == dimension,
                table.c.asserted_direction == direction,
                table.c.withdrawn_at.is_(None),
                table.c.id != exclude,
            )
        ).all()
        return any(normalise(row[0]) == folded for row in rows)


class PostgresScoreOverrideRepository(TenantScopedRepository):
    """This user's local score overrides, and no one else's.

    Kept apart from the pushback repository on purpose. An override is not a
    correction the tool learned from -- it is the user setting the displayed
    number by hand for one application, and it must not be able to reach a
    dimension's displacement even by accident. Two repositories with no method
    in common is a cheaper guarantee than a comment saying not to.
    """

    def set_override(
        self,
        application_id: uuid.UUID,
        *,
        axis: Axis,
        value: int | None,
        note: str = "",
    ) -> ScoreOverride:
        """Set or clear the override for one axis of one application.

        Appends: clearing an override leaves the row that set it, because the
        record of what you told the tool in March stays readable.
        """
        row = self._conn.execute(
            overrides_table.insert()
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                axis=axis,
                value=value,
                note=note,
            )
            .returning(*_OVERRIDE_COLUMNS)
        ).one()
        return ScoreOverride(
            id=row.id,
            application_id=row.application_id,
            axis=row.axis,
            value=row.value,
            note=row.note,
            created_at=row.created_at,
        )

    def current(self, application_id: uuid.UUID) -> dict[str, ScoreOverride]:
        """Axis -> the live override on it, if any. Latest row per axis wins."""
        rows = self._conn.execute(
            select(*_OVERRIDE_COLUMNS)
            .where(
                overrides_table.c.application_id == application_id,
                overrides_table.c.user_id == self._user_id,
            )
            .order_by(overrides_table.c.created_at.asc())
        ).all()
        live: dict[str, ScoreOverride] = {}
        for row in rows:
            override = ScoreOverride(
                id=row.id,
                application_id=row.application_id,
                axis=row.axis,
                value=row.value,
                note=row.note,
                created_at=row.created_at,
            )
            if override.value is None:
                live.pop(override.axis, None)
            else:
                live[override.axis] = override
        return live


__all__ = [
    "PostgresPushbackRepository",
    "PostgresScoreOverrideRepository",
    "normalise",
]
