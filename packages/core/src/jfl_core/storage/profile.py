"""Profile setup storage -- PLAN.md slice B3a, tenancy-scoped.

Three tables, three shapes of "not the same as everything else":

* **Simple keyed answers** (`profile_answers`) are append-only. Saving a
  question a second time never overwrites the first row -- it inserts a new
  one, so "what the user said, when" stays readable after a change. The
  current value of a question is its latest row; `get_current_answers`
  resolves that with `DISTINCT ON`. Saving the same text again (no change) is
  a no-op, so re-submitting an untouched section does not manufacture history.
  A question a user has never answered simply has no row -- never a default
  row, never an inferred one, per PLAN.md's "a skipped question is never
  guessed at".

* **Objectives** (`profile_objectives`, questions 10/11) are up to four
  separate slots, one per ordinal 1-4, versioned exactly like
  `profile_answers`: saving a slot a second time never overwrites the first
  row -- it inserts a new one, so what the user once said an objective was is
  never lost. The current value of a slot is its latest row, resolved the
  same way as answers. Clearing both fields is itself saved as a new version
  (with blank text) rather than deleting anything; the *current view* treats
  an all-blank latest version as "no objective" -- absent from
  `list_objectives`, rendered as "not stated" -- without erasing the fact
  that an objective once stood there.

* **Ruled-out decisions** (`profile_ruled_out`, question 17) are add-only and
  never deleted, each with its own `recorded_at`. Marking one reopened sets
  `reopened_at` rather than removing it, so a decision that gets revisited is
  still on the record for later slices to flag against.

No model call anywhere in this module -- these are the user's own words,
verbatim, and nothing here normalises or tidies them. Nothing here logs answer
text; callers must not either (comp and deal-breakers are sensitive, and the
project's standing rule is never in `runs`, never in a trace, never logged).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from sqlalchemy import func, insert, select, update

from jfl_core.db.tables import profile_answers as answers_table
from jfl_core.db.tables import profile_objectives as objectives_table
from jfl_core.db.tables import profile_ruled_out as ruled_out_table
from jfl_core.models import ProfileAnswer, ProfileObjective, ProfileRuledOut
from jfl_core.profile_questions import QUESTION_KEYS
from jfl_core.storage.tenancy import TenantScopedRepository


class UnknownQuestionKeyError(ValueError):
    """`question_key` is not one of `jfl_core.profile_questions.QUESTION_KEYS`.

    A defensive check ahead of the database's own CHECK constraint: the
    constraint is the guarantee, this is a clearer failure than an
    `IntegrityError` from inside a caller that built the key programmatically.
    """

    def __init__(self, question_key: str) -> None:
        super().__init__(f"{question_key!r} is not a known profile question key")


_ANSWER_COLUMNS = (
    answers_table.c.id,
    answers_table.c.question_key,
    answers_table.c.answer_text,
    answers_table.c.structured,
    answers_table.c.created_at,
)

_OBJECTIVE_COLUMNS = (
    objectives_table.c.id,
    objectives_table.c.ordinal,
    objectives_table.c.objective_text,
    objectives_table.c.evidence_text,
    objectives_table.c.created_at,
)

_RULED_OUT_COLUMNS = (
    ruled_out_table.c.id,
    ruled_out_table.c.decision_text,
    ruled_out_table.c.recorded_at,
    ruled_out_table.c.reopened_at,
)


def _answer_from_row(row: Any) -> ProfileAnswer:
    return ProfileAnswer(
        id=row.id,
        question_key=row.question_key,
        answer_text=row.answer_text,
        structured=row.structured,
        created_at=row.created_at,
    )


def _objective_from_row(row: Any) -> ProfileObjective:
    return ProfileObjective(
        id=row.id,
        ordinal=row.ordinal,
        objective_text=row.objective_text,
        evidence_text=row.evidence_text,
        created_at=row.created_at,
    )


def _objective_is_blank(objective_text: str, evidence_text: str) -> bool:
    """Both fields empty once surrounding whitespace is ignored -- the same
    test `save_objective` uses to decide whether a slot has anything in it.
    A slot whose *latest version* is blank reads as "no objective" even
    though the version itself is a real, kept row.
    """
    return not objective_text.strip() and not evidence_text.strip()


def _ruled_out_from_row(row: Any) -> ProfileRuledOut:
    return ProfileRuledOut(
        id=row.id,
        decision_text=row.decision_text,
        recorded_at=row.recorded_at,
        reopened_at=row.reopened_at,
    )


class PostgresProfileRepository(TenantScopedRepository):
    """One user's profile setup answers, objectives and ruled-out decisions."""

    # -- simple keyed answers -------------------------------------------------

    def get_current_answers(self) -> dict[str, ProfileAnswer]:
        """The latest answer for every question this user has ever answered,
        keyed by `question_key`. A key absent from the result means the
        question has never been answered -- render "not stated", never a
        default.
        """
        rows = self._conn.execute(
            select(*_ANSWER_COLUMNS)
            .distinct(answers_table.c.question_key)
            .where(answers_table.c.user_id == self._user_id)
            .order_by(answers_table.c.question_key, answers_table.c.created_at.desc())
        ).all()
        return {row.question_key: _answer_from_row(row) for row in rows}

    def save_answer(
        self,
        question_key: str,
        *,
        answer_text: str,
        structured: dict[str, Any] | None = None,
    ) -> ProfileAnswer | None:
        """Append a new version if it differs from the current one; otherwise
        do nothing and return None, so re-saving an untouched section does not
        manufacture history. An empty `answer_text` and `structured=None` is a
        legitimate value (the question left blank) and is still versioned if
        it differs from what came before -- e.g. clearing a previously typed
        answer is itself a change worth recording.

        A never-answered question submitted blank writes nothing at all --
        there is no way to tell "left blank" apart from "not touched" at the
        form layer, and PLAN.md's rule is that an unanswered question is
        simply absent, never a row saying so.

        Raises `UnknownQuestionKeyError` for a key outside `QUESTION_KEYS`,
        ahead of the database's own CHECK constraint.
        """
        if question_key not in QUESTION_KEYS:
            raise UnknownQuestionKeyError(question_key)
        current = self.get_current_answers().get(question_key)
        blank = not answer_text and not structured
        if current is None and blank:
            return None
        if (
            current is not None
            and current.answer_text == answer_text
            and current.structured == (structured or None)
        ):
            return None
        row = self._conn.execute(
            insert(answers_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                question_key=question_key,
                answer_text=answer_text,
                structured=structured,
            )
            .returning(*_ANSWER_COLUMNS)
        ).one()
        return _answer_from_row(row)

    def save_answers(self, answers: Mapping[str, tuple[str, dict[str, Any] | None]]) -> None:
        """Save several answers in one call -- one section's worth of a form
        submission. Each is independently versioned by `save_answer`; a
        question submitted blank and never answered before writes nothing.
        """
        for key, (text_value, structured) in answers.items():
            self.save_answer(key, answer_text=text_value, structured=structured)

    def history(self, question_key: str) -> list[ProfileAnswer]:
        """Every version of one question's answer, oldest first."""
        rows = self._conn.execute(
            select(*_ANSWER_COLUMNS)
            .where(
                answers_table.c.user_id == self._user_id,
                answers_table.c.question_key == question_key,
            )
            .order_by(answers_table.c.created_at.asc())
        ).all()
        return [_answer_from_row(r) for r in rows]

    # -- objectives (questions 10/11) -----------------------------------------

    def _latest_objective_version(self, ordinal: int) -> ProfileObjective | None:
        """The most recent row for this ordinal, blank or not -- used to
        decide whether a save is a no-op, a new version, or a first version,
        never to decide what the page shows (that is `list_objectives`).
        """
        row = self._conn.execute(
            select(*_OBJECTIVE_COLUMNS)
            .where(
                objectives_table.c.user_id == self._user_id,
                objectives_table.c.ordinal == ordinal,
            )
            .order_by(objectives_table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _objective_from_row(row)

    def list_objectives(self) -> list[ProfileObjective]:
        """The current, in-use objective slots, ordinal ascending -- a slot
        whose latest version is blank (cleared, or never set) is absent, same
        as "an unanswered question is simply absent" for the rest of the
        profile. Its history is not lost; see `objective_history`.
        """
        rows = self._conn.execute(
            select(*_OBJECTIVE_COLUMNS)
            .distinct(objectives_table.c.ordinal)
            .where(objectives_table.c.user_id == self._user_id)
            .order_by(objectives_table.c.ordinal, objectives_table.c.created_at.desc())
        ).all()
        objectives = [_objective_from_row(r) for r in rows]
        return [o for o in objectives if not _objective_is_blank(o.objective_text, o.evidence_text)]

    def objective_history(self, ordinal: int) -> list[ProfileObjective]:
        """Every version ever saved for this ordinal, oldest first -- the
        objectives analogue of `history`, including blank (cleared) versions.
        """
        rows = self._conn.execute(
            select(*_OBJECTIVE_COLUMNS)
            .where(
                objectives_table.c.user_id == self._user_id,
                objectives_table.c.ordinal == ordinal,
            )
            .order_by(objectives_table.c.created_at.asc())
        ).all()
        return [_objective_from_row(r) for r in rows]

    def save_objective(
        self, ordinal: int, *, objective_text: str, evidence_text: str
    ) -> ProfileObjective | None:
        """Append a new version for objective `ordinal` (1-4) if it differs
        from the current one; otherwise do nothing and return None, exactly
        `save_answer`'s semantics. Both fields blank is a legitimate value --
        clearing a previously set objective -- and is still versioned if it
        differs from what came before, so the fact that an objective once
        stood there is never lost.

        A slot that has never been set, submitted blank, writes nothing at
        all: there is no way to tell "left blank" apart from "not touched" at
        the form layer, and an unused slot must stay simply absent.

        Nothing here is ever deleted. Use `list_objectives` for the current,
        in-use slots -- a blank latest version reads there as "no objective".
        """
        current = self._latest_objective_version(ordinal)
        blank = _objective_is_blank(objective_text, evidence_text)
        if current is None and blank:
            return None
        if (
            current is not None
            and current.objective_text == objective_text
            and current.evidence_text == evidence_text
        ):
            return None
        row = self._conn.execute(
            insert(objectives_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                ordinal=ordinal,
                objective_text=objective_text,
                evidence_text=evidence_text,
            )
            .returning(*_OBJECTIVE_COLUMNS)
        ).one()
        return _objective_from_row(row)

    # -- ruled-out decisions (question 17) ------------------------------------

    def list_ruled_out(self) -> list[ProfileRuledOut]:
        """Oldest first -- the same ordering convention as job filter
        exceptions, so the earliest decision reads first.
        """
        rows = self._conn.execute(
            select(*_RULED_OUT_COLUMNS)
            .where(ruled_out_table.c.user_id == self._user_id)
            .order_by(ruled_out_table.c.recorded_at.asc(), ruled_out_table.c.id.asc())
        ).all()
        return [_ruled_out_from_row(r) for r in rows]

    def add_ruled_out(self, decision_text: str) -> ProfileRuledOut:
        row = self._conn.execute(
            insert(ruled_out_table)
            .values(id=uuid.uuid4(), user_id=self._user_id, decision_text=decision_text)
            .returning(*_RULED_OUT_COLUMNS)
        ).one()
        return _ruled_out_from_row(row)

    def mark_reopened(self, ruled_out_id: uuid.UUID) -> ProfileRuledOut | None:
        """Record that this ruled-out decision no longer holds. Never deletes
        the row -- the original decision and its date stay on the record.
        None if there is no such entry for this user.
        """
        row = self._conn.execute(
            update(ruled_out_table)
            .where(
                ruled_out_table.c.id == ruled_out_id,
                ruled_out_table.c.user_id == self._user_id,
            )
            .values(reopened_at=func.now())
            .returning(*_RULED_OUT_COLUMNS)
        ).first()
        return None if row is None else _ruled_out_from_row(row)
