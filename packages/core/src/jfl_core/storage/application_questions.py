"""Application questions -- two equal paths, tenancy-scoped.

See CLAUDE.md's 2026-09-18 decision ("check my answer" / "draft one for me"
side by side, the page advises, it never prescribes) and NEXT.md's task 4.

Two tables, one repository. `application_questions` holds the question text,
written once. `application_question_answers` holds every *attempt* to answer
it -- append-only, never an UPDATE to a previous row, the same rule
`jfl_core.storage.profile` follows for `profile_answers` and for the same
reason: a tool whose whole claim is measuring distance from what someone
actually said must never let that record be edited out from under them.
Pressing "check my answer" or "draft one for me" a second time creates a new
row; the previous one is still there, in `list_answers`.

No model call anywhere in this module. The two model calls this feature makes
(the assessment call, `jfl_generate.answers.assess_answer`; the claim gate's own
automatic pass) happen in the worker, which reads a `pending` row through
`get_answer` and writes the result back through `mark_done` / `mark_failed` --
same shape as `jfl_core.storage.title_suggestions`.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import insert, select, update

from jfl_core.db.tables import application_question_answers as answers_table
from jfl_core.db.tables import application_questions as questions_table
from jfl_core.models import AnswerErrorCode, ApplicationQuestion, ApplicationQuestionAnswer
from jfl_core.storage.tenancy import TenantScopedRepository

_QUESTION_COLUMNS = (
    questions_table.c.id,
    questions_table.c.user_id,
    questions_table.c.application_id,
    questions_table.c.question_text,
    questions_table.c.created_at,
)

_ANSWER_COLUMNS = (
    answers_table.c.id,
    answers_table.c.user_id,
    answers_table.c.question_id,
    answers_table.c.kind,
    answers_table.c.answer_text,
    answers_table.c.status,
    answers_table.c.error_code,
    answers_table.c.gate_result,
    answers_table.c.assessment,
    answers_table.c.model,
    answers_table.c.trace_id,
    answers_table.c.created_at,
    answers_table.c.updated_at,
)


def _question_from_row(row: Any) -> ApplicationQuestion:
    return ApplicationQuestion(
        id=row.id,
        user_id=row.user_id,
        application_id=row.application_id,
        question_text=row.question_text,
        created_at=row.created_at,
    )


def _answer_from_row(row: Any) -> ApplicationQuestionAnswer:
    return ApplicationQuestionAnswer(
        id=row.id,
        user_id=row.user_id,
        question_id=row.question_id,
        kind=row.kind,
        answer_text=row.answer_text,
        status=row.status,
        error_code=row.error_code,
        gate_result=row.gate_result,
        assessment=row.assessment,
        model=row.model,
        trace_id=row.trace_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresApplicationQuestionRepository(TenantScopedRepository):
    """One user's application questions and answer attempts, and no one else's."""

    # -- questions -------------------------------------------------------

    def add_question(self, application_id: uuid.UUID, question_text: str) -> ApplicationQuestion:
        row = self._conn.execute(
            insert(questions_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                question_text=question_text,
            )
            .returning(*_QUESTION_COLUMNS)
        ).one()
        return _question_from_row(row)

    def get_question(self, question_id: uuid.UUID) -> ApplicationQuestion | None:
        row = self._conn.execute(
            select(*_QUESTION_COLUMNS).where(
                questions_table.c.id == question_id, questions_table.c.user_id == self._user_id
            )
        ).first()
        return None if row is None else _question_from_row(row)

    def list_questions(self, application_id: uuid.UUID) -> list[ApplicationQuestion]:
        """Every question added under this application, in the order they were
        added -- the order the page offers them for answering.
        """
        rows = self._conn.execute(
            select(*_QUESTION_COLUMNS)
            .where(
                questions_table.c.application_id == application_id,
                questions_table.c.user_id == self._user_id,
            )
            .order_by(questions_table.c.created_at.asc())
        ).all()
        return [_question_from_row(r) for r in rows]

    # -- answer attempts ---------------------------------------------------

    def create_user_answer(
        self, question_id: uuid.UUID, answer_text: str
    ) -> ApplicationQuestionAnswer | None:
        """A new `pending`, `kind='user'` attempt: the answer text is already
        known (the user wrote it), and the worker only has to run the claim
        gate and the assessment call over it. None if there is no such
        question for this user -- nothing is written.
        """
        if self.get_question(question_id) is None:
            return None
        row = self._conn.execute(
            insert(answers_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                question_id=question_id,
                kind="user",
                answer_text=answer_text,
            )
            .returning(*_ANSWER_COLUMNS)
        ).one()
        return _answer_from_row(row)

    def create_draft_answer(self, question_id: uuid.UUID) -> ApplicationQuestionAnswer | None:
        """A new `pending`, `kind='draft'` attempt: `answer_text` starts empty
        and is filled in by the worker once it has generated something. None
        if there is no such question for this user.
        """
        if self.get_question(question_id) is None:
            return None
        row = self._conn.execute(
            insert(answers_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                question_id=question_id,
                kind="draft",
                answer_text="",
            )
            .returning(*_ANSWER_COLUMNS)
        ).one()
        return _answer_from_row(row)

    def get_answer(self, answer_id: uuid.UUID) -> ApplicationQuestionAnswer | None:
        row = self._conn.execute(
            select(*_ANSWER_COLUMNS).where(
                answers_table.c.id == answer_id, answers_table.c.user_id == self._user_id
            )
        ).first()
        return None if row is None else _answer_from_row(row)

    def list_answers(self, question_id: uuid.UUID) -> list[ApplicationQuestionAnswer]:
        """Every attempt at this question, oldest first -- the history."""
        rows = self._conn.execute(
            select(*_ANSWER_COLUMNS)
            .where(
                answers_table.c.question_id == question_id,
                answers_table.c.user_id == self._user_id,
            )
            .order_by(answers_table.c.created_at.asc())
        ).all()
        return [_answer_from_row(r) for r in rows]

    def latest_answer(self, question_id: uuid.UUID) -> ApplicationQuestionAnswer | None:
        row = self._conn.execute(
            select(*_ANSWER_COLUMNS)
            .where(
                answers_table.c.question_id == question_id,
                answers_table.c.user_id == self._user_id,
            )
            .order_by(answers_table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _answer_from_row(row)

    def mark_done(
        self,
        answer_id: uuid.UUID,
        *,
        answer_text: str | None,
        gate_result: dict[str, object],
        assessment: dict[str, object] | None,
        model: str,
        trace_id: uuid.UUID,
    ) -> None:
        """Record a successful attempt. `answer_text` is given only by
        `draft_application_answer` (the generated text); a check leaves it as
        None and the column keeps what the user typed at creation.
        """
        values: dict[str, object] = {
            "status": "done",
            "error_code": None,
            "gate_result": gate_result,
            "assessment": assessment,
            "model": model,
            "trace_id": trace_id,
        }
        if answer_text is not None:
            values["answer_text"] = answer_text
        self._conn.execute(
            update(answers_table)
            .where(answers_table.c.id == answer_id, answers_table.c.user_id == self._user_id)
            .values(**values)
        )

    def mark_failed(self, answer_id: uuid.UUID, code: AnswerErrorCode) -> None:
        self._conn.execute(
            update(answers_table)
            .where(answers_table.c.id == answer_id, answers_table.c.user_id == self._user_id)
            .values(status="failed", error_code=code)
        )
