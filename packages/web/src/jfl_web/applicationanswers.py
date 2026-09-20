"""What the application-questions panel needs before it renders.

Mirrors `jfl_web.jobads`'s split: the failure wording lives here rather than
in the worker, because the worker writes `application_question_answers.error_code`
while holding the user's decrypted API key, and a closed set of codes cannot
carry a secret. Wording is a UI concern, changeable without a migration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from jfl_core.models import AnswerErrorCode, ApplicationQuestion, ApplicationQuestionAnswer
from jfl_core.storage.application_questions import PostgresApplicationQuestionRepository


@dataclass(frozen=True, slots=True)
class AnswerFailure:
    """What to tell the user, and where to send them to fix it. Same shape as
    `jfl_web.jobads.ExtractionFailure`, deliberately duplicated rather than
    shared -- the two closed sets of codes are different, and a shared type
    would invite someone to widen one FAILURES table with a code the other
    schema has never heard of.
    """

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[AnswerErrorCode, AnswerFailure] = {
    "no_api_key": AnswerFailure(
        "This needs your own Anthropic API key -- checking or drafting an answer is a "
        "model call, and it is billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": AnswerFailure(
        "Anthropic rejected the API key stored here. Replace it and try again.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "model_refused": AnswerFailure(
        "The model declined to respond to that text. Trying again with the same "
        "wording usually works."
    ),
    "model_error": AnswerFailure("That attempt failed. Trying again is worth a go."),
    "credential_unreadable": AnswerFailure(
        "Your stored API key could not be unlocked on the server. Setting it again will fix it.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
    "no_requirements": AnswerFailure(
        "This job's requirements have not been read yet, so there is nothing to draft "
        "against -- read the job ad first."
    ),
}

_UNKNOWN = AnswerFailure("That attempt failed. Trying again is worth a go.")


def failure_for(code: AnswerErrorCode | None) -> AnswerFailure | None:
    """None means nothing failed -- there is no row, or the row is not
    `failed`, and the caller is expected to check that first.
    """
    return None if code is None else _FAILURES.get(code, _UNKNOWN)


@dataclass(frozen=True, slots=True)
class QuestionView:
    """One question plus its latest answer attempt, and the failure message
    for it if that attempt failed -- everything `_application_question.html`
    needs, built once so the full panel and the standalone polled row
    (`GET /applications/{id}/questions/{qid}`) can never render a question's
    state differently.
    """

    question: ApplicationQuestion
    latest: ApplicationQuestionAnswer | None
    failure: AnswerFailure | None


def question_views(
    questions: PostgresApplicationQuestionRepository, application_id: uuid.UUID
) -> list[QuestionView]:
    """Every question under this application, oldest first, each with its
    latest answer attempt resolved. One extra query per question -- the
    per-application question count is small (a handful of application-form
    questions, never hundreds), so this stays simple rather than trying to
    batch it.
    """
    views: list[QuestionView] = []
    for question in questions.list_questions(application_id):
        latest = questions.latest_answer(question.id)
        error_code = latest.error_code if latest and latest.status == "failed" else None
        views.append(
            QuestionView(question=question, latest=latest, failure=failure_for(error_code))
        )
    return views
