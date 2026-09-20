"""Application questions -- two equal paths, advised not prescribed.

See CLAUDE.md's 2026-09-18 decision and NEXT.md's task 4. Under an
application, the user pastes a question and then has two buttons of equal
standing: **"Check my answer"** (they write it, the claim gate and a model
assessment run over it) and **"Draft one for me"** (generated from the corpus,
then gated the same way). Both calls to the model happen in the worker -- see
`jfl_worker.handlers.application_questions` -- so every POST here creates a
row, enqueues a task and redirects. Fast input, slow processing, same shape as
`jfl_web.routes.applications`.

Screens:

  POST /applications/{id}/questions              -- add a question
  POST /applications/{id}/questions/{qid}/check   -- "check my answer"
  POST /applications/{id}/questions/{qid}/draft   -- "draft one for me"
  GET  /applications/{id}/questions/{qid}         -- one question, for htmx to
                                                      poll while an answer is
                                                      in flight
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import ApplicationQuestion

from jfl_web.applicationanswers import QuestionView, failure_for
from jfl_web.deps import (
    ApplicationQuestionRepoDep,
    ApplicationRepoDep,
    CsrfDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.templating import render

router = APIRouter()

CHECK_ANSWER_KIND = "check_application_answer"
DRAFT_ANSWER_KIND = "draft_application_answer"

# A question is a line or two, not a document -- generous headroom against
# someone pasting a whole recruiter email by mistake.
MAX_QUESTION_CHARS = 2000
# An answer someone writes themselves before pressing "check". Comfortably
# above what any real application-question answer runs to.
MAX_ANSWER_CHARS = 8000

_NOT_FOUND = "No question found -- it may belong to another account."


def _error(request: Request, session: SessionDep, message: str, status_code: int) -> Response:
    return render(
        request,
        "error.html",
        {"session": session, "user": session.user, "message": message},
        status_code=status_code,
    )


def _get_owned_question(
    questions: ApplicationQuestionRepoDep, application_id: uuid.UUID, question_id: uuid.UUID
) -> ApplicationQuestion | None:
    """The question, only if it is this user's AND belongs to this application
    -- the second check guards a URL with mismatched ids; tenancy itself is
    already enforced by `questions` being bound to the signed-in user.
    """
    question = questions.get_question(question_id)
    if question is None or question.application_id != application_id:
        return None
    return question


@router.post("/applications/{application_id}/questions")
def add_question(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    questions: ApplicationQuestionRepoDep,
    _csrf: CsrfDep,
    question_text: Annotated[str, Form()],
) -> Response:
    if applications.get_application(application_id) is None:
        return _error(request, session, _NOT_FOUND, 404)

    text_value = question_text.strip()
    if not text_value or len(text_value) > MAX_QUESTION_CHARS:
        return _error(
            request,
            session,
            "Paste the application question to add it."
            if not text_value
            else "That is much longer than an application question.",
            400,
        )
    questions.add_question(application_id, text_value)
    return RedirectResponse(f"/applications/{application_id}", status_code=303)


@router.post("/applications/{application_id}/questions/{question_id}/check")
def check_answer(
    request: Request,
    application_id: uuid.UUID,
    question_id: uuid.UUID,
    session: SessionDep,
    questions: ApplicationQuestionRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    answer_text: Annotated[str, Form()],
) -> Response:
    """ "Check my answer" -- the user's own words, checked by the claim gate
    and assessed against the question and the role. The page advises
    answering first; it never requires it -- this route and `draft_answer`
    below are reached from the same two equally-sized buttons.
    """
    question = _get_owned_question(questions, application_id, question_id)
    if question is None:
        return _error(request, session, _NOT_FOUND, 404)

    text_value = answer_text.strip()
    if not text_value or len(text_value) > MAX_ANSWER_CHARS:
        return RedirectResponse(f"/applications/{application_id}#question-{question_id}", 303)

    answer = questions.create_user_answer(question_id, text_value)
    if answer is not None:
        tasks.enqueue(kind=CHECK_ANSWER_KIND, payload={"answer_id": str(answer.id)})
    return RedirectResponse(f"/applications/{application_id}#question-{question_id}", 303)


@router.post("/applications/{application_id}/questions/{question_id}/draft")
def draft_answer(
    request: Request,
    application_id: uuid.UUID,
    question_id: uuid.UUID,
    session: SessionDep,
    questions: ApplicationQuestionRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """ "Draft one for me" -- generated from the corpus, then gated
    automatically, exactly like a CV draft. Equally reachable, never hidden
    behind having written something first.
    """
    question = _get_owned_question(questions, application_id, question_id)
    if question is None:
        return _error(request, session, _NOT_FOUND, 404)

    answer = questions.create_draft_answer(question_id)
    if answer is not None:
        tasks.enqueue(kind=DRAFT_ANSWER_KIND, payload={"answer_id": str(answer.id)})
    return RedirectResponse(f"/applications/{application_id}#question-{question_id}", 303)


@router.get("/applications/{application_id}/questions/{question_id}")
def question_row(
    request: Request,
    application_id: uuid.UUID,
    question_id: uuid.UUID,
    session: SessionDep,
    questions: ApplicationQuestionRepoDep,
) -> Response:
    """One question, standalone -- what `_application_question.html`'s rows
    poll while an answer is in flight. Same view-building function the full
    panel uses, so the two can never render a question's state differently.
    """
    question = _get_owned_question(questions, application_id, question_id)
    if question is None:
        return _error(request, session, _NOT_FOUND, 404)

    latest = questions.latest_answer(question_id)
    error_code = latest.error_code if latest and latest.status == "failed" else None
    view = QuestionView(question=question, latest=latest, failure=failure_for(error_code))
    return render(
        request,
        "_application_question.html",
        {"session": session, "application_id": application_id, "view": view},
    )
