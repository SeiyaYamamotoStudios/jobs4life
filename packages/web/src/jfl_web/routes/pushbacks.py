"""Disagreeing with a score: one box, a plain result, and an undo.

Design: `~/jobs4life-profile-research/feedback-loops.md`, "The loop we should
build". The rule is `jfl_core.pushback`, the log is
`jfl_core.storage.pushbacks`, the wording is `jfl_web.pushbacks`, and this file
is the handful of things a person does.

**One box, then the result, then an undo.** The owner found the earlier flow
-- type, wait, confirm a kind on radio buttons, apply -- too many steps, and
still could not tell what had changed. So the box takes words and nothing else;
the worker reads them and applies the reading in one task; and the panel shows
one card: what we took it to mean, the number before -> after (or "stays at"),
and what would move it. "Not what I meant" withdraws that correction and
re-applies under the reading the person picks, in one POST.

Undo-after is only as safe as the result is visible, and the guards that make
it safe are not here: `jfl_core.pushback.decide` gives no reading a way to move
"could I get this" upward, and a CHECK in the database says the same.

**No model call in any handler here.** Recording a pushback is an INSERT and an
enqueue; the reading happens in the worker on the user's own key.

Screens:

  POST /applications/{id}/pushback        -- the box: record it, verbatim
  GET  /applications/{id}/pushbacks       -- the panel, for htmx to poll
  POST /pushbacks/{id}/reading            -- "Not what I meant": undo, and
                                             optionally re-apply as another reading
  GET  /pushbacks/{id}/evidence           -- the fact that would move a
  POST /pushbacks/{id}/evidence              "stronger fit" claim, in their words
  POST /applications/{id}/override        -- the local, labelled escape hatch
  GET  /pushbacks                         -- every correction, as plain history

Errors are rendered on the page, never passed through a query string: a message
in a query string is a message an attacker can write.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import ApplicationScore, Pushback
from jfl_core.pushback import WANT_OVERALL, Axis, Direction
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.ui_sections import SectionState

from jfl_web.deps import (
    ApplicationRepoDep,
    CsrfDep,
    PushbackRepoDep,
    ScoreOverrideRepoDep,
    ScoreRepoDep,
    SectionRepoDep,
    SessionDep,
    TaskRepoDep,
    UserCorpusRepoDep,
)
from jfl_web.pushbacks import (
    BOX_BUTTON,
    BOX_LABEL,
    DRIFT_LINK,
    DRIFT_LOUD,
    EVIDENCE_NOTE,
    EVIDENCE_QUESTION,
    EVIDENCE_SECTION,
    JUST_UNDO,
    NEVER_CHANGED,
    NEVER_CHANGED_HEADING,
    NOT_MEANT,
    NOT_MEANT_INTRO,
    card,
    drift_line,
    history_line,
    reading_by_key,
    reading_of,
    was_sent,
)
from jfl_web.scores import COST_NOTE
from jfl_web.sections import pushbacks_section as build_pushbacks_section
from jfl_web.templating import render

# The worker's kind for "read this pushback". A string on both sides, for the
# same reason `jfl_web.routes.applications` gives: importing `jfl_worker` here
# would make the web container carry the worker.
CLASSIFY_PUSHBACK_KIND = "classify_pushback"

router = APIRouter()

# Long enough for a real objection, short enough that this is a text box and
# not a document store. A pushback is a sentence or two.
MAX_PUSHBACK_CHARS = 2000

# The evidence answer goes into the user's confirmed facts verbatim and becomes
# one statement, not an essay.
MAX_EVIDENCE_CHARS = 1000

# The one-box form does not ask how far out the number is. One point is the
# smallest thing the rule accepts, and the caps mean a bigger figure would
# barely matter for "do I want this"; where it would matter -- "you've overrated
# me", applied in full -- one point per correction is the conservative answer,
# and saying it again moves it again.
ASSERTED_POINTS = 1.0

_NOT_FOUND = "No application found -- it may belong to another account."
_NO_PUSHBACK = "No correction found -- it may belong to another account."


@router.post("/applications/{application_id}/pushback")
def record_pushback(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    pushbacks: PushbackRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    user_text: Annotated[str, Form()],
) -> Response:
    """The box. Record the words verbatim and queue the reading.

    Which score the words are about, and which way they push, are not asked:
    the reading supplies both, and until it does the row is inert -- it counts
    for nothing and moves nothing. The row is recorded against "do I want this"
    pushed up until then, which is the side the drift meter would rather
    over-count than miss.

    Both writes are in the request's single transaction, so there is no state
    where a pushback sits unread with nothing queued to read it.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        return _error(request, session, _NOT_FOUND, 404)

    score = scores.latest(application_id)
    if score is None or score.status != "done":
        return _error(request, session, "There is no finished score here to disagree with.", 400)

    text = user_text.strip()
    if not text or len(text) > MAX_PUSHBACK_CHARS:
        message = (
            "Say what you disagree with, in your own words."
            if not text
            else "That is longer than this box needs -- a sentence or two is plenty."
        )
        return _error(request, session, message, 400)

    row = pushbacks.record(
        application_id=application_id,
        score_id=score.id,
        axis="want",
        dimension=WANT_OVERALL,
        shown_score=score.want_it_score,
        shown_explanation=score.want_it_assessment,
        user_text=text,
        asserted_direction="up",
        asserted_points=ASSERTED_POINTS,
    )
    tasks.enqueue(
        kind=CLASSIFY_PUSHBACK_KIND,
        # Ids only. The user's words are already stored once.
        payload={"pushback_id": str(row.id)},
    )
    # POST/redirect/GET: a refresh must not record the same disagreement twice.
    return RedirectResponse(f"/applications/{application_id}#pushbacks", status_code=303)


@router.post("/pushbacks/{pushback_id}/reading")
def choose_reading(
    request: Request,
    pushback_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    pushbacks: PushbackRepoDep,
    _csrf: CsrfDep,
    reading: Annotated[str, Form()],
) -> Response:
    """ "Not what I meant" -- and the same list when the reading failed.

    `reading` is a key from `jfl_web.pushbacks.READINGS`, or `undo` to take the
    correction back and put nothing in its place.

    On an applied correction: it is withdrawn (kept in the log, counted by
    nothing), and the same words are recorded again as a new row under the
    reading the person picked, applied at once. A new row rather than an edit
    because the log is append-only -- the record shows both what we read and
    what they said they meant. On a row that was never applied (the reading
    failed), the pick is applied to that row directly.
    """
    existing = pushbacks.get(pushback_id)
    if existing is None:
        return _error(request, session, _NO_PUSHBACK, 404)
    back = RedirectResponse(f"/applications/{existing.application_id}#pushbacks", status_code=303)

    if reading == "undo":
        pushbacks.withdraw(pushback_id)
        return back

    chosen = reading_by_key(reading)
    if chosen is None:
        return _error(request, session, "Pick one of the readings on the list.", 400)
    if existing.withdrawn:
        # Already undone. Nothing to re-read; the page says so.
        return back

    axis: Axis = "get" if chosen.kind == "capability" else "want"
    direction: Direction = chosen.direction or (
        "down" if existing.asserted_direction == "down" else "up"
    )
    score = scores.get(existing.score_id)
    shown_score, shown_explanation = _stimulus(score, axis)
    new_information = True if existing.new_information is None else existing.new_information

    target = existing
    if existing.status == "applied":
        if reading_of(existing) == chosen:
            # "Not what I meant" and then the same meaning: nothing to change.
            return back
        pushbacks.withdraw(pushback_id)
        target = pushbacks.record(
            application_id=existing.application_id,
            score_id=existing.score_id,
            axis=axis,
            dimension=WANT_OVERALL,
            shown_score=shown_score,
            shown_explanation=shown_explanation,
            user_text=existing.user_text,
            asserted_direction=direction,
            asserted_points=existing.asserted_points,
        )

    pushbacks.set_reading(
        target.id,
        axis=axis,
        direction=direction,
        shown_score=shown_score,
        shown_explanation=shown_explanation,
    )
    pushbacks.set_classification(
        target.id,
        classification=chosen.kind,
        new_information=new_information,
        source="user",
    )
    pushbacks.apply(
        target.id,
        classification=chosen.kind,
        new_information=new_information,
        evidence_question=EVIDENCE_QUESTION,
        submitted_applications=_submitted(applications),
    )
    return back


def _stimulus(score: ApplicationScore | None, axis: Axis) -> tuple[int | None, str]:
    """The number and sentence on the axis the words are about."""
    if score is None:
        return None, ""
    if axis == "want":
        return score.want_it_score, score.want_it_assessment
    return score.could_get_score, score.could_get_assessment


def _submitted(applications: ApplicationRepoDep) -> int:
    """How many applications this user has actually sent.

    The behavioural channel in the shrinkage denominator: enacting a preference
    is better evidence of it than asserting one, so a submitted application
    counts for three observations against a pushback's one. Counted across the
    account rather than per dimension -- see `jfl_core.pushback.observations`.
    """
    live = applications.list_applications(archived=False)
    return sum(
        1 for a in live if a.status in ("applied", "screening", "interviewing", "offer", "rejected")
    )


@router.get("/pushbacks/{pushback_id}/evidence")
def evidence_form(
    request: Request,
    pushback_id: uuid.UUID,
    session: SessionDep,
    pushbacks: PushbackRepoDep,
) -> Response:
    """The one question a "stronger fit" correction opens, on its own page, so
    the card that links here can stay one short paragraph.
    """
    existing = pushbacks.get(pushback_id)
    if existing is None:
        return _error(request, session, _NO_PUSHBACK, 404)
    if existing.disposition != "pending_evidence" or existing.withdrawn:
        return RedirectResponse(
            f"/applications/{existing.application_id}#pushbacks", status_code=303
        )
    return render(
        request,
        "pushback_evidence.html",
        {
            "session": session,
            "user": session.user,
            "pushback": existing,
            "question": existing.evidence_question or EVIDENCE_QUESTION,
            "evidence_note": EVIDENCE_NOTE,
        },
    )


@router.post("/pushbacks/{pushback_id}/evidence")
def record_evidence(
    request: Request,
    pushback_id: uuid.UUID,
    session: SessionDep,
    pushbacks: PushbackRepoDep,
    corpus: UserCorpusRepoDep,
    _csrf: CsrfDep,
    answer: Annotated[str, Form()],
) -> Response:
    """The user's own words, into their confirmed facts, verbatim.

    No model is on this path and none may be put on one. A model tidying an
    answer into a neater fact is the ratchet in miniature: the user is then
    held to wording they did not choose. CLAUDE.md, 2026-09-01, unchanged.

    The number still does not move here. The fact now exists; the score is
    recomputed when the user scores the job again, which is a model call on
    their own key and therefore theirs to ask for.
    """
    existing = pushbacks.get(pushback_id)
    if existing is None:
        return _error(request, session, _NO_PUSHBACK, 404)

    text = " ".join(answer.split())
    if not text or len(text) > MAX_EVIDENCE_CHARS:
        message = (
            "Write the sentence in your own words -- it is kept exactly as you type it."
            if not text
            else "That is longer than one statement should be."
        )
        return _error(request, session, message, 400)

    span_id = corpus.add_statement(EVIDENCE_SECTION, text)
    pushbacks.record_evidence(pushback_id, span_id)
    return RedirectResponse(f"/applications/{existing.application_id}#pushbacks", status_code=303)


@router.post("/applications/{application_id}/override")
def set_override(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    overrides: ScoreOverrideRepoDep,
    _csrf: CsrfDep,
    axis: Annotated[str, Form()],
    value: Annotated[str, Form()] = "",
    note: Annotated[str, Form()] = "",
) -> Response:
    """Set or clear a displayed number by hand, for this application only.

    Offered honestly and labelled everywhere. It feeds no correction, so a user
    who overrides every number has changed nothing about how the next job is
    scored, and the claim gate does not know this exists.
    """
    if applications.get_application(application_id) is None:
        return _error(request, session, _NOT_FOUND, 404)
    if axis not in ("want", "get"):
        return _error(request, session, "Unknown axis.", 400)
    narrow_axis: Axis = "want" if axis == "want" else "get"

    cleared = not value.strip()
    number: int | None = None
    if not cleared:
        try:
            number = int(value)
        except ValueError:
            return _error(request, session, "An override is a number from 1 to 10.", 400)
        if not 1 <= number <= 10:
            return _error(request, session, "An override is a number from 1 to 10.", 400)

    overrides.set_override(
        application_id, axis=narrow_axis, value=number, note=" ".join(note.split())
    )
    return RedirectResponse(f"/applications/{application_id}#score", status_code=303)


@router.get("/pushbacks")
def pushback_log(
    request: Request,
    session: SessionDep,
    pushbacks: PushbackRepoDep,
) -> Response:
    """Every correction, newest first, as plain history: when, what you said,
    what it changed.

    The log is the store -- there is no preference weight anywhere for this
    page to be a view of -- so this reads as an audit, not a summary of one.
    """
    rows = pushbacks.recent()
    return render(
        request,
        "pushbacks.html",
        {
            "session": session,
            "user": session.user,
            "items": [{"pushback": row, "changed": history_line(row)} for row in rows],
            "drift": drift_line(pushbacks.drift_meter()),
            "drift_loud": DRIFT_LOUD,
            "awaiting_evidence": pushbacks.awaiting_evidence(),
            "evidence_question": EVIDENCE_QUESTION,
        },
    )


@router.get("/applications/{application_id}/pushbacks")
def pushback_panel(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    pushbacks: PushbackRepoDep,
    ui_sections: SectionRepoDep,
) -> Response:
    """The panel on its own, for htmx to poll while a reading is in flight.

    The fragment carries its polling trigger only while the latest correction
    is still being read, so the poll stops by virtue of what came back.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        return _error(request, session, _NOT_FOUND, 404)
    score = scores.latest(application_id)
    return render(
        request,
        "_pushbacks.html",
        {
            "session": session,
            "application_id": application_id,
            "score": score,
            **pushback_context(detail, score, pushbacks, ui_sections.states()),
        },
    )


def pushback_context(
    detail: object,
    score: ApplicationScore | None,
    pushbacks: PushbackRepoDep,
    states: dict[str, SectionState] | None = None,
) -> dict[str, object]:
    """One shape for both the detail page and the polled fragment, so the panel
    cannot render differently depending on which route produced it.

    The latest correction gets the card; every earlier one is a line in the
    folded history under it.
    """
    rows = pushbacks.for_application(detail.application.id)  # type: ignore[attr-defined]
    sent = was_sent(detail)  # type: ignore[arg-type]
    latest: Pushback | None = rows[-1] if rows else None
    latest_card = (
        card(latest, score=score, log=pushbacks.applied_log(), sent=sent) if latest else None
    )
    earlier = [{"pushback": row, "changed": history_line(row)} for row in reversed(rows[:-1])]
    return {
        "latest_pushback": latest,
        "pushback_card": latest_card,
        "pushback_history": earlier,
        "pushback_polling": latest is not None
        and latest.status == "awaiting_classification"
        and latest.error_code is None,
        "drift": drift_line(pushbacks.drift_meter()),
        "drift_link": DRIFT_LINK,
        "drift_loud": DRIFT_LOUD,
        "box_label": BOX_LABEL,
        "box_button": BOX_BUTTON,
        "not_meant": NOT_MEANT,
        "not_meant_intro": NOT_MEANT_INTRO,
        "just_undo": JUST_UNDO,
        "never_changed": NEVER_CHANGED,
        "never_changed_heading": NEVER_CHANGED_HEADING,
        "sent": sent,
        "score_cost_note": COST_NOTE,
        "pushbacks_section": build_pushbacks_section(states or {}, earlier),
    }


def _error(request: Request, session: AuthenticatedSession, message: str, status: int) -> Response:
    return render(
        request,
        "error.html",
        {"session": session, "user": session.user, "message": message},
        status_code=status,
    )
