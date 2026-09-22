"""Disagreeing with a score, and what that is allowed to change.

Design: `~/jobs4life-profile-research/feedback-loops.md`, "The loop we should
build". The rule is `jfl_core.pushback`, the log is
`jfl_core.storage.pushbacks`, the wording is `jfl_web.pushbacks`, and this file
is the four things a person does.

**Nothing is applied until the user has seen the classification.** The three
kinds do sharply different things -- a preference is accepted and shrunk, a
capability claim upward moves nothing at all, a factual objection is about the
ad -- so classifying a capability claim as a preference is precisely how the
ratchet gets in. A cheap model call proposes the classification in the
background; the POST that applies it takes the kind from the form the human was
looking at. There is no route on which a model's answer is applied unseen.

**No model call in any handler here.** Recording a pushback is an INSERT and an
enqueue; the classification happens in the worker on the user's own key. Fast
input, slow processing, the same shape as pasting a job ad.

Screens:

  POST /applications/{id}/pushback        -- record it, verbatim, apply nothing
  GET  /applications/{id}/pushbacks       -- the panel, for htmx to poll
  POST /pushbacks/{id}/apply              -- the user confirms or corrects the
                                             classification; the effect is computed
  POST /pushbacks/{id}/evidence           -- their own words into the corpus
  POST /applications/{id}/override        -- the local, labelled escape hatch
  GET  /pushbacks                         -- every correction, and the drift meter

Errors are rendered on the page, never passed through a query string: a message
in a query string is a message an attacker can write.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import ApplicationScore, Pushback
from jfl_core.pushback import (
    MAX_ASSERTED_POINTS,
    PUSHBACK_KINDS,
    Axis,
    Direction,
    PushbackKind,
    valid_dimension,
)
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
    EVIDENCE_SECTION,
    dimension_label,
    dimension_options,
    evidence_question,
    was_sent,
)
from jfl_web.sections import pushbacks_section as build_pushbacks_section
from jfl_web.templating import render

# The worker's kind for "classify this pushback". A string on both sides, for
# the same reason `jfl_web.routes.applications` gives: importing `jfl_worker`
# here would make the web container carry the worker.
CLASSIFY_PUSHBACK_KIND = "classify_pushback"

router = APIRouter()

# Long enough for a real objection, short enough that this is a text box and
# not a document store. A pushback is a sentence or two.
MAX_PUSHBACK_CHARS = 2000

# The evidence answer goes into the corpus verbatim and becomes one span, so it
# is one statement, not an essay.
MAX_EVIDENCE_CHARS = 1000

_NOT_FOUND = "No application found -- it may belong to another account."
_NO_PUSHBACK = "No pushback found -- it may belong to another account."


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
    axis: Annotated[str, Form()],
    dimension: Annotated[str, Form()],
    direction: Annotated[str, Form()],
    user_text: Annotated[str, Form()],
    points: Annotated[str, Form()] = "1",
) -> Response:
    """Record the disagreement. Nothing moves and nothing is classified yet.

    The user's words go in byte for byte, with the exact number and sentence
    they were shown beside them -- a pushback typed straight after reading our
    explanation is partly a response to our explanation, so the stimulus is
    part of the record rather than something to reconstruct later.

    Both writes are in the request's single transaction, so the row and its
    classification task are committed together: there is no state where a
    pushback sits unclassified with nothing queued to classify it.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        return _error(request, session, _NOT_FOUND, 404)

    score = scores.latest(application_id)
    if score is None or score.status != "done":
        return _error(request, session, "There is no finished score here to push back on.", 400)

    text = user_text.strip()
    if not text or len(text) > MAX_PUSHBACK_CHARS:
        message = (
            "Say what is wrong with this in your own words."
            if not text
            else "That is longer than a pushback needs to be -- a sentence or two is plenty."
        )
        return _error(request, session, message, 400)

    if axis not in ("want", "get") or direction not in ("up", "down"):
        return _error(request, session, "Unknown axis or direction.", 400)
    if not valid_dimension(dimension):
        # An allowlist, and anything off it is refused rather than silently
        # retargeted: the list is what keeps the claim gate, the corpus and the
        # coverage statuses out of reach, and quietly accepting something not on
        # it would be exactly the wrong failure mode.
        return _error(request, session, "That is not something a pushback can change.", 400)

    narrow_axis: Axis = "want" if axis == "want" else "get"
    narrow_direction: Direction = "up" if direction == "up" else "down"
    shown_score, shown_explanation = _stimulus(score, narrow_axis)

    row = pushbacks.record(
        application_id=application_id,
        score_id=score.id,
        axis=narrow_axis,
        dimension=dimension,
        shown_score=shown_score,
        shown_explanation=shown_explanation,
        user_text=text,
        asserted_direction=narrow_direction,
        asserted_points=_points(points),
    )
    tasks.enqueue(
        kind=CLASSIFY_PUSHBACK_KIND,
        # Ids only. The user's words are already stored once, and a second copy
        # in a payload that admin queries read back buys nothing.
        payload={"pushback_id": str(row.id)},
    )
    # POST/redirect/GET: a refresh must not record the same disagreement twice.
    return RedirectResponse(f"/applications/{application_id}#pushbacks", status_code=303)


def _points(raw: str) -> float:
    """1, 2 or 3 -- how far out they say it is, not what the number should be.

    Anything else lands on 1. People are far more reliable at relative
    judgements than absolute ones, so this figure is a coarse hint and the caps
    mean it barely matters; a request asserting 40 gets the smallest sensible
    answer rather than an error page.
    """
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    if value <= 0:
        return 1.0
    return min(value, MAX_ASSERTED_POINTS)


def _stimulus(score: ApplicationScore, axis: Axis) -> tuple[int | None, str]:
    """The exact number and sentence the user was arguing with."""
    if axis == "want":
        return score.want_it_score, score.want_it_assessment
    return score.could_get_score, score.could_get_assessment


@router.post("/pushbacks/{pushback_id}/apply")
def apply_pushback(
    request: Request,
    pushback_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    pushbacks: PushbackRepoDep,
    _csrf: CsrfDep,
    classification: Annotated[str, Form()],
    new_information: Annotated[str, Form()] = "",
) -> Response:
    """Apply the classification the human confirmed or corrected.

    `classification` comes from the radio they were looking at, never from the
    stored model answer -- the model's proposal is only the default selection.
    That is the whole protection against a misclassification silently changing
    the wrong thing, and it is one form field rather than a guard rail.

    `new_information` is likewise theirs to set, and is not the last word: an
    exact restatement of something already on the record for this dimension is
    forced to False in the repository, whatever the box says.
    """
    existing = pushbacks.get(pushback_id)
    if existing is None:
        return _error(request, session, _NO_PUSHBACK, 404)
    if classification not in PUSHBACK_KINDS:
        return _error(request, session, "Unknown kind.", 400)
    kind: PushbackKind = classification  # type: ignore[assignment]

    score = scores.get(existing.score_id)
    label = _label(existing, score)
    applied = pushbacks.apply(
        pushback_id,
        classification=kind,
        new_information=new_information == "yes",
        evidence_question=evidence_question(label),
        submitted_applications=_submitted(applications),
    )
    target = applied.application_id if applied else existing.application_id
    return RedirectResponse(f"/applications/{target}#pushbacks", status_code=303)


def _submitted(applications: ApplicationRepoDep) -> int:
    """How many applications this user has actually sent.

    The behavioural channel in the shrinkage denominator: enacting a preference
    is better evidence of it than asserting one, so a submitted application
    counts for three observations against a pushback's one. Counted across the
    account rather than per dimension -- the per-dimension refinement needs a
    structural query over stored JSONB verdicts, and the honest approximation is
    named here rather than hidden. See `jfl_core.pushback.observations`.
    """
    live = applications.list_applications(archived=False)
    return sum(
        1 for a in live if a.status in ("applied", "screening", "interviewing", "offer", "rejected")
    )


def _label(pushback: Pushback, score: ApplicationScore | None) -> str:
    if score is None:
        return dimension_label(pushback.dimension, [])
    axis: Axis = "want" if pushback.axis == "want" else "get"
    return dimension_label(pushback.dimension, dimension_options(score, axis))


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
    """The user's own words, into their corpus, verbatim.

    No model is on this path and none may be put on one. A model tidying an
    answer into a neater corpus fact is the ratchet in miniature: the user is
    then held to wording they did not choose, by a tool whose whole claim is
    that it measures distance from what they actually said. CLAUDE.md,
    2026-09-01, unchanged.

    The number still does not move here. The corpus now holds the fact; the
    score is recomputed when the user scores the job again, which is a model
    call on their own key and therefore theirs to ask for.
    """
    existing = pushbacks.get(pushback_id)
    if existing is None:
        return _error(request, session, _NO_PUSHBACK, 404)

    text = " ".join(answer.split())
    if not text or len(text) > MAX_EVIDENCE_CHARS:
        message = (
            "Write the sentence in your own words -- it is stored exactly as you type it."
            if not text
            else "That is longer than one corpus statement should be."
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

    Offered honestly and labelled everywhere. People will use an imperfect tool
    if they are allowed to modify it, even slightly, and an unbounded lever
    destroys the product -- so the lever is real, it is local, and it is never
    mistaken for what the tool said. It feeds no dimension's displacement, so a
    user who overrides every number has changed nothing about how the next job
    is scored, and the claim gate does not know this exists.
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
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    pushbacks: PushbackRepoDep,
) -> Response:
    """Every correction, in the user's own words, newest first, with the drift
    meter over it.

    The log is the store -- there is no preference weight anywhere for this
    page to be a view of. What is on it is what moved the numbers, which is why
    it can be read as an audit rather than as a summary of one.
    """
    rows = pushbacks.recent()
    items = []
    for row in rows:
        score = scores.get(row.score_id)
        items.append({"pushback": row, "label": _label(row, score)})
    return render(
        request,
        "pushbacks.html",
        {
            "session": session,
            "user": session.user,
            "items": items,
            "meter": pushbacks.drift_meter(),
            "displacements": pushbacks.displacements(),
            "awaiting_evidence": pushbacks.awaiting_evidence(),
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
    """The pushback panel on its own, for htmx to poll while a classification
    is in flight.

    The fragment carries its own polling trigger only while something is still
    `awaiting_classification`, so the poll stops by virtue of what came back --
    the same shape as the extraction and scoring panels.
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

    Imported by `jfl_web.routes.applications` rather than duplicated: the score
    panel and this one are read together, and two different ideas of what a
    pushback did on one screen would be worse than a shared import.
    """
    from jfl_web.pushbacks import receipt as build_receipt

    rows = pushbacks.for_application(detail.application.id)  # type: ignore[attr-defined]
    axis_options = {
        "want": dimension_options(score, "want") if score else [],
        "get": dimension_options(score, "get") if score else [],
    }
    views = []
    for row in rows:
        options = axis_options.get(row.axis, [])
        label = dimension_label(row.dimension, options)
        views.append(
            {
                "pushback": row,
                "label": label,
                "receipt": build_receipt(row, label=label) if row.status == "applied" else None,
            }
        )
    return {
        "pushbacks": views,
        "pushback_polling": any(
            row.status == "awaiting_classification" and row.error_code is None for row in rows
        ),
        "want_dimensions": axis_options["want"],
        "get_dimensions": axis_options["get"],
        "drift_meter": pushbacks.drift_meter(),
        "sent": was_sent(detail),  # type: ignore[arg-type]
        # The corrections list folds behind its count once every one of them
        # has been applied or set aside. The drift meter above it never folds:
        # it is the one number no other measure in this product can catch.
        "pushbacks_section": build_pushbacks_section(states or {}, views),
    }


def _error(request: Request, session: AuthenticatedSession, message: str, status: int) -> Response:
    return render(
        request,
        "error.html",
        {"session": session, "user": session.user, "message": message},
        status_code=status,
    )
