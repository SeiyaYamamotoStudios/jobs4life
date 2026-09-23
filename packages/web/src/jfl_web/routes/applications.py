"""The application tracker -- slice A5/A6, the reason this whole hosted app
exists. See CLAUDE.md's 2026-09-07 decision log and `PLAN.md`'s slice A: the
owner's own words for the gap conversations cannot close are "a clear list of
all the applications I have going."

**No model call anywhere in this file, and that is a hard rule rather than a
description.** Slice B3 replaced the six-field add form with a paste box, and
reading the pasted ad is a ~30-second Anthropic call on the user's own key --
so the POST creates the row, enqueues an `extract_job_ad` task and redirects,
and the work happens in the worker. Fast input, slow processing.

Screens:

  GET  /applications                 -- the list, most recently updated first,
                                         optionally filtered by `?status=`
  GET  /applications/new             -- the paste box
  POST /applications                 -- create, enqueue the read, redirect
  GET  /applications/{id}            -- one application: fields, full timeline,
                                         a status control, editable notes
  POST /applications/{id}/status     -- change status; appends an event
  POST /applications/{id}/notes      -- replace the notes field
  GET  /applications/{id}/extraction -- the extraction panel, for htmx polling
  GET  /applications/{id}/score      -- the scoring panel, for htmx polling
  POST /applications/{id}/score      -- score it: two axes, never composited;
                                         explicit, because it costs the user
  POST /applications/{id}/extract    -- read the ad again; explicit, never
                                         automatic, because it costs the user
  POST /applications/{id}/archive    -- soft delete: off the lists, status and
                                         timeline untouched, reversible
  POST /applications/{id}/unarchive  -- restore it

`POST /applications/{id}/status` answers two different callers with one route
rather than two: the **list** screen calls it over htmx (`HX-Request` header
present) and gets back just the updated `<tr>`, satisfying "the list updates
without a full page reload"; the **detail** page's status form is a plain
submit and gets a redirect back to itself, which is exactly the reload that
screen already does for every other change on it. One handler, one truth
about what a status change does, two response shapes.
"""

from __future__ import annotations

import uuid
from typing import Annotated, get_args

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import (
    ApplicationDetail,
    ApplicationExtraction,
    ApplicationScore,
    ApplicationStatus,
)
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.applications import ApplicationNotFoundError
from jfl_core.storage.ui_sections import SectionState

from jfl_web.applicationanswers import question_views
from jfl_web.deps import (
    ApplicationQuestionRepoDep,
    ApplicationRepoDep,
    CsrfDep,
    JobRepoDep,
    PushbackRepoDep,
    ScoreOverrideRepoDep,
    ScoreRepoDep,
    SectionRepoDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.jobads import (
    MAX_AD_CHARS,
    extraction_failure,
    normalise_url,
    provisional_title,
)
from jfl_web.pushbacks import (
    ADJUSTED_NOTE,
    ASSERTED_POINTS_CHOICES,
    ASSERTED_POINTS_LABEL,
    CLASSIFY_FAILED,
    CLASSIFY_HEADING,
    CLASSIFY_INTRO,
    COMPARISON_HEADING,
    COMPARISON_INTRO,
    DIRECTION_CHOICES,
    DIRECTION_LABEL,
    DRIFT_METER_HEADING,
    DRIFT_METER_NOTE,
    EVIDENCE_HEADING,
    EVIDENCE_NOTE,
    EVIDENCE_RECORDED,
    KIND_CONSEQUENCE,
    KIND_WORDING,
    NEVER_CHANGED,
    NEVER_CHANGED_HEADING,
    NEW_INFORMATION_LABEL,
    NEW_INFORMATION_NOTE,
    OVERRIDE_HEADING,
    OVERRIDE_LABEL,
    OVERRIDE_NOTE,
    PUSHBACK_HEADING,
    PUSHBACK_INTRO,
    RECEIPT_CHANGED,
    RECEIPT_UNCHANGED,
    RECEIPT_WOULD,
    SENT_NOTE,
    axis_displays,
    drift_meter_sentence,
)
from jfl_web.routes.drafts import cv_panel_context
from jfl_web.routes.pushbacks import pushback_context
from jfl_web.scores import (
    COST_NOTE,
    COULD_GET_LABEL,
    NO_WANT_IT_SCORE,
    SILENCE_NOTE,
    STANCE_WORDING,
    UNMEASURED,
    VERDICT_WORDING,
    WANT_IT_LABEL,
    WANT_IT_SUBTITLE,
    score_failure,
    want_it_summary,
)
from jfl_web.sections import (
    ad_section,
    notes_section,
    questions_section,
    score_detail_sections,
    score_section,
    status_section,
    timeline_section,
)
from jfl_web.templating import render

# The worker's kind for "read this pasted ad". A string on both sides, on
# purpose: importing `jfl_worker` here would make the web container carry the
# worker, and the queue's whole point is that the two deploy separately. A kind
# no deployed worker knows stays `pending` rather than failing, which is the
# safe direction for a rolling deploy.
EXTRACT_JOB_AD_KIND = "extract_job_ad"

# The worker's kind for "score this application". A string on both sides, for
# the same reason as above.
SCORE_APPLICATION_KIND = "score_application"

router = APIRouter()

STATUSES: tuple[ApplicationStatus, ...] = get_args(ApplicationStatus)

# The happy path, in order, for the "next step" quick action. A dropdown plus a
# submit is the wrong control for the common case -- almost every status change
# is a move one step along this line, and it should take one click.
#
# `rejected` and `withdrawn` are deliberately not on it: they can follow any
# active state rather than a particular one, so `rejected` is offered as its own
# secondary action and `withdrawn` stays in the full picker, which remains for
# jumps, reversals and corrections.
_PIPELINE: tuple[ApplicationStatus, ...] = (
    "interested",
    "applied",
    "screening",
    "interviewing",
    "offer",
)

_NEXT_LABEL: dict[str, str] = {
    "applied": "Mark as applied",
    "screening": "Move to screening",
    "interviewing": "Move to interviewing",
    "offer": "Record an offer",
}


def next_status(current: ApplicationStatus) -> ApplicationStatus | None:
    """The state one step along, or None at the end of the line.

    Returns None for `rejected` and `withdrawn` too: they are outcomes, not
    stages, so there is nothing to advance to.
    """
    try:
        index = _PIPELINE.index(current)
    except ValueError:
        return None
    return _PIPELINE[index + 1] if index + 1 < len(_PIPELINE) else None


# Deliberately the same message whether the id never existed or belongs to
# another user -- distinguishing the two would tell a caller which ids are
# real, which is a tenancy leak in miniature.
_NOT_FOUND = "No application found -- it may belong to another account."


@router.get("/applications")
def list_applications(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
) -> Response:
    # `archived=1` swaps the whole screen for the archived list rather than
    # combining with the status filter -- the archived list is not another
    # slice of the live one, it is a different question ("what did I put
    # away") from the default's ("what is live").
    show_archived = request.query_params.get("archived") == "1"
    raw_status = request.query_params.get("status")
    status = raw_status if raw_status in STATUSES else None
    items = applications.list_applications(
        status=None if show_archived else status, archived=show_archived
    )
    # Always known, even on the live list, so the "Archived (N)" link can
    # decide whether to render itself without a second round trip.
    archived_count = len(applications.list_applications(archived=True))
    return render(
        request,
        "applications_list.html",
        {
            "session": session,
            "user": session.user,
            "applications": items,
            "statuses": STATUSES,
            "active_status": status,
            "show_archived": show_archived,
            "archived_count": archived_count,
            "just_archived": _just_archived_title(applications, request),
        },
    )


def _just_archived_title(applications: ApplicationRepoDep, request: Request) -> str | None:
    """The title for the "Archived ..." confirmation, looked up -- never echoed.

    The redirect carries the application's id, not its title. Rendering text taken
    straight from the query string would let anyone craft a link that puts words of
    their choosing on this page, which is why the login flow already returns a fixed
    error code instead of a message. The id is resolved through the signed-in user's
    own repository, so another user's id, a stale id or a garbage value simply shows
    no banner.
    """
    raw = request.query_params.get("just_archived")
    if not raw:
        return None
    try:
        application_id = uuid.UUID(raw)
    except ValueError:
        return None
    detail = applications.get_application(application_id)
    if detail is None or detail.application.archived_at is None:
        return None
    return detail.application.title


@router.get("/applications/new")
def new_application_form(request: Request, session: SessionDep) -> Response:
    return render(
        request,
        "application_form.html",
        {"session": session, "user": session.user},
    )


@router.post("/applications")
def create_application(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    job_ad: Annotated[str, Form()],
    url: Annotated[str, Form()] = "",
) -> Response:
    """Two fields, one required, and no model call.

    Everything that used to be typed here -- title, employer, source -- is in
    the ad, and asking someone to retype it is the friction that gets a tool
    abandoned. So the ad goes in verbatim, the row gets a provisional title, and
    an `extract_job_ad` task reads it properly in the background.

    Both writes are in the request's single transaction (`deps.db_conn`), so the
    application and its task are committed together: there is no state where a
    row sits `pending` with nothing queued to move it, or a task names an
    application that was rolled back.
    """
    ad = job_ad.strip()
    if not ad or len(ad) > MAX_AD_CHARS:
        message = (
            "Paste the job ad to add an application."
            if not ad
            else "That is much longer than a job ad -- paste just the role and its requirements."
        )
        return _form(request, session, error=message, job_ad=job_ad, url=url)

    try:
        link = normalise_url(url)
    except ValueError as exc:
        return _form(request, session, error=str(exc), job_ad=job_ad, url=url)

    application = applications.create_application(
        # A placeholder, and the row says so: extraction may replace a
        # provisional title, and may never replace a typed one.
        title=provisional_title(ad),
        url=link,
        raw_job_text=ad,
        title_is_provisional=True,
        extraction_status="pending",
    )
    tasks.enqueue(
        kind=EXTRACT_JOB_AD_KIND,
        # Ids only. The ad text is already stored once in `jobs.raw_text` and
        # the handler reads it from there under its own tenancy scope; a second
        # copy in a payload that admin queries read back buys nothing. The API
        # key is never here at all -- it is unsealed in the worker.
        payload={"application_id": str(application.id)},
    )
    # POST/redirect/GET: a refresh must not resubmit the form.
    return RedirectResponse(f"/applications/{application.id}", status_code=303)


def _form(
    request: Request, session: AuthenticatedSession, *, error: str, job_ad: str, url: str
) -> Response:
    """Re-render the paste box with what was typed still in it. Losing a pasted
    ad to a validation error is the kind of small insult that stops a tool being
    used.
    """
    return render(
        request,
        "application_form.html",
        {
            "session": session,
            "user": session.user,
            "error": error,
            "values": {"job_ad": job_ad, "url": url},
        },
        status_code=400,
    )


def _detail_context(
    session: AuthenticatedSession,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    application_id: uuid.UUID,
    detail: ApplicationDetail,
    *,
    questions: ApplicationQuestionRepoDep,
    pushbacks: PushbackRepoDep,
    overrides: ScoreOverrideRepoDep,
    ui_sections: SectionRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
    **extra: object,
) -> dict[str, object]:
    """Everything the detail page needs, in one place -- shared with
    `attach_ad`, which re-renders this same page on a validation error rather
    than duplicating its context by hand.

    `ui_sections.states()` is one query for the whole page: which panels this
    user has folded away, read once and looked up per section. Nothing is
    written here -- a render never writes -- so this stays a read the page was
    going to make anyway.
    """
    extraction = applications.get_extraction(application_id)
    states = ui_sections.states()
    views = question_views(questions, application_id)
    return {
        "session": session,
        "user": session.user,
        "application": detail.application,
        "events": detail.events,
        "statuses": STATUSES,
        "next_status": next_status(detail.application.status),
        "next_labels": _NEXT_LABEL,
        "application_id": application_id,
        # NEXT.md's task 4: "check my answer" / "draft one for me", side by
        # side -- see jfl_web.applicationanswers and jfl_web.routes.application_questions.
        "question_views": views,
        # The collapsible sections this page owns directly. The ad, the score
        # and the corrections panel build their own, inside the context helpers
        # they share with their polling routes -- see `docs/ui-sections.md`.
        "questions_section": questions_section(states, views),
        "status_section": status_section(states),
        "notes_section": notes_section(states, detail.application.notes),
        "timeline_section": timeline_section(states, detail.events),
        **_extraction_context(extraction, states),
        # "CV for this job": the steps and the one button, on this page.
        **cv_panel_context(session, applications, jobs, tasks, states, application_id, detail),
        **_score_context(
            scores.latest(application_id),
            detail=detail,
            pushbacks=pushbacks,
            overrides=overrides,
            states=states,
        ),
        **extra,
    }


@router.get("/applications/{application_id}")
def application_detail(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    questions: ApplicationQuestionRepoDep,
    pushbacks: PushbackRepoDep,
    overrides: ScoreOverrideRepoDep,
    ui_sections: SectionRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
) -> Response:
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return render(
        request,
        "application_detail.html",
        _detail_context(
            session,
            applications,
            scores,
            application_id,
            detail,
            questions=questions,
            pushbacks=pushbacks,
            overrides=overrides,
            ui_sections=ui_sections,
            jobs=jobs,
            tasks=tasks,
        ),
    )


@router.get("/applications/{application_id}/extraction")
def extraction_panel(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    ui_sections: SectionRepoDep,
) -> Response:
    """The extraction panel on its own, for htmx to poll while it is pending.

    The fragment carries its own polling trigger only while `status` is
    `pending`, so the poll stops by virtue of what came back rather than by
    anything having to cancel it -- there is no timer left running against a
    finished job, and no client-side state to get out of step with the row.
    """
    extraction = applications.get_extraction(application_id)
    if extraction is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return render(
        request,
        "_extraction.html",
        {
            "session": session,
            "application_id": application_id,
            **_extraction_context(extraction, ui_sections.states()),
        },
    )


@router.post("/applications/{application_id}/extract")
def extract_again(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Read the ad again. A button, never automatic.

    Extraction is a model call on the user's own key, so a re-run spends their
    money: it happens because a person asked, not because a page was refreshed
    or a task was redelivered. `request_extraction` returns False when there is
    no ad stored, and then nothing is enqueued -- a task that could only fail is
    not worth queueing.
    """
    if applications.get_application(application_id) is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    if applications.request_extraction(application_id):
        tasks.enqueue(kind=EXTRACT_JOB_AD_KIND, payload={"application_id": str(application_id)})
    return RedirectResponse(f"/applications/{application_id}", status_code=303)


@router.post("/applications/{application_id}/ad")
def attach_ad(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    questions: ApplicationQuestionRepoDep,
    pushbacks: PushbackRepoDep,
    overrides: ScoreOverrideRepoDep,
    ui_sections: SectionRepoDep,
    tasks: TaskRepoDep,
    jobs: JobRepoDep,
    _csrf: CsrfDep,
    job_ad: Annotated[str, Form()],
) -> Response:
    """The paste box offered when "Track as application" (slice C7) could not
    read a description off the board -- the application exists, with no ad
    text and `extraction_error_code == "description_unavailable"`, and this is
    how a person finishes it by hand.

    Same paste box as `create_application`, reached through a different door,
    so the same length validation applies. `attach_job_ad` is what
    `request_extraction` cannot be here: that method requires a `job_id`
    already on the row, which is exactly what this application does not have
    yet.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    ad = job_ad.strip()
    if not ad or len(ad) > MAX_AD_CHARS:
        message = (
            "Paste the job ad to attach it."
            if not ad
            else "That is much longer than a job ad -- paste just the role and its requirements."
        )
        return render(
            request,
            "application_detail.html",
            _detail_context(
                session,
                applications,
                scores,
                application_id,
                detail,
                questions=questions,
                pushbacks=pushbacks,
                overrides=overrides,
                ui_sections=ui_sections,
                jobs=jobs,
                tasks=tasks,
                ad_error=message,
                ad_value=job_ad,
            ),
            status_code=400,
        )

    applications.attach_job_ad(application_id, ad)
    tasks.enqueue(kind=EXTRACT_JOB_AD_KIND, payload={"application_id": str(application_id)})
    return RedirectResponse(f"/applications/{application_id}", status_code=303)


@router.get("/applications/{application_id}/score")
def score_panel(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    pushbacks: PushbackRepoDep,
    overrides: ScoreOverrideRepoDep,
    ui_sections: SectionRepoDep,
) -> Response:
    """The scoring panel on its own, for htmx to poll while a run is pending.

    The fragment carries its own polling trigger only while the latest run is
    `pending`, so the poll stops by virtue of what came back -- same shape as
    the extraction panel, and for the same reason: no timer left running, no
    client-side state to get out of step with the row.

    The application is looked up first so an id belonging to someone else is a
    404 here exactly as it is on the page, rather than an empty panel.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return render(
        request,
        "_score.html",
        {
            "session": session,
            "application_id": application_id,
            **_score_context(
                scores.latest(application_id),
                detail=detail,
                pushbacks=pushbacks,
                overrides=overrides,
                states=ui_sections.states(),
            ),
        },
    )


@router.post("/applications/{application_id}/score")
def score_application(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Score this application. A button, never automatic.

    CLAUDE.md's 2026-09-15 decision: a job is scored only when the user turns
    it into an application and asks, never on arrival and never for a job they
    merely browsed -- they pay for the call with their own key. So this row and
    this task exist because a person pressed something.

    A run already in flight is not duplicated: pressing twice while the panel
    says "scoring" would buy a second charge for the same answer. A finished
    run *is* re-scored, because that is what the button is for, and the earlier
    row is kept rather than overwritten.

    Both writes are in the request's single transaction, so the row and its
    task are committed together: there is no state where a `pending` score sits
    with nothing queued to move it.
    """
    if applications.get_application(application_id) is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    latest = scores.latest(application_id)
    if latest is None or latest.status != "pending":
        row = scores.create_pending(application_id)
        tasks.enqueue(
            kind=SCORE_APPLICATION_KIND,
            # Ids only. Nothing about the job, the profile or the key is in a
            # payload that admin queries read back.
            payload={"score_id": str(row.id)},
        )
    # POST/redirect/GET: a refresh must not enqueue a second run.
    return RedirectResponse(f"/applications/{application_id}", status_code=303)


def _score_context(
    score: ApplicationScore | None,
    *,
    detail: ApplicationDetail,
    pushbacks: PushbackRepoDep,
    overrides: ScoreOverrideRepoDep,
    states: dict[str, SectionState],
) -> dict[str, object]:
    """One shape for both the full page and the polled fragment, so the panel
    cannot render differently depending on which route produced it.

    The two axes are handed over as two separate values with two separate
    labels, and nothing here derives a third from them.

    A **third** value goes with each of them, and also never merges into it:
    what the user's own corrections have moved that number to. The stored run
    keeps the number the pipeline produced -- corrections are a labelled layer
    over it, read from the append-only pushback log, so the panel can always
    say both what the tool said and what you moved it to.
    """
    failed = score is not None and score.status == "failed"
    failure = score_failure(score.error_code) if score is not None and failed else None
    live_overrides = overrides.current(detail.application.id)
    panel = pushback_context(detail, score, pushbacks, states)
    displays = axis_displays(
        score,
        pushbacks.displacements(),
        live_overrides,
        sent=bool(panel["sent"]),
    )
    return {
        **panel,
        # The panel itself, and the four long lists inside it. The verdicts stay
        # open by default: the silences are the product, and folding them away
        # would hide what the panel is for.
        "score_section": score_section(states, score),
        "score_sections": score_detail_sections(states, score),
        "want_display": displays.get("want"),
        "could_get_display": displays.get("get"),
        "overrides": live_overrides,
        "drift_meter_heading": DRIFT_METER_HEADING,
        "drift_meter_note": DRIFT_METER_NOTE,
        "drift_meter_sentence": drift_meter_sentence(panel["drift_meter"]),  # type: ignore[arg-type]
        "pushback_heading": PUSHBACK_HEADING,
        "pushback_intro": PUSHBACK_INTRO,
        "asserted_points_label": ASSERTED_POINTS_LABEL,
        "asserted_points_choices": ASSERTED_POINTS_CHOICES,
        "direction_label": DIRECTION_LABEL,
        "direction_choices": DIRECTION_CHOICES,
        "classify_heading": CLASSIFY_HEADING,
        "classify_intro": CLASSIFY_INTRO,
        "classify_failed": CLASSIFY_FAILED,
        "kind_wording": KIND_WORDING,
        "kind_consequence": KIND_CONSEQUENCE,
        "new_information_label": NEW_INFORMATION_LABEL,
        "new_information_note": NEW_INFORMATION_NOTE,
        "receipt_changed": RECEIPT_CHANGED,
        "receipt_unchanged": RECEIPT_UNCHANGED,
        "receipt_would": RECEIPT_WOULD,
        "never_changed": NEVER_CHANGED,
        "never_changed_heading": NEVER_CHANGED_HEADING,
        "comparison_heading": COMPARISON_HEADING,
        "comparison_intro": COMPARISON_INTRO,
        "override_heading": OVERRIDE_HEADING,
        "override_note": OVERRIDE_NOTE,
        "override_label": OVERRIDE_LABEL,
        "evidence_heading": EVIDENCE_HEADING,
        "evidence_note": EVIDENCE_NOTE,
        "evidence_recorded": EVIDENCE_RECORDED,
        "adjusted_note": ADJUSTED_NOTE,
        "sent_note": SENT_NOTE,
        "score": score,
        "score_failure": failure,
        "score_unmeasured": UNMEASURED,
        "score_cost_note": COST_NOTE,
        "could_get_label": COULD_GET_LABEL,
        "want_it_label": WANT_IT_LABEL,
        "want_it_subtitle": WANT_IT_SUBTITLE,
        # Recomputed from the stored verdicts, so the tally under the number
        # can never disagree with the verdicts listed under it.
        "want_it_summary": want_it_summary(score) if score is not None else "",
        "no_want_it_score": NO_WANT_IT_SCORE,
        "verdict_wording": VERDICT_WORDING,
        "stance_wording": STANCE_WORDING,
        "silence_note": SILENCE_NOTE,
    }


def _extraction_context(
    extraction: ApplicationExtraction | None, states: dict[str, SectionState]
) -> dict[str, object]:
    """One shape for both the full page and the polled fragment, so the panel
    cannot render differently depending on which route produced it -- including
    whether it is folded, which is why the section is built here rather than
    once on the page and once again in the poll.
    """
    section = ad_section(states, extraction)
    if extraction is None:
        return {"extraction": None, "failure": None, "ad_section": section}
    failure = extraction_failure(extraction.error_code) if extraction.status == "failed" else None
    return {"extraction": extraction, "failure": failure, "ad_section": section}


@router.post("/applications/{application_id}/status")
def change_status(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    _csrf: CsrfDep,
    to_status: Annotated[str, Form()],
    note: Annotated[str, Form()] = "",
) -> Response:
    if to_status not in STATUSES:
        context = {"session": session, "user": session.user, "message": "Unknown status."}
        return render(request, "error.html", context, status_code=400)

    try:
        # `to_status not in STATUSES` above already narrows this to
        # ApplicationStatus for mypy -- no cast needed.
        application = applications.change_status(
            application_id, to_status=to_status, note=note.strip() or None
        )
    except ApplicationNotFoundError:
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _NOT_FOUND},
            status_code=404,
        )

    if request.headers.get("HX-Request") == "true":
        return render(
            request,
            "_application_row.html",
            {
                "session": session,
                "application": application,
                "statuses": STATUSES,
                "next_status": next_status(application.status),
                "next_labels": _NEXT_LABEL,
            },
        )
    return RedirectResponse(f"/applications/{application_id}", status_code=303)


@router.post("/applications/{application_id}/notes")
def update_notes(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    _csrf: CsrfDep,
    notes: Annotated[str, Form()] = "",
) -> Response:
    try:
        applications.update_notes(application_id, notes.strip() or None)
    except ApplicationNotFoundError:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return RedirectResponse(f"/applications/{application_id}", status_code=303)


@router.post("/applications/{application_id}/archive")
def archive_application(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Off the lists, nothing else touched.

    Status and timeline are exactly what they were -- `archive` sets only
    `archived_at`. This is the fix for an application that never happened
    rather than an honest one: see migration `e5396ef31c67`. Redirects to
    the live list, not back to the now-hidden detail page, since that is
    where the owner is once the row is out of play.
    """
    try:
        application = applications.archive(application_id)
    except ApplicationNotFoundError:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return RedirectResponse(f"/applications?just_archived={application.id}", status_code=303)


@router.post("/applications/{application_id}/unarchive")
def unarchive_application(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    _csrf: CsrfDep,
) -> Response:
    try:
        applications.unarchive(application_id)
    except ApplicationNotFoundError:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return RedirectResponse(f"/applications/{application_id}", status_code=303)
