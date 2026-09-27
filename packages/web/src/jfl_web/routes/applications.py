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
                                         optionally filtered by `?status=` and
                                         sorted by one column via `?sort=` and
                                         `?dir=` (the column headers); both
                                         scores per row, never combined, and an
                                         "Archive..." confirm per row
  GET  /applications/new             -- the paste box
  POST /applications                 -- create, enqueue the read, redirect
  GET  /applications/{id}            -- one application: fields, full timeline,
                                         a status control, editable notes
  POST /applications/{id}/status     -- change status; appends an event
  POST /applications/{id}/notes      -- replace the notes field
  GET  /applications/{id}/extraction -- the extraction panel, for htmx polling
  GET  /applications/{id}/score      -- the scoring panel, for htmx polling
  POST /applications/{id}/score      -- Re-score: two axes, never composited.
                                         The first score is chained from the
                                         read of the ad; this is the button
  POST /applications/{id}/extract    -- read the ad again; explicit, never
                                         automatic, because it costs the user
  POST /applications/{id}/archive    -- soft delete: off the lists, status and
                                         timeline untouched, reversible
  POST /applications/{id}/unarchive  -- restore it
  POST /applications/bulk-archive    -- archive every checked row at once
  POST /applications/bulk-unarchive  -- the undo for the above, and what the
                                         Archived view's own bulk restore uses
  POST /applications/archive-by-rule/preview  -- "this will archive N" -- reads
                                         only, never writes
  POST /applications/archive-by-rule/confirm  -- re-runs the same rule and
                                         archives what it matches
  POST /applications/rescore/preview          -- "this will re-score N,
                                         about $X" -- reads only
  POST /applications/rescore/confirm          -- re-scores exactly the
                                         eligible subset of the submission
  POST /applications/retry-failed/preview     -- "this will retry N,
                                         about $X" -- reads only
  POST /applications/retry-failed/confirm     -- retries every currently
                                         failed read or score on the live list

Bulk archiving (owner feedback, 2026-09-27) adds a second way onto the same
`archive_many`/`unarchive_many` repository methods the per-row buttons already
use: pick rows by hand, or describe them with a rule ("scored below 6",
"rejected", "no activity in 30 days" -- see `jfl_web.archive_rules`). Both
paths end at the same place, so both get the same undo: the redirect carries
the ids that actually changed, resolved back through this user's own
repository rather than trusted from the request, and the banner's "Undo"
button is `bulk-unarchive` with exactly that set of ids as hidden fields.
Rule matching never trusts an id list either -- `archive-by-rule/confirm`
re-parses the same rule fields the preview showed and recomputes the match at
write time, so there is nothing to tamper with between "this will archive"
and the button that does it.

Bulk re-score and "retry everything that failed" (owner feedback, same day)
follow the identical preview-then-confirm shape, over `jfl_web.bulk_actions`'
pure selection logic and the `_enqueue_rescore`/`_enqueue_read_retry` helpers
this module shares with the single-application buttons -- a bulk press does
exactly what N presses of the existing button would do, never a shortcut that
skips a check the single button makes. Neither ever re-reads an ad that was
already read successfully: the only re-read path here is retrying a *failed*
one. Both show a cost estimate measured from the user's own `runs` history
(`jfl_web.bulk_actions.estimate_for_count`/`estimate_retry_cost`), labelled as
an estimate, with a stated fallback range when there is no history yet.

`POST /applications/{id}/status` answers two different callers with one route
rather than two: the **list** screen calls it over htmx (`HX-Request` header
present) and gets back just the updated `<tr>`, satisfying "the list updates
without a full page reload"; the **detail** page's status form is a plain
submit and gets a redirect back to itself, which is exactly the reload that
screen already does for every other change on it. One handler, one truth
about what a status change does, two response shapes.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from typing import Annotated, get_args
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import (
    Application,
    ApplicationDetail,
    ApplicationExtraction,
    ApplicationScore,
    ApplicationStatus,
    TableKey,
)
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.applications import ApplicationNotFoundError
from jfl_core.storage.credentials import ANTHROPIC_API_KEY
from jfl_core.storage.ui_sections import SectionState

from jfl_web.applicationanswers import question_views
from jfl_web.archive_rules import (
    AXIS_LABELS,
    RULE_ERROR_MESSAGES,
    SCORE_AXES,
    TERMINAL_STATUSES,
    ArchiveRules,
    matching_applications,
    parse_rule_form,
)
from jfl_web.bulk_actions import (
    SKIP_REASON_LABELS,
    CostEstimate,
    RescoreSkip,
    RetryPlan,
    estimate_for_count,
    estimate_retry_cost,
    partition_rescore,
    plan_retry_failed,
)
from jfl_web.deps import (
    ApplicationQuestionRepoDep,
    ApplicationRepoDep,
    CredentialRepoDep,
    CsrfDep,
    JobRepoDep,
    PushbackRepoDep,
    RunRepoDep,
    ScoreOverrideRepoDep,
    ScoreRepoDep,
    SectionRepoDep,
    SessionDep,
    TableSortRepoDep,
    TaskRepoDep,
)
from jfl_web.jobads import (
    EXTRACTION_RETRYING_NOTE,
    FETCH_RETRYING_NOTE,
    MAX_AD_CHARS,
    extraction_failure,
    normalise_url,
    provisional_title,
)
from jfl_web.pushbacks import (
    ADJUSTED_NOTE,
    OVERRIDE_HEADING,
    OVERRIDE_LABEL,
    OVERRIDE_NOTE,
    SENT_NOTE,
    axis_displays,
)
from jfl_web.routes.drafts import cv_panel_context
from jfl_web.routes.pushbacks import pushback_context
from jfl_web.scores import (
    ADD_COST_NOTE,
    AWAITS_AD_NOTE,
    COST_NOTE,
    COULD_GET_LABEL,
    NO_KEY_ADD_NOTE,
    NO_WANT_IT_SCORE,
    RETRYING_NOTE,
    SILENCE_NOTE,
    STANCE_WORDING,
    UNMEASURED,
    VERDICT_WORDING,
    WANT_IT_LABEL,
    WANT_IT_SUBTITLE,
    RowScore,
    ScoreFailure,
    SortDirection,
    SortHeader,
    SortKey,
    parse_sort,
    row_score,
    score_failure,
    sort_applications,
    sort_headers,
    sort_query,
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

# `runs.stage` values a score's button press can write -- `jfl_generate.scoring`
# and `jfl_generate.coverage` respectively (a score runs coverage first only
# when none is recorded yet). Grouped by trace in `recent_costs`, this is what
# `application_scores.cost_usd` already totals for one run; used here only to
# estimate a *bulk* re-score before it is pressed. See `jfl_web.bulk_actions`.
_SCORE_COST_STAGES = ("score", "coverage")
# `jfl_generate.extract`'s stage -- what one read of a job ad costs.
_READ_COST_STAGES = ("extract_requirements",)
# How many recent traces the cost estimate samples from. Not the whole
# history: a modest, recent sample is enough for a rough estimate, and an old,
# no-longer-representative run should not count as much as a recent one.
_COST_SAMPLE_SIZE = 20

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

# The table key this list's sort is saved under -- see
# `jfl_core.storage.ui_table_sorts` and CLAUDE.md's owner feedback that
# sorting "needs to be persistent on any tables".
_SORT_TABLE_KEY: TableKey = "applications"


def _resolved_sort(
    request: Request, table_sorts: TableSortRepoDep
) -> tuple[SortKey, SortDirection]:
    """The sort in effect for this render.

    A header click puts `?sort=` on the URL, so its presence (not its value)
    is what triggers a save -- `parse_sort` still normalises whatever value
    arrived. A plain visit (`?sort=` absent) reads back what was last saved; no
    saved row, or one naming a column this list no longer has, falls back to
    the default silently. See `jfl_web.sorting.resolve_sort`, which this
    mirrors for `jfl_web.scores`'s own (differently shaped) sort helpers --
    kept separate because `packages/web/tests/test_list_scores.py` pins their
    exact public signatures.
    """
    raw_sort = request.query_params.get("sort")
    if raw_sort is not None:
        sort, direction = parse_sort(raw_sort, request.query_params.get("dir"))
        table_sorts.save_sort(_SORT_TABLE_KEY, sort, direction)
        return sort, direction
    saved = table_sorts.get_sort(_SORT_TABLE_KEY)
    if saved is None:
        return parse_sort(None, None)
    return parse_sort(saved.sort_key, saved.direction)


def _persistent_sort_headers(
    sort: SortKey, direction: SortDirection, *, status: str | None
) -> dict[SortKey, SortHeader]:
    """`jfl_web.scores.sort_headers`, with every href made explicit.

    That function omits `?sort=` for the one link that lands on the table's
    global default, to keep that single URL clean -- fine when sort is
    stateless. Once sort is persisted, that omission makes the click
    indistinguishable from a plain visit: `_resolved_sort` would read back the
    previously *saved* order instead of the column just clicked, so clicking
    "Updated" from any other sort would silently do nothing. So every header
    this list renders carries its sort explicitly, and only the `href` field
    is rewritten -- `active`, `direction` and `aria_sort` come straight from
    the wrapped call.
    """
    headers = sort_headers(sort, direction, status=status)
    rewritten: dict[SortKey, SortHeader] = {}
    for key, header in headers.items():
        params: dict[str, str] = {"status": status} if status else {}
        params["sort"] = header.key
        params["dir"] = header.next_direction
        rewritten[key] = dataclasses.replace(header, href=f"/applications?{urlencode(params)}")
    return rewritten


@router.get("/applications")
def list_applications(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    table_sorts: TableSortRepoDep,
) -> Response:
    # `archived=1` swaps the whole screen for the archived list rather than
    # combining with the status filter -- the archived list is not another
    # slice of the live one, it is a different question ("what did I put
    # away") from the default's ("what is live").
    show_archived = request.query_params.get("archived") == "1"
    raw_status = request.query_params.get("status")
    status = raw_status if raw_status in STATUSES else None
    sort, direction = _resolved_sort(request, table_sorts)
    items = applications.list_applications(
        status=None if show_archived else status, archived=show_archived
    )
    # Both axes per row, from one query. Two numbers, two cells, and the only
    # orderings on offer are by one axis or by neither -- see `jfl_web.scores`.
    row_scores = _row_scores(scores, [a.id for a in items])
    if not show_archived:
        items = sort_applications(items, row_scores, sort, direction)
    # Always known, even on the live list, so the "Archived (N)" link can
    # decide whether to render itself without a second round trip.
    archived_count = len(applications.list_applications(archived=True))
    bulk_archived = _bulk_archived_context(applications, request)
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
            "retry_failed_count": _retry_failed_count(
                applications,
                scores,
                show_archived=show_archived,
                status=status,
                items=items,
                row_scores=row_scores,
            ),
            "just_archived": _just_archived_title(applications, request),
            "queued_rescores": _queued_count(request, "queued_rescores"),
            "queued_retries": _queued_count(request, "queued_retries"),
            "row_scores": row_scores,
            # The column headers are the sort control; the status filter
            # carries the sort, and the headers carry the filter. Every link
            # is explicit about the sort it sets, so a click always overrides
            # whatever was last saved -- see `_persistent_sort_headers`.
            "sort_headers": _persistent_sort_headers(sort, direction, status=status),
            "status_links": _status_links(status, sort_query(sort, direction)),
            **bulk_archived,
            # The "Archive by rule..." form -- live view only, but harmless to
            # hand the archived view too since it never renders the fieldset.
            "score_axes": SCORE_AXES,
            "axis_labels": AXIS_LABELS,
            "terminal_statuses": TERMINAL_STATUSES,
            "rule_error": RULE_ERROR_MESSAGES.get(request.query_params.get("rule_error", "")),
        },
    )


def _status_links(active: str | None, sort_params: dict[str, str]) -> list[tuple[str, str, bool]]:
    """(label, href, active) for "All" and each status, keeping the sort."""
    links: list[tuple[str, str, bool]] = []
    for status in (None, *STATUSES):
        params = ({"status": status} if status else {}) | sort_params
        href = "/applications" + ("?" + urlencode(params) if params else "")
        links.append((status or "All", href, status == active))
    return links


def _row_scores(
    scores: ScoreRepoDep, application_ids: list[uuid.UUID]
) -> dict[uuid.UUID, RowScore]:
    pairs = scores.latest_for_applications(application_ids)
    return {app_id: row_score(*pairs.get(app_id, (None, None))) for app_id in application_ids}


def _retry_failed_count(
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    *,
    show_archived: bool,
    status: str | None,
    items: list[Application],
    row_scores: dict[uuid.UUID, RowScore],
) -> int:
    """How many live applications "Retry everything that failed" would touch --
    what decides whether the button shows on the list at all (owner feedback,
    2026-09-27, alongside the bulk-archive slice).

    Never computed for the archived view -- retrying is for what is still
    live. "Everything that failed" is also never scoped to the current
    `?status=` filter, so when one is active this re-reads the unfiltered live
    list; with no filter, `items`/`row_scores` already *are* that list, one
    query saved.
    """
    if show_archived:
        return 0
    if status is None:
        return plan_retry_failed(items, row_scores).count
    all_live = applications.list_applications(archived=False)
    all_row_scores = _row_scores(scores, [a.id for a in all_live])
    return plan_retry_failed(all_live, all_row_scores).count


def _queued_count(request: Request, param: str) -> int | None:
    """A non-negative integer straight off a redirect's own query string (never
    user-tampered in a way that matters -- it only ever names a count this
    same request cycle just wrote), or None if it is absent or malformed.
    """
    raw = request.query_params.get(param)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


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


# Shown next to the bulk toolbar when a bulk action could not do anything --
# a fixed set of codes, never a message built from the request, same
# discipline as `ExtractionErrorCode`/`ScoreErrorCode`.
BULK_ERROR_MESSAGES: dict[str, str] = {
    "none_selected": "Select at least one application first.",
}


def _parse_ids(raw: list[str] | None) -> list[uuid.UUID]:
    """Form values -> ids, dropping anything that is not a UUID. A tampered or
    stale checkbox value must not 500 the request -- it just does not match
    any application this user owns, which `list_by_ids`/`archive_many` already
    handle by silently ignoring it.
    """
    ids: list[uuid.UUID] = []
    for value in raw or []:
        try:
            ids.append(uuid.UUID(value))
        except ValueError:
            continue
    return ids


def _bulk_redirect(path: str, ids: list[uuid.UUID]) -> str:
    """`path` plus one `bulk_archived=<id>` per id -- what a bulk archive (by
    selection or by rule) redirects to, so the list page can look the ids back
    up and show "Archived N. Undo" for exactly the set that changed.
    """
    if not ids:
        return path
    return path + "?" + urlencode({"bulk_archived": [str(i) for i in ids]}, doseq=True)


def _bulk_archived_context(applications: ApplicationRepoDep, request: Request) -> dict[str, object]:
    """The "Archived N. Undo" banner's data, resolved through this user's own
    repository rather than trusted from the query string -- same "looked up,
    never echoed" discipline as `_just_archived_title`, extended to a list.
    Ids for another user, or ids that are no longer archived (raced with an
    unarchive elsewhere), are simply absent, so the count can never overstate
    what is really sitting in Archived right now.
    """
    raw_ids = _parse_ids(request.query_params.getlist("bulk_archived"))
    resolved = [a for a in applications.list_by_ids(raw_ids) if a.archived_at is not None]
    bulk_error_code = request.query_params.get("bulk_error")
    return {
        "bulk_archived_applications": resolved,
        "bulk_error": BULK_ERROR_MESSAGES.get(bulk_error_code) if bulk_error_code else None,
    }


@router.get("/applications/new")
def new_application_form(
    request: Request, session: SessionDep, credentials: CredentialRepoDep
) -> Response:
    return render(
        request,
        "application_form.html",
        {"session": session, "user": session.user, **_cost_context(credentials)},
    )


def _cost_context(credentials: CredentialRepoDep) -> dict[str, object]:
    """What adding sets off, said before the user presses anything: the calls
    it triggers on their key, or -- with no key stored -- that nothing will be
    read or scored, and why.
    """
    return {
        "has_api_key": credentials.summary(ANTHROPIC_API_KEY) is not None,
        "add_cost_note": ADD_COST_NOTE,
        "no_key_add_note": NO_KEY_ADD_NOTE,
    }


@router.post("/applications")
def create_application(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    tasks: TaskRepoDep,
    credentials: CredentialRepoDep,
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

    A successful read queues the first score by itself (see
    `jfl_worker.handlers.extraction._chain_first_score`), and the form said so
    before this was pressed. **With no API key stored nothing is queued at
    all**: the read could only fail, so the application is saved with its read
    marked `no_api_key` -- the panel then says why and links to Settings, and
    "Try reading it again" after adding a key is what starts read and score.
    """
    ad = job_ad.strip()
    if not ad or len(ad) > MAX_AD_CHARS:
        message = (
            "Paste the job ad to add an application."
            if not ad
            else "That is much longer than a job ad -- paste just the role and its requirements."
        )
        return _form(request, session, credentials, error=message, job_ad=job_ad, url=url)

    try:
        link = normalise_url(url)
    except ValueError as exc:
        return _form(request, session, credentials, error=str(exc), job_ad=job_ad, url=url)

    application = applications.create_application(
        # A placeholder, and the row says so: extraction may replace a
        # provisional title, and may never replace a typed one.
        title=provisional_title(ad),
        url=link,
        raw_job_text=ad,
        title_is_provisional=True,
        extraction_status="pending",
    )
    if credentials.summary(ANTHROPIC_API_KEY) is None:
        applications.fail_extraction(application.id, "no_api_key")
        return RedirectResponse(f"/applications/{application.id}", status_code=303)
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
    request: Request,
    session: AuthenticatedSession,
    credentials: CredentialRepoDep,
    *,
    error: str,
    job_ad: str,
    url: str,
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
            **_cost_context(credentials),
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


def _enqueue_read_retry(
    applications: ApplicationRepoDep, tasks: TaskRepoDep, application_id: uuid.UUID
) -> bool:
    """Re-read this application's ad, unless there is no ad to read. Shared by
    the single "Read it again" button and the bulk "Retry everything that
    failed" action below, so both press exactly the same button under the
    hood. Returns whether a task was actually enqueued -- `request_extraction`
    returns False when there is no ad stored (`description_unavailable`'s own
    case), and a task that could only fail is not worth queueing.
    """
    if not applications.request_extraction(application_id):
        return False
    tasks.enqueue(kind=EXTRACT_JOB_AD_KIND, payload={"application_id": str(application_id)})
    return True


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
    or a task was redelivered.
    """
    if applications.get_application(application_id) is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    _enqueue_read_retry(applications, tasks, application_id)
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


def _enqueue_rescore(
    scores: ScoreRepoDep, tasks: TaskRepoDep, application_id: uuid.UUID
) -> uuid.UUID | None:
    """Create a pending score and enqueue it, unless one is already in flight.
    Shared by the single Re-score button and the bulk "Re-score selected" and
    "Retry everything that failed" actions below, so all three press exactly
    the same button under the hood -- one place that decides what a score
    press does, rather than three copies that could drift.

    A run already in flight is not duplicated: pressing twice while the panel
    says "scoring" would buy a second charge for the same answer. A finished
    *or failed* run is re-scored, because that is what the button is for, and
    the earlier row is kept rather than overwritten. Returns the new score's
    id, or None if nothing was queued.
    """
    latest = scores.latest(application_id)
    if latest is not None and latest.status == "pending":
        return None
    row = scores.create_pending(application_id)
    tasks.enqueue(
        kind=SCORE_APPLICATION_KIND,
        # Ids only. Nothing about the job, the profile or the key is in a
        # payload that admin queries read back.
        payload={"score_id": str(row.id)},
    )
    return row.id


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
    """Re-score this application -- or score one that has no run yet.

    CLAUDE.md's 2026-09-15 decision: a job is scored only when the user turns
    it into an application, never on arrival and never for a job they merely
    browsed -- they pay for the call with their own key. The first score is
    chained from the read of the ad, because adding the application *is* that
    choice; every later one is this button, because a person pressed it.

    Both writes `_enqueue_rescore` makes are in the request's single
    transaction, so the row and its task are committed together: there is no
    state where a `pending` score sits with nothing queued to move it.
    """
    if applications.get_application(application_id) is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    _enqueue_rescore(scores, tasks, application_id)
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
    # No run yet, and the ad is still being read (or fetched): the first score
    # is chained from that read, so the panel says so and polls for it.
    awaits_ad = score is None and detail.application.extraction_status == "pending"
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
        "override_heading": OVERRIDE_HEADING,
        "override_note": OVERRIDE_NOTE,
        "override_label": OVERRIDE_LABEL,
        "adjusted_note": ADJUSTED_NOTE,
        "sent_note": SENT_NOTE,
        "score": score,
        "score_failure": failure,
        "score_unmeasured": UNMEASURED,
        "score_cost_note": COST_NOTE,
        "score_awaits_ad": awaits_ad,
        "score_awaits_ad_note": AWAITS_AD_NOTE,
        "score_retrying_note": RETRYING_NOTE,
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
    notes = {
        "extraction_retrying_note": EXTRACTION_RETRYING_NOTE,
        "fetch_retrying_note": FETCH_RETRYING_NOTE,
    }
    if extraction is None:
        return {"extraction": None, "failure": None, "ad_section": section, **notes}
    failure = extraction_failure(extraction.error_code) if extraction.status == "failed" else None
    return {"extraction": extraction, "failure": failure, "ad_section": section, **notes}


@router.post("/applications/{application_id}/status")
def change_status(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
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
                # The swapped row carries both scores exactly as the full list
                # rendered them -- same partial, same context.
                "row_scores": _row_scores(scores, [application.id]),
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


# --------------------------------------------------------------------------
# Bulk archive: select a batch of rows and archive them together, plus the
# undo for that batch. Owner feedback, 2026-09-27.
# --------------------------------------------------------------------------


@router.post("/applications/bulk-archive")
def bulk_archive(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    _csrf: CsrfDep,
    application_id: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """Archive every checked row at once. Ids for another user's application,
    or a stale id, are silently dropped by `archive_many` -- the response is
    the same either way, so this cannot be used to probe which ids exist.
    """
    ids = _parse_ids(application_id)
    archived = applications.archive_many(ids) if ids else []
    if not archived:
        return RedirectResponse("/applications?bulk_error=none_selected", status_code=303)
    return RedirectResponse(_bulk_redirect("/applications", archived), status_code=303)


@router.post("/applications/bulk-unarchive")
def bulk_unarchive(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    _csrf: CsrfDep,
    application_id: Annotated[list[str] | None, Form()] = None,
    back: Annotated[str, Form()] = "/applications?archived=1",
) -> Response:
    """The undo for `bulk_archive` (redirected back to the live list, where the
    "Archived N. Undo" banner lives), and also what the Archived view's own
    "Restore selected" button uses (redirected back to `?archived=1`, via
    `back`). `back` is never trusted as an arbitrary redirect target -- it is
    accepted only if it already starts with `/applications`, so a tampered
    value falls back to the live list rather than sending a signed-in session
    somewhere else.
    """
    ids = _parse_ids(application_id)
    if ids:
        applications.unarchive_many(ids)
    destination = back if back.startswith("/applications") else "/applications"
    return RedirectResponse(destination, status_code=303)


# --------------------------------------------------------------------------
# Archive by rule: "archive every application where ..." -- preview, then
# confirm. See `jfl_web.archive_rules` for the matching logic.
# --------------------------------------------------------------------------


def _rule_matches(
    applications: ApplicationRepoDep, scores: ScoreRepoDep, rules: ArchiveRules
) -> list[Application]:
    """Read the live list fresh and apply `rules` to it -- called from both the
    preview and the confirm route, so "what this will archive" and "what this
    archives" are always the exact same computation.
    """
    items = applications.list_applications(archived=False)
    row_scores = _row_scores(scores, [a.id for a in items])
    return matching_applications(items, row_scores, rules, now=dt.datetime.now(dt.UTC))


@router.post("/applications/archive-by-rule/preview")
def archive_by_rule_preview(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    _csrf: CsrfDep,
    score_enabled: Annotated[str | None, Form()] = None,
    score_axis: Annotated[str, Form()] = "either",
    score_threshold: Annotated[str, Form()] = "",
    status: Annotated[list[str] | None, Form()] = None,
    activity_enabled: Annotated[str | None, Form()] = None,
    activity_days: Annotated[str, Form()] = "",
) -> Response:
    """ "This will archive N applications:" -- read-only. Nothing here calls
    `archive_many`; the confirm route below is the only write, and it
    recomputes the match itself rather than trusting a list of ids this page
    handed back, so nothing here needs to be tamper-proof to stay safe.
    """
    rules, error = parse_rule_form(
        score_enabled=bool(score_enabled),
        score_axis=score_axis,
        score_threshold=score_threshold,
        statuses=status,
        activity_enabled=bool(activity_enabled),
        activity_days=activity_days,
    )
    if error is not None or rules is None:
        return RedirectResponse(f"/applications?rule_error={error}", status_code=303)

    matched = _rule_matches(applications, scores, rules)
    return render(
        request,
        "archive_rule_preview.html",
        {
            "session": session,
            "user": session.user,
            "matched": matched,
            "row_scores": _row_scores(scores, [a.id for a in matched]),
            "rules": rules,
        },
    )


@router.post("/applications/archive-by-rule/confirm")
def archive_by_rule_confirm(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    _csrf: CsrfDep,
    score_enabled: Annotated[str | None, Form()] = None,
    score_axis: Annotated[str, Form()] = "either",
    score_threshold: Annotated[str, Form()] = "",
    status: Annotated[list[str] | None, Form()] = None,
    activity_enabled: Annotated[str | None, Form()] = None,
    activity_days: Annotated[str, Form()] = "",
) -> Response:
    """The confirm button on the preview page. Parses the same hidden fields
    the preview rendered and re-runs the same match -- an application that
    changed state between preview and confirm (a score landed, a status
    changed) is measured as it is now, never as the stale preview said, and an
    id is never read off the request at all.
    """
    rules, error = parse_rule_form(
        score_enabled=bool(score_enabled),
        score_axis=score_axis,
        score_threshold=score_threshold,
        statuses=status,
        activity_enabled=bool(activity_enabled),
        activity_days=activity_days,
    )
    if error is not None or rules is None:
        return RedirectResponse(f"/applications?rule_error={error}", status_code=303)

    matched = _rule_matches(applications, scores, rules)
    archived = applications.archive_many([a.id for a in matched])
    return RedirectResponse(_bulk_redirect("/applications", archived), status_code=303)


# --------------------------------------------------------------------------
# Bulk re-score: score a selection of rows at once, skipping anything already
# scoring or not yet read. Owner feedback, 2026-09-27.
# --------------------------------------------------------------------------


def _has_requirements(
    applications: ApplicationRepoDep, application_ids: list[uuid.UUID]
) -> dict[uuid.UUID, bool]:
    """Which of these applications have at least one requirement recorded --
    what a bulk re-score's preflight check reads, so it can skip an
    application scoring would only fail on (`no_requirements`) instead of
    spending a queued, doomed run to discover the same thing.
    """
    return {
        application_id: bool(extraction.requirements)
        for application_id in application_ids
        if (extraction := applications.get_extraction(application_id)) is not None
    }


def _rescore_selection(
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    ids: list[uuid.UUID],
) -> tuple[list[Application], list[RescoreSkip]]:
    """The eligible/skipped split for a set of submitted ids, resolved fresh
    through this user's own repositories -- shared by the preview and confirm
    routes so "what this will score" and "what this scores" are always the
    same computation, the same discipline `_rule_matches` uses for
    archive-by-rule.
    """
    apps = applications.list_by_ids(ids)
    row_scores = _row_scores(scores, [a.id for a in apps])
    has_requirements = _has_requirements(applications, [a.id for a in apps])
    return partition_rescore(apps, row_scores, has_requirements)


@router.post("/applications/rescore/preview")
def rescore_preview(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    runs: RunRepoDep,
    _csrf: CsrfDep,
    application_id: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """ "This will re-score N applications, about $X on your API key" --
    read-only. Nothing here enqueues anything; the confirm route below is the
    only write, and it recomputes the eligible set itself rather than trusting
    what this page listed.
    """
    ids = _parse_ids(application_id)
    if not ids:
        return RedirectResponse("/applications?bulk_error=none_selected", status_code=303)

    eligible, skipped = _rescore_selection(applications, scores, ids)
    costs = runs.recent_costs(session.user.id, stages=_SCORE_COST_STAGES, limit=_COST_SAMPLE_SIZE)
    estimate: CostEstimate = estimate_for_count(costs, len(eligible))
    return render(
        request,
        "rescore_preview.html",
        {
            "session": session,
            "user": session.user,
            "eligible": eligible,
            "skipped": skipped,
            "skip_reason_labels": SKIP_REASON_LABELS,
            "row_scores": _row_scores(scores, [a.id for a in eligible]),
            "estimate": estimate,
            "submitted_ids": ids,
        },
    )


@router.post("/applications/rescore/confirm")
def rescore_confirm(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    application_id: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """The confirm button on the preview page. Recomputes the eligible set from
    the submitted ids rather than trusting the preview's -- an application that
    started scoring, or had its ad read, between preview and confirm is
    measured as it stands now. Ids for another user, or a stale id, resolve to
    nothing through `list_by_ids` and are silently dropped, same as every
    other bulk action here.
    """
    ids = _parse_ids(application_id)
    eligible, _skipped = _rescore_selection(applications, scores, ids)
    queued = [a.id for a in eligible if _enqueue_rescore(scores, tasks, a.id) is not None]
    if not queued:
        # Nothing to say -- same "silent when nothing happened" convention as
        # `_bulk_redirect` above.
        return RedirectResponse("/applications", status_code=303)
    return RedirectResponse(f"/applications?queued_rescores={len(queued)}", status_code=303)


# --------------------------------------------------------------------------
# Retry everything that failed: no selection, no rule form -- it retries
# whatever the live list currently shows as failed. Owner feedback,
# 2026-09-27.
# --------------------------------------------------------------------------


def _retry_plan(
    applications: ApplicationRepoDep, scores: ScoreRepoDep
) -> tuple[RetryPlan, dict[uuid.UUID, RowScore]]:
    """Read the live (non-archived) list fresh and classify it -- called from
    the list page (to decide whether the button shows at all), the preview and
    the confirm route, so all three agree on what "has failed" means right
    now. Never scoped to the current status filter: "everything that failed"
    means everything, not whatever `?status=` happens to be on the URL.
    """
    items = applications.list_applications(archived=False)
    row_scores = _row_scores(scores, [a.id for a in items])
    return plan_retry_failed(items, row_scores), row_scores


@router.post("/applications/retry-failed/preview")
def retry_failed_preview(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    runs: RunRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """ "This will retry N applications, about $X on your API key" -- read-only,
    same discipline as every other preview here: nothing is enqueued until
    confirm, which recomputes the plan itself rather than trusting this page.
    """
    plan, row_scores = _retry_plan(applications, scores)
    if plan.is_empty:
        return RedirectResponse("/applications", status_code=303)

    read_costs = runs.recent_costs(
        session.user.id, stages=_READ_COST_STAGES, limit=_COST_SAMPLE_SIZE
    )
    score_costs = runs.recent_costs(
        session.user.id, stages=_SCORE_COST_STAGES, limit=_COST_SAMPLE_SIZE
    )
    estimate: CostEstimate = estimate_retry_cost(
        read_costs, score_costs, reads=len(plan.retry_reads), scores=len(plan.retry_scores)
    )
    # Why each one failed, in the same words the single-application panels
    # use -- `extraction_failure`/`score_failure` are the same lookups
    # `_extraction_context`/`_score_context` call for the detail page.
    read_failures = {
        a.id: extraction_failure(a.extraction_error_code)
        for a in (*plan.retry_reads, *plan.needs_paste)
    }
    score_failures: dict[uuid.UUID, ScoreFailure] = {}
    for application in plan.retry_scores:
        latest = scores.latest(application.id)
        score_failures[application.id] = score_failure(latest.error_code if latest else None)
    return render(
        request,
        "retry_failed_preview.html",
        {
            "session": session,
            "user": session.user,
            "plan": plan,
            "row_scores": row_scores,
            "read_failures": read_failures,
            "score_failures": score_failures,
            "estimate": estimate,
        },
    )


@router.post("/applications/retry-failed/confirm")
def retry_failed_confirm(
    request: Request,
    session: SessionDep,
    applications: ApplicationRepoDep,
    scores: ScoreRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """The confirm button on the preview page. Re-reads the live list and
    recomputes the plan at write time -- an application fixed, archived or
    newly failed between preview and confirm is retried (or not) as it stands
    now, never as the stale preview said.

    A failed read wins over a failed score on the same application: retrying
    the read is enough (a successful re-read chains its own first score, see
    `jfl_worker.handlers.extraction._chain_first_score`), so `plan_retry_failed`
    never puts one application in both buckets.
    """
    plan, _row_scores = _retry_plan(applications, scores)
    read_count = sum(1 for a in plan.retry_reads if _enqueue_read_retry(applications, tasks, a.id))
    score_count = sum(
        1 for a in plan.retry_scores if _enqueue_rescore(scores, tasks, a.id) is not None
    )
    total = read_count + score_count
    if not total:
        # Nothing to say -- same "silent when nothing happened" convention as
        # `_bulk_redirect` above.
        return RedirectResponse("/applications", status_code=303)
    return RedirectResponse(f"/applications?queued_retries={total}", status_code=303)
