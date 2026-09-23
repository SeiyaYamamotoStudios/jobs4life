"""B5's drafting screen: "Write the CV" (and a cover letter) for an
application, behind the queue, on the user's own key -- see PLAN.md's slice B5
and `NEXT.md` task 3.

**Three steps, one button.** A CV needs the ad read (extraction) and the job's
requirements checked against the user's confirmed facts (coverage) before it
can be written. The screen shows the three as one sequence -- "Read the ad ✓ →
Check it against your facts → Write the CV" -- with what each costs on the
user's key, and the one button runs whichever are still missing, in order.
It does that by queueing the first missing step with the rest named in its
payload (`then`); each handler queues the next when it succeeds -- see
`jfl_worker.chain`. Nothing is silently satisfied: the cost line under the
button says what the press will run and what it will cost before anyone
presses it, and a step that fails stops the rest and is shown as the step that
stopped.

**Drafts are anchored on the job, not the application**, the same way
`jfl_generate.draft.generate_draft` stores them (`drafts.job_id`) -- so this
screen resolves `application.job_id` once and reads and writes through that.

**No status column tracks a press in flight.** It is watched through the task
queue itself: `?task=<id>` names the first task of a press, and the page
follows it forward through `PostgresTaskRepository.follow_up`. With no
`?task=`, an unfinished step for this application is found and shown anyway,
so a second press while one is running queues nothing.

**The CV leads.** The newest draft is the first thing on the page once there
is one: its headline count, the text itself (copyable, downloadable), and the
sentence-by-sentence check beside it -- flagged sentences first, each with its
next action.

**A flagged draft is still shown in full.** Nothing here inspects a draft's
verdicts to decide what to render -- see CLAUDE.md, "How the claim gate
behaves": it informs, it never blocks. Framing renders as not checked, never
as supported -- `jfl_web.drafts.verdict_key` is the one place that rule is
expressed for this screen.

Screens:

  GET  /applications/{id}/drafts                   -- the CV, the steps, the
                                                        earlier versions;
                                                        `?task=<id>` follows a press
  GET  /applications/{id}/drafts/steps             -- the steps panel on its own,
                                                        for htmx to poll
  GET  /applications/{id}/drafts/tasks/{task_id}   -- the same, for one press
  GET  /applications/{id}/drafts/{draft_id}/download -- the text, as a .txt file
  POST /applications/{id}/drafts/coverage          -- check against your facts again
  POST /applications/{id}/drafts                   -- write one (`kind`:
                                                        cv_bullets | cover_letter),
                                                        running any missing step first
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Annotated, Any, get_args

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import ApplicationDetail, DraftKind, Task
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_core.storage.tasks import UNFINISHED_STATUSES, PostgresTaskRepository
from jfl_core.storage.ui_sections import SectionState

from jfl_web.deps import (
    ApplicationRepoDep,
    CsrfDep,
    CvDocumentRepoDep,
    GroundingRepoDep,
    JobRepoDep,
    RunRepoDep,
    SectionRepoDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.drafts import (
    ChainTask,
    CvPlan,
    check_summary,
    download_name,
    next_action,
    plan_steps,
    sentence_meaning,
    step_failure,
)
from jfl_web.routes.cv_documents import cv_document_context
from jfl_web.sections import (
    cv_section,
    draft_history_section,
    draft_section,
    generate_section,
    requirements_section,
)
from jfl_web.templating import render

EXTRACT_JOB_AD_KIND = "extract_job_ad"
GENERATE_COVERAGE_KIND = "generate_coverage"
GENERATE_CV_DRAFT_KIND = "generate_cv_draft"

_STEP_KINDS = (EXTRACT_JOB_AD_KIND, GENERATE_COVERAGE_KIND, GENERATE_CV_DRAFT_KIND)

_DRAFT_KINDS: tuple[DraftKind, ...] = get_args(DraftKind)

router = APIRouter()

# Same wording and the same reasoning as `jfl_web.routes.applications._NOT_FOUND`:
# one message whether the id never existed or belongs to another account, so
# neither answer tells a caller which ids are real. Not imported from that
# module -- a private name there, and this file should not break if its
# wording changes.
_NOT_FOUND = "No application found -- it may belong to another account."

# A press is at most three tasks; the walk never needs more.
_MAX_CHAIN = 3


# --------------------------------------------------------------------------
# Where this application is in the three steps
# --------------------------------------------------------------------------


def _belongs(task: Task, application_id: uuid.UUID, job_id: uuid.UUID | None) -> bool:
    """Whether a task is one of this application's three steps."""
    if task.kind not in _STEP_KINDS:
        return False
    payload = task.payload
    if payload.get("application_id") == str(application_id):
        return True
    return job_id is not None and payload.get("job_id") == str(job_id)


def _active_task(
    tasks: PostgresTaskRepository,
    application_id: uuid.UUID,
    job_id: uuid.UUID | None,
    *,
    ad_read: bool,
) -> Task | None:
    """The newest unfinished step for this application, if there is one.

    A re-read of an ad that has already been read does not count: the
    requirements are there, so it blocks nothing, and hiding the button behind
    it would be a wait for no reason.
    """
    for status in UNFINISHED_STATUSES:
        for task in tasks.list_tasks(status=status, limit=50):
            if task.kind == EXTRACT_JOB_AD_KIND and ad_read:
                continue
            if _belongs(task, application_id, job_id):
                return task
    return None


def _chain(tasks: PostgresTaskRepository, root: Task) -> list[Task]:
    """`root` and every step queued after it, in order."""
    chain = [root]
    current = root
    while len(chain) < _MAX_CHAIN and current.status == "succeeded" and current.payload.get("then"):
        following = tasks.follow_up(current.id)
        if following is None:
            break
        chain.append(following)
        current = following
    return chain


def _chain_view(chain: Sequence[Task]) -> list[ChainTask]:
    return [
        ChainTask(
            kind=task.kind,
            status=task.status,
            failure=step_failure(task.kind, task.last_error) if task.status == "failed" else None,
            has_next=bool(task.payload.get("then")),
        )
        for task in chain
    ]


def _writing(chain: Sequence[Task]) -> DraftKind:
    """Which kind of draft a press is writing -- named in the draft step's
    payload, whether that step is queued yet or still in a `then` list."""
    for task in chain:
        candidates = [task.payload, *(s.get("payload", {}) for s in task.payload.get("then", []))]
        for payload in candidates:
            kind = payload.get("kind") if isinstance(payload, dict) else None
            if kind in _DRAFT_KINDS:
                return kind  # type: ignore[no-any-return]
    return "cv_bullets"


def _parse_task_id(raw: str | None) -> uuid.UUID | None:
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None


def plan_context(
    session: AuthenticatedSession,
    applications: PostgresApplicationRepository,
    jobs: PostgresJobRepository,
    tasks: PostgresTaskRepository,
    application_id: uuid.UUID,
    detail: ApplicationDetail,
    task_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """The steps panel's context: where this application is, and the press
    being watched -- `task_id` if it is one of this application's, else any
    step still running for it.

    Shared with the application page, which shows the same panel.
    `jobs` is `jfl_core.storage.postgres`'s CLI-era repository (see
    `jfl_web.deps.job_repo`), so `session.user.id` goes on every call.
    """
    job_id = detail.application.job_id
    extraction = applications.get_extraction(application_id)
    job = None
    requirements: list[Any] = []
    coverage: list[Any] = []
    if job_id is not None:
        found = jobs.get_job(session.user.id, job_id)
        if found is not None:
            job, requirements = found
            coverage = jobs.latest_coverage(session.user.id, job_id)

    has_ad = job is not None and extraction is not None and extraction.has_job_ad
    ad_read = bool(requirements) or (extraction is not None and extraction.status == "done")
    ad_reading = extraction is not None and extraction.status == "pending" and not ad_read

    root: Task | None = None
    if task_id is not None:
        candidate = tasks.get_task(task_id)
        if candidate is not None and _belongs(candidate, application_id, job_id):
            root = candidate
    if root is None:
        root = _active_task(tasks, application_id, job_id, ad_read=ad_read)
    chain = _chain(tasks, root) if root is not None else []

    plan = plan_steps(
        has_ad=has_ad,
        ad_read=ad_read,
        ad_reading=ad_reading,
        checked=bool(coverage),
        chain=_chain_view(chain),
        writing=_writing(chain),
    )
    steps_url = f"/applications/{application_id}/drafts/steps"
    if root is not None:
        steps_url += f"?task={root.id}"
    return {
        "plan": plan,
        "steps_url": steps_url,
        "job": job,
        "job_id": job_id,
        "requirements": requirements,
        "coverage": coverage,
        "watched": root,
        "draft_kinds": _DRAFT_KINDS,
    }


# --------------------------------------------------------------------------
# The drafts themselves
# --------------------------------------------------------------------------


def _draft_entries(
    jobs: PostgresJobRepository,
    run_repo: PostgresRunRepository | None,
    session: AuthenticatedSession,
    job_id: uuid.UUID,
    states: dict[str, SectionState],
) -> list[dict[str, Any]]:
    """Every stored draft for this job, most recent first, each with its check
    counted and grouped (`check_summary`) and what it cost
    (`RunRepository.cost_for_trace`, summing the draft call and its automatic
    sentence check under one `trace_id`) -- because the user is paying for it
    on their own key.

    Each also carries its own section. The newest draft is open and the rest
    fold behind their date and count.
    """
    entries: list[dict[str, Any]] = []
    for index, draft in enumerate(jobs.list_drafts(session.user.id, job_id)):
        entry: dict[str, Any] = {
            "draft": draft,
            "check": check_summary(draft.gate_result),
            "cost": None
            if run_repo is None
            else run_repo.cost_for_trace(session.user.id, draft.trace_id),
        }
        entry["section"] = draft_section(states, entry, newest=index == 0)
        entries.append(entry)
    return entries


# A cited fact is shown at this length and then trimmed on a word boundary.
# Long enough to recognise which fact it is, short enough that a sentence with
# four citations does not bury the draft it belongs to.
_CITED_FACT_CHARS = 160


def _cited_facts(
    grounding: PostgresGroundingRepository,
    session: AuthenticatedSession,
    entries: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """`{fact id: the fact's own words}` for every fact the drafts on this page
    cite, so the screen can say what a sentence rests on rather than printing a
    UUID at someone.

    A missing id is left out rather than guessed at: a fact the user has since
    retired is exactly the case where inventing a plausible sentence would be
    worst, so the template says the fact has changed instead.
    """
    wanted: set[str] = set()
    for entry in entries:
        gate_result = entry["draft"].gate_result or {}
        for sentence in gate_result.get("sentences", []):
            wanted.update(str(span_id) for span_id in sentence.get("cited_span_ids", []))
    facts: dict[str, str] = {}
    for raw_id in wanted:
        try:
            span_id = uuid.UUID(raw_id)
        except ValueError:
            continue
        span = grounding.get_span(session.user.id, span_id)
        if span is None:
            continue
        text = " ".join(span.text.split())
        if len(text) > _CITED_FACT_CHARS:
            text = text[:_CITED_FACT_CHARS].rsplit(" ", 1)[0] + "..."
        facts[raw_id] = text
    return facts


def cv_panel_context(
    session: AuthenticatedSession,
    applications: PostgresApplicationRepository,
    jobs: PostgresJobRepository,
    tasks: PostgresTaskRepository,
    states: dict[str, SectionState],
    application_id: uuid.UUID,
    detail: ApplicationDetail,
) -> dict[str, Any]:
    """What the application page's "CV for this job" panel needs: the same
    steps and button as the drafting screen, and the newest draft's headline.
    """
    context = plan_context(session, applications, jobs, tasks, application_id, detail)
    latest = None
    job_id = detail.application.job_id
    if context["job"] is not None and job_id is not None:
        drafts = jobs.list_drafts(session.user.id, job_id)
        if drafts:
            latest = {"draft": drafts[0], "check": check_summary(drafts[0].gate_result)}
    plan: CvPlan = context["plan"]
    return {
        "cv_plan": plan,
        "cv_steps_url": context["steps_url"],
        "cv_latest": latest,
        "draft_kinds": _DRAFT_KINDS,
        "cv_section": cv_section(states, latest, pending=plan.in_progress),
    }


def _page_context(
    session: AuthenticatedSession,
    applications: PostgresApplicationRepository,
    jobs: PostgresJobRepository,
    run_repo: PostgresRunRepository,
    grounding: PostgresGroundingRepository,
    tasks: PostgresTaskRepository,
    states: dict[str, SectionState],
    application_id: uuid.UUID,
    detail: ApplicationDetail,
    task_id: uuid.UUID | None,
) -> dict[str, Any]:
    """Everything `application_drafts.html` needs, in one place."""
    context = plan_context(session, applications, jobs, tasks, application_id, detail, task_id)
    entries: list[dict[str, Any]] = []
    if context["job"] is not None and context["job_id"] is not None:
        entries = _draft_entries(jobs, run_repo, session, context["job_id"], states)
    plan: CvPlan = context["plan"]
    context.update(
        {
            "session": session,
            "user": session.user,
            "application_id": application_id,
            "latest": entries[0] if entries else None,
            "older": entries[1:],
            "cited_facts": _cited_facts(grounding, session, entries),
            "next_action": next_action,
            "sentence_meaning": sentence_meaning,
            "generate_section": generate_section(states, pending=plan.in_progress),
            "requirements_section": requirements_section(
                states, context["requirements"], context["coverage"]
            ),
            "draft_history_section": draft_history_section(states, entries[1:]),
        }
    )
    return context


def _not_found(request: Request, session: AuthenticatedSession) -> Response:
    context = {"session": session, "user": session.user, "message": _NOT_FOUND}
    return render(request, "error.html", context, status_code=404)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.get("/applications/{application_id}/drafts")
def drafting_screen(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
    run_repo: RunRepoDep,
    grounding: GroundingRepoDep,
    ui_sections: SectionRepoDep,
    cv_documents: CvDocumentRepoDep,
) -> Response:
    detail = applications.get_application(application_id)
    if detail is None:
        return _not_found(request, session)
    states = ui_sections.states()
    context = _page_context(
        session,
        applications,
        jobs,
        run_repo,
        grounding,
        tasks,
        states,
        application_id,
        detail,
        _parse_task_id(request.query_params.get("task")),
    )
    # The generated CV document, when there is one -- its preview, its check,
    # its versions and its downloads (`jfl_web.routes.cv_documents`).
    context.update(
        cv_document_context(
            cv_documents, tasks, states, application_id, request.query_params.get("cv")
        )
    )
    return render(request, "application_drafts.html", context)


def _steps_fragment(
    request: Request,
    session: AuthenticatedSession,
    applications: PostgresApplicationRepository,
    jobs: PostgresJobRepository,
    tasks: PostgresTaskRepository,
    application_id: uuid.UUID,
    detail: ApplicationDetail,
    task_id: uuid.UUID | None,
) -> Response:
    """The steps panel on its own. Polled only while something is running; the
    poll that finds it stopped asks htmx to reload the page (`HX-Refresh`), so
    the finished CV -- or the step that stopped -- is shown by the full page
    rather than squeezed into the panel. A browser without htmx ignores the
    header and simply shows the panel.
    """
    context = plan_context(session, applications, jobs, tasks, application_id, detail, task_id)
    context.update({"session": session, "application_id": application_id})
    response = render(request, "_cv_steps.html", context)
    if not context["plan"].in_progress:
        response.headers["HX-Refresh"] = "true"
    return response


@router.get("/applications/{application_id}/drafts/steps")
def drafting_steps(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
) -> Response:
    detail = applications.get_application(application_id)
    if detail is None:
        return _not_found(request, session)
    return _steps_fragment(
        request,
        session,
        applications,
        jobs,
        tasks,
        application_id,
        detail,
        _parse_task_id(request.query_params.get("task")),
    )


@router.get("/applications/{application_id}/drafts/tasks/{task_id}")
def drafting_task(
    request: Request,
    application_id: uuid.UUID,
    task_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
) -> Response:
    """The steps panel for one press -- 404 for a task that is not one of this
    application's, so an id cannot be used to probe for another's."""
    detail = applications.get_application(application_id)
    if detail is None:
        return _not_found(request, session)
    task = tasks.get_task(task_id)
    if task is None or not _belongs(task, application_id, detail.application.job_id):
        return _not_found(request, session)
    return _steps_fragment(
        request, session, applications, jobs, tasks, application_id, detail, task_id
    )


@router.get("/applications/{application_id}/drafts/{draft_id}/download")
def download_draft(
    request: Request,
    application_id: uuid.UUID,
    draft_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
) -> Response:
    """The draft's text as a plain-text file -- exactly the text shown in the
    box, nothing added."""
    detail = applications.get_application(application_id)
    job_id = None if detail is None else detail.application.job_id
    if detail is None or job_id is None:
        return _not_found(request, session)
    draft = next((d for d in jobs.list_drafts(session.user.id, job_id) if d.id == draft_id), None)
    if draft is None:
        return _not_found(request, session)
    name = download_name(draft.kind, detail.application.title)
    return Response(
        content=draft.text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.post("/applications/{application_id}/drafts/coverage")
def check_coverage_now(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Check the job against your facts again -- worth it after confirming more."""
    detail = applications.get_application(application_id)
    if detail is None:
        return _not_found(request, session)
    job_id = detail.application.job_id
    if job_id is None:
        # Nothing to check against yet -- the same "could only fail" reasoning
        # `applications.extract_again` uses when there is no ad.
        return RedirectResponse(f"/applications/{application_id}/drafts", status_code=303)

    task = tasks.enqueue(kind=GENERATE_COVERAGE_KIND, payload={"job_id": str(job_id)})
    return RedirectResponse(
        f"/applications/{application_id}/drafts?task={task.id}", status_code=303
    )


@router.post("/applications/{application_id}/drafts")
def request_draft(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    kind: Annotated[str, Form()],
) -> Response:
    """Write one -- running whichever of the two steps before it are missing.

    Queues the first missing step and names the rest in its payload, so one
    press is one of each step (`jfl_worker.chain`). A press while a step for
    this application is already running queues nothing and shows that one.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        return _not_found(request, session)
    drafts_url = f"/applications/{application_id}/drafts"
    job_id = detail.application.job_id
    if kind not in _DRAFT_KINDS or job_id is None:
        # An unrecognised kind is a tampered form, not a user mistake worth a
        # message; no job to write against is the "could only fail" case.
        return RedirectResponse(drafts_url, status_code=303)

    context = plan_context(session, applications, jobs, tasks, application_id, detail)
    watched: Task | None = context["watched"]
    if watched is not None and context["plan"].in_progress:
        return RedirectResponse(f"{drafts_url}?task={watched.id}", status_code=303)
    plan: CvPlan = context["plan"]
    if not plan.can_start:
        return RedirectResponse(drafts_url, status_code=303)

    todo = {step.key for step in plan.steps if step.state != "done"}
    draft_payload = {"application_id": str(application_id), "kind": kind}
    draft_step = {"kind": GENERATE_CV_DRAFT_KIND, "payload": draft_payload}
    coverage_step = {"kind": GENERATE_COVERAGE_KIND, "payload": {"job_id": str(job_id)}}
    if "ad" in todo:
        if not applications.request_extraction(application_id):
            return RedirectResponse(drafts_url, status_code=303)
        task = tasks.enqueue(
            kind=EXTRACT_JOB_AD_KIND,
            payload={"application_id": str(application_id), "then": [coverage_step, draft_step]},
        )
    elif "check" in todo:
        task = tasks.enqueue(
            kind=GENERATE_COVERAGE_KIND, payload={"job_id": str(job_id), "then": [draft_step]}
        )
    else:
        task = tasks.enqueue(kind=GENERATE_CV_DRAFT_KIND, payload=dict(draft_payload))
    return RedirectResponse(f"{drafts_url}?task={task.id}", status_code=303)
