"""B5's drafting screen: "Generate a CV" (and a cover letter) for an
application, behind the queue, on the user's own key -- see PLAN.md's slice B5
and `NEXT.md` task 3.

**Prerequisites are explicit, never silent.** Drafting needs the job's
requirements (extraction) and recorded corpus coverage -- both model calls, so
neither runs on the way to another. This screen (`GET .../drafts`) reads what
is already true and offers a button for whatever is missing: read the ad first
(the existing extraction panel handles that), check coverage
(`POST .../drafts/coverage`), then generate (`POST .../drafts`). Nothing here
calls `generate_draft` or `run_coverage` directly -- both happen in the
worker, via `jfl_worker.handlers.draft_generation` and `.coverage_generation`.

**Drafts are anchored on the job, not the application**, the same way
`jfl_generate.draft.generate_draft` stores them (`drafts.job_id`) -- so this
screen resolves `application.job_id` once and reads and writes through that.

**No status column tracks a generation in flight.** A draft or coverage task
is watched through the task queue itself (`tasks.get_task`), the same way its
worker handler proves it did not double-spend -- see
`jfl_worker.handlers.draft_generation`'s module docstring. `?task=<id>` on the
GET route is what a poll (and the redirect straight after enqueueing) carries
that task id in.

**A flagged draft is still shown in full.** Nothing here inspects a draft's
gate verdicts to decide what to render -- see CLAUDE.md, "How the claim gate
behaves": the claim gate informs, it never blocks. Framing renders as NOT
CHECKED, never as supported -- `jfl_web.drafts.sentence_label` is the one
place that rule is expressed for this screen.

Screens:

  GET  /applications/{id}/drafts                  -- prerequisites, buttons,
                                                       the draft history;
                                                       `?task=<id>` also shows
                                                       that task's progress
  POST /applications/{id}/drafts/coverage          -- enqueue a coverage check
  POST /applications/{id}/drafts                   -- enqueue a draft (`kind`
                                                       form field: cv_bullets
                                                       | cover_letter)
  GET  /applications/{id}/drafts/tasks/{task_id}   -- the polling fragment on
                                                       its own, for htmx
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, get_args

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import DraftKind
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.postgres import PostgresJobRepository, PostgresRunRepository
from jfl_core.storage.tasks import PostgresTaskRepository

from jfl_web.deps import (
    ApplicationRepoDep,
    CsrfDep,
    JobRepoDep,
    RunRepoDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.templating import render

GENERATE_COVERAGE_KIND = "generate_coverage"
GENERATE_CV_DRAFT_KIND = "generate_cv_draft"

_DRAFT_KINDS: tuple[DraftKind, ...] = get_args(DraftKind)

router = APIRouter()

# Same wording and the same reasoning as `jfl_web.routes.applications._NOT_FOUND`:
# one message whether the id never existed or belongs to another account, so
# neither answer tells a caller which ids are real. Not imported from that
# module -- a private name there, and this file should not break if its
# wording changes.
_NOT_FOUND = "No application found -- it may belong to another account."


def _draft_entries(
    jobs: PostgresJobRepository,
    run_repo: PostgresRunRepository,
    session: AuthenticatedSession,
    job_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Every stored draft for this job, most recent first, each paired with
    what it cost (`RunRepository.cost_for_trace`, summing the draft call and
    its automatic claim-gate pass under one `trace_id`) -- because the user is
    paying for it on their own key.
    """
    return [
        {"draft": d, "cost": run_repo.cost_for_trace(session.user.id, d.trace_id)}
        for d in jobs.list_drafts(session.user.id, job_id)
    ]


def _task_context(tasks: PostgresTaskRepository, task_id: str | None) -> dict[str, Any] | None:
    """None means "nothing to show" -- no `?task=` at all. A task id that does
    not parse or does not belong to this user is folded into that same "show
    nothing" answer for the inline case (`drafting_screen`), and into a 404
    for the standalone fragment (`drafting_task`), which is why this returns
    `None` for both and the two callers decide what `None` means for them.
    """
    if not task_id:
        return None
    try:
        parsed = uuid.UUID(task_id)
    except ValueError:
        return None
    task = tasks.get_task(parsed)
    if task is None:
        return None
    return {"id": task.id, "kind": task.kind, "status": task.status, "last_error": task.last_error}


def _page_context(
    session: AuthenticatedSession,
    jobs: PostgresJobRepository,
    run_repo: PostgresRunRepository,
    tasks: PostgresTaskRepository,
    application_id: uuid.UUID,
    job_id: uuid.UUID | None,
    task_id: str | None,
) -> dict[str, Any]:
    """Everything `application_drafts.html` needs, in one place.

    `jobs` and `run_repo` are `jfl_core.storage.postgres`'s CLI-era, per-call
    repositories (see `jfl_web.deps.job_repo`'s docstring) -- every call below
    passes `session.user.id` explicitly, standing in for the structural
    tenancy the rest of this app enforces at construction.
    """
    context: dict[str, Any] = {
        "session": session,
        "user": session.user,
        "application_id": application_id,
        "job_id": job_id,
        "job": None,
        "requirements": [],
        "coverage": [],
        "drafts": [],
        "draft_kinds": _DRAFT_KINDS,
    }
    draft_entries: list[dict[str, Any]] = []
    if job_id is not None:
        found = jobs.get_job(session.user.id, job_id)
        if found is not None:
            job, requirements = found
            context["job"] = job
            context["requirements"] = requirements
            context["coverage"] = jobs.latest_coverage(session.user.id, job_id)
            draft_entries = _draft_entries(jobs, run_repo, session, job_id)
            context["drafts"] = draft_entries

    task = _task_context(tasks, task_id)
    if task is not None and task["kind"] == GENERATE_CV_DRAFT_KIND:
        # The list just built (most recent first) is exactly where a
        # just-succeeded draft lives -- one query, not two. A coverage task
        # needs no equivalent lookup: `context["coverage"]` above is already
        # the latest row per requirement, which is this run's own output once
        # it has succeeded.
        task["draft"] = next((e for e in draft_entries if e["draft"].trace_id == task["id"]), None)
        if task["draft"] is not None:
            # Shown once, inline in the task panel -- drop it from the history
            # list below so a just-generated draft is never rendered twice on
            # the same page.
            context["drafts"] = [e for e in draft_entries if e is not task["draft"]]
    context["task"] = task
    return context


@router.get("/applications/{application_id}/drafts")
def drafting_screen(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
    run_repo: RunRepoDep,
) -> Response:
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    context = _page_context(
        session,
        jobs,
        run_repo,
        tasks,
        application_id,
        detail.application.job_id,
        request.query_params.get("task"),
    )
    return render(request, "application_drafts.html", context)


@router.get("/applications/{application_id}/drafts/tasks/{task_id}")
def drafting_task(
    request: Request,
    application_id: uuid.UUID,
    task_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    jobs: JobRepoDep,
    tasks: TaskRepoDep,
    run_repo: RunRepoDep,
) -> Response:
    """The polling fragment on its own -- what `_draft_task.html` polls while
    a task is still pending or running.
    """
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    task = _task_context(tasks, str(task_id))
    if task is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    job_id = detail.application.job_id
    if task["kind"] == GENERATE_CV_DRAFT_KIND and job_id is not None:
        entries = _draft_entries(jobs, run_repo, session, job_id)
        task["draft"] = next((e for e in entries if e["draft"].trace_id == task["id"]), None)

    return render(
        request,
        "_draft_task.html",
        {"session": session, "application_id": application_id, "task": task},
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
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    job_id = detail.application.job_id
    if job_id is None:
        # Nothing to check coverage against yet -- the same "could only fail"
        # reasoning `applications.extract_again` uses when there is no ad.
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
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    kind: Annotated[str, Form()],
) -> Response:
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    if kind not in _DRAFT_KINDS or detail.application.job_id is None:
        # An unrecognised kind is a tampered form, not a user mistake worth a
        # message; no job to draft against is the same "could only fail" case
        # `check_coverage_now` guards above.
        return RedirectResponse(f"/applications/{application_id}/drafts", status_code=303)

    task = tasks.enqueue(
        kind=GENERATE_CV_DRAFT_KIND,
        payload={"application_id": str(application_id), "kind": kind},
    )
    return RedirectResponse(
        f"/applications/{application_id}/drafts?task={task.id}", status_code=303
    )
