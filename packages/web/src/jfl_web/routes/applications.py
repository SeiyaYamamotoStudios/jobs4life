"""The application tracker -- slice A5/A6, the reason this whole hosted app
exists. See CLAUDE.md's 2026-09-07 decision log and `PLAN.md`'s slice A: the
owner's own words for the gap conversations cannot close are "a clear list of
all the applications I have going."

No model call anywhere in this file -- see `jfl_core.storage.applications`'s
module docstring for why a pasted job ad is stored verbatim and never parsed.

Screens:

  GET  /applications                 -- the list, most recently updated first,
                                         optionally filtered by `?status=`
  GET  /applications/new             -- the add form
  POST /applications                 -- create, then redirect to the detail page
  GET  /applications/{id}            -- one application: fields, full timeline,
                                         a status control, editable notes
  POST /applications/{id}/status     -- change status; appends an event
  POST /applications/{id}/notes      -- replace the notes field

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
from jfl_core.models import ApplicationStatus
from jfl_core.storage.applications import ApplicationNotFoundError

from jfl_web.deps import ApplicationRepoDep, CsrfDep, SessionDep
from jfl_web.templating import render

router = APIRouter()

STATUSES: tuple[ApplicationStatus, ...] = get_args(ApplicationStatus)

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
    raw_status = request.query_params.get("status")
    status = raw_status if raw_status in STATUSES else None
    items = applications.list_applications(status=status)
    return render(
        request,
        "applications_list.html",
        {
            "session": session,
            "user": session.user,
            "applications": items,
            "statuses": STATUSES,
            "active_status": status,
        },
    )


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
    _csrf: CsrfDep,
    title: Annotated[str, Form()],
    employer: Annotated[str, Form()] = "",
    url: Annotated[str, Form()] = "",
    source: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
    job_ad: Annotated[str, Form()] = "",
) -> Response:
    title = title.strip()
    if not title:
        return render(
            request,
            "application_form.html",
            {
                "session": session,
                "user": session.user,
                "error": "A title is required.",
                "values": {
                    "title": title,
                    "employer": employer,
                    "url": url,
                    "source": source,
                    "notes": notes,
                    "job_ad": job_ad,
                },
            },
            status_code=400,
        )

    application = applications.create_application(
        title=title,
        employer=employer.strip() or None,
        url=url.strip() or None,
        source=source.strip() or None,
        notes=notes.strip() or None,
        # Stored verbatim, never parsed -- see the module docstring.
        raw_job_text=job_ad.strip() or None,
    )
    # POST/redirect/GET: a refresh must not resubmit the form.
    return RedirectResponse(f"/applications/{application.id}", status_code=303)


@router.get("/applications/{application_id}")
def application_detail(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
) -> Response:
    detail = applications.get_application(application_id)
    if detail is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)
    return render(
        request,
        "application_detail.html",
        {
            "session": session,
            "user": session.user,
            "application": detail.application,
            "events": detail.events,
            "statuses": STATUSES,
        },
    )


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
            {"session": session, "application": application, "statuses": STATUSES},
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
