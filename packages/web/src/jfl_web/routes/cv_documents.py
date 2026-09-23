"""Review, edit and export the generated CV document for an application.

The CV is shown on the application's CV page (`/applications/{id}/drafts`) as
it reads -- through `render_cv_html`, so the preview on screen is the same
document the PDF is made from -- with each generated line's check beside it in
the page's four marks.

**Every change is a new version.** The store is append-only
(`PostgresCvDocumentRepository.add_version`): an edit, a template switch, a
header refresh and a check each write one, and every version stays listed and
downloadable. A download serves exactly the version asked for -- never
regenerated, never re-rendered from anything but that stored document.

**Wording is editable; facts are not.** Every generated line, the descriptors,
the skill labels and the headings can be rewritten here. Titles, employers,
dates and education are copied from confirmed facts, and the header and
interests from the profile: they are shown, not offered as fields, and a
crafted POST that names one is refused with nothing stored
(`jfl_core.cv_lines.apply_edits`). The screen says where each is changed.

**An edited line loses its verdict** -- the check was of different words -- and
reads "not checked since you edited it" until the user asks for "Check my
edits": one background task, on their key, over the edited lines only.

**The claim gate informs, it never blocks.** A version with lines the user's
confirmed facts do not back still downloads; the download says so plainly.

Screens:

  GET  /applications/{id}/cv/edit               -- the edit form, latest version
  POST /applications/{id}/cv/edit               -- save edits as a new version
  POST /applications/{id}/cv/template           -- classic / modern, new version
  POST /applications/{id}/cv/header             -- header + interests from the profile
  POST /applications/{id}/cv/check              -- "Check my edits" (a task)
  GET  /applications/{id}/cv/{version_id}/pdf   -- that version, as a PDF
  GET  /applications/{id}/cv/{version_id}/text  -- that version, as plain text
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.cv_document import CvDocument, CvHeader, CvLink
from jfl_core.cv_lines import (
    InvalidEditError,
    ProtectedFieldError,
    apply_edits,
    editable_fields,
    unchecked_edits,
)
from jfl_core.cv_render import render_cv_html, render_cv_pdf
from jfl_core.profile import Profile
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.cv_documents import CvDocumentVersion, PostgresCvDocumentRepository
from jfl_core.storage.tasks import UNFINISHED_STATUSES, PostgresTaskRepository
from jfl_core.storage.ui_sections import SectionState

from jfl_web.cvdocs import (
    STATUS_WORDS,
    TEMPLATE_LABELS,
    check_edits_cost,
    check_failure,
    cv_check,
    cv_filename,
    cv_plain_text,
    export_warning,
)
from jfl_web.deps import (
    ApplicationRepoDep,
    CsrfDep,
    CvDocumentRepoDep,
    ProfileRepoDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.sections import cv_document_section, cv_versions_section
from jfl_web.templating import render

router = APIRouter()

CHECK_CV_EDITS_KIND = "check_cv_edits"

_NOT_FOUND = "No application found -- it may belong to another account."

# `?cv=<flag>` after a POST. Matched against this table and never rendered
# itself -- a query string is not somewhere page text comes from.
CV_MESSAGES: dict[str, tuple[str, str]] = {
    "saved": ("ok", "Saved as a new version. The lines you changed are marked as not checked."),
    "unchanged": ("note", "Nothing had changed, so no new version was saved."),
    "stale": (
        "error",
        "A newer version was saved while you had this open, so nothing was changed. "
        "Here is the newest one.",
    ),
    "checking": ("ok", "Checking your edits. Reload the page in a minute to see the result."),
    "nothing": ("note", "There are no edited lines waiting to be checked."),
    "template": ("ok", "Template changed. Saved as a new version."),
    "header": ("ok", "The top of the CV now matches your profile. Saved as a new version."),
}

_TEMPLATES = tuple(TEMPLATE_LABELS)


def _pending_check(tasks: PostgresTaskRepository, application_id: uuid.UUID) -> bool:
    for status in UNFINISHED_STATUSES:
        for task in tasks.list_tasks(status=status, kind=CHECK_CV_EDITS_KIND, limit=50):
            if task.payload.get("application_id") == str(application_id):
                return True
    return False


def _failed_check(
    tasks: PostgresTaskRepository, application_id: uuid.UUID, latest: CvDocumentVersion
) -> Any:
    """The newest failed check of this application's edits, if it failed
    after the version on screen was saved -- otherwise it is old news."""
    for task in tasks.list_tasks(status="failed", kind=CHECK_CV_EDITS_KIND, limit=20):
        if task.payload.get("application_id") != str(application_id):
            continue
        if task.scheduled_at >= latest.created_at:
            return check_failure(task.last_error)
        return None
    return None


def cv_document_context(
    cv_documents: PostgresCvDocumentRepository,
    tasks: PostgresTaskRepository,
    states: dict[str, SectionState],
    application_id: uuid.UUID,
    flag: str | None,
) -> dict[str, Any]:
    """What the CV page's document panel needs. Empty (`cvdoc: None`) for an
    application with no generated CV document yet."""
    versions = cv_documents.list_versions(application_id)
    if not versions:
        return {"cvdoc": None, "cv_message": None}
    latest, older = versions[0], versions[1:]
    check = cv_check(latest.doc)
    pending = _pending_check(tasks, application_id)
    previews = {
        name: render_cv_html(latest.doc.model_copy(update={"template": name}))
        for name in _TEMPLATES
    }
    return {
        "cvdoc": {
            "version": latest,
            "doc": latest.doc,
            "check": check,
            "warning": export_warning(latest.doc),
            "text": cv_plain_text(latest.doc),
            "previews": previews,
            "templates": TEMPLATE_LABELS,
            "unchecked_edits": len(unchecked_edits(latest.doc)),
            "check_pending": pending,
            "check_failed": None if pending else _failed_check(tasks, application_id, latest),
            "check_cost": check_edits_cost(),
            "status_words": STATUS_WORDS,
            "section": cv_document_section(states, latest, check, pending=pending),
            "older": [
                {"version": v, "warning": export_warning(v.doc), "check": cv_check(v.doc)}
                for v in older
            ],
            "versions_section": cv_versions_section(states, older),
        },
        "cv_message": CV_MESSAGES.get(flag or ""),
    }


def _not_found(request: Request, session: AuthenticatedSession) -> Response:
    context = {"session": session, "user": session.user, "message": _NOT_FOUND}
    return render(request, "error.html", context, status_code=404)


def _drafts_url(application_id: uuid.UUID, flag: str) -> RedirectResponse:
    return RedirectResponse(
        f"/applications/{application_id}/drafts?cv={flag}#cv-document", status_code=303
    )


def _owned_latest(
    applications: ApplicationRepoDep,
    cv_documents: PostgresCvDocumentRepository,
    application_id: uuid.UUID,
) -> tuple[Any, CvDocumentVersion | None]:
    detail = applications.get_application(application_id)
    if detail is None:
        return None, None
    return detail, cv_documents.latest(application_id)


def _edit_page(
    request: Request,
    session: AuthenticatedSession,
    application_id: uuid.UUID,
    detail: Any,
    version: CvDocumentVersion,
    *,
    error: str | None = None,
    status_code: int = 200,
    stale: bool = False,
    submitted: dict[str, str] | None = None,
) -> Response:
    fields = editable_fields(version.doc)
    values = {f.path: (submitted or {}).get(f.path, f.value) for f in fields}
    return render(
        request,
        "cv_edit.html",
        {
            "session": session,
            "user": session.user,
            "application_id": application_id,
            "application": detail.application,
            "version": version,
            "doc": version.doc,
            "fields": {f.path: f for f in fields},
            "values": values,
            "error": error,
            "stale": stale,
        },
        status_code=status_code,
    )


@router.get("/applications/{application_id}/cv/edit")
def edit_cv(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
) -> Response:
    detail, latest = _owned_latest(applications, cv_documents, application_id)
    if detail is None or latest is None:
        return _not_found(request, session)
    return _edit_page(
        request,
        session,
        application_id,
        detail,
        latest,
        stale=request.query_params.get("stale") == "1",
    )


def _base_is_latest(form: Any, latest: CvDocumentVersion) -> bool:
    base = form.get("base_version")
    return isinstance(base, str) and base == str(latest.id)


@router.post("/applications/{application_id}/cv/edit")
async def save_cv_edits(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Save the submitted wording as a new version.

    Every posted field other than the token and the base version must be an
    editable path of the latest version, or the whole submission is refused
    and nothing is stored -- see `jfl_core.cv_lines.apply_edits`.
    """
    detail, latest = _owned_latest(applications, cv_documents, application_id)
    if detail is None or latest is None:
        return _not_found(request, session)
    form = await request.form()
    if not _base_is_latest(form, latest):
        return RedirectResponse(f"/applications/{application_id}/cv/edit?stale=1", status_code=303)
    submitted: dict[str, str] = {}
    for key, value in form.multi_items():
        if key in ("csrf_token", "base_version"):
            continue
        if not isinstance(value, str) or key in submitted:
            return _edit_page(
                request,
                session,
                application_id,
                detail,
                latest,
                error="That form was not one this page sent, so nothing was saved.",
                status_code=400,
            )
        submitted[key] = value
    try:
        result = apply_edits(latest.doc, submitted)
    except ProtectedFieldError:
        return _edit_page(
            request,
            session,
            application_id,
            detail,
            latest,
            error=(
                "Nothing was saved: titles, employers, dates and education come from "
                "your confirmed facts and can't be changed here."
            ),
            status_code=400,
        )
    except InvalidEditError as exc:
        return _edit_page(
            request,
            session,
            application_id,
            detail,
            latest,
            error=str(exc),
            status_code=400,
            submitted=submitted,
        )
    if not result.changed:
        return _drafts_url(application_id, "unchanged")
    cv_documents.add_version(application_id, result.doc, status="edited", trace_id=None)
    return _drafts_url(application_id, "saved")


@router.post("/applications/{application_id}/cv/template")
async def switch_template(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Keep the other template. The page already previewed it; this makes it
    the version that downloads."""
    detail, latest = _owned_latest(applications, cv_documents, application_id)
    if detail is None or latest is None:
        return _not_found(request, session)
    form = await request.form()
    template = form.get("template")
    if template not in _TEMPLATES:
        return RedirectResponse(f"/applications/{application_id}/drafts", status_code=303)
    if not _base_is_latest(form, latest):
        return _drafts_url(application_id, "stale")
    if template == latest.doc.template:
        return _drafts_url(application_id, "unchanged")
    doc = latest.doc.model_copy(update={"template": template})
    cv_documents.add_version(application_id, doc, status="template", trace_id=None)
    return _drafts_url(application_id, "template")


def header_from_profile(profile: Profile, current: CvDocument) -> CvDocument:
    """The CV with its header and interests taken from the profile's settings.

    Settings, not claims: nothing here is checked. A name left blank on the
    profile keeps the one the CV already has, since a CV cannot have no name.
    """
    settings = profile.cv_header
    header = CvHeader(
        name=settings.name or current.header.name,
        tagline=settings.tagline,
        contact=settings.contact,
        links=[CvLink(label=link.label, url=link.url) for link in settings.links],
    )
    return current.model_copy(update={"header": header, "interests": list(profile.interests)})


@router.post("/applications/{application_id}/cv/header")
async def refresh_header(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
    profile_store: ProfileRepoDep,
    _csrf: CsrfDep,
) -> Response:
    detail, latest = _owned_latest(applications, cv_documents, application_id)
    if detail is None or latest is None:
        return _not_found(request, session)
    form = await request.form()
    if not _base_is_latest(form, latest):
        return _drafts_url(application_id, "stale")
    doc = header_from_profile(profile_store.current(), latest.doc)
    if doc == latest.doc:
        return _drafts_url(application_id, "unchanged")
    cv_documents.add_version(application_id, doc, status="header", trace_id=None)
    return _drafts_url(application_id, "header")


@router.post("/applications/{application_id}/cv/check")
def check_my_edits(
    request: Request,
    application_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """One task, over the edited lines of the latest version only. A second
    press while one is waiting or running queues nothing."""
    detail, latest = _owned_latest(applications, cv_documents, application_id)
    if detail is None or latest is None:
        return _not_found(request, session)
    paths = unchecked_edits(latest.doc)
    if not paths:
        return _drafts_url(application_id, "nothing")
    if not _pending_check(tasks, application_id):
        tasks.enqueue(
            kind=CHECK_CV_EDITS_KIND,
            payload={
                "application_id": str(application_id),
                "version_id": str(latest.id),
                "paths": paths,
            },
        )
    return _drafts_url(application_id, "checking")


def _version(
    applications: ApplicationRepoDep,
    cv_documents: PostgresCvDocumentRepository,
    application_id: uuid.UUID,
    version_id: uuid.UUID,
) -> tuple[Any, CvDocumentVersion | None]:
    detail = applications.get_application(application_id)
    if detail is None:
        return None, None
    found = next(
        (v for v in cv_documents.list_versions(application_id) if v.id == version_id), None
    )
    return detail, found


@router.get("/applications/{application_id}/cv/{version_id}/pdf")
def download_pdf(
    request: Request,
    application_id: uuid.UUID,
    version_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
) -> Response:
    """Exactly that stored version, rendered -- no regeneration, no marks."""
    detail, version = _version(applications, cv_documents, application_id, version_id)
    if detail is None or version is None:
        return _not_found(request, session)
    name = cv_filename(version.doc.header.name, detail.application.employer, "pdf")
    return Response(
        content=render_cv_pdf(version.doc),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/applications/{application_id}/cv/{version_id}/text")
def download_text(
    request: Request,
    application_id: uuid.UUID,
    version_id: uuid.UUID,
    session: SessionDep,
    applications: ApplicationRepoDep,
    cv_documents: CvDocumentRepoDep,
) -> Response:
    detail, version = _version(applications, cv_documents, application_id, version_id)
    if detail is None or version is None:
        return _not_found(request, session)
    name = cv_filename(version.doc.header.name, detail.application.employer, "txt")
    return Response(
        content=cv_plain_text(version.doc),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
