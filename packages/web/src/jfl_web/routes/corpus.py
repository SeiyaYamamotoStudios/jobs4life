"""CV upload -- PLAN.md slice B6, the first half of onboarding from CVs.

A new user's corpus is too small to score or draft against, so they start by
uploading **every** CV they have. Each one is stored verbatim in the
sent-document store -- form, never truth -- and a background task reads it and
proposes candidate facts. Nothing here puts anything in the corpus: only the
user confirming a fact does that, one fact at a time, on the confirmation
screen.

Screens:

  GET  /corpus         -- what has been uploaded, and how each CV's read went
  POST /corpus/upload  -- one or more .md/.txt files
  POST /corpus/paste   -- a CV pasted as text

**Fast input, slow processing.** The POSTs store bytes and enqueue; the model
call happens in the worker, on the user's own key. A request that called a
model would hold a connection for half a minute and would make uploading
thirty-three CVs unusable.

**Re-uploading changes nothing.** `add_cv` is idempotent on content, so a CV the
user already has is not stored twice and no second task is queued for it --
which matters, because a second task is a second charge on their key.

Errors are rendered on the page itself rather than passed through a redirect: a
message in a query string is a message an attacker can write, and this page
never renders text that came from one.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import RedirectResponse, Response

from jfl_web.corpus import (
    MAX_CV_CHARS,
    MAX_FILES_PER_UPLOAD,
    UploadedCv,
    UploadRejected,
    check_length,
    check_suffix,
    decode,
    extraction_failure,
)
from jfl_web.deps import (
    CandidateFactRepoDep,
    CsrfDep,
    SentDocumentRepoDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.templating import render

router = APIRouter()

CV_FACTS_KIND = "extract_cv_facts"

# A count read back off the URL after a redirect. Parsed as an integer and
# clamped, so what reaches the template is a number this code produced -- never
# text from a query string.
_MAX_REPORTED = 999


def _count(request: Request, name: str) -> int:
    raw = request.query_params.get(name)
    try:
        value = int(raw) if raw is not None else 0
    except ValueError:
        return 0
    return max(0, min(value, _MAX_REPORTED))


def _context(
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
    **extra: Any,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "session": session,
        "user": session.user,
        "cvs": cvs.list_cvs(),
        "counts": facts.counts(),
        "roles": facts.roles(),
        "max_cv_chars": MAX_CV_CHARS,
        "max_files": MAX_FILES_PER_UPLOAD,
        "extraction_failure": extraction_failure,
    }
    ctx.update(extra)
    return ctx


@router.get("/corpus")
def corpus_page(
    request: Request,
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
) -> Response:
    return render(
        request,
        "corpus.html",
        _context(
            session,
            cvs,
            facts,
            added=_count(request, "added"),
            already=_count(request, "already"),
        ),
    )


def _error(
    request: Request,
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
    message: str,
) -> Response:
    return render(
        request,
        "corpus.html",
        _context(session, cvs, facts, error=message),
        status_code=400,
    )


def _store(
    uploaded: list[UploadedCv],
    cvs: SentDocumentRepoDep,
    tasks: TaskRepoDep,
) -> tuple[int, int]:
    """Store each CV and queue a read for the ones that are new.

    Returns (added, already). A CV this user has already uploaded is stored
    once and read once: queuing a second read would charge them again on their
    own key for an answer that is already coming.
    """
    added = 0
    already = 0
    for item in uploaded:
        if cvs.find_by_content(item.text) is not None:
            already += 1
            continue
        stored = cvs.add_cv(filename=item.filename, text=item.text)
        added += 1
        tasks.enqueue(kind=CV_FACTS_KIND, payload={"sent_document_id": str(stored.id)})
    return added, already


@router.post("/corpus/upload")
async def upload_cvs(
    request: Request,
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    files: list[UploadFile] | None = None,
) -> Response:
    chosen = [f for f in (files or []) if f.filename]
    if not chosen:
        return _error(request, session, cvs, facts, "Choose at least one file.")
    if len(chosen) > MAX_FILES_PER_UPLOAD:
        return _error(
            request,
            session,
            cvs,
            facts,
            f"That is {len(chosen)} files; {MAX_FILES_PER_UPLOAD} at a time is the limit.",
        )

    uploaded: list[UploadedCv] = []
    try:
        for upload in chosen:
            name = upload.filename or "cv"
            check_suffix(name)
            uploaded.append(UploadedCv(filename=name, text=decode(name, await upload.read())))
    except UploadRejected as exc:
        # Nothing is stored when one file is rejected: a partial upload the user
        # has to reason about is worse than doing it again.
        return _error(request, session, cvs, facts, str(exc))

    added, already = _store(uploaded, cvs, tasks)
    return RedirectResponse(f"/corpus?added={added}&already={already}", status_code=303)


@router.post("/corpus/paste")
def paste_cv(
    request: Request,
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    cv_text: Annotated[str, Form()] = "",
    name: Annotated[str, Form()] = "",
) -> Response:
    label = " ".join(name.split()) or "Pasted CV"
    try:
        text = check_length(label, cv_text)
    except UploadRejected as exc:
        return _error(request, session, cvs, facts, str(exc))

    added, already = _store([UploadedCv(filename=f"{label}.txt", text=text)], cvs, tasks)
    return RedirectResponse(f"/corpus?added={added}&already={already}", status_code=303)
