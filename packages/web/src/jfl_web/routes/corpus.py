"""CV upload -- PLAN.md slice B6, the first half of onboarding from CVs.

A new user's corpus is too small to score or draft against, so they start by
uploading **every** CV they have. Each one is stored verbatim in the
sent-document store -- form, never truth -- and a background task reads it and
proposes candidate facts. Nothing here puts anything in the corpus: only the
user confirming a fact does that, one fact at a time, on the confirmation
screen.

The word "corpus" stays in the code and off the screens: these pages are
**Background**, and they talk about CVs and about facts you have confirmed.

Screens:

  GET  /background          -- what has been uploaded, and how each CV's read went
  POST /background/upload   -- one or more PDF/.md/.txt files
  POST /background/confirm  -- the reviewed text, as the author left it
  POST /background/paste    -- a CV pasted as text
  GET  /corpus              -- permanent redirect to /background, for bookmarks

**A PDF is shown back before it is believed.** Extraction is lossy in ways
nothing in the file announces -- two columns interleave, a header lands
mid-sentence, a ligature comes out as one character -- so a PDF upload renders
a review screen instead of storing anything. What the author leaves in the box
is what is stored, verbatim, and what every later quotation is taken from. This
is what made it possible to accept PDFs at all: the protection is not a better
extractor, it is that nothing is quoted back to somebody as their own words
until they have read it.

The reviewed text travels in the form itself rather than in a staging table.
Nothing is stored until the author says so, which is the property the screen
exists for, and a row written before that point would be a CV they never
confirmed -- visible on their own page and counted in their own totals.

**A plain-text upload does not get the screen.** There is nothing to review: a
`.md` file is the bytes the author wrote. A batch with a PDF anywhere in it
goes to review whole, because splitting one upload into two outcomes is worse
than one extra screen.

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
from jfl_core.cv_limits import MAX_CV_READ_CHARS

from jfl_web.corpus import (
    MAX_CV_CHARS,
    MAX_FILE_BYTES,
    MAX_FILES_PER_UPLOAD,
    MAX_UPLOAD_BYTES,
    UploadedCv,
    UploadRejected,
    check_length,
    check_suffix,
    check_upload_size,
    clean_filename,
    extraction_failure,
    read_note,
    read_upload,
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
_MAX_REPORTED = 9_999


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
        "max_read_chars": MAX_CV_READ_CHARS,
        "max_files": MAX_FILES_PER_UPLOAD,
        "max_file_mb": MAX_FILE_BYTES // (1024 * 1024),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "extraction_failure": extraction_failure,
        "read_note": read_note,
    }
    ctx.update(extra)
    return ctx


@router.get("/background")
def background_page(
    request: Request,
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
) -> Response:
    return render(
        request,
        "background.html",
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
        "background.html",
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


def _done(added: int, already: int) -> Response:
    return RedirectResponse(f"/background?added={added}&already={already}", status_code=303)


@router.post("/background/upload")
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
    total = 0
    try:
        for upload in chosen:
            name = clean_filename(upload.filename or "cv")
            check_suffix(name)
            raw = await upload.read()
            total += len(raw)
            check_upload_size(name, raw, total)
            uploaded.append(read_upload(name, raw))
    except UploadRejected as exc:
        # Nothing is stored when one file is rejected: a partial upload the user
        # has to reason about is worse than doing it again.
        return _error(request, session, cvs, facts, str(exc))

    if not any(item.extracted for item in uploaded):
        # Plain text: the bytes are the author's own words already, so there is
        # nothing for a review screen to protect against.
        return _done(*_store(uploaded, cvs, tasks))

    return render(
        request,
        "background_review.html",
        _context(session, cvs, facts, review=uploaded),
    )


@router.post("/background/confirm")
def confirm_cvs(
    request: Request,
    session: SessionDep,
    cvs: SentDocumentRepoDep,
    facts: CandidateFactRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    filenames: Annotated[list[str] | None, Form()] = None,
    texts: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """Store the text the author confirmed, exactly as they left it.

    The two lists are parallel and come from the review form in document order.
    Nothing here re-reads the file or re-runs extraction: an edit the author
    made is the point, so what is stored is the box, not the PDF.
    """
    names = filenames or []
    bodies = texts or []
    if not bodies or len(bodies) != len(names):
        return _error(
            request,
            session,
            cvs,
            facts,
            "That upload could not be confirmed. Choose the files again.",
        )

    confirmed: list[UploadedCv] = []
    try:
        for filename, text in zip(names, bodies, strict=True):
            name = clean_filename(filename)
            confirmed.append(UploadedCv(filename=name, text=check_length(name, text)))
    except UploadRejected as exc:
        return _error(request, session, cvs, facts, str(exc))

    return _done(*_store(confirmed, cvs, tasks))


@router.post("/background/paste")
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

    return _done(*_store([UploadedCv(filename=f"{label}.txt", text=text)], cvs, tasks))


# -- the old paths -------------------------------------------------------------
#
# These screens lived under /corpus until the word came off the UI. A bookmark
# is not a reason to keep a word nobody uses, and a 404 is a bad way to learn it
# changed -- so the old paths answer 301 and nothing else. No session is
# required: this is a path, not a page, and it renders nothing.


@router.get("/corpus", include_in_schema=False)
def legacy_corpus_path() -> Response:
    return RedirectResponse("/background", status_code=301)
