"""Every matching open job across all watched boards, in one list -- and the
filter, board setting and board exceptions that decide "matching".

**Filtering is a lens over stored data, never a fetch parameter.** Boards are
watched whole; nothing here touches a board's history or makes a request to a
board's site. Matching is `jfl_intake.filtering` (pure); this file only reads,
saves and renders.

**The page never hides how much it filtered out.** The summary is always
"N matching of M open jobs across B boards", never N alone; jobs hidden only
because their workplace is not stated are counted, with a one-click "show them"
for that view; a board with no complete check yet is named ("first check
pending") rather than silently absent; and a held board is named too. Rows say
"seen since" -- when we first saw the job -- never "posted", which we do not know.

Screens:

  GET  /jobs                                           -- the list, filter form on top
  POST /jobs/filter                                    -- save the filter
  POST /boards/{id}/unstated                           -- the board's include-unstated setting
  POST /boards/{id}/hybrid                             -- the board's hybrid under remote friendly
  POST /boards/{id}/exceptions                         -- add a board exception
  POST /boards/{id}/exceptions/{exception_id}          -- edit one
  POST /boards/{id}/exceptions/{exception_id}/remove   -- remove one

The exception routes live here rather than in `boards.py` because they are part
of the filter, even though the forms sit on the board's page. An exception's
note is stored exactly as submitted -- no model, no tidying.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response

from jfl_web.boards import platform_label
from jfl_web.deps import BoardRepoDep, CsrfDep, JobFilterRepoDep, SessionDep
from jfl_web.jobfilter import (
    JOBS_PAGE_CAP,
    MAX_FILTER_TEXT,
    MAX_NOTE_TEXT,
    WORKPLACE_MODE_NAMES,
    WORKPLACE_NAMES,
    FormTooLongError,
    checked_text,
    filter_open_jobs,
    parse_workplace_mode,
    parse_workplaces,
)
from jfl_web.templating import render

router = APIRouter()

_BOARD_NOT_FOUND = "No board found -- it may belong to another account."
_EXCEPTION_NOT_FOUND = "No exception found -- it may belong to another account."


def _error(request: Request, session: SessionDep, message: str, status_code: int) -> Response:
    return render(
        request,
        "error.html",
        {"session": session, "user": session.user, "message": message},
        status_code=status_code,
    )


@router.get("/jobs")
def list_jobs(
    request: Request,
    session: SessionDep,
    boards: BoardRepoDep,
    filters: JobFilterRepoDep,
) -> Response:
    show_unstated = request.query_params.get("show_unstated") == "1"
    saved = filters.get_filter()
    all_boards = boards.list_boards()
    board_by_id = {b.id: b for b in all_boards}
    exceptions = filters.list_exceptions()
    result = filter_open_jobs(
        boards.list_open_jobs(),
        saved,
        all_boards,
        exceptions,
        show_hidden_unstated=show_unstated,
    )
    return render(
        request,
        "jobs_list.html",
        {
            "session": session,
            "user": session.user,
            "saved": saved,
            "workplace_names": WORKPLACE_NAMES,
            "workplace_mode_names": WORKPLACE_MODE_NAMES,
            "result": result,
            "rows": result.matches[:JOBS_PAGE_CAP],
            "capped": len(result.matches) > JOBS_PAGE_CAP,
            "cap": JOBS_PAGE_CAP,
            "board_by_id": board_by_id,
            "checked_board_count": sum(1 for b in all_boards if b.baseline_check_id is not None),
            "pending_boards": [b for b in all_boards if b.baseline_check_id is None],
            "held_boards": [
                b for b in all_boards if b.held_check_id is not None and b.baseline_check_id
            ],
            "has_exceptions": bool(exceptions),
            "show_unstated": show_unstated,
            "platform_label": platform_label,
            "checked_status": request.query_params.get("status"),
            "max_filter_text": MAX_FILTER_TEXT,
        },
    )


@router.post("/jobs/filter")
def save_filter(
    request: Request,
    session: SessionDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    workplace: Annotated[list[str] | None, Form()] = None,
    title_includes: Annotated[str, Form()] = "",
    title_excludes: Annotated[str, Form()] = "",
    location: Annotated[str, Form()] = "",
    workplace_mode: Annotated[str, Form()] = "custom",
) -> Response:
    mode = parse_workplace_mode(workplace_mode)
    if mode is None:
        return _error(request, session, "That workplace choice is not one of the options.", 400)
    try:
        filters.save_filter(
            workplace_mode=mode,
            workplaces=parse_workplaces(workplace or []),
            title_includes=checked_text(title_includes, MAX_FILTER_TEXT),
            title_excludes=checked_text(title_excludes, MAX_FILTER_TEXT),
            location=checked_text(location, MAX_FILTER_TEXT),
        )
    except FormTooLongError as exc:
        return _error(request, session, str(exc), 400)
    return RedirectResponse("/jobs?status=saved", status_code=303)


@router.post("/boards/{board_id}/unstated")
def set_include_unstated(
    request: Request,
    board_id: uuid.UUID,
    session: SessionDep,
    boards: BoardRepoDep,
    _csrf: CsrfDep,
    setting: Annotated[str, Form()] = "default",
) -> Response:
    """`setting` is one of a closed set: `default`, `include`, `hide`."""
    value = {"include": True, "hide": False}.get(setting)
    if setting not in ("default", "include", "hide"):
        return _error(request, session, "That setting is not one of the choices.", 400)
    if not boards.set_include_unstated_workplace(board_id, value):
        return _error(request, session, _BOARD_NOT_FOUND, 404)
    return RedirectResponse(f"/boards/{board_id}?status=setting_saved", status_code=303)


@router.post("/boards/{board_id}/hybrid")
def set_hybrid_too_heavy(
    request: Request,
    board_id: uuid.UUID,
    session: SessionDep,
    boards: BoardRepoDep,
    _csrf: CsrfDep,
    setting: Annotated[str, Form()] = "include",
) -> Response:
    """`setting` is one of a closed set: `include` (hybrid shown under remote
    friendly, badged "days not stated") or `too_heavy` (left out of it).
    """
    if setting not in ("include", "too_heavy"):
        return _error(request, session, "That setting is not one of the choices.", 400)
    if not boards.set_hybrid_too_heavy(board_id, setting == "too_heavy"):
        return _error(request, session, _BOARD_NOT_FOUND, 404)
    return RedirectResponse(f"/boards/{board_id}?status=setting_saved", status_code=303)


@router.post("/boards/{board_id}/exceptions")
def add_exception(
    request: Request,
    board_id: uuid.UUID,
    session: SessionDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    workplace: Annotated[list[str] | None, Form()] = None,
    location: Annotated[str, Form()] = "",
    note: Annotated[str, Form()] = "",
) -> Response:
    try:
        added = filters.add_exception(
            board_id,
            workplaces=parse_workplaces(workplace or []),
            location=checked_text(location, MAX_FILTER_TEXT),
            note=checked_text(note, MAX_NOTE_TEXT),
        )
    except FormTooLongError as exc:
        return _error(request, session, str(exc), 400)
    if added is None:
        return _error(request, session, _BOARD_NOT_FOUND, 404)
    return RedirectResponse(f"/boards/{board_id}?status=exception_added", status_code=303)


@router.post("/boards/{board_id}/exceptions/{exception_id}")
def edit_exception(
    request: Request,
    board_id: uuid.UUID,
    exception_id: uuid.UUID,
    session: SessionDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    workplace: Annotated[list[str] | None, Form()] = None,
    location: Annotated[str, Form()] = "",
    note: Annotated[str, Form()] = "",
) -> Response:
    existing = filters.get_exception(exception_id)
    if existing is None or existing.board_id != board_id:
        return _error(request, session, _EXCEPTION_NOT_FOUND, 404)
    try:
        filters.update_exception(
            exception_id,
            workplaces=parse_workplaces(workplace or []),
            location=checked_text(location, MAX_FILTER_TEXT),
            note=checked_text(note, MAX_NOTE_TEXT),
        )
    except FormTooLongError as exc:
        return _error(request, session, str(exc), 400)
    return RedirectResponse(f"/boards/{board_id}?status=exception_saved", status_code=303)


@router.post("/boards/{board_id}/exceptions/{exception_id}/remove")
def remove_exception(
    request: Request,
    board_id: uuid.UUID,
    exception_id: uuid.UUID,
    session: SessionDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
) -> Response:
    existing = filters.get_exception(exception_id)
    if existing is None or existing.board_id != board_id:
        return _error(request, session, _EXCEPTION_NOT_FOUND, 404)
    filters.remove_exception(exception_id)
    return RedirectResponse(f"/boards/{board_id}?status=exception_removed", status_code=303)
