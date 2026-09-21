"""Watched job boards -- the screen that makes the watching engine reachable.

The engine (`jfl_intake`, `jfl_core.storage.boards`, the worker's `check_board`
handler) already runs daily checks; nothing before this screen could add a
board, so nothing was being watched. Deliberately minimal, per the brief: paste
a URL, see what is watched and the status of its last check, check one now,
stop watching one. **The rich "what changed" feed (new / gone / returned /
reposted) is out of scope** -- its design is still being settled with the
owner -- so this file never touches `events_for_check` or `events_since`.

**History cannot be backfilled**, which is why this screen is urgent: every day
a board is not added is a day of history that can never be recovered (see
CLAUDE.md's 2026-09-10 decision log). It is also why "adding a board" and
"the first check" are two different acts here: `add_board` is a fast database
write with no network call, and the fetch itself happens in the worker via
`enqueue_board_check` -- exactly the split slice B3 already established for
reading a job ad, and for the same reason (a ~2-minute Workday check must never
run in a request).

Screens:

  GET  /boards               -- the list, and the add-a-board form above it
  POST /boards                -- detect the platform, add it, queue its baseline
  POST /boards/check-all      -- "Check all boards now"
  GET  /boards/{id}           -- one board: open jobs, recent checks
  POST /boards/{id}/check     -- "check now"
  POST /boards/{id}/remove    -- stop watching (with a confirm step in the form)

`check_all` is not a new check mechanism: it calls the same
`enqueue_board_check` "check now" does, once per watched board, through
`jfl_intake.scheduling.enqueue_all_board_checks` -- so a board already
mid-check is skipped by the exact rule that already governs one board, and
the staggering a worker restart's backlog already gets is what spreads the
enqueue across `scheduled_at` rather than firing every board "now". See that
function's docstring for why staggering matters even though the worker runs
one task at a time.

No model call anywhere in this file, and no HTTP request to a board's own site
either -- `detect_board` is pattern matching against the pasted URL, nothing
more. The fetch belongs to the worker, which is the whole reason the "must not
make an HTTP request in the request path" rule is trivially true here rather
than something this file has to be careful about.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import BoardCheck, WatchedBoard
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_intake.detect import BoardRef, BoardUrlError, detect_board
from jfl_intake.scheduling import CHECK_BOARD_KIND, enqueue_all_board_checks, enqueue_board_check

from jfl_web.boards import default_label, platform_label
from jfl_web.deps import (
    ApplicationRepoDep,
    BoardRepoDep,
    CsrfDep,
    JobFilterRepoDep,
    SessionDep,
    TaskRepoDep,
)
from jfl_web.jobfilter import (
    MAX_FILTER_TEXT,
    MAX_NOTE_TEXT,
    WORKPLACE_NAMES,
    BoardMatchCount,
    match_counts_by_board,
    unstated_setting_view,
)
from jfl_web.templating import render

router = APIRouter()

# Same message whether the id never existed or belongs to another user -- see
# applications.py's `_NOT_FOUND` for why distinguishing the two is a tenancy
# leak in miniature.
_NOT_FOUND = "No board found -- it may belong to another account."


def _existing_board(boards: PostgresBoardRepository, ref: BoardRef) -> WatchedBoard | None:
    """Is this board already watched? `add_board` is itself idempotent (a
    unique index on `user_id, platform, board_key`), but it gives no way to
    tell "just created" from "already there" -- and only the former should
    queue a baseline check. So this checks first, against the same key
    `add_board` would use.
    """
    for board in boards.list_boards():
        if board.platform == ref.platform and board.board_key == ref.board_key:
            return board
    return None


def _board_view(
    boards: PostgresBoardRepository,
    board: WatchedBoard,
    match_count: BoardMatchCount | None = None,
) -> dict[str, object]:
    """Everything one row (or the detail page's header) needs about one board.

    `open_job_count` is None, not 0, when the board has no baseline yet --
    "first check pending" and "zero jobs open" are different facts and the
    template must not conflate them. It is the true open count even while the
    board is held: a held check changes no job's state, so `list_jobs` already
    reflects reality; `held` is reported alongside as its own flag rather than
    substituted for the count.

    `match_count` is "N of M match" under the saved job filter -- the same
    computation /jobs uses, so the two pages can never disagree.

    `checking` is whether a check of this board is pending or running right
    now -- from "check now", the daily schedule, or "check all boards". No
    spinner that lies: this is the same read `enqueue_board_check` itself
    uses to decide whether to skip, so it can never claim a board is checking
    when nothing is actually queued.
    """
    checks = boards.list_checks(board.id, limit=1)
    open_job_count = (
        None if board.baseline_check_id is None else len(boards.list_jobs(board.id, open_only=True))
    )
    return {
        "board": board,
        "platform_label": platform_label(board.platform),
        "last_check": checks[0] if checks else None,
        "open_job_count": open_job_count,
        "held": board.held_check_id is not None,
        "checking": boards.check_task_queued(board.id, kind=CHECK_BOARD_KIND),
        "match_count": match_count,
    }


def _list_context(
    request: Request,
    session: SessionDep,
    boards: PostgresBoardRepository,
    filters: JobFilterRepoDep,
    **extra: object,
) -> dict[str, object]:
    all_boards = boards.list_boards()
    counts = match_counts_by_board(
        boards.list_open_jobs(), filters.get_filter(), all_boards, filters.list_exceptions()
    )
    return {
        "session": session,
        "user": session.user,
        "boards": [_board_view(boards, b, counts.get(b.id)) for b in all_boards],
        **extra,
    }


def _count_param(request: Request, name: str) -> int:
    """A bounded, digit-only count off the query string -- never free text.

    `str.isdigit()` admits only the characters `0`-`9` (no sign, no
    whitespace, no HTML), so there is nothing here for the query string to
    inject; anything else, including absence, reads as zero. The cap matches
    nothing about board counts specifically -- it exists only so a
    hand-edited URL cannot make the template render an ungainly number.
    """
    raw = request.query_params.get(name)
    if raw is None or not raw.isdigit():
        return 0
    return min(int(raw), 100_000)


@router.get("/boards")
def list_boards(
    request: Request, session: SessionDep, boards: BoardRepoDep, filters: JobFilterRepoDep
) -> Response:
    return render(
        request,
        "boards_list.html",
        _list_context(
            request,
            session,
            boards,
            filters,
            checked_status=request.query_params.get("status"),
            check_all_queued=_count_param(request, "queued"),
            check_all_skipped=_count_param(request, "skipped"),
        ),
    )


@router.post("/boards")
def add_board(
    request: Request,
    session: SessionDep,
    boards: BoardRepoDep,
    filters: JobFilterRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    url: Annotated[str, Form()],
) -> Response:
    """Detect the platform, add it if it is new, and queue the baseline check.

    A `BoardUrlError` carries its own user-facing message -- LinkedIn/Indeed
    refused by design, an unrecognised site listing what is supported, a
    malformed URL saying so -- so the three failure branches in the brief are
    one branch here, and the message is `detect_board`'s to own.
    """
    try:
        ref = detect_board(url)
    except BoardUrlError as exc:
        return render(
            request,
            "boards_list.html",
            _list_context(request, session, boards, filters, error=str(exc), url_value=url),
            status_code=400,
        )

    existing = _existing_board(boards, ref)
    if existing is not None:
        return RedirectResponse("/boards?status=already_watched", status_code=303)

    board = boards.add_board(
        platform=ref.platform,
        board_url=ref.board_url,
        board_key=ref.board_key,
        label=default_label(ref),
    )
    # The baseline starts now, or it never truly starts -- see the module
    # docstring. `enqueue_board_check` is a no-op only if a check is already
    # queued, which cannot be true for a board that did not exist a moment ago.
    enqueue_board_check(boards, tasks, board.id)
    return RedirectResponse("/boards?status=added", status_code=303)


@router.post("/boards/check-all")
def check_all(
    request: Request,
    session: SessionDep,
    boards: BoardRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Queue a check of every watched board at once. See the module
    docstring: this enqueues through `enqueue_all_board_checks`, which is
    `enqueue_board_check` in a loop, so a board with a check already pending
    or running is skipped exactly as it would be from its own "check now".

    An empty board list queues nothing and reports that honestly rather than
    pretending a check was requested.
    """
    board_ids = [board.id for board in boards.list_boards()]
    queued, skipped = enqueue_all_board_checks(
        boards, tasks, board_ids, now=dt.datetime.now(dt.UTC)
    )
    return RedirectResponse(
        f"/boards?status=check_all&queued={queued}&skipped={skipped}", status_code=303
    )


@router.get("/boards/{board_id}")
def board_detail(
    request: Request,
    board_id: uuid.UUID,
    session: SessionDep,
    boards: BoardRepoDep,
    filters: JobFilterRepoDep,
    applications: ApplicationRepoDep,
) -> Response:
    board = boards.get_board(board_id)
    if board is None:
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _NOT_FOUND},
            status_code=404,
        )
    open_jobs = (
        None if board.baseline_check_id is None else boards.list_jobs(board_id, open_only=True)
    )
    checks: list[BoardCheck] = boards.list_checks(board_id, limit=10)
    return render(
        request,
        "board_detail.html",
        {
            "session": session,
            "user": session.user,
            "board": board,
            "platform_label": platform_label(board.platform),
            "held": board.held_check_id is not None,
            "open_jobs": open_jobs,
            "checks": checks,
            "checked_status": request.query_params.get("status"),
            "unstated": unstated_setting_view(board),
            "exceptions": filters.list_exceptions(board_id),
            "workplace_names": WORKPLACE_NAMES,
            "max_filter_text": MAX_FILTER_TEXT,
            "max_note_text": MAX_NOTE_TEXT,
            # Slice C7: "Track as application" renders as "Tracked" for a job
            # that already has a live application from it.
            "tracked": applications.tracked_board_jobs(
                [j.id for j in open_jobs] if open_jobs else []
            ),
        },
    )


@router.post("/boards/{board_id}/check")
def check_now(
    request: Request,
    board_id: uuid.UUID,
    session: SessionDep,
    boards: BoardRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    next: Annotated[str, Form()] = "list",
) -> Response:
    """`next` says which page asked -- "list" (default, the boards screen) or
    "detail" -- so the redirect lands back where the button was pressed rather
    than always at the list. A closed set of two values, not a URL, so there is
    nothing here an open redirect could be built from.
    """
    if boards.get_board(board_id) is None:
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _NOT_FOUND},
            status_code=404,
        )
    task = enqueue_board_check(boards, tasks, board_id)
    status = "queued" if task is not None else "already_queued"
    destination = f"/boards/{board_id}" if next == "detail" else "/boards"
    return RedirectResponse(f"{destination}?status={status}", status_code=303)


@router.post("/boards/{board_id}/remove")
def remove_board(
    request: Request,
    board_id: uuid.UUID,
    session: SessionDep,
    boards: BoardRepoDep,
    _csrf: CsrfDep,
) -> Response:
    """Stop watching. The confirmation step lives in the form (a collapsed
    `<details>` the user has to open before the button that submits this is
    even visible) rather than as a second route -- there is nothing here to
    confirm that isn't already confirmed by the extra click it took to reach
    this button.
    """
    if not boards.remove_board(board_id):
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _NOT_FOUND},
            status_code=404,
        )
    return RedirectResponse("/boards?status=removed", status_code=303)
