"""The "what changed" feed: new, gone, returned and reposted jobs on watched boards,
since the user last looked (PLAN.md C7).

Screens:

  GET  /changes                      -- the feed
  POST /changes/{mark_id}/dismiss    -- dismiss one event
  POST /changes/dismiss-all          -- dismiss everything currently shown

**Viewing is the act of looking, so the GET writes -- deliberately.** "Since you
last looked" needs a record of looking, and the only honest record is the page
being rendered: a separate "mark as read" button is a step that eventually does
not get pressed, and then the feed shows yesterday's news as today's. So one GET,
in the request's one transaction: derive events, filter them, apply the
visibility rule (`jfl_intake.feed`), mark the newly shown events as first seen
now, and move the last-looked time to now. A GET that fails rolls all of it back,
so an error page can never count as having looked. The write is idempotent in
effect -- reloading keeps every event's original first-seen time, so the 24 hours
never restart.

**Only what is displayed is marked.** An event the saved filter hides gets no
mark; once the last-looked time passes it, it is not news for this user. That is
the filter doing its job, and the summary still says how many were filtered out
-- "N changes matching your filter of M", never N alone, as on `/jobs`.

Events come from `PostgresBoardRepository.events_since`, which never yields a
baseline's jobs (C4) and nothing from a check that was not complete (C3). Nothing
here re-derives either rule.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.models import BoardJob, BoardJobEvent, JobFeedMark
from jfl_core.storage.credentials import ANTHROPIC_API_KEY
from jfl_intake.feed import (
    FIRST_VISIT_LOOKBACK,
    VISIBLE_FOR,
    derive_since,
    event_key,
    mark_key,
    visible_events,
)
from jfl_intake.filtering import Match

from jfl_web.boards import platform_label
from jfl_web.deps import (
    ApplicationRepoDep,
    BoardRepoDep,
    CredentialRepoDep,
    CsrfDep,
    JobFeedRepoDep,
    JobFilterRepoDep,
    SessionDep,
)
from jfl_web.jobfilter import filter_open_jobs
from jfl_web.scores import track_context
from jfl_web.templating import render

router = APIRouter()

_MARK_NOT_FOUND = "No change found -- it may belong to another account."

# `?status=` is looked up in this closed set; the query string's text is never rendered.
_STATUS_MESSAGES = {
    "dismissed": "Dismissed.",
    "dismissed_all": "Dismissed everything that was shown.",
}


@dataclass(frozen=True, slots=True)
class FeedRow:
    event: BoardJobEvent
    match: Match[BoardJob]
    mark: JobFeedMark | None
    # True if this user had already been shown the event before this view.
    seen_before: bool


@router.get("/changes")
def list_changes(
    request: Request,
    session: SessionDep,
    boards: BoardRepoDep,
    filters: JobFilterRepoDep,
    feed: JobFeedRepoDep,
    applications: ApplicationRepoDep,
    credentials: CredentialRepoDep,
) -> Response:
    now = dt.datetime.now(dt.UTC)
    show_unstated = request.query_params.get("show_unstated") == "1"
    previous_look = feed.last_looked_at()

    since = derive_since(previous_look, feed.live_marks(now=now, visible_for=VISIBLE_FOR), now=now)
    items = visible_events(
        boards.events_since(since),
        feed.marks_for_events_after(since),
        last_looked_at=previous_look,
        now=now,
    )

    # One entry per event, not per job: a job that went and came back is two
    # changes, and the counts on the page are counts of changes.
    all_boards = boards.list_boards()
    result = filter_open_jobs(
        [item.event.job for item in items],
        filters.get_filter(),
        all_boards,
        filters.list_exceptions(),
        show_hidden_unstated=show_unstated,
    )
    match_by_job = {m.job.id: m for m in result.matches}
    shown = [item for item in items if item.event.job.id in match_by_job]

    recorded = feed.record_seen([i.event for i in shown if i.mark is None], now=now)
    new_marks = {mark_key(m): m for m in recorded}
    rows = [
        FeedRow(
            event=item.event,
            match=match_by_job[item.event.job.id],
            mark=item.mark or new_marks.get(event_key(item.event)),
            seen_before=item.mark is not None,
        )
        for item in shown
    ]
    feed.set_last_looked_at(now)

    return render(
        request,
        "changes.html",
        {
            "session": session,
            "user": session.user,
            "rows": rows,
            "event_total": len(items),
            "hidden_unstated": result.hidden_unstated,
            "show_unstated": show_unstated,
            "previous_look": previous_look,
            "first_visit_days": FIRST_VISIT_LOOKBACK.days,
            "visible_hours": int(VISIBLE_FOR.total_seconds() // 3600),
            # Slice C7: the same "Track as application" button /jobs shows, so a
            # change is actionable where it is read.
            "tracked": applications.tracked_board_jobs([r.event.job.id for r in rows]),
            **track_context(credentials.summary(ANTHROPIC_API_KEY) is not None),
            "board_by_id": {b.id: b for b in all_boards},
            "board_count": len(all_boards),
            "platform_label": platform_label,
            "status_message": _STATUS_MESSAGES.get(request.query_params.get("status", "")),
        },
    )


@router.post("/changes/{mark_id}/dismiss")
def dismiss_change(
    request: Request,
    mark_id: uuid.UUID,
    session: SessionDep,
    feed: JobFeedRepoDep,
    _csrf: CsrfDep,
) -> Response:
    if not feed.dismiss(mark_id, now=dt.datetime.now(dt.UTC)):
        return render(
            request,
            "error.html",
            {"session": session, "user": session.user, "message": _MARK_NOT_FOUND},
            status_code=404,
        )
    return RedirectResponse("/changes?status=dismissed", status_code=303)


@router.post("/changes/dismiss-all")
def dismiss_all_changes(feed: JobFeedRepoDep, _csrf: CsrfDep) -> Response:
    """Every event still on the page. See `PostgresJobFeedRepository.dismiss_live`
    for why an event that arrived after the page was rendered survives this.
    """
    feed.dismiss_live(now=dt.datetime.now(dt.UTC), visible_for=VISIBLE_FOR)
    return RedirectResponse("/changes?status=dismissed_all", status_code=303)
