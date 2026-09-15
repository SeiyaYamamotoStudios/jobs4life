"""Suggested title expansions for a saved filter's title includes -- slice C7a.

See CLAUDE.md's 2026-09-15 decision and PLAN.md's C7a. When the user adds a
title phrase to their saved filter, a cheap Claude Haiku 4.5 call
(`jfl_generate.titles`) suggests adjacent titles through the background queue,
on the user's own key, once per phrase, under the model kill switch. They are
never added to the filter silently: each suggestion carries a tickbox, and only
ticked titles become match terms.

Enqueueing itself lives in `jfl_web.routes.jobs.save_filter`, not here -- a
suggestion call is a side effect of saving the filter, not its own screen.

Screens:

  GET  /jobs/filter/titles/{id}          -- one row, for htmx to poll while pending
  POST /jobs/filter/titles/{id}/accept   -- add ticked titles to title_includes
  POST /jobs/filter/titles/{id}/dismiss  -- hide the row
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response

from jfl_web.deps import CsrfDep, JobFilterRepoDep, SessionDep, TitleSuggestionRepoDep
from jfl_web.jobfilter import MAX_FILTER_TEXT, FormTooLongError, checked_text
from jfl_web.templating import render
from jfl_web.titlesuggestions import split_phrases, suggestion_row_view

router = APIRouter()

_NOT_FOUND = "No suggestion found -- it may belong to another account."


def _error(request: Request, session: SessionDep, message: str, status_code: int) -> Response:
    return render(
        request,
        "error.html",
        {"session": session, "user": session.user, "message": message},
        status_code=status_code,
    )


@router.get("/jobs/filter/titles/{suggestion_id}")
def title_suggestion_row(
    request: Request,
    suggestion_id: uuid.UUID,
    session: SessionDep,
    suggestions: TitleSuggestionRepoDep,
    filters: JobFilterRepoDep,
) -> Response:
    """The one row, standalone -- what `_title_suggestions.html`'s pending rows
    poll. Same view-building function the full panel uses
    (`jfl_web.titlesuggestions.suggestion_row_view`), so the two can never
    render a phrase's state differently.
    """
    row = suggestions.get(suggestion_id)
    if row is None or row.dismissed_at is not None:
        return _error(request, session, _NOT_FOUND, 404)
    saved = filters.get_filter()
    view = suggestion_row_view(row.phrase, row, saved.title_includes)
    return render(
        request,
        "_title_suggestion_row.html",
        {
            "session": session,
            "phrase": view.phrase,
            "row": view.row,
            "failure": view.failure,
            "visible": view.visible,
        },
    )


@router.post("/jobs/filter/titles/{suggestion_id}/accept")
def accept_title_suggestions(
    request: Request,
    suggestion_id: uuid.UUID,
    session: SessionDep,
    suggestions: TitleSuggestionRepoDep,
    filters: JobFilterRepoDep,
    _csrf: CsrfDep,
    title: Annotated[list[str] | None, Form()] = None,
) -> Response:
    """Add only titles that are actually in this row's own stored suggestions
    -- anything else submitted is ignored rather than trusted. Rejects (never
    truncates) if the merged text would exceed `MAX_FILTER_TEXT`, the same rule
    `save_filter` applies to a typed filter.
    """
    row = suggestions.get(suggestion_id)
    if row is None:
        return _error(request, session, _NOT_FOUND, 404)

    offered = {s.title for s in row.suggestions}
    ticked = [t for t in (title or []) if t in offered]
    if not ticked:
        return RedirectResponse("/jobs", status_code=303)

    saved = filters.get_filter()
    existing = split_phrases(saved.title_includes)
    merged = existing + [t for t in ticked if t not in existing]
    combined = ", ".join(merged)
    try:
        checked_text(combined, MAX_FILTER_TEXT)
    except FormTooLongError as exc:
        return _error(request, session, str(exc), 400)

    filters.save_filter(
        workplaces=saved.workplaces,
        title_includes=combined,
        title_excludes=saved.title_excludes,
        location=saved.location,
    )
    return RedirectResponse("/jobs?status=titles_added", status_code=303)


@router.post("/jobs/filter/titles/{suggestion_id}/dismiss")
def dismiss_title_suggestion(
    request: Request,
    suggestion_id: uuid.UUID,
    session: SessionDep,
    suggestions: TitleSuggestionRepoDep,
    _csrf: CsrfDep,
) -> Response:
    if not suggestions.dismiss(suggestion_id):
        return _error(request, session, _NOT_FOUND, 404)
    return RedirectResponse("/jobs?status=title_suggestion_dismissed", status_code=303)
