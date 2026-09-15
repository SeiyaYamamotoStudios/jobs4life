"""`POST /jobs/{board_job_id}/track` -- slice C7's "Track as application"
button.

Turns a watched-board job into an application "just like how it would be added
manually via copy/paste" (the owner's words): the row is created here, fast,
with no network call and no model call, and a `fetch_job_description` task is
enqueued in the same transaction to fetch the posting's description lazily,
which in turn enqueues the existing B3 `extract_job_ad` once it has one. Fast
input, slow processing -- the same split `applications.py` already established
for a pasted ad, applied to a different door in.

**Nothing is fetched or extracted for a job the user only browses.** This
route is the one and only place that changes -- pressing the button is the
explicit action; landing on `/jobs` or a board's page is not.

A router of its own rather than a route added to `jobs.py` or `applications.py`:
the entry point is a *job*, not an application, which rules out
`applications.py`, and `jobs.py` is where the parallel workplace-preset and
title-suggestion slices are landing, which made it the wrong place to add a
POST route neither of them touches.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from jfl_web.deps import ApplicationRepoDep, BoardRepoDep, CsrfDep, SessionDep, TaskRepoDep
from jfl_web.templating import render

router = APIRouter()

# The worker's kind for "fetch this job's description off its board". A
# string on both sides, on purpose -- same reasoning as
# `applications.EXTRACT_JOB_AD_KIND`: importing `jfl_worker` here would make
# the web container carry the worker, and the queue's whole point is that the
# two deploy separately.
FETCH_JOB_DESCRIPTION_KIND = "fetch_job_description"

# Same message whether the job never existed or belongs to another user -- see
# `applications._NOT_FOUND` for why distinguishing the two is a tenancy leak
# in miniature.
_NOT_FOUND = "No job found -- it may belong to another account."


@router.post("/jobs/{board_job_id}/track")
def track_as_application(
    request: Request,
    board_job_id: uuid.UUID,
    session: SessionDep,
    boards: BoardRepoDep,
    applications: ApplicationRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
) -> Response:
    job = boards.get_job(board_job_id)
    if job is None:
        context = {"session": session, "user": session.user, "message": _NOT_FOUND}
        return render(request, "error.html", context, status_code=404)

    existing = applications.find_live_by_board_job(board_job_id)
    if existing is not None:
        return RedirectResponse(f"/applications/{existing}", status_code=303)

    board = boards.get_board(job.board_id)
    application = applications.create_application(
        # The employer's own words, not a placeholder -- `job.title` is what
        # the board actually published, unlike `provisional_title`'s guess off
        # the top of a pasted ad. `title_is_provisional` stays False, so
        # `finish_extraction` will never overwrite it; only `employer`, which
        # is genuinely a guess (the board's label, not the employer's own
        # copy), stays open to being filled in once extraction reads the ad.
        title=job.title,
        employer=board.label if board is not None else None,
        url=job.url,
        source="Watched board",
        extraction_status="pending",
        board_job_id=job.id,
    )
    tasks.enqueue(
        kind=FETCH_JOB_DESCRIPTION_KIND,
        # Ids only, same rule as every other task payload in this app: the
        # description is fetched, and the API key is unsealed, in the worker.
        payload={"application_id": str(application.id)},
    )
    return RedirectResponse(f"/applications/{application.id}", status_code=303)
