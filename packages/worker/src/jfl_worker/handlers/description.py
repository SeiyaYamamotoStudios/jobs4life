"""`fetch_job_description`: read one watched-board job's full description, on
demand -- slice C7's "Track as application" button.

The button uses the model exactly as a manual paste does: it fetches the
posting's description from the platform, lazily, for this one job only, and
then hands off to the existing B3 `extract_job_ad`. This file is the first
half of that; `jfl_worker.handlers.extraction` is the second, unchanged.

**Registered `calls_model=False`.** This handler makes HTTP requests to a job
board's own API -- the same kind `check_board` makes -- and no Anthropic call.
It therefore runs even under `JFL_DISABLE_MODEL_CALLS`, which is correct: the
switch exists to stop spending a user's key, and this spends none of it. The
`extract_job_ad` task it enqueues on success is a model call and is still held
by the switch, exactly as a hand-pasted ad is.

**Not fetching twice, and not enqueueing extraction twice.** `attach_job_ad`
and the `extract_job_ad` enqueue happen in one transaction, so `job_id` on the
application is set if and only if a description has been attached and a read
is already queued or already ran. At-least-once delivery means this handler
can be run again on work that already finished -- a redelivery, or the user
pasting an ad by hand while the fetch was still in flight -- and finding
`job_id` already set is what tells it there is nothing left to do.

**Every attempt records its own outcome, and only the last one says
"failed".** A transient failure with attempts left is noted on the application
(`pending`, `description_unavailable`) so the panel says "retrying"; the last
attempt, or a permanent failure, writes `failed` before the exception goes to
the queue. So an exhausted ladder leaves the application in exactly that
terminal state instead of `pending` forever -- see `jfl_worker.queue`'s
`mark_failed` for where the ladder itself gives up.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from jfl_core.models import BoardJob, WatchedBoard
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.descriptions import fetch_description

from jfl_worker.handlers.boards import TransportFactory, polite_httpx_transport
from jfl_worker.handlers.extraction import KIND as EXTRACT_JOB_AD_KIND
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "fetch_job_description"

# Mirrors `jfl_web.jobads.MAX_AD_CHARS`. Duplicated, not imported: the worker
# does not depend on the web package, and never should -- that dependency
# would run backwards, a backend process reaching into a presentation-layer
# constant. `packages/worker/tests/test_description_handler.py` pins the two
# values equal, so drift fails a test rather than silently truncating
# differently depending on which door the ad came in.
MAX_AD_CHARS = 100_000


def _application_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("application_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no application_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # Never format the value in: `last_error` is read back in admin
        # queries and quoted into logs, same rule as `extraction._application_id`.
        raise PermanentTaskError("payload application_id is not a uuid") from None


def build_ad_text(job: BoardJob, board: WatchedBoard, description: str) -> str:
    """Header the description with what the board told us, then the text
    itself -- the same shape a hand-pasted ad has, so `extract_job_ad`
    downstream needs no special case for where the text came from.
    """
    lines = [job.title]
    if board.label:
        lines.append(board.label)
    locations = ", ".join(job.locations) if job.locations else (job.location or "")
    if locations:
        lines.append(locations)
    workplace = job.workplace_label or (None if job.workplace == "unknown" else job.workplace)
    if workplace:
        lines.append(workplace)
    if job.url:
        lines.append(job.url)
    text = "\n".join(lines) + "\n\n" + description
    return text[:MAX_AD_CHARS]


def build_fetch_job_description(*, transport_factory: TransportFactory | None = None) -> Handler:
    """Bind the handler to its transport. Tests pass a fake factory; production
    passes none and gets the same politeness-wrapped httpx client
    `check_board` uses.
    """
    factory = transport_factory or polite_httpx_transport()

    def handler(ctx: TaskContext) -> Mapping[str, object] | None:
        return _fetch_job_description(ctx, factory)

    return handler


def _fetch_job_description(
    ctx: TaskContext, transport_factory: TransportFactory
) -> Mapping[str, object] | None:
    application_id = _application_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        detail = PostgresApplicationRepository(conn, ctx.user_id).get_application(application_id)
    if detail is None:
        # No such application for this user. Nothing a retry could find.
        return {"application_id": str(application_id), "skipped": "no such application"}
    application = detail.application

    if application.job_id is not None:
        # A description is already attached -- a prior attempt succeeded, or
        # the user pasted one by hand after a failure (`POST
        # /applications/{id}/ad`, which enqueues `extract_job_ad` itself).
        # Either way there is nothing left for this handler to do, and running
        # it again must not fetch or enqueue a second time.
        return {"application_id": str(application_id), "skipped": "ad already attached"}

    if application.board_job_id is None:
        # Defensive: the web route only ever enqueues this kind for
        # applications it creates with a `board_job_id` set. Not something a
        # retry can fix.
        _fail(ctx, application_id)
        raise PermanentTaskError("application has no board_job_id")

    with ctx.engine.begin() as conn:
        boards = PostgresBoardRepository(conn, ctx.user_id)
        job = boards.get_job(application.board_job_id)
        board = boards.get_board(job.board_id) if job is not None else None
    if job is None or board is None:
        # The board job -- or the board it belonged to -- is gone. A genuine
        # delete would already have cleared `board_job_id` via its `ON DELETE
        # SET NULL`, so reaching this is the narrow window where that removal
        # happened between the read above and this one. Not something a retry
        # can fix either way.
        _fail(ctx, application_id)
        raise PermanentTaskError("the board job this application was tracked from is gone")

    with transport_factory() as transport:
        result = fetch_description(
            board.platform, board.board_key, job.external_id, job.url, transport
        )

    if result.text is None:
        if result.is_transient and not ctx.is_last_attempt:
            # A retry is coming: stay `pending` with the code noted, so the
            # panel says "retrying" rather than offering a paste box the next
            # attempt may make unnecessary.
            with ctx.engine.begin() as conn:
                PostgresApplicationRepository(conn, ctx.user_id).note_extraction_retry(
                    application_id, "description_unavailable"
                )
        else:
            _fail(ctx, application_id)
        if result.is_transient:
            # An ordinary exception, not `PermanentTaskError`: the queue's
            # backoff ladder rides this out, same as `check_board`'s
            # `BoardUnreachableError`. `_fail` above already recorded this
            # attempt, so exhausting the ladder leaves the application exactly
            # where it should be rather than spinning forever as `pending`.
            raise RuntimeError(f"description fetch unreachable: {result.error_code}")
        raise PermanentTaskError(f"description fetch failed permanently: {result.error_code}")

    ad_text = build_ad_text(job, board, result.text)
    with ctx.engine.begin() as conn:
        apps = PostgresApplicationRepository(conn, ctx.user_id)
        apps.attach_job_ad(application_id, ad_text)
        if PostgresCredentialRepository(conn, ctx.user_id).summary(ANTHROPIC_API_KEY) is None:
            # No key, so the read could only fail -- and the score chained
            # from it could never be queued. Nothing is enqueued; the
            # application says why, the same as a pasted ad added without a
            # key. The fetched ad is kept, so "Read the job ad" after adding a
            # key starts read and score without fetching again.
            apps.fail_extraction(application_id, "no_api_key")
            queued = False
        else:
            PostgresTaskRepository(conn, ctx.user_id).enqueue(
                kind=EXTRACT_JOB_AD_KIND, payload={"application_id": str(application_id)}
            )
            queued = True

    return {
        "application_id": str(application_id),
        "job_id": str(job.id),
        "requests": result.requests,
        "extraction_queued": queued,
    }


def _fail(ctx: TaskContext, application_id: uuid.UUID) -> None:
    """Record the failure on the application, in its own transaction -- same
    reasoning as `extraction._fail`: the transaction being rolled back must not
    take "something needs your attention" down with it.
    """
    with ctx.engine.begin() as conn:
        PostgresApplicationRepository(conn, ctx.user_id).fail_extraction(
            application_id, "description_unavailable"
        )
