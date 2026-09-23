"""`extract_job_ad`: read a pasted job ad, in the background, on the user's key.

Slice B3, and the first handler that spends money. The form the user submits is
a paste box and nothing else; it returns immediately with a provisional title
and a task in the queue, and this is what turns that paste into an employer, a
title and a list of requirements. Fast input, slow processing -- the owner's
words, and the whole reason the queue exists.

**Registered with `calls_model=True`.** That is what puts it behind
`JFL_DISABLE_MODEL_CALLS`, and getting it wrong would make the incident lever a
lie: the switch is what stops the spending when a key starts burning money.

Four things this file is careful about.

**The key is never data.** It is fetched from `user_credentials`, unsealed with
the master key from the worker's environment, held in one local, handed to
`extract_requirements`, and dropped. It is not in the task payload -- the
payload is one id -- not in a log line, not in `tasks.last_error`, not in the
`runs` row, and not in the `applications` row this handler updates. The
application's failure column takes a **code from a closed set**, never a
formatted exception, precisely because a formatted exception is where a
credential leaks.

**Not extracting twice for one paste.** Delivery is at-least-once, so this can
be run again on work that already finished; `claim_extraction` returns None for
an application whose extraction is already `done`, and the handler returns
without calling anything. A genuine re-read is a button the user presses, which
enqueues a new task -- explicit, because it is their money.

**Some failures do not deserve a retry.** No key stored, a key Anthropic
rejects, an ad too long for one call, a refusal: nothing changes between
attempts, and three of them would buy two more charges and a worse message.
Those raise `PermanentTaskError`. Transient trouble -- a 429, a 5xx, a dropped
connection -- falls through to the ordinary backoff ladder, which is also the
default for anything unrecognised.

**A successful read queues the first score.** Adding an application is the
user choosing the job, so the score is chained here, in the transaction that
records the read, exactly once -- see `_chain_first_score`. A transient failure
with attempts left is noted (`note_extraction_retry`) rather than written as
`failed`, so the page says "retrying" until the attempts run out.

**One `runs` row per model call, always.** `extract_requirements` writes it on
every path including failure, and it is given a recorder that commits on its own
connection, so the cost is attributed even if the writes that follow are rolled
back. That table is the cost-attribution record and nothing here may bypass it.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import ExtractionErrorCode, RunRecord
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresJobRepository, PostgresRunRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_generate.errors import GenerateError
from jfl_generate.jobs import add_job
from sqlalchemy.engine import Connection, Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.handlers.scoring import KIND as SCORE_APPLICATION_KIND
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "extract_job_ad"


class _RunRecorder:
    """A `RunRepository` that commits each row on its own connection.

    Deliberately not sharing the handler's write transaction: a `runs` row is
    the record that money was spent, and money spent is true whether or not the
    writes that follow succeed. If the job write fails or the process dies
    afterwards, the cost must still be attributable.

    Not named `*Repository` on purpose -- it is a sink for one table, not a
    tenancy-scoped repository, and the name should not invite it to be read as
    one.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, run: RunRecord) -> None:
        with self._engine.begin() as conn:
            PostgresRunRepository(conn).record(run)


# How a failure from `extract_requirements` is classified. Keys are the prefixes
# that `jfl_generate.extract` builds its `GenerateError` messages from; the
# coupling is real and is pinned by a test, and anything unmatched falls through
# to a retryable `model_error`, so a wording change there degrades to "retried
# once too often", never to "given up on wrongly".
_PERMANENT_FAILURES: tuple[tuple[str, ExtractionErrorCode], ...] = (
    ("no text found in the job ad", "no_job_ad"),
    # The key authenticates nowhere, or authenticates and has no access. Both
    # are answered in Settings, and neither is answered by waiting.
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    # The ad is longer than one call can answer in. Retrying sends the same ad.
    ("model output was truncated", "ad_too_long"),
    ("model refused to respond", "model_refused"),
    # A malformed request is a bug in this code, not weather.
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[ExtractionErrorCode, bool]:
    """(code, permanent) for a failed extraction."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _application_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("application_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no application_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload application_id is not a uuid") from None


def build_extract_job_ad(*, master_key: MasterKey | None, model: str) -> Handler:
    """Bind the handler to the two things it needs from the process environment.

    A closure rather than a module-level environment read: `WorkerSettings` is
    the one place this process reads its environment, same rule as
    `RequestContext.from_env` at the CLI boundary.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _extract_job_ad(ctx, master_key=master_key, model=model)

    return handler


def _extract_job_ad(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str
) -> Mapping[str, object]:
    application_id = _application_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        claimed = PostgresApplicationRepository(conn, ctx.user_id).claim_extraction(application_id)
    if claimed is None:
        # No such application for this user, no ad against it, or an extraction
        # that already succeeded. All three mean "do not call the model", and
        # none of them is a failure worth retrying.
        return {"application_id": str(application_id), "skipped": "nothing to extract"}

    if master_key is None:
        # An operator problem, not the user's: the worker was started without
        # JFL_MASTER_KEY, so no stored credential can be read at all.
        _fail(ctx, application_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        # `from None` and a literal message: the exceptions here name key ids
        # and nothing more, but the rule is that nothing on this path formats an
        # exception into a row.
        _fail(ctx, application_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, application_id, "no_api_key")
        # Permanent, and the reason is worth stating plainly: a key does not
        # appear by itself between attempts. Burning three of them to discover
        # that is noise in the log and twenty minutes of a spinner on a screen
        # that should already be saying "add your API key".
        raise PermanentTaskError("no Anthropic API key stored for this user")

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one. The
        # engine's own URL is the honest value to carry.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        # This transaction spans the model call, which is ~30 seconds of holding
        # one pooled connection idle-in-transaction. Accepted deliberately: it
        # is the handler's own write, not the claim (the claim was committed
        # before this handler ran, which is the thing that must never be held --
        # see `jfl_worker.queue`), and it buys atomicity between the job rows and
        # the application's state, so there is no window where requirements
        # exist but the page still says "reading". The `runs` row is the one
        # thing deliberately outside it, because a cost is true whether or not
        # the writes that follow commit.
        with ctx.engine.begin() as conn:
            # `add_job` is the same call `jfl job add` makes: extract, upsert the
            # job row, replace its requirements. The job id it derives is
            # `job_id(user_id, raw_text)` -- the same derivation that produced
            # `claimed.job_id` when the ad was pasted -- so this updates the row
            # the application already points at rather than minting a second.
            job, requirements = add_job(
                request, recorder, PostgresJobRepository(conn), claimed.raw_text
            )
            PostgresApplicationRepository(conn, ctx.user_id).finish_extraction(
                application_id, title=job.title, employer=job.employer
            )
            scored = _chain_first_score(conn, ctx, application_id)
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        if permanent or ctx.is_last_attempt:
            _fail(ctx, application_id, code)
        else:
            # A retry is coming: stay `pending`, note the code, and let the
            # panel say "retrying" rather than show an error it may not keep.
            _note_retry(ctx, application_id, code)
        if permanent:
            # The code, not the message: `last_error` must not carry SDK text
            # from a call that was authenticated with the user's key.
            raise PermanentTaskError(f"extraction failed permanently: {code}") from None
        raise

    return {
        "application_id": str(application_id),
        "job_id": str(job.id),
        "requirements": len(requirements),
        "score_id": None if scored is None else str(scored),
    }


def _chain_first_score(
    conn: Connection, ctx: TaskContext, application_id: uuid.UUID
) -> uuid.UUID | None:
    """Queue the application's first score, in the transaction that records
    the read. Returns the new score's id, or None if nothing was queued.

    **Adding an application is the user choosing the job** -- CLAUDE.md's
    2026-09-15 rule is that a job is scored when it becomes an application,
    never on arrival -- so the owner's "scoring needs to kick off instantly
    with the application being added" is that rule, applied without a second
    button press. It covers both doors in: a pasted ad, and "Track as
    application" (fetch -> this read -> score). The add form and the track
    button say so, and name the calls, before anyone presses them.

    **Exactly once, structurally.** Three things hold it:

      * Same transaction as `finish_extraction`: the score row, its task and
        the `done` extraction commit together or not at all. A redelivered
        extraction finds `done` in `claim_extraction` and never reaches here.
      * Only if the application has **no scoring run at all**. A re-read of the
        ad (a button) or a run the user already started does not queue another
        -- re-scoring after a re-read is the Re-score button, explicitly.
      * Ordered after `finish_extraction`'s UPDATE, which row-locks the
        application: a second extraction racing this one blocks on that lock
        until this commits, and its `has_any` (a fresh READ COMMITTED
        statement) then sees the row written here.

    No key, no chain: the key was needed to reach this point at all, so an
    application added without one fails its read with `no_api_key` and the
    page says so; nothing is queued behind it.
    """
    scores = PostgresScoreRepository(conn, ctx.user_id)
    if scores.has_any(application_id):
        return None
    row = scores.create_pending(application_id)
    PostgresTaskRepository(conn, ctx.user_id).enqueue(
        kind=SCORE_APPLICATION_KIND,
        # Ids only -- same rule as every payload in this app.
        payload={"score_id": str(row.id)},
    )
    return row.id


def _note_retry(ctx: TaskContext, application_id: uuid.UUID, code: ExtractionErrorCode) -> None:
    """Record a failed attempt the queue will retry, in its own transaction."""
    with ctx.engine.begin() as conn:
        PostgresApplicationRepository(conn, ctx.user_id).note_extraction_retry(application_id, code)


def _fail(ctx: TaskContext, application_id: uuid.UUID, code: ExtractionErrorCode) -> None:
    """Record the failure on the application, in its own transaction.

    Its own, because the transaction that failed is being rolled back and this
    has to survive that: the screen saying "reading the job ad..." forever is a
    worse failure than the one that caused it.
    """
    with ctx.engine.begin() as conn:
        PostgresApplicationRepository(conn, ctx.user_id).fail_extraction(application_id, code)
