"""`extract_cv_facts`: read one uploaded CV, in the background, on the user's key.

Slice B6. The upload form returns immediately with the CV stored verbatim and a
task in the queue; this is what turns it into candidate facts for the user to
confirm. Fast input, slow processing -- the same shape as `extract_job_ad`, and
this handler is deliberately its near-twin.

**Registered with `calls_model=True`.** That is what puts it behind
`JFL_DISABLE_MODEL_CALLS`. A handler that spends a user's key and says
`calls_model=False` would make the incident lever a lie.

**The key is never data.** Fetched from `user_credentials`, unsealed with the
worker's master key, held in one local, handed to `extract_cv_facts`, dropped.
Not in the task payload (one id), not in a log line, not in
`cv_extractions.error_code` -- which takes a **code from a closed set**,
never a formatted exception, precisely because a formatted exception is where a
credential leaks.

**Not reading one CV twice.** Delivery is at-least-once.
`claim_extraction` returns None for a CV whose extraction is already `done`, and
the handler returns without calling anything. Re-reading a CV is a button the
user presses, which enqueues a new task -- explicit, because it is their money,
and thirty-three CVs is thirty-three calls.

**Some failures do not deserve a retry.** No key stored, a key Anthropic
rejects, a CV too long for one call, a refusal: nothing changes between
attempts. Those raise `PermanentTaskError`. Transient trouble -- a 429, a 5xx, a
dropped connection -- falls through to the ordinary backoff ladder, which is
also the default for anything unrecognised.

**One `runs` row per model call, always**, on its own connection, so the cost is
attributed even if the writes that follow are rolled back.

**Nothing here writes a corpus span.** The handler's output is `proposed` rows
and nothing else; only the user confirming a fact puts anything in the corpus,
and that path goes through markdown (`jfl_core.corpus_source`). A worker that
could ground on a CV would quietly undo the whole measurement.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import CvExtractionErrorCode, RunRecord
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresRunRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_generate.cv_facts import extract_cv_facts as call_extract_cv_facts
from jfl_generate.cv_facts import to_proposed_facts
from jfl_generate.errors import GenerateError
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "extract_cv_facts"


class _RunRecorder:
    """A `RunRepository` that commits each row on its own connection. See
    `jfl_worker.handlers.extraction._RunRecorder` -- the reasoning is identical
    and duplicated rather than imported for the reason given there.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, run: RunRecord) -> None:
        with self._engine.begin() as conn:
            PostgresRunRepository(conn).record(run)


# Keys are the prefixes `jfl_generate.cv_facts` builds its `GenerateError`
# messages from. The coupling is real, is pinned by a test, and anything
# unmatched falls through to a retryable `model_error` -- so a wording change
# there degrades to "retried once too often", never to "given up on wrongly".
_PERMANENT_FAILURES: tuple[tuple[str, CvExtractionErrorCode], ...] = (
    ("no text found in the CV", "no_cv_text"),
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model output was truncated", "cv_too_long"),
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[CvExtractionErrorCode, bool]:
    """(code, permanent) for a failed CV read."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _sent_document_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("sent_document_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no sent_document_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload sent_document_id is not a uuid") from None


def build_extract_cv_facts(*, master_key: MasterKey | None, model: str) -> Handler:
    """Bind the handler to the two things it needs from the process environment.
    A closure rather than a module-level environment read: `WorkerSettings` is
    this process's one environment read.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _extract_cv_facts(ctx, master_key=master_key, model=model)

    return handler


def _extract_cv_facts(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str
) -> Mapping[str, object]:
    document_id = _sent_document_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        cv_text = PostgresSentDocumentRepository(conn, ctx.user_id).claim_extraction(document_id)
    if cv_text is None:
        # No such CV for this user, no extraction row against it, or one that
        # already succeeded. All three mean "do not call the model", and none of
        # them is a failure worth retrying.
        return {"sent_document_id": str(document_id), "skipped": "nothing to extract"}

    if master_key is None:
        # An operator problem, not the user's: the worker was started without
        # JFL_MASTER_KEY, so no stored credential can be read at all.
        _fail(ctx, document_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        # `from None` and a literal message: nothing on this path formats an
        # exception into a row.
        _fail(ctx, document_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, document_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        items = call_extract_cv_facts(request, recorder, cv_text=cv_text, now=ctx.now)
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, document_id, code)
        if permanent:
            # The code, not the message: `last_error` must not carry SDK text
            # from a call that was authenticated with the user's key.
            raise PermanentTaskError(f"reading the CV failed permanently: {code}") from None
        raise

    proposed = to_proposed_facts(items, sent_document_id=document_id)
    with ctx.engine.begin() as conn:
        inserted = PostgresCandidateFactRepository(conn, ctx.user_id).add_proposed(proposed)
        PostgresSentDocumentRepository(conn, ctx.user_id).finish_extraction(
            document_id, facts_proposed=inserted
        )

    # `proposed` minus `inserted` is the dedupe doing its job across CVs, which
    # is the number worth seeing in the log when thirty-three near-identical CVs
    # collapse into one list.
    return {
        "sent_document_id": str(document_id),
        "facts_returned": len(proposed),
        "facts_new": inserted,
    }


def _fail(ctx: TaskContext, document_id: uuid.UUID, code: CvExtractionErrorCode) -> None:
    """Record the failure on the CV, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`; the reasoning is identical.
    """
    with ctx.engine.begin() as conn:
        PostgresSentDocumentRepository(conn, ctx.user_id).fail_extraction(document_id, code)
