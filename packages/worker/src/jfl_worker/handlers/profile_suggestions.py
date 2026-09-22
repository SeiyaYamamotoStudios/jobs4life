"""`suggest_profile_settings`: read a user's CVs and propose profile settings.

Mirrors `jfl_worker.handlers.capability_clusters` closely, and deliberately:
same credential discipline, same failure classification shape, same append-only
`runs` recording. Differences follow from what this call needs.

**Always `claude-haiku-4-5`, never the deployment's configured model.** No
`model` keyword is threaded through `build_suggest_profile_settings` --
`jfl_generate.profile_suggestions.suggest_profile_settings` hard-codes it,
because this is the second, cheaper model CLAUDE.md's 2026-09-05 decision keeps
selectable per call site.

**The key is never data.** Fetched from `user_credentials`, unsealed with the
worker's master key, held in one local, handed to the call, and dropped. Never
in the task payload (one id), never in a log line, never in
`profile_suggestions.error_code` (a closed set), never in the `runs` row.

**Idempotent under redelivery.** Delivery is at-least-once, so the handler
re-reads the row and returns without calling anything if it is no longer
`pending`. The web route also refuses to enqueue a second run while one is in
flight, so the common case never gets that far.

**The CVs are read, never grounded on.** Text comes out of the sent-document
store and goes into one call. Nothing here writes a span, a candidate fact or a
corpus document, and nothing here reads the corpus -- see
`jfl_core.storage.sent_documents`, which is separate from `GroundingRepository`
precisely so this cannot be got wrong by accident.

**Nothing here writes to `profiles`.** The handler's output is proposals. A
proposal becomes a setting only when the user accepts it, which is the same
rule `extract_cv_facts` follows for the corpus and for the same reason.

**A suggestion the user already answered is never proposed again.**
`answered_keys` from earlier runs is subtracted before the row is written, so
"no" sticks across runs rather than only until the next press.

**One `runs` row per model call, always**, on its own connection -- see
`_RunRecorder`, duplicated from `extraction.py`'s rather than shared, for the
reason `jfl_worker.credentials` gives: the cost is attributed even if the
writes that follow are rolled back.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import ProfileSuggestionErrorCode, RunRecord
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresRunRepository
from jfl_core.storage.profile_suggestions import PostgresProfileSuggestionRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_generate.errors import GenerateError
from jfl_generate.profile_suggestions import select_cv_texts, to_proposals
from jfl_generate.profile_suggestions import (
    suggest_profile_settings as call_suggest_profile_settings,
)
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "suggest_profile_settings"


class _RunRecorder:
    """A `RunRepository` that commits each row on its own connection. See
    `jfl_worker.handlers.extraction._RunRecorder`'s docstring -- the reasoning
    is identical and duplicated rather than imported for the same reason.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, run: RunRecord) -> None:
        with self._engine.begin() as conn:
            PostgresRunRepository(conn).record(run)


# Keys are the prefixes `jfl_generate.profile_suggestions` builds its
# `GenerateError` messages from -- the same coupling `capability_clusters.py`
# has to `jfl_generate.capabilities`, pinned the same way by a test. Anything
# unmatched falls through to a retryable `model_error`, so a wording change
# there degrades to "retried once too often", never to "given up on wrongly".
_PERMANENT_FAILURES: tuple[tuple[str, ProfileSuggestionErrorCode], ...] = (
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("model output was truncated", "model_error"),
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[ProfileSuggestionErrorCode, bool]:
    """(code, permanent) for a failed suggestion call."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _run_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("suggestion_run_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no suggestion_run_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload suggestion_run_id is not a uuid") from None


def build_suggest_profile_settings(*, master_key: MasterKey | None) -> Handler:
    """Bind the handler to the one thing it needs from the process environment.
    No `model` keyword -- see the module docstring.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _suggest_profile_settings(ctx, master_key=master_key)

    return handler


def _suggest_profile_settings(
    ctx: TaskContext, *, master_key: MasterKey | None
) -> Mapping[str, object]:
    run_id = _run_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        row = PostgresProfileSuggestionRepository(conn, ctx.user_id).get(run_id)
    if row is None or row.status != "pending":
        # No such run for this user, or one already answered: both mean "do not
        # call the model". A row missing for this user is silent rather than an
        # error -- the web route already refuses a cross-user id at the door.
        return {"suggestion_run_id": str(run_id), "skipped": "nothing to suggest"}

    if master_key is None:
        # An operator problem, not the user's: the worker was started without
        # JFL_MASTER_KEY, so no stored credential can be read at all.
        _fail(ctx, run_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        # `from None` and a literal message: nothing on this path formats an
        # exception into a row.
        _fail(ctx, run_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, run_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")

    with ctx.engine.begin() as conn:
        documents = PostgresSentDocumentRepository(conn, ctx.user_id)
        # Newest first, which is also the order the prompt asks locations to be
        # returned in.
        stored = documents.list_cvs()
        texts = [text for text in (documents.cv_text(cv.id) for cv in stored) if text]
        answered = PostgresProfileSuggestionRepository(conn, ctx.user_id).answered_keys()
    sending = select_cv_texts(texts)

    if not sending:
        # Every CV was deleted between pressing the button and this running, or
        # they are all empty. Finishing empty is the honest answer and costs
        # nothing; failing would be a lie about what happened.
        with ctx.engine.begin() as conn:
            PostgresProfileSuggestionRepository(conn, ctx.user_id).mark_done(run_id, [], cv_count=0)
        return {"suggestion_run_id": str(run_id), "skipped": "no CV text to read"}

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        # The trace the run row already carries, so the screen can price this
        # call without the handler telling it what it cost.
        trace_id=row.trace_id,
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        items = call_suggest_profile_settings(request, recorder, cv_texts=sending, now=ctx.now)
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, run_id, code)
        if permanent:
            # The code, not the message: `last_error` must not carry SDK text
            # from a call that was authenticated with the user's key.
            raise PermanentTaskError(
                f"suggesting profile settings failed permanently: {code}"
            ) from None
        raise

    proposals = to_proposals(items, cv_texts=sending, answered_keys=sorted(answered))
    with ctx.engine.begin() as conn:
        PostgresProfileSuggestionRepository(conn, ctx.user_id).mark_done(
            run_id, proposals, cv_count=len(sending)
        )

    # `items` minus `proposals` is the whitelist and the fabricated-quote check
    # doing their job, which is the number worth seeing in the log.
    return {
        "suggestion_run_id": str(run_id),
        "cvs_read": len(sending),
        "suggestions_returned": len(items),
        "proposals": len(proposals),
    }


def _fail(ctx: TaskContext, run_id: uuid.UUID, code: ProfileSuggestionErrorCode) -> None:
    """Record the failure, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`'s docstring; the reasoning is
    identical.
    """
    with ctx.engine.begin() as conn:
        PostgresProfileSuggestionRepository(conn, ctx.user_id).mark_failed(run_id, code)
