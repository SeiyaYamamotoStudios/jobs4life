"""`suggest_titles`: adjacent title suggestions for one phrase just added to a
saved job filter's title includes -- slice C7a.

Mirrors `jfl_worker.handlers.extraction` closely, and deliberately: same
credential discipline, same failure classification shape, same append-only
`runs` recording. Differences follow from what this call actually needs.

**Always `claude-haiku-4-5`, never the user's configured model.** No `model`
keyword is threaded through `build_suggest_titles` the way `extract_job_ad`
threads `settings.model` -- `jfl_generate.titles.suggest_titles` hard-codes it,
because this is the second, cheaper model CLAUDE.md's 2026-09-05 decision log
says stays selectable per call site, not the product model.

**The key is never data.** Same custody as extraction: fetched from
`user_credentials`, unsealed with the worker's master key, held in one local,
handed to `suggest_titles`, and dropped. Never in the task payload (one id),
never in a log line, never in `title_suggestions.error_code` (a closed set),
never in the `runs` row.

**Not suggesting twice for one phrase.** `title_suggestions` has a unique
`(user_id, phrase_key)` and is only ever enqueued once, at save time (see
`jfl_web.routes.jobs.save_filter`) -- but delivery is at-least-once, so this
still checks `status == "done"` and returns without calling anything if a
redelivered task finds the work already finished.

**One `runs` row per model call, always**, on its own connection -- see
`_RunRecorder`, deliberately duplicated from `extraction.py`'s rather than
shared, for the reason `jfl_worker.credentials` gives for duplicating
`jfl_web.credentials`: the cost is attributed even if the writes that follow
are rolled back.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import RunRecord, TitleSuggestionErrorCode
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_core.storage.postgres import PostgresRunRepository
from jfl_core.storage.title_suggestions import PostgresTitleSuggestionRepository
from jfl_generate.errors import GenerateError
from jfl_generate.titles import suggest_titles as call_suggest_titles
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "suggest_titles"

# Live tracked applications given as context, capped -- see PLAN.md's C7a: a
# short excerpt of "what the user is currently pursuing", not the whole table.
MAX_APPLICATION_TITLES = 30


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


# Keys are the prefixes `jfl_generate.titles.suggest_titles` builds its
# `GenerateError` messages from -- the same coupling `extraction.py`'s
# `_PERMANENT_FAILURES` has to `jfl_generate.extract`, pinned the same way by
# `tests/test_title_suggestions_handler.py`.
_PERMANENT_FAILURES: tuple[tuple[str, TitleSuggestionErrorCode], ...] = (
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[TitleSuggestionErrorCode, bool]:
    """(code, permanent) for a failed title-suggestion call."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _suggestion_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("suggestion_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no suggestion_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload suggestion_id is not a uuid") from None


def build_suggest_titles(*, master_key: MasterKey | None) -> Handler:
    """Bind the handler to the one thing it needs from the process environment.
    No `model` keyword -- see the module docstring.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _suggest_titles(ctx, master_key=master_key)

    return handler


def _split_phrases(text: str) -> list[str]:
    """Comma-separated phrases, trimmed, in order, deduplicated by exact text --
    what the filter form actually holds. Deliberately duplicated from
    `jfl_web.titlesuggestions.split_phrases` rather than imported: the worker
    must not depend on `jfl_web` (see `jfl_worker.credentials`'s docstring for
    the same rule applied to credential loading), and this is four lines.
    """
    seen: list[str] = []
    for part in text.split(","):
        phrase = part.strip()
        if phrase and phrase not in seen:
            seen.append(phrase)
    return seen


def _suggest_titles(ctx: TaskContext, *, master_key: MasterKey | None) -> Mapping[str, object]:
    suggestion_id = _suggestion_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        row = PostgresTitleSuggestionRepository(conn, ctx.user_id).get(suggestion_id)
    if row is None or row.status == "done":
        # No such row for this user, or one already answered: both mean "do not
        # call the model". A row missing for this user is silent rather than an
        # error -- the web route already refuses a cross-user id at the door.
        return {"suggestion_id": str(suggestion_id), "skipped": "nothing to suggest"}

    if master_key is None:
        _fail(ctx, suggestion_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        _fail(ctx, suggestion_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, suggestion_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")

    with ctx.engine.begin() as conn:
        saved = PostgresJobFilterRepository(conn, ctx.user_id).get_filter()
        live_applications = PostgresApplicationRepository(conn, ctx.user_id).list_applications(
            archived=False
        )

    other_includes = [p for p in _split_phrases(saved.title_includes) if p != row.phrase]
    excludes = _split_phrases(saved.title_excludes)
    application_titles = [a.title for a in live_applications][:MAX_APPLICATION_TITLES]

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        suggestions = call_suggest_titles(
            request,
            recorder,
            phrase=row.phrase,
            other_includes=other_includes,
            excludes=excludes,
            application_titles=application_titles,
            now=ctx.now,
        )
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, suggestion_id, code)
        if permanent:
            raise PermanentTaskError(f"title suggestion failed permanently: {code}") from None
        raise

    with ctx.engine.begin() as conn:
        PostgresTitleSuggestionRepository(conn, ctx.user_id).mark_done(suggestion_id, suggestions)

    return {"suggestion_id": str(suggestion_id), "count": len(suggestions)}


def _fail(ctx: TaskContext, suggestion_id: uuid.UUID, code: TitleSuggestionErrorCode) -> None:
    """Record the failure, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`'s docstring; the reasoning is
    identical.
    """
    with ctx.engine.begin() as conn:
        PostgresTitleSuggestionRepository(conn, ctx.user_id).mark_failed(suggestion_id, code)
