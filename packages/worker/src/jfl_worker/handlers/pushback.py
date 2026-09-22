"""`classify_pushback`: propose what kind one score pushback is.

Mirrors `jfl_worker.handlers.title_suggestions` closely, and deliberately: same
credential discipline, same failure classification shape, same append-only
`runs` recording. Differences follow from what this call actually needs.

**A failure here is not much of a failure.** `jfl_core.storage.pushbacks`'s
`mark_classification_failed` leaves the row `awaiting_classification` with a
code on it, and the screen simply asks the user which of the three kinds this
is -- which it was always going to ask, since the classification is theirs to
confirm before anything applies (see `jfl_core.pushback`'s module docstring
and `PostgresPushbackRepository.apply`'s). A model outage here costs the user
a moment of picking a radio button themselves, never the loop.

**The key is never data.** Same custody as `title_suggestions.py`: fetched
from `user_credentials`, unsealed with the worker's master key, held in one
local, handed to `classify_pushback`, and dropped. Never in the task payload
(one id), never in a log line, never in `score_pushbacks.error_code` (a closed
set), never in the `runs` row.

**One `runs` row per model call, always**, on its own connection -- see
`_RunRecorder`, deliberately duplicated rather than shared, for the reason
`jfl_worker.credentials` gives for duplicating `jfl_web.credentials`: the cost
is attributed even if the writes that follow are rolled back.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Protocol

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import Pushback, PushbackErrorCode, RunRecord
from jfl_core.pushback import COULD_GET_OVERALL, WANT_OVERALL
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresRunRepository
from jfl_core.storage.pushbacks import PostgresPushbackRepository
from jfl_generate.errors import GenerateError
from jfl_generate.pushback import classify_pushback as call_classify_pushback
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "classify_pushback"

# How many of this user's earlier pushbacks on the same dimension are given to
# the model as context for judging `new_information`. Recent ones, not all of
# them -- a dimension with a long history does not need its whole log
# re-sent on every new correction, and `PostgresPushbackRepository.recent()`
# is already newest-first.
MAX_EARLIER_TEXTS = 5


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


# Keys are the prefixes `jfl_generate.pushback.classify_pushback` builds its
# `GenerateError` messages from -- the same coupling `title_suggestions.py`'s
# `_PERMANENT_FAILURES` has to `jfl_generate.titles`, pinned the same way by
# `tests/test_pushback_handler.py`.
_PERMANENT_FAILURES: tuple[tuple[str, PushbackErrorCode], ...] = (
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[PushbackErrorCode, bool]:
    """(code, permanent) for a failed classification call."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _pushback_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("pushback_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no pushback_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload pushback_id is not a uuid") from None


def dimension_label(dimension: str) -> str:
    """Plain words for a dimension string, for the prompt and nowhere else --
    it is not stored and not shown to the user, who sees their own panel, not
    this label.

    Deliberately dumb: it reads the prefix and nothing more. `capability:<key>`
    collapses to "a capability" rather than trying to prettify `<key>`, which
    is an internal id (see `jfl_core.profile.Capability`) with no guarantee of
    being readable prose.
    """
    if dimension == WANT_OVERALL:
        return 'the whole "do I want this" number'
    if dimension == COULD_GET_OVERALL:
        return 'the whole "could I get this" number'
    if dimension.startswith("constraint:"):
        return dimension.removeprefix("constraint:")
    if dimension.startswith("objective:"):
        return "objective " + dimension.removeprefix("objective:")
    if dimension.startswith("capability:"):
        return "a capability"
    return dimension


class _RecentReader(Protocol):
    """What `_earlier_texts` needs from a pushback repository -- narrower than
    `PostgresPushbackRepository` so a test double can satisfy it without
    standing in for the whole concrete class.
    """

    def recent(self, limit: int = ...) -> list[Pushback]: ...


def _earlier_texts(repo: _RecentReader, row: Pushback) -> list[str]:
    """This user's own words from earlier pushbacks on the same dimension,
    oldest excluded beyond the cap -- what `new_information` is judged
    against. Simple: one table scan via `recent()`, filtered in Python. This
    call is cheap and the log is not large enough yet to need a purpose-built
    query.
    """
    older = [
        p.user_text
        for p in repo.recent()
        if p.dimension == row.dimension and p.id != row.id and p.created_at < row.created_at
    ]
    return older[:MAX_EARLIER_TEXTS]


def build_classify_pushback(*, master_key: MasterKey | None) -> Handler:
    """Bind the handler to the one thing it needs from the process environment.
    No `model` keyword -- `jfl_generate.pushback.classify_pushback` always
    calls claude-haiku-4-5, never the deployment's configured model.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _classify_pushback(ctx, master_key=master_key)

    return handler


def _classify_pushback(ctx: TaskContext, *, master_key: MasterKey | None) -> Mapping[str, object]:
    pushback_id = _pushback_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        row = PostgresPushbackRepository(conn, ctx.user_id).get(pushback_id)
    if row is None or row.status != "awaiting_classification":
        # No such row for this user, one already classified, or one already
        # applied: all three mean "do not call the model". Delivery is
        # at-least-once, and a redelivered task must not spend the user's key
        # twice on a row the first attempt already finished.
        return {"pushback_id": str(pushback_id), "skipped": "nothing to classify"}

    if master_key is None:
        _fail(ctx, pushback_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        _fail(ctx, pushback_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, pushback_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")

    with ctx.engine.begin() as conn:
        earlier_texts = _earlier_texts(PostgresPushbackRepository(conn, ctx.user_id), row)

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        result = call_classify_pushback(
            request,
            recorder,
            user_text=row.user_text,
            axis=row.axis,
            direction=row.asserted_direction,
            shown_score=row.shown_score,
            shown_explanation=row.shown_explanation,
            dimension_label=dimension_label(row.dimension),
            earlier_texts=earlier_texts,
            now=ctx.now,
        )
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, pushback_id, code)
        if permanent:
            raise PermanentTaskError(
                f"pushback classification failed permanently: {code}"
            ) from None
        raise

    with ctx.engine.begin() as conn:
        PostgresPushbackRepository(conn, ctx.user_id).set_classification(
            pushback_id,
            classification=result.kind,
            new_information=result.new_information,
            note=result.note,
            source="model",
            trace_id=request.trace_id,
        )

    return {"pushback_id": str(pushback_id), "classification": result.kind}


def _fail(ctx: TaskContext, pushback_id: uuid.UUID, code: PushbackErrorCode) -> None:
    """Record the failure, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`'s docstring; the reasoning is
    identical.
    """
    with ctx.engine.begin() as conn:
        PostgresPushbackRepository(conn, ctx.user_id).mark_classification_failed(pushback_id, code)
