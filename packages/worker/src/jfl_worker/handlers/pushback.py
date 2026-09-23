"""`classify_pushback`: read one score pushback, and apply what it read.

Mirrors `jfl_worker.handlers.title_suggestions` closely, and deliberately: same
credential discipline, same failure classification shape, same append-only
`runs` recording. Differences follow from what this call actually needs.

**It applies what it read.** The pushback box is one textarea, and the owner's
complaint about the earlier design was that confirming a classification before
anything happened was a step too many and still left him unsure what changed.
So the reading is applied here, in the same task, and the screen shows the
result -- what was taken, before -> after, what did not move and what would --
with "Not what I meant" beside it, which withdraws this row and re-applies
under the reading the user picks. What makes applying unseen acceptable is the
rule, not this handler: `jfl_core.pushback.decide` gives no reading a way to
move "could I get this" upward, and the database refuses it too.

**A failure here is not much of a failure.** `mark_classification_failed`
leaves the row `awaiting_classification` with a code on it and nothing moved,
and the screen asks the user which of a short list of plain readings they
meant. A model outage costs one click, never the loop.

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
from jfl_core.models import ApplicationScore, Pushback, PushbackErrorCode, RunRecord
from jfl_core.pushback import Axis
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresRunRepository
from jfl_core.storage.pushbacks import PostgresPushbackRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_generate.errors import GenerateError
from jfl_generate.pushback import classify_pushback as call_classify_pushback
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "classify_pushback"

# How many of this user's earlier pushbacks are given to the model as context
# for judging `new_information`. Recent ones, not all of them -- a long history
# does not need re-sending on every new correction, and
# `PostgresPushbackRepository.recent()` is already newest-first.
MAX_EARLIER_TEXTS = 5

# Statuses that mean the application was actually sent. Same list as
# `jfl_web.routes.pushbacks._submitted`, which the web path uses when the user
# picks a reading themselves; duplicated rather than imported because the
# worker does not carry the web package.
_SENT_STATUSES = frozenset({"applied", "screening", "interviewing", "offer", "rejected"})


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


class _RecentReader(Protocol):
    """What `_earlier_texts` needs from a pushback repository -- narrower than
    `PostgresPushbackRepository` so a test double can satisfy it without
    standing in for the whole concrete class.
    """

    def recent(self, limit: int = ...) -> list[Pushback]: ...


def _earlier_texts(repo: _RecentReader, row: Pushback) -> list[str]:
    """This user's own words from their earlier pushbacks, newest first and
    capped -- what `new_information` is judged against.

    Any score, not one dimension: the box no longer asks which part of the
    score the words are about, so every earlier correction is a candidate for
    "said this before". Withdrawn ones are left out -- the user said they were
    misread, so they are not a record of what was meant.
    """
    older = [
        p.user_text
        for p in repo.recent()
        if p.id != row.id and p.created_at < row.created_at and p.withdrawn_at is None
    ]
    return older[:MAX_EARLIER_TEXTS]


def _stimulus(score: ApplicationScore | None, axis: Axis) -> tuple[int | None, str]:
    """The number and sentence on the axis the words turned out to be about."""
    if score is None:
        return None, ""
    if axis == "want":
        return score.want_it_score, score.want_it_assessment
    return score.could_get_score, score.could_get_assessment


def _submitted(applications: PostgresApplicationRepository) -> int:
    """How many applications this user has actually sent -- the behavioural
    channel in the shrinkage denominator (`jfl_core.pushback.observations`).
    """
    return sum(
        1 for a in applications.list_applications(archived=False) if a.status in _SENT_STATUSES
    )


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
        score = PostgresScoreRepository(conn, ctx.user_id).get(row.score_id)

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
            could_get_score=score.could_get_score if score else None,
            could_get_explanation=score.could_get_assessment if score else "",
            want_score=score.want_it_score if score else None,
            want_explanation=score.want_it_assessment if score else "",
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

    # One transaction: read, classified and applied together, so the screen
    # never shows a reading that has not been applied. Every write below is a
    # no-op on a row that is already applied -- which is what happens if the
    # user picked a reading themselves while this call was in flight, and then
    # their pick stands.
    axis: Axis = "get" if result.kind == "capability" else "want"
    shown_score, shown_explanation = _stimulus(score, axis)
    with ctx.engine.begin() as conn:
        repo = PostgresPushbackRepository(conn, ctx.user_id)
        repo.set_reading(
            pushback_id,
            axis=axis,
            direction=result.direction,
            shown_score=shown_score,
            shown_explanation=shown_explanation,
        )
        repo.set_classification(
            pushback_id,
            classification=result.kind,
            new_information=result.new_information,
            note=result.note,
            source="model",
            trace_id=request.trace_id,
        )
        repo.apply(
            pushback_id,
            classification=result.kind,
            new_information=result.new_information,
            submitted_applications=_submitted(PostgresApplicationRepository(conn, ctx.user_id)),
        )

    return {
        "pushback_id": str(pushback_id),
        "classification": result.kind,
        "direction": result.direction,
    }


def _fail(ctx: TaskContext, pushback_id: uuid.UUID, code: PushbackErrorCode) -> None:
    """Record the failure, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`'s docstring; the reasoning is
    identical.
    """
    with ctx.engine.begin() as conn:
        PostgresPushbackRepository(conn, ctx.user_id).mark_classification_failed(pushback_id, code)
