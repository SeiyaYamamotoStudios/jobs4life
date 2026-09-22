"""`cluster_capabilities`: group one user's confirmed facts into capabilities.

Mirrors `jfl_worker.handlers.title_suggestions` closely, and deliberately: same
credential discipline, same failure classification shape, same append-only
`runs` recording. Differences follow from what this call needs.

**Always `claude-haiku-4-5`, never the deployment's configured model.** No
`model` keyword is threaded through `build_cluster_capabilities` --
`jfl_generate.capabilities.cluster_capabilities` hard-codes it, because this is
the second, cheaper model CLAUDE.md's 2026-09-05 decision keeps selectable per
call site.

**The key is never data.** Fetched from `user_credentials`, unsealed with the
worker's master key, held in one local, handed to `cluster_capabilities`, and
dropped. Never in the task payload (one id), never in a log line, never in
`capability_clusters.error_code` (a closed set), never in the `runs` row.

**Idempotent under redelivery.** Delivery is at-least-once, so the handler
re-reads the row and returns without calling anything if it is no longer
`pending`. The web route also refuses to enqueue a second run while one is in
flight, so the common case never gets that far.

**Only confirmed facts, and only ones nothing already accounts for.**
`jfl_core.profile.facts_to_cluster` picks them: a `proposed` fact is a CV's
claim rather than the user's, and a fact whose span is already somebody's
evidence has been answered. What does not fit one bounded call is recorded as
`omitted_fact_ids` and picked up next time -- **never dropped silently**.

**Nothing here writes to `profiles`.** The handler's output is proposals. A
proposal becomes a capability only when the user accepts it, which is the same
rule `extract_cv_facts` follows for the corpus and for the same reason.

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
from jfl_core.models import CapabilityClusterErrorCode, RunRecord
from jfl_core.profile import MAX_FACTS_PER_CLUSTER_CALL, facts_to_cluster
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.capability_clusters import PostgresCapabilityClusterRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresRunRepository
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_generate.capabilities import cluster_capabilities as call_cluster_capabilities
from jfl_generate.capabilities import unplaced_facts
from jfl_generate.errors import GenerateError
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "cluster_capabilities"

# How many confirmed facts one call is given. Read here at call time rather than
# taken as `facts_to_cluster`'s default, so the bound is one named thing a test
# can move and an operator can find.
MAX_FACTS = MAX_FACTS_PER_CLUSTER_CALL


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


# Keys are the prefixes `jfl_generate.capabilities` builds its `GenerateError`
# messages from -- the same coupling `title_suggestions.py` has to
# `jfl_generate.titles`, pinned the same way by a test. Anything unmatched
# falls through to a retryable `model_error`, so a wording change there degrades
# to "retried once too often", never to "given up on wrongly".
_PERMANENT_FAILURES: tuple[tuple[str, CapabilityClusterErrorCode], ...] = (
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("model output was truncated", "model_error"),
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[CapabilityClusterErrorCode, bool]:
    """(code, permanent) for a failed clustering call."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _cluster_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("cluster_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no cluster_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload cluster_id is not a uuid") from None


def build_cluster_capabilities(*, master_key: MasterKey | None) -> Handler:
    """Bind the handler to the one thing it needs from the process environment.
    No `model` keyword -- see the module docstring.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _cluster_capabilities(ctx, master_key=master_key)

    return handler


def _cluster_capabilities(
    ctx: TaskContext, *, master_key: MasterKey | None
) -> Mapping[str, object]:
    cluster_id = _cluster_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        row = PostgresCapabilityClusterRepository(conn, ctx.user_id).get(cluster_id)
    if row is None or row.status != "pending":
        # No such run for this user, or one already answered: both mean "do not
        # call the model". A row missing for this user is silent rather than an
        # error -- the web route already refuses a cross-user id at the door.
        return {"cluster_id": str(cluster_id), "skipped": "nothing to cluster"}

    if master_key is None:
        # An operator problem, not the user's: the worker was started without
        # JFL_MASTER_KEY, so no stored credential can be read at all.
        _fail(ctx, cluster_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        # `from None` and a literal message: nothing on this path formats an
        # exception into a row.
        _fail(ctx, cluster_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, cluster_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")

    with ctx.engine.begin() as conn:
        confirmed = PostgresCandidateFactRepository(conn, ctx.user_id).list_facts(state="confirmed")
        profile = PostgresProfileRepository(conn, ctx.user_id).current()
    sending, omitted = facts_to_cluster(confirmed, existing=profile.capabilities, limit=MAX_FACTS)

    if not sending:
        # Every confirmed fact is already accounted for, or the user rejected
        # them between pressing the button and this running. Finishing empty is
        # the honest answer and costs nothing; failing would be a lie about
        # what happened.
        with ctx.engine.begin() as conn:
            PostgresCapabilityClusterRepository(conn, ctx.user_id).mark_done(
                cluster_id, [], fact_count=0, omitted_fact_ids=[f.id for f in omitted]
            )
        return {"cluster_id": str(cluster_id), "skipped": "no unaccounted confirmed facts"}

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
        proposals = call_cluster_capabilities(request, recorder, facts=sending, now=ctx.now)
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, cluster_id, code)
        if permanent:
            # The code, not the message: `last_error` must not carry SDK text
            # from a call that was authenticated with the user's key.
            raise PermanentTaskError(
                f"clustering capabilities failed permanently: {code}"
            ) from None
        raise

    with ctx.engine.begin() as conn:
        PostgresCapabilityClusterRepository(conn, ctx.user_id).mark_done(
            cluster_id,
            proposals,
            fact_count=len(sending),
            unclustered_fact_ids=unplaced_facts(sending, proposals),
            omitted_fact_ids=[f.id for f in omitted],
        )

    return {
        "cluster_id": str(cluster_id),
        "facts_sent": len(sending),
        "capabilities": len(proposals),
        "facts_omitted": len(omitted),
    }


def _fail(ctx: TaskContext, cluster_id: uuid.UUID, code: CapabilityClusterErrorCode) -> None:
    """Record the failure, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`'s docstring; the reasoning is
    identical.
    """
    with ctx.engine.begin() as conn:
        PostgresCapabilityClusterRepository(conn, ctx.user_id).mark_failed(cluster_id, code)
