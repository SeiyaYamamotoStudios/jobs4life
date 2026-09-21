"""`score_application`: two scores for one tracked application -- PLAN.md B4.

Mirrors `jfl_worker.handlers.extraction` deliberately: same credential
discipline, same closed-set failure codes, same append-only `runs` recording on
its own connection. What differs follows from what this call needs.

**Scoring is spent only on a job the user chose.** CLAUDE.md's 2026-09-15
decision: users pay for model calls with their own key, so nothing scores a job
on arrival or because a page was refreshed. A task exists here because a person
pressed "Score this application".

**Coverage first, if it is missing, and the user was told.** "Could I get this"
is judged from the job's requirements and the corpus coverage recorded for
them; with no coverage recorded there is nothing honest to judge from, so this
runs `run_coverage` first. That is a second model call on the user's key, and
the button that enqueues this says so -- it is never silent.

**Only confirmed corpus facts are evidence.** Coverage is computed against
spans, which are the confirmed corpus. The user's *unconfirmed* CV-derived
facts are read too, but only so the model can name which of them would move the
first number if confirmed (`ScoreLever`). They never raise a score by
themselves.

**The key is never data.** Fetched from `user_credentials`, unsealed with the
worker's master key, held in one local, handed to the generate calls, and
dropped. Never in the task payload (one id), never in a log line, never in
`application_scores.error_code` (a closed set), never in a `runs` row.

**Not scoring twice for one press.** Delivery is at-least-once, so a
redelivered task finds its row already `done` or `failed` and returns without
calling anything. A re-score is a button, which mints a new row.

**One `runs` row per model call, always** -- see `_RunRecorder`, duplicated
from `extraction.py`'s for the reason `jfl_worker.credentials` gives. It also
totals what this run cost, which is what the page shows: the user paid for the
coverage call too when this run had to make one.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from decimal import Decimal

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import RunRecord, ScoreErrorCode
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_generate.errors import GenerateError
from jfl_generate.jobs import run_coverage
from jfl_generate.prompts import ProposedFactView, ScoreInputs
from jfl_generate.scoring import score_application as call_score_application
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "score_application"

# Enough to name what the user could confirm next, without turning a scoring
# prompt into a second copy of their CV history. The list is ordered by the
# repository; this is a ceiling on it, not a ranking.
MAX_PROPOSED_FACTS = 60


class _RunRecorder:
    """A `RunRepository` that commits each row on its own connection, and adds
    up what this scoring run cost.

    Its own connection for the reason `extraction.py`'s gives: a `runs` row is
    the record that money was spent, and money spent is true whether or not the
    writes that follow succeed.

    The total is what the page shows beside the score. It covers every call
    this run made -- the scoring call, and the coverage call when there was no
    coverage recorded yet -- because that is what the user was billed for
    pressing one button. Summing here rather than querying `runs` back keeps
    the handler free of SQL above the repository layer.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.total_cost_usd: Decimal | None = None

    def record(self, run: RunRecord) -> None:
        with self._engine.begin() as conn:
            PostgresRunRepository(conn).record(run)
        if run.cost_usd is not None:
            self.total_cost_usd = (self.total_cost_usd or Decimal(0)) + run.cost_usd


# How a failure from the generate calls is classified. Keys are the prefixes
# `jfl_generate.scoring` and `jfl_generate.coverage` build their `GenerateError`
# messages from; the coupling is real, is pinned by a test, and anything
# unmatched falls through to a retryable `model_error` -- so a wording change
# there degrades to "retried once too often", never to "given up on wrongly".
_PERMANENT_FAILURES: tuple[tuple[str, ScoreErrorCode], ...] = (
    # The key authenticates nowhere, or authenticates and has no access. Both
    # are answered in Settings, and neither is answered by waiting.
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    # A malformed request is a bug in this code, not weather.
    ("bad_request", "model_error"),
    # Retrying sends the same job and the same profile, and gets the same size
    # of answer back.
    ("model output was truncated", "model_error"),
    # Nothing to score against: the ad has not been read. Waiting does not read
    # it -- the user re-reads the ad, and that is a different button.
    ("job has no requirements", "no_requirements"),
)


def _classify(message: str) -> tuple[ScoreErrorCode, bool]:
    """(code, permanent) for a failed scoring run."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _score_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("score_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no score_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload score_id is not a uuid") from None


def build_score_application(*, master_key: MasterKey | None, model: str) -> Handler:
    """Bind the handler to the two things it needs from the process
    environment. A closure rather than a module-level environment read:
    `WorkerSettings` is the one place this process reads its environment.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _score_application(ctx, master_key=master_key, model=model)

    return handler


def _score_application(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str
) -> Mapping[str, object]:
    score_id = _score_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        row = PostgresScoreRepository(conn, ctx.user_id).get(score_id)
    if row is None or row.status != "pending":
        # No such row for this user, or one this or another attempt already
        # finished. All of those mean "do not call the model", and none is a
        # failure worth retrying.
        return {"score_id": str(score_id), "skipped": "nothing to score"}

    if master_key is None:
        # An operator problem, not the user's: the worker was started without
        # JFL_MASTER_KEY, so no stored credential can be read at all.
        _fail(ctx, score_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        # `from None` and a literal message: the rule is that nothing on this
        # path formats an exception into a row.
        _fail(ctx, score_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None

    if api_key is None:
        _fail(ctx, score_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")

    with ctx.engine.begin() as conn:
        detail = PostgresApplicationRepository(conn, ctx.user_id).get_application(
            row.application_id
        )
        job_id = detail.application.job_id if detail is not None else None
        found = None if job_id is None else PostgresJobRepository(conn).get_job(ctx.user_id, job_id)
        coverage = (
            []
            if job_id is None
            else PostgresJobRepository(conn).latest_coverage(ctx.user_id, job_id)
        )
        # One read for the whole profile -- constraints, capabilities,
        # disciplines, objectives and the self-assessment
        # (docs/profile-schema.md). Empty for a user who has filled nothing in,
        # never None, so there is no "has a profile?" branch here.
        profile = PostgresProfileRepository(conn, ctx.user_id).current()
        # Unconfirmed, and therefore never evidence -- see the module docstring
        # and `jfl_core.storage.candidate_facts`.
        proposed = PostgresCandidateFactRepository(conn, ctx.user_id).list_facts(state="proposed")

    if found is None or not found[1]:
        # No job linked, or an ad that has not been read into requirements yet.
        # Scoring "could I get this" from an unread ad would be a number with
        # nothing behind it.
        _fail(ctx, score_id, "no_requirements")
        raise PermanentTaskError("scoring failed permanently: no_requirements")
    job, requirements = found

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
    )
    recorder = _RunRecorder(ctx.engine)

    ran_coverage = False
    try:
        if not coverage:
            # The one place this handler holds a transaction across a model
            # call, and for the same reason `extraction.py` does: the coverage
            # rows and the gap questions it derives belong together. The `runs`
            # row is deliberately outside it.
            ran_coverage = True
            with ctx.engine.begin() as conn:
                coverage = run_coverage(
                    request,
                    PostgresGroundingRepository(conn),
                    recorder,
                    PostgresJobRepository(conn),
                    job.id,
                )

        result = call_score_application(
            request,
            recorder,
            ScoreInputs(
                job=job,
                requirements=requirements,
                coverage=coverage,
                profile=profile,
                proposed_facts=[
                    ProposedFactView(
                        fact_text=fact.fact_text,
                        role_label=fact.role_label,
                        source_line=fact.source_line,
                    )
                    for fact in proposed[:MAX_PROPOSED_FACTS]
                ],
                now=ctx.now,
            ),
        )
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, score_id, code)
        if permanent:
            # The code, not the message: `last_error` must not carry SDK text
            # from a call that was authenticated with the user's key.
            raise PermanentTaskError(f"scoring failed permanently: {code}") from None
        raise

    with ctx.engine.begin() as conn:
        PostgresScoreRepository(conn, ctx.user_id).mark_done(
            score_id,
            could_get_score=result.could_get_score,
            could_get_assessment=result.could_get_assessment,
            want_it_score=result.want_it_score,
            want_it_assessment=result.want_it_assessment,
            objective_verdicts=result.objective_verdicts,
            hard_gate_breaches=result.hard_gate_breaches,
            levers=result.levers,
            not_stated=result.not_stated,
            model=model,
            cost_usd=recorder.total_cost_usd,
            trace_id=request.trace_id,
        )

    # Both numbers, separately, and no third one -- the success log line is not
    # the place a composite gets invented either.
    return {
        "score_id": str(score_id),
        "could_get_score": result.could_get_score,
        "want_it_score": result.want_it_score,
        "ran_coverage": ran_coverage,
    }


def _fail(ctx: TaskContext, score_id: uuid.UUID, code: ScoreErrorCode) -> None:
    """Record the failure on the score row, in its own transaction.

    Its own, because the transaction that failed is being rolled back and this
    has to survive that: a panel saying "scoring..." forever is a worse failure
    than the one that caused it.
    """
    with ctx.engine.begin() as conn:
        PostgresScoreRepository(conn, ctx.user_id).mark_failed(score_id, code)
