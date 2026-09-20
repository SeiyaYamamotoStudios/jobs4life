"""`check_application_answer` and `draft_application_answer`: the two equal
paths for an application question, in the background, on the user's key.

See CLAUDE.md's 2026-09-18 decision ("check my answer" / "draft one for me"
side by side, the page advises, it never prescribes) and NEXT.md's task 4.
Mirrors `jfl_worker.handlers.extraction` and `.title_suggestions` closely, and
deliberately: same credential discipline, same failure classification shape,
same append-only `runs` recording, same `calls_model=True` registration --
getting that wrong would make the `JFL_DISABLE_MODEL_CALLS` incident lever a
lie for two more calls that spend a user's key.

**The key is never data.** Fetched from `user_credentials`, unsealed with the
worker's master key, held in one local, handed to `jfl_generate.answers`
and/or `jfl_gate.gate.check_text`, and dropped. Never in the task payload (one
id), never in a log line, never in `application_question_answers.error_code`
(a closed set), never in the `runs` row.

**Not running twice for one attempt that already succeeded.** Delivery is
at-least-once; both handlers read the row through `get_answer` and return
without calling anything if its status is already `done`. A `failed` row is
*not* treated as finished -- the runner's own retry (a redelivery after a
transient failure) must still be able to try again against the same row, the
same convention `jfl_worker.handlers.title_suggestions` uses. Only a fresh
press of the button (a new row, a new task) retries a *permanent* failure --
exactly as `jfl_web.routes.applications.extract_again` does for extraction.

**One `runs` row per model call, always**, on its own connection -- see
`_RunRecorder`, deliberately duplicated rather than shared, for the reason
`jfl_worker.credentials`'s docstring gives for duplicating `jfl_web.credentials`:
the cost is attributed even if the writes that follow are rolled back. A check
attempt makes two model calls (the assessment call, then the claim gate's own
automatic pass) and so writes two rows; a draft attempt makes the same two
(the draft call, then the gate). Both calls in one attempt share one
`RequestContext`, and therefore one `trace_id` -- an attempt's total cost is
`SELECT sum(cost_usd) FROM runs WHERE trace_id = ...`, the same convention
`drafts.trace_id` uses.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import AnswerErrorCode, Job, JobRequirement, RunRecord
from jfl_core.storage.application_questions import PostgresApplicationQuestionRepository
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_gate.gate import GateError, check_text
from jfl_generate.answers import assess_answer, draft_application_answer
from jfl_generate.errors import GenerateError
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

CHECK_KIND = "check_application_answer"
DRAFT_KIND = "draft_application_answer"


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


# Keys are the prefixes `jfl_generate.answers` and `jfl_gate.gate.check_text`
# build their error messages from -- the same coupling `extraction.py`'s
# `_PERMANENT_FAILURES` has to `jfl_generate.extract`, pinned the same way by
# `tests/test_application_questions_handler.py`. `model output was truncated`
# is permanent here (unlike a whole CV, a short answer or assessment hitting
# the ceiling will not fit on a retry of the same input either).
_PERMANENT_FAILURES: tuple[tuple[str, AnswerErrorCode], ...] = (
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
    ("model output was truncated", "model_error"),
)


def _classify(message: str) -> tuple[AnswerErrorCode, bool]:
    """(code, permanent) for a failed check or draft attempt."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _answer_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("answer_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no answer_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        # The message names neither the value nor the payload: `last_error` is
        # read back in admin queries and quoted into logs.
        raise PermanentTaskError("payload answer_id is not a uuid") from None


def _job_and_requirements(
    ctx: TaskContext, job_id: uuid.UUID | None
) -> tuple[Job | None, list[JobRequirement]]:
    """The job and its requirements, or `(None, [])` if the application has no
    linked job yet -- both callers must still produce something rather than
    fail: `_format_question_job`/`_format_question_requirements` in
    `jfl_generate.prompts` render the absence honestly instead.
    """
    if job_id is None:
        return None, []
    with ctx.engine.begin() as conn:
        found = PostgresJobRepository(conn).get_job(ctx.user_id, job_id)
    return found if found is not None else (None, [])


def build_check_application_answer(*, master_key: MasterKey | None, model: str) -> Handler:
    """Bind the handler to the two things it needs from the process
    environment. See `jfl_worker.handlers.extraction.build_extract_job_ad`'s
    docstring -- same reasoning.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _check_application_answer(ctx, master_key=master_key, model=model)

    return handler


def build_draft_application_answer(*, master_key: MasterKey | None, model: str) -> Handler:
    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _draft_application_answer(ctx, master_key=master_key, model=model)

    return handler


def _load_key_or_fail(
    ctx: TaskContext, answer_id: uuid.UUID, *, master_key: MasterKey | None
) -> str:
    """The user's Anthropic key, or a permanent failure recorded and raised.
    Shared by both handlers below -- identical to the credential half of
    `extraction._extract_job_ad`.
    """
    if master_key is None:
        _fail(ctx, answer_id, "credential_unreadable")
        raise PermanentTaskError("worker has no master key, so no credential can be unsealed")
    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        _fail(ctx, answer_id, "credential_unreadable")
        raise PermanentTaskError("stored credential could not be unsealed") from None
    if api_key is None:
        _fail(ctx, answer_id, "no_api_key")
        raise PermanentTaskError("no Anthropic API key stored for this user")
    return api_key


def _check_application_answer(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str
) -> Mapping[str, object]:
    answer_id = _answer_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        repo = PostgresApplicationQuestionRepository(conn, ctx.user_id)
        answer = repo.get_answer(answer_id)
        question = None if answer is None else repo.get_question(answer.question_id)
    if answer is None or answer.status == "done" or question is None:
        # No such answer for this user, one already finished, or (should the
        # FK cascade ever race with a redelivery) a question that is gone too.
        # All mean "do not call the model".
        return {"answer_id": str(answer_id), "skipped": "nothing to check"}

    api_key = _load_key_or_fail(ctx, answer_id, master_key=master_key)

    with ctx.engine.begin() as conn:
        detail = PostgresApplicationRepository(conn, ctx.user_id).get_application(
            question.application_id
        )
    job_id = detail.application.job_id if detail is not None else None
    job, requirements = _job_and_requirements(ctx, job_id)

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        # Only the CLI builds engines from this; the worker already has one.
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        with ctx.engine.begin() as conn:
            gate_output = check_text(
                request, PostgresGroundingRepository(conn), recorder, answer.answer_text
            )
    except GateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, answer_id, code)
        if permanent:
            raise PermanentTaskError(f"check failed permanently: {code}") from None
        raise

    try:
        assessment = assess_answer(
            request,
            recorder,
            question_text=question.question_text,
            answer_text=answer.answer_text,
            job=job,
            requirements=requirements,
            now=ctx.now,
        )
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, answer_id, code)
        if permanent:
            raise PermanentTaskError(f"assessment failed permanently: {code}") from None
        raise

    with ctx.engine.begin() as conn:
        PostgresApplicationQuestionRepository(conn, ctx.user_id).mark_done(
            answer_id,
            answer_text=None,
            gate_result=gate_output.model_dump(mode="json"),
            assessment=assessment.model_dump(),
            model=model,
            trace_id=request.trace_id,
        )
    return {"answer_id": str(answer_id)}


def _draft_application_answer(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str
) -> Mapping[str, object]:
    answer_id = _answer_id(ctx.task.payload)

    with ctx.engine.begin() as conn:
        repo = PostgresApplicationQuestionRepository(conn, ctx.user_id)
        answer = repo.get_answer(answer_id)
        question = None if answer is None else repo.get_question(answer.question_id)
    if answer is None or answer.status == "done" or question is None:
        return {"answer_id": str(answer_id), "skipped": "nothing to draft"}

    with ctx.engine.begin() as conn:
        detail = PostgresApplicationRepository(conn, ctx.user_id).get_application(
            question.application_id
        )
    job_id = detail.application.job_id if detail is not None else None
    job, requirements = _job_and_requirements(ctx, job_id)
    if job is None or not requirements:
        # There is nothing job-specific to draft from yet -- the ad has not
        # been read (or named no requirements). Permanent: nothing changes
        # between attempts until the ad is read, which is a different button.
        _fail(ctx, answer_id, "no_requirements")
        raise PermanentTaskError("job has no extracted requirements to draft against")

    api_key = _load_key_or_fail(ctx, answer_id, master_key=master_key)

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        with ctx.engine.begin() as conn:
            draft_text = draft_application_answer(
                request,
                PostgresGroundingRepository(conn),
                recorder,
                question_text=question.question_text,
                job=job,
                requirements=requirements,
                now=ctx.now,
            )
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, answer_id, code)
        if permanent:
            raise PermanentTaskError(f"draft failed permanently: {code}") from None
        raise

    try:
        with ctx.engine.begin() as conn:
            gate_output = check_text(
                request, PostgresGroundingRepository(conn), recorder, draft_text
            )
    except GateError as exc:
        code, permanent = _classify(str(exc))
        _fail(ctx, answer_id, code)
        if permanent:
            raise PermanentTaskError(f"draft's gate pass failed permanently: {code}") from None
        raise

    with ctx.engine.begin() as conn:
        PostgresApplicationQuestionRepository(conn, ctx.user_id).mark_done(
            answer_id,
            answer_text=draft_text,
            gate_result=gate_output.model_dump(mode="json"),
            assessment=None,
            model=model,
            trace_id=request.trace_id,
        )
    return {"answer_id": str(answer_id)}


def _fail(ctx: TaskContext, answer_id: uuid.UUID, code: AnswerErrorCode) -> None:
    """Record the failure, in its own transaction -- see
    `jfl_worker.handlers.extraction._fail`'s docstring; the reasoning is
    identical.
    """
    with ctx.engine.begin() as conn:
        PostgresApplicationQuestionRepository(conn, ctx.user_id).mark_failed(answer_id, code)
