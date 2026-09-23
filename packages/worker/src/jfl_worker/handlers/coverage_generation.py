"""`generate_coverage`: corpus coverage for a job's requirements, in the
background, on the user's key -- B5's prerequisite for drafting.

Mirrors `jfl_worker.handlers.extraction` closely: same credential custody,
same failure classification shape, same append-only `runs` recording. What
differs follows from what this call needs and from *why* it exists here at
all -- see `jfl_generate.draft.generate_draft`'s docstring: drafting requires
coverage to already be recorded, and will not run it silently, because that
would be a second, unbudgeted model call the user never asked for. This
handler is the button B5's screen offers instead: "check coverage" is
something a person presses, explicitly, and it is billed to them.

**Registered with `calls_model=True`.** One Anthropic call
(`jfl_generate.coverage.check_coverage`) on the user's own key.

**The key is never data.** Same custody as extraction: unsealed from
`user_credentials` with the worker's master key, held in one local, handed to
`run_coverage`, and dropped.

**Idempotent under at-least-once delivery, without a status column.** Unlike
extraction, there is no `coverage_status` on the job to check "already done" --
coverage is meant to be *re-runnable* (the whole point of `jfl job coverage` is
that a requirement can move off `absent` once a gap question is answered), so
"already done" is the wrong question. The right one is "did *this task* already
write its rows": `RequestContext.trace_id` is set to the task's own id, and
`JobRepository.coverage_run_exists` checks for a `requirement_coverage` row
carrying it. A redelivery of the same task sees its own earlier rows and skips
the model; a person pressing the button again gets a new task, a new trace_id,
and a fresh check.

**Some failures do not deserve a retry.** No key, a key Anthropic rejects, a
refusal, a job with no requirements to check, a job that does not belong to
this user -- nothing changes between attempts. Those raise
`PermanentTaskError`. Everything else falls through to the ordinary backoff
ladder.

**One `runs` row per model call, always**, on its own connection -- see
`extraction._RunRecorder`'s docstring; duplicated here for the same reason.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Literal

from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import RunRecord
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_generate.errors import GenerateError
from jfl_generate.jobs import run_coverage
from sqlalchemy.engine import Engine

from jfl_worker.chain import queue_next
from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "generate_coverage"

# A closed set, read back only by `jfl_web.drafts` -- see that module for the
# friendly wording. Never a formatted exception: this handler holds the user's
# decrypted API key while it runs, same discipline as `extraction.py`.
CoverageErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "no_job",
    "no_requirements",
    "model_refused",
    "model_error",
    "credential_unreadable",
]


class _RunRecorder:
    """A `RunRepository` that commits each row on its own connection. See
    `jfl_worker.handlers.extraction._RunRecorder`'s docstring -- duplicated
    rather than imported for the same reason.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, run: RunRecord) -> None:
        with self._engine.begin() as conn:
            PostgresRunRepository(conn).record(run)


# Keys are the prefixes `jfl_generate.jobs.run_coverage` and
# `jfl_generate.coverage.check_coverage` build their `GenerateError` messages
# from -- pinned by `tests/test_coverage_handler.py`, the same coupling
# `extraction.py`'s `_PERMANENT_FAILURES` has to `jfl_generate.extract`.
_PERMANENT_FAILURES: tuple[tuple[str, CoverageErrorCode], ...] = (
    ("no job ", "no_job"),
    ("job has no requirements to check", "no_requirements"),
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
)


def _classify(message: str) -> tuple[CoverageErrorCode, bool]:
    """(code, permanent) for a failed coverage run."""
    for prefix, code in _PERMANENT_FAILURES:
        if message.startswith(prefix):
            return code, True
    return "model_error", False


def _job_id(payload: Mapping[str, object]) -> uuid.UUID:
    raw = payload.get("job_id")
    if not isinstance(raw, str):
        raise PermanentTaskError("payload has no job_id")
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise PermanentTaskError("payload job_id is not a uuid") from None


def build_generate_coverage(*, master_key: MasterKey | None, model: str) -> Handler:
    """Bind the handler to the two things it needs from the process
    environment -- same pattern as `extraction.build_extract_job_ad`.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _generate_coverage(ctx, master_key=master_key, model=model)

    return handler


def _permanent(code: CoverageErrorCode) -> PermanentTaskError:
    """The one message shape every permanent failure raises.

    There is no `applications`-style status column to write a code into --
    coverage has no per-job state of its own -- so the code travels in
    `PermanentTaskError`'s own message instead, which already carries the
    credential discipline `tasks.last_error` requires (a closed set, never a
    formatted exception; see the module docstring). `jfl_web.drafts` parses
    this one shape back into a sentence for the person, the same way
    `jfl_web.jobads.extraction_failure` reads `ExtractionErrorCode`.
    """
    return PermanentTaskError(f"coverage generation failed permanently: {code}")


def _generate_coverage(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str
) -> Mapping[str, object]:
    job_id = _job_id(ctx.task.payload)

    with ctx.engine.connect() as conn:
        already_ran = PostgresJobRepository(conn).coverage_run_exists(ctx.user_id, ctx.task.id)
    if already_ran:
        # This exact task already wrote its coverage rows on an earlier
        # delivery -- at-least-once redelivery, not a genuine re-check. A
        # person pressing the button again gets a new task and a new
        # trace_id, so this never blocks that. The next step, if this was a
        # chain, may not have been queued before the redelivery -- `queue_next`
        # queues it at most once either way.
        queue_next(ctx)
        return {"job_id": str(job_id), "skipped": "coverage already recorded for this task"}

    if master_key is None:
        raise _permanent("credential_unreadable")

    try:
        with ctx.engine.begin() as conn:
            api_key = load_api_key(PostgresCredentialRepository(conn, ctx.user_id), master_key)
    except (MasterKeyError, SecretUnsealError):
        raise _permanent("credential_unreadable") from None

    if api_key is None:
        raise _permanent("no_api_key")

    request = RequestContext(
        user_id=ctx.user_id,
        anthropic_api_key=api_key,
        database_url=ctx.engine.url.render_as_string(hide_password=False),
        model=model,
        # The task's own id, so a redelivery is detectable -- see the module
        # docstring's idempotency note.
        trace_id=ctx.task.id,
    )
    recorder = _RunRecorder(ctx.engine)

    try:
        with ctx.engine.begin() as conn:
            rows = run_coverage(
                request,
                PostgresGroundingRepository(conn),
                recorder,
                PostgresJobRepository(conn),
                job_id,
            )
    except GenerateError as exc:
        code, permanent = _classify(str(exc))
        if permanent:
            raise _permanent(code) from None
        raise

    # "Write the CV" pressed before this check had run: the draft is next.
    queue_next(ctx)
    return {"job_id": str(job_id), "requirements_checked": len(rows)}
