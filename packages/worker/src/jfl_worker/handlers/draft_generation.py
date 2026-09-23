"""`generate_cv_draft`: a CV or a cover letter for an application's job,
generated against the corpus, gated automatically -- B5's "Write the CV"
screen, behind the queue, and the last step of its chain (`jfl_worker.chain`).

**Payload `kind` picks what is written.** `cv_document` -- what the web's
"Write the CV" now queues -- is the complete CV
(`jfl_generate.cv_document.generate_cv_document`), stored as a new version in
`cv_documents`. `cv_bullets` and `cover_letter` are the older text drafts
(`jfl_generate.draft.generate_draft`), stored in `drafts`; the CLI still writes
bullets that way. One task kind for all three, so the chain, the steps panel
and the failure codes the web already reads are unchanged: the complete CV
takes the bullets draft's place at the end of the same chain.

Mirrors `jfl_worker.handlers.extraction` and `.coverage_generation` closely:
same credential custody, same failure classification shape, same append-only
`runs` recording, same reasoning for a task-id trace rather than a status
column. What this file adds is `jfl_generate.draft.generate_draft` itself,
which already does two things no other handler in this worker does:

  * it makes **two** model calls, not one -- the draft, then the claim gate,
    automatically (see CLAUDE.md's decisions log, "The claim gate runs
    automatically on generated text") -- and both write their own `runs` row
    under the same `trace_id`, so `RunRepository.cost_for_trace` sums a
    draft's whole cost in one query;
  * on the second call it can raise `jfl_gate.gate.GateError`, not only
    `GenerateError`. Both build their failure text the same way (the same
    exception ladder, the same literal prefixes -- "authentication_error:",
    "rate_limited:", and so on), so one `_classify` handles both.

**A flagged draft is still stored and returned in full.** Nothing here
inspects `draft.gate_result` to decide whether to keep it -- the claim gate
informs, it never blocks (CLAUDE.md, "How the claim gate behaves"). Framing
renders as NOT CHECKED, never as supported, but that is a rendering rule for
`jfl_web.drafts`, not a reason to withhold anything here.

**Registered with `calls_model=True`.**

**The key is never data.** Same custody as extraction and coverage: unsealed
from `user_credentials`, held in one local, handed to `generate_draft`, and
dropped.

**Idempotent under at-least-once delivery, the same way coverage is.** There
is no per-draft status column to check "already done" against, and there
should not be one: drafts are deliberately append-only history (CLAUDE.md
build order, B5), so "already generated" is not even the right question for a
person pressing the button twice on purpose -- each press is a new draft.
What must not happen is a *redelivered* task minting a second draft for one
press. `RequestContext.trace_id` is set to the task's own id, and
`drafts.trace_id` carries it through, so a redelivery finds its own earlier
draft (by trace_id, among this job's drafts) and skips the model.

**Prerequisites are read, never silently satisfied.** `generate_draft` itself
refuses to run without a job, requirements, and recorded coverage -- see its
docstring -- and raises a `GenerateError` naming which. This handler
classifies each of those as permanent (`no_job`, `no_requirements`,
`no_coverage`): nothing about retrying calls coverage for the user, because
that would spend their money on a call the screen never asked for. The screen
offers the coverage button itself (`generate_coverage`).

**One `runs` row per model call, always**, on its own connection -- see
`extraction._RunRecorder`'s docstring; duplicated here for the same reason
every other handler in this worker duplicates it.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Literal, cast, get_args

from jfl_core.context import GATE_MODEL as DEFAULT_GATE_MODEL
from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, MasterKeyError, SecretUnsealError
from jfl_core.models import DraftKind, RunRecord
from jfl_core.profile import Capability, Profile
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.cv_documents import PostgresCvDocumentRepository
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_gate.gate import GateError
from jfl_generate.cv_document import generate_cv_document
from jfl_generate.draft import generate_draft
from jfl_generate.errors import GenerateError
from sqlalchemy.engine import Engine

from jfl_worker.credentials import load_api_key
from jfl_worker.registry import Handler, PermanentTaskError, TaskContext

KIND = "generate_cv_draft"

# The complete CV, stored in `cv_documents` -- beside the two text drafts.
CV_DOCUMENT = "cv_document"
WriteKind = DraftKind | Literal["cv_document"]

_WRITE_KINDS: tuple[str, ...] = (*get_args(DraftKind), CV_DOCUMENT)

# A closed set, read back only by `jfl_web.drafts`. Never a formatted
# exception -- this handler holds the user's decrypted API key while it runs.
DraftErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "no_job",
    "no_requirements",
    "no_coverage",
    "ad_too_long",
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


# Keys are the prefixes `jfl_generate.draft.generate_draft` and
# `jfl_gate.gate.check_text` build their error messages from -- pinned by
# `tests/test_draft_handler.py`, the same coupling `extraction.py`'s
# `_PERMANENT_FAILURES` has to `jfl_generate.extract`.
_PERMANENT_FAILURES: tuple[tuple[str, DraftErrorCode], ...] = (
    ("no job ", "no_job"),
    ("job has no requirements to draft against", "no_requirements"),
    ("no coverage recorded for job", "no_coverage"),
    ("authentication_error", "api_key_rejected"),
    ("permission_denied", "api_key_rejected"),
    (
        "model output was truncated",
        "ad_too_long",
    ),  # the draft, not an ad, but the shape ("too long for one call") is the same
    ("model refused to respond", "model_refused"),
    ("bad_request", "model_error"),
    # The CV's gate results could not be mapped back onto its lines. Nothing a
    # retry changes -- it would only pay for both calls again.
    ("claim gate output does not line up", "model_error"),
)


def _classify(message: str) -> tuple[DraftErrorCode, bool]:
    """(code, permanent) for a failed draft generation -- `GenerateError` or
    `GateError` alike, since both build their API-failure text off the same
    exception ladder (see the module docstring).
    """
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
        raise PermanentTaskError("payload application_id is not a uuid") from None


def _kind(payload: Mapping[str, object]) -> WriteKind:
    raw = payload.get("kind")
    if raw not in _WRITE_KINDS:
        # The message names neither the value nor the payload -- same
        # discipline as `_application_id`'s malformed-uuid branch, even though
        # `kind` is never a secret: consistency with the rest of this file's
        # error text is worth more than one extra word here.
        raise PermanentTaskError("payload kind is not a recognised draft kind")
    return raw  # type: ignore[return-value]


def build_generate_cv_draft(
    *, master_key: MasterKey | None, model: str, gate_model: str = DEFAULT_GATE_MODEL
) -> Handler:
    """Bind the handler to the two things it needs from the process
    environment -- same pattern as `extraction.build_extract_job_ad`.
    """

    def handler(ctx: TaskContext) -> Mapping[str, object]:
        return _generate_cv_draft(ctx, master_key=master_key, model=model, gate_model=gate_model)

    return handler


def _permanent(code: DraftErrorCode) -> PermanentTaskError:
    """The one message shape every permanent failure raises -- see
    `jfl_worker.handlers.coverage_generation._permanent`, which this mirrors.
    """
    return PermanentTaskError(f"draft generation failed permanently: {code}")


def _generate_cv_draft(
    ctx: TaskContext, *, master_key: MasterKey | None, model: str, gate_model: str
) -> Mapping[str, object]:
    application_id = _application_id(ctx.task.payload)
    kind = _kind(ctx.task.payload)

    with ctx.engine.connect() as conn:
        application = PostgresApplicationRepository(conn, ctx.user_id).get_application(
            application_id
        )
    if application is None:
        # No such application for this user. Not a formatted id -- see the
        # sibling handlers' identical reasoning.
        raise PermanentTaskError("no application for this user")
    job_id = application.application.job_id
    if job_id is None:
        raise _permanent("no_job")

    if kind == CV_DOCUMENT:
        with ctx.engine.connect() as conn:
            done = PostgresCvDocumentRepository(conn, ctx.user_id).version_for_trace(ctx.task.id)
        if done is not None:
            # Redelivery of a task that already stored its CV -- see the module
            # docstring's idempotency note; the same reasoning as for drafts.
            return {
                "application_id": str(application_id),
                "cv_document_id": str(done.id),
                "skipped": "cv document already recorded for this task",
            }
    else:
        with ctx.engine.connect() as conn:
            existing = PostgresJobRepository(conn).list_drafts(ctx.user_id, job_id)
        if any(d.trace_id == ctx.task.id for d in existing):
            # This exact task already wrote its draft on an earlier delivery --
            # at-least-once redelivery, not a second press of the button. A
            # genuine second press is a new task with a new trace_id, so this
            # never suppresses an intentional re-generation.
            matched = next(d for d in existing if d.trace_id == ctx.task.id)
            return {
                "application_id": str(application_id),
                "draft_id": str(matched.id),
                "skipped": "draft already recorded for this task",
            }

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
        gate_model=gate_model,
        # The task's own id, so a redelivery is detectable -- see the module
        # docstring's idempotency note. `generate_draft` carries this straight
        # into `drafts.trace_id` and into both `runs` rows it writes.
        trace_id=ctx.task.id,
    )
    recorder = _RunRecorder(ctx.engine)

    # The confirmed depths, read before the call: the prompt names them as the
    # ceiling on what the draft may claim (`docs/profile-schema.md`). Read in
    # its own short transaction rather than inside the one that holds the model
    # call.
    with ctx.engine.begin() as conn:
        profile = PostgresProfileRepository(conn, ctx.user_id).current()
        display_name = PostgresCvDocumentRepository(conn, ctx.user_id).account_display_name()
    capabilities = profile.capabilities

    if kind == CV_DOCUMENT:
        return _write_cv_document(
            ctx,
            request,
            recorder,
            application_id,
            job_id,
            capabilities=capabilities,
            name=header_name(profile, display_name),
        )

    try:
        with ctx.engine.begin() as conn:
            draft = generate_draft(
                request,
                PostgresJobRepository(conn),
                PostgresGroundingRepository(conn),
                recorder,
                job_id,
                cast(DraftKind, kind),  # cv_document returned above
                capabilities,
            )
    except (GenerateError, GateError) as exc:
        code, permanent = _classify(str(exc))
        if permanent:
            raise _permanent(code) from None
        raise

    # "draft_kind", not "kind" -- the runner's own success log line already
    # writes `kind=task.kind` from every handler's result dict merged in
    # (`jfl_worker.runner.Worker._dispatch`), and a second `kind` keyword
    # collides with it.
    return {"application_id": str(application_id), "draft_id": str(draft.id), "draft_kind": kind}


def header_name(profile: Profile, display_name: str) -> str:
    """The CV header's name: the profile's, if it states one, else the
    account's display name, else "" -- in which case
    `jfl_generate.cv_document` falls back to the corpus document's own title.

    The profile's contact settings are read by attribute because they belong to
    the profile editing screen, not to this handler: a profile that has no
    `contact` section (or no name in it) simply falls through. Never inferred
    -- a name is only ever one the user typed or the account carries.
    """
    contact = getattr(profile, "contact", None)
    name = getattr(contact, "name", None) or getattr(profile, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return display_name


def _write_cv_document(
    ctx: TaskContext,
    request: RequestContext,
    recorder: _RunRecorder,
    application_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    capabilities: Sequence[Capability],
    name: str,
) -> Mapping[str, object]:
    """The complete CV: two model calls, then one new `cv_documents` row.

    The version is written in its own transaction after both calls, so a
    failed call leaves no half-written CV -- only its `runs` rows, which the
    recorder has already committed on their own connection.
    """
    try:
        with ctx.engine.connect() as conn:
            generated = generate_cv_document(
                request,
                PostgresJobRepository(conn),
                PostgresGroundingRepository(conn),
                recorder,
                job_id,
                capabilities=capabilities,
                name=name,
            )
    except (GenerateError, GateError) as exc:
        code, permanent = _classify(str(exc))
        if permanent:
            raise _permanent(code) from None
        raise

    with ctx.engine.begin() as conn:
        version = PostgresCvDocumentRepository(conn, ctx.user_id).add_version(
            application_id,
            generated.document,
            status="generated",
            gate_result=generated.gate_result,
            trace_id=ctx.task.id,
        )
    if version is None:
        # The application went away between the read above and this write.
        raise PermanentTaskError("no application for this user")
    return {
        "application_id": str(application_id),
        "cv_document_id": str(version.id),
        "draft_kind": CV_DOCUMENT,
    }
