"""Domain schemas. Pydantic, no HTTP or framework types anywhere in core."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

Provenance = Literal["document", "adjudicated"]
SpanKind = Literal["bullet", "paragraph", "heading"]
JobSource = Literal["paste", "file"]
Necessity = Literal["essential", "desirable", "unstated"]
# Deliberately not met/unmet -- coverage is measured against the corpus, never
# against the candidate. See CLAUDE.md's decisions log.
CoverageStatus = Literal["evidenced", "partial", "absent", "contradicted"]
QuestionStatus = Literal["open", "answered", "dismissed"]


class Sentence(BaseModel):
    idx: int
    start_offset: int
    end_offset: int


class Span(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    document_id: uuid.UUID | None = None
    provenance: Provenance
    kind: SpanKind
    section_path: str | None = None
    ordinal: int | None = None
    text: str
    content_hash: str
    char_start: int | None = None  # offsets into the source document; None for adjudicated spans
    char_end: int | None = None
    sentences: list[Sentence] = Field(default_factory=list)


class SpanCandidate(BaseModel):
    span: Span
    score: float


class RunRecord(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    trace_id: uuid.UUID
    parent_run_id: uuid.UUID | None = None
    component: Literal["gate", "evals", "ingest", "generate"]
    stage: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: Decimal | None = None
    latency_ms: int | None = None
    outcome: Literal["ok", "error", "refused", "skipped"] = "ok"
    error: str | None = None
    attributes: dict[str, object] | None = None
    started_at: dt.datetime


# --------------------------------------------------------------------------
# Generation (domain 2a): a job anchors extracted requirements, which anchor
# per-run coverage rows and gap questions. Server-defaulted timestamps
# (`created_at`) are left off these models, same convention as `Span` above --
# they are not known until the row is written, and no caller here needs them
# back. `JobSummary` is the one exception, built for `jfl job list`'s display.
# --------------------------------------------------------------------------


class Job(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    source: JobSource
    employer: str | None = None
    title: str | None = None
    location: str | None = None
    url: str | None = None
    raw_text: str
    content_hash: str


class JobRequirement(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID
    ordinal: int  # order in the ad; ordering only, NOT part of the id
    text: str
    necessity: Necessity


class JobSummary(BaseModel):
    """One row of `jfl job list` -- not a table, just what that listing needs."""

    id: uuid.UUID
    employer: str | None
    title: str | None
    requirement_count: int
    created_at: dt.datetime


class RequirementCoverage(BaseModel):
    """One row of an append-only history: a fresh row per coverage run, never an
    update. See `requirement_coverage` in tables.py for why.
    """

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    requirement_id: uuid.UUID
    trace_id: uuid.UUID  # groups every requirement checked in one coverage run
    status: CoverageStatus
    cited_span_ids: list[uuid.UUID]
    evidence_note: str


class GapQuestion(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    requirement_id: uuid.UUID
    question: str
    status: QuestionStatus = "open"
    answer_text: str | None = None
    answered_at: dt.datetime | None = None
    resulting_span_id: uuid.UUID | None = None


# --------------------------------------------------------------------------
# Generation (domain 2b-core): a draft anchored on a job, gated automatically.
# See CLAUDE.md's decisions log, "The claim gate runs automatically on
# generated text." `gate_result` is `jfl_gate.schema.GateOutput.model_dump()`
# -- kept as a plain dict here rather than typed against that model, since
# `jfl_core` has no dependency on `jfl_gate` (core holds no HTTP/framework/
# other-package types; see CLAUDE.md's architectural constraints).
# --------------------------------------------------------------------------

DraftKind = Literal["cv_bullets", "cover_letter"]


class Draft(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    job_id: uuid.UUID
    kind: DraftKind
    text: str
    gate_result: dict[str, object]
    trace_id: uuid.UUID


# --------------------------------------------------------------------------
# Application tracker (slice A5). No model call anywhere in this slice --
# see CLAUDE.md's build order. `created_at`/`updated_at`/`occurred_at` are
# NOT left off these models the way `Job`'s are: the list and detail screens
# these exist for show timestamps on every row, so the repository always
# populates them from what Postgres actually wrote (via `RETURNING`), never
# guessed client-side.
# --------------------------------------------------------------------------

ApplicationStatus = Literal[
    "interested", "applied", "screening", "interviewing", "offer", "rejected", "withdrawn"
]

# Slice B3. Where the background read of the pasted ad has got to -- deliberately
# a separate axis from `ApplicationStatus`, which is where the application is in
# the world. An extraction that failed says nothing about whether the user has
# applied, and collapsing the two would make one lie about the other.
ExtractionStatus = Literal["none", "pending", "done", "failed"]

# A closed set, and never a message. The worker writes this while holding the
# user's decrypted API key, and a free-text error column is exactly where a
# careless `str(exc)` from the SDK ends up. Wording lives in the web layer, where
# it can change without a migration.
ExtractionErrorCode = Literal[
    "no_api_key",
    "api_key_rejected",
    "no_job_ad",
    "ad_too_long",
    "model_refused",
    "model_error",
    "credential_unreadable",
]


class Application(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    job_id: uuid.UUID | None = None
    title: str
    employer: str | None = None
    url: str | None = None
    status: ApplicationStatus
    source: str | None = None
    notes: str | None = None
    extraction_status: ExtractionStatus = "none"
    extraction_error_code: ExtractionErrorCode | None = None
    extracted_at: dt.datetime | None = None
    # True while `title` is a placeholder taken from the ad's first line. See
    # `jfl_core.db.tables.applications` for why this is a column and not a guess.
    title_is_provisional: bool = False
    created_at: dt.datetime
    updated_at: dt.datetime


class ApplicationEvent(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    application_id: uuid.UUID
    from_status: ApplicationStatus | None = None
    to_status: ApplicationStatus
    note: str | None = None
    occurred_at: dt.datetime
    created_at: dt.datetime


class ApplicationDetail(BaseModel):
    """What the detail screen needs: the row, plus its full timeline in
    chronological order (oldest first -- how a timeline reads).
    """

    application: Application
    events: list[ApplicationEvent]


class ExtractionInput(BaseModel):
    """What the `extract_job_ad` handler needs to do its work, read out of the
    database under the task's own `user_id` rather than carried in the payload.

    The ad text is deliberately NOT in the task payload. It is already stored
    once, verbatim, in `jobs.raw_text`; a second copy in `tasks.payload` would
    be a second place a user's job ad lives, read back by admin queries and
    quoted into log lines, for no gain over a scoped read of the row.
    """

    application_id: uuid.UUID
    job_id: uuid.UUID
    raw_text: str


class ApplicationExtraction(BaseModel):
    """The extraction panel's whole state, in one read.

    `status` is the only thing the UI needs while work is in flight; the rest is
    the result, and is empty until `status == "done"`.
    """

    application_id: uuid.UUID
    status: ExtractionStatus
    error_code: ExtractionErrorCode | None = None
    extracted_at: dt.datetime | None = None
    has_job_ad: bool = False
    employer: str | None = None
    title: str | None = None
    location: str | None = None
    requirements: list[JobRequirement] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Background work (slice B1). A row in `tasks`, as the queue and the worker see
# it -- see `jfl_core.storage.tasks`.
# --------------------------------------------------------------------------

TaskStatus = Literal["pending", "running", "succeeded", "failed"]


class Task(BaseModel):
    """One unit of background work.

    `payload` carries `repr=False` deliberately. This object reaches log lines
    and tracebacks, and while a payload is only ever meant to hold arguments
    (see `tables.py`), "only ever meant to" is not a guarantee -- so the default
    repr shows the id, kind and status and leaves the arguments out. Code that
    genuinely needs the payload asks for `task.payload`, which is a decision
    someone made rather than a field that came along for the ride.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict, repr=False)
    status: TaskStatus
    attempts: int
    max_attempts: int
    last_error: str | None = None
    scheduled_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class ReclaimResult(BaseModel):
    """What one sweep of the stale-`running` reclaim did.

    Two lists, not one count, because they mean different things to whoever is
    reading the logs: `requeued` is a worker that died and a task that will run
    again, `failed` is a task that died for the last time and now needs a human.
    """

    requeued: list[uuid.UUID] = Field(default_factory=list)
    failed: list[uuid.UUID] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.requeued or self.failed)
