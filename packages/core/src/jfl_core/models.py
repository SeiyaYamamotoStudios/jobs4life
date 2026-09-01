"""Domain schemas. Pydantic, no HTTP or framework types anywhere in core."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from typing import Literal

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
    reason: str


class GapQuestion(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    requirement_id: uuid.UUID
    question: str
    status: QuestionStatus = "open"
    answer_text: str | None = None
    answered_at: dt.datetime | None = None
    resulting_span_id: uuid.UUID | None = None
