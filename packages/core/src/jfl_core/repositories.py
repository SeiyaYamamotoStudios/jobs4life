"""Storage interfaces. No SQL exists above this layer.

Note what is absent: GroundingRepository has no method that returns a sent
document or a sent span. The isolation of the sent-document store is a property
of the interface, not of a WHERE clause someone has to remember.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from jfl_core.models import (
    DocumentStorageKind,
    Draft,
    GapQuestion,
    Job,
    JobRequirement,
    JobSummary,
    RequirementCoverage,
    RunRecord,
    Span,
    SpanCandidate,
)


class IngestRepository(Protocol):
    """What corpus ingestion needs. Separate from GroundingRepository because
    ingestion writes and the gate only reads -- splitting them keeps the
    write surface small enough to audit.
    """

    def upsert_document(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        source_uri: str,
        title: str | None,
        content_hash: str,
        storage_kind: DocumentStorageKind = "local_file",
        text: str | None = None,
    ) -> bool:
        """Insert or refresh a document row. Returns True if this created a new row.

        `storage_kind` defaults to `local_file`, which is the CLI walking
        `corpus/*.md`: the markdown is a file on disk and this row only indexes
        it. A `hosted` document has no file -- this deployment holds the
        markdown in `text`, which is what keeps "markdown is the source of
        truth" true for a user who has no disk here. See
        `jfl_core.corpus_source`.
        """
        ...

    def document_text(self, user_id: uuid.UUID, document_id: uuid.UUID) -> str | None:
        """The stored markdown for a hosted document, or None.

        None for a `local_file` document too: the file is the source of truth
        there, and `jfl_core.ingest.source` is what reads it.
        """
        ...

    def upsert_span(self, span: Span) -> bool:
        """Insert or refresh a span and its sentences. Returns True if newly created."""
        ...

    def retire_document_spans(
        self, user_id: uuid.UUID, document_id: uuid.UUID, seen: set[uuid.UUID]
    ) -> int:
        """Retire one document's spans not in `seen`, leaving every other
        document alone. What re-parsing a single edited document needs.
        """
        ...

    def retire_missing_documents(self, user_id: uuid.UUID, seen: set[uuid.UUID]) -> int:
        """Retire documents not in `seen`. Returns the count retired."""
        ...

    def retire_missing_spans(self, user_id: uuid.UUID, seen: set[uuid.UUID]) -> int:
        """Retire provenance='document' spans not in `seen`. Never touches adjudicated
        spans -- they have no source file and nothing ingestion sees could confirm
        or refute their presence.
        """
        ...


class GroundingRepository(Protocol):
    """Everything the gate is allowed to ground a claim against."""

    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None: ...

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]: ...

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        """Used by the stuff-everything baseline."""
        ...

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID: ...


class SentDocumentRepository(Protocol):
    """Consistency comparison only. Deliberately a separate interface."""

    def recent(self, user_id: uuid.UUID, limit: int = 20) -> list[str]: ...


class RunRepository(Protocol):
    def record(self, run: RunRecord) -> None: ...


class JobRepository(Protocol):
    """What slice 2a (job intake, requirement extraction, coverage, gap questions)
    needs. Separate from `GroundingRepository` because coverage *reads* the corpus
    through that interface and writes an answered gap back through its
    `add_adjudicated_span` -- this interface owns the job-shaped tables only.
    """

    def upsert_job(self, job: Job) -> bool:
        """Insert or refresh a job row (id is deterministic, see ids.py). Returns
        True if this created a new row.
        """
        ...

    def replace_requirements(
        self, user_id: uuid.UUID, job_id: uuid.UUID, requirements: list[JobRequirement]
    ) -> None:
        """Replace every requirement row for a job: requirement ids are deterministic
        from (job_id, text), so re-extraction after an edited ad is a clean swap, not
        a diff.
        """
        ...

    def list_jobs(self, user_id: uuid.UUID) -> list[JobSummary]: ...

    def get_job(
        self, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> tuple[Job, list[JobRequirement]] | None: ...

    def record_coverage(self, coverage: RequirementCoverage) -> None:
        """Append-only insert: one row per requirement, per coverage run."""
        ...

    def latest_coverage(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[RequirementCoverage]:
        """The most recent coverage row per requirement -- `DISTINCT ON
        (requirement_id) ... ORDER BY requirement_id, created_at DESC`.
        """
        ...

    def upsert_gap_question(self, question: GapQuestion) -> None:
        """Insert a gap question, or refresh its `question` text -- but only while
        the existing row is still `open`. An `answered` or `dismissed` row is never
        overwritten by a later coverage run.
        """
        ...

    def list_open_questions(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[GapQuestion]: ...

    def get_question(self, user_id: uuid.UUID, question_id: uuid.UUID) -> GapQuestion | None: ...

    def mark_question_answered(
        self,
        user_id: uuid.UUID,
        question_id: uuid.UUID,
        answer_text: str,
        resulting_span_id: uuid.UUID,
    ) -> None: ...

    def record_draft(self, draft: Draft) -> None:
        """Insert a draft row. Id is random (see `Draft.id`), so this is a plain
        insert, never an upsert.
        """
        ...

    def list_drafts(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[Draft]:
        """Drafts for a job, most recent first."""
        ...


class JobSource(Protocol):
    """Pluggable intake.

    ATS JSON endpoints, RSS, a forwarded-email inbox and manual paste all land in
    the same queue, so they resolve to the same shape here. `cursor` is opaque and
    per-kind -- an etag, a last-seen id, a date -- and is round-tripped through
    `job_sources.cursor` rather than interpreted by the caller.

    No implementations yet: intake is domain 3 and RawPosting is not designed.
    """

    kind: str

    def fetch(
        self, config: dict[str, object], cursor: str | None
    ) -> tuple[list[dict[str, object]], str | None]: ...
