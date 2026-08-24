"""Storage interfaces. No SQL exists above this layer.

Note what is absent: GroundingRepository has no method that returns a sent
document or a sent span. The isolation of the sent-document store is a property
of the interface, not of a WHERE clause someone has to remember.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from jfl_core.models import RunRecord, Span, SpanCandidate


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
    ) -> bool:
        """Insert or refresh a document row (source_uri, storage_kind='local_file').

        Returns True if this created a new row.
        """
        ...

    def upsert_span(self, span: Span) -> bool:
        """Insert or refresh a span and its sentences. Returns True if newly created."""
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
    ) -> tuple[list[dict], str | None]: ...
