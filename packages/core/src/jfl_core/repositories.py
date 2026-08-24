"""Storage interfaces. No SQL exists above this layer.

Note what is absent: GroundingRepository has no method that returns a sent
document or a sent span. The isolation of the sent-document store is a property
of the interface, not of a WHERE clause someone has to remember.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from jfl_core.models import RunRecord, Span, SpanCandidate


class GroundingRepository(Protocol):
    """Everything the gate is allowed to ground a claim against."""

    def get_span(self, user_id: str, span_id: uuid.UUID) -> Span | None: ...

    def search(
        self, user_id: str, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]: ...

    def all_spans(self, user_id: str, include_retired: bool = False) -> list[Span]:
        """Used by the stuff-everything baseline."""
        ...

    def add_adjudicated_span(self, user_id: str, span: Span) -> uuid.UUID: ...


class SentDocumentRepository(Protocol):
    """Consistency comparison only. Deliberately a separate interface."""

    def recent(self, user_id: str, limit: int = 20) -> list[str]: ...


class RunRepository(Protocol):
    def record(self, run: RunRecord) -> None: ...
