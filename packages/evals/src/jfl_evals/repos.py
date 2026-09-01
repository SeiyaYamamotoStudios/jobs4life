"""In-memory implementations of the two Protocols `check_text` needs
(`jfl_core.repositories.GroundingRepository` and `RunRepository`), so the eval runs
with no Postgres anywhere.

CLAUDE.md: "Inspect eval logs stay as Inspect's own files on disk, never in
Postgres." These exist so nothing in the eval path ever imports
`jfl_core.storage.postgres` or opens a connection.

Both are Protocols (structural typing), not ABCs -- there is nothing to subclass,
only methods to match. mypy's structural check is what actually verifies these
satisfy the interfaces `check_text` was written against; `tests/test_repos.py`
additionally exercises the behaviour.
"""

from __future__ import annotations

import uuid

from jfl_core.models import RunRecord, Span, SpanCandidate


class InMemoryGroundingRepository:
    """Seeded once, per golden item, with that item's evidence spans -- see
    `jfl_evals.spans.build_spans`. `all_spans` is the only method the baseline
    gate actually calls (`jfl_gate.gate.check_text`); the rest exist to satisfy
    the Protocol and behave sanely if something ever does call them.
    """

    def __init__(self, spans: list[Span]) -> None:
        self._spans: list[Span] = list(spans)

    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None:
        return next((s for s in self._spans if s.user_id == user_id and s.id == span_id), None)

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]:
        # Retrieval is unused in v1 (CLAUDE.md, "Retrieval unused in v1") and the
        # baseline gate never calls this -- there is no sensible in-memory
        # embedding search to fake, so a real caller finds out immediately.
        raise NotImplementedError("search() is unused by the baseline gate in v1")

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        return [s for s in self._spans if s.user_id == user_id]

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID:
        self._spans.append(span)
        return span.id


class InMemoryRunRepository:
    """Collects every `RunRecord` `check_text` writes, in call order, so the eval
    can total tokens and cost across a run without a `runs` table.
    """

    def __init__(self) -> None:
        self.records: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self.records.append(run)
