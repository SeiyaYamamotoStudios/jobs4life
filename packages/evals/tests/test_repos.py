"""Unit tests for the in-memory repositories (jfl_evals.repos).

No Postgres, no network -- these just exercise the plain Python behaviour and,
via mypy strict elsewhere in the toolchain, structurally satisfy
jfl_core.repositories.GroundingRepository / RunRepository.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from jfl_core.models import RunRecord, Span
from jfl_core.repositories import GroundingRepository, RunRepository
from jfl_evals.repos import InMemoryGroundingRepository, InMemoryRunRepository

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
OTHER_USER = uuid.uuid4()


def _span(text: str = "Evidence sentence.") -> Span:
    return Span(
        id=uuid.uuid4(),
        user_id=USER,
        document_id=uuid.uuid4(),
        provenance="document",
        kind="paragraph",
        section_path="FEVER evidence [fever-1]",
        ordinal=0,
        text=text,
        content_hash="0" * 64,
    )


class TestInMemoryGroundingRepository:
    def test_satisfies_the_grounding_repository_protocol(self) -> None:
        repo: GroundingRepository = InMemoryGroundingRepository([])
        assert repo is not None

    def test_all_spans_returns_only_the_seeded_spans_for_that_user(self) -> None:
        mine = _span()
        repo = InMemoryGroundingRepository([mine])
        assert repo.all_spans(USER) == [mine]
        assert repo.all_spans(OTHER_USER) == []

    def test_get_span_finds_a_seeded_span_by_id(self) -> None:
        span = _span()
        repo = InMemoryGroundingRepository([span])
        assert repo.get_span(USER, span.id) is span

    def test_get_span_returns_none_for_an_unknown_id(self) -> None:
        repo = InMemoryGroundingRepository([_span()])
        assert repo.get_span(USER, uuid.uuid4()) is None

    def test_get_span_is_scoped_to_the_requesting_user(self) -> None:
        span = _span()
        repo = InMemoryGroundingRepository([span])
        assert repo.get_span(OTHER_USER, span.id) is None

    def test_add_adjudicated_span_appends_and_is_then_visible_to_all_spans(self) -> None:
        repo = InMemoryGroundingRepository([])
        new_span = _span("A verbatim gap answer.")
        returned_id = repo.add_adjudicated_span(USER, new_span)
        assert returned_id == new_span.id
        assert repo.all_spans(USER) == [new_span]

    def test_search_is_not_implemented(self) -> None:
        """Retrieval is unused in v1 and the baseline gate never calls this -- a
        real caller should find out immediately rather than get silent zero results.
        """
        repo = InMemoryGroundingRepository([])
        with pytest.raises(NotImplementedError):
            repo.search(USER, [0.1, 0.2])


class TestInMemoryRunRepository:
    def test_satisfies_the_run_repository_protocol(self) -> None:
        repo: RunRepository = InMemoryRunRepository()
        assert repo is not None

    def test_record_appends_in_call_order(self) -> None:
        repo = InMemoryRunRepository()
        first = RunRecord(
            user_id=USER,
            trace_id=uuid.uuid4(),
            component="evals",
            stage="baseline",
            outcome="ok",
            started_at=datetime.now(UTC),
        )
        second = RunRecord(
            user_id=USER,
            trace_id=uuid.uuid4(),
            component="evals",
            stage="baseline",
            outcome="error",
            started_at=datetime.now(UTC),
        )
        repo.record(first)
        repo.record(second)
        assert repo.records == [first, second]
