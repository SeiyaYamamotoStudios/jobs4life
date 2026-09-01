"""Unit tests for turning one golden item's evidence into in-memory Spans
(jfl_evals.spans.build_spans).
"""

from __future__ import annotations

import uuid

from jfl_evals.dataset import GoldenItem
from jfl_evals.spans import build_spans

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


def _item(evidence: list[str], item_id: str = "fever-1") -> GoldenItem:
    return GoldenItem(
        id=item_id,
        claim="A claim about something.",
        expected_verdict="supported",
        evidence=evidence,
        fever_label="SUPPORTS",
    )


class TestBuildSpans:
    def test_one_span_per_evidence_sentence(self) -> None:
        item = _item(["First evidence sentence.", "Second evidence sentence."])
        spans = build_spans(item, USER)
        assert len(spans) == 2
        assert [s.text for s in spans] == item.evidence

    def test_empty_evidence_yields_an_empty_corpus(self) -> None:
        item = _item([])
        assert build_spans(item, USER) == []

    def test_every_span_is_document_provenance_with_a_document_id(self) -> None:
        """tables.py's CheckConstraint ties provenance='document' to a non-null
        document_id -- these are in-memory only and the constraint never actually
        runs, but the objects should still be internally consistent with it.
        """
        item = _item(["One sentence."])
        spans = build_spans(item, USER)
        assert spans[0].provenance == "document"
        assert spans[0].document_id is not None

    def test_span_ids_are_deterministic_for_the_same_item(self) -> None:
        item = _item(["Evidence text."])
        first = build_spans(item, USER)
        second = build_spans(item, USER)
        assert first[0].id == second[0].id

    def test_different_items_never_collide_even_with_identical_evidence_text(self) -> None:
        a = build_spans(_item(["Shared evidence text."], item_id="fever-1"), USER)
        b = build_spans(_item(["Shared evidence text."], item_id="fever-2"), USER)
        assert a[0].id != b[0].id

    def test_duplicate_evidence_sentences_within_one_item_get_distinct_ids(self) -> None:
        item = _item(["Repeated sentence.", "Repeated sentence."])
        spans = build_spans(item, USER)
        assert spans[0].id != spans[1].id

    def test_spans_are_ordered_by_evidence_position(self) -> None:
        item = _item(["First.", "Second.", "Third."])
        spans = build_spans(item, USER)
        assert [s.ordinal for s in spans] == [0, 1, 2]
