"""Tests for the deterministic rule tier.

The most important test in this file is the framing one: framing is the only path
through the gate with no check anywhere, and a rule must never touch it.
"""

from __future__ import annotations

import uuid

from jfl_core.models import Span
from jfl_gate.rules import apply_rules
from jfl_gate.schema import GateOutput, SentenceResult

USER = uuid.uuid4()


def _span(text: str, section: str = "Northwind > Platform") -> Span:
    return Span(
        id=uuid.uuid4(),
        user_id=USER,
        document_id=uuid.uuid4(),
        provenance="document",
        kind="bullet",
        section_path=section,
        ordinal=0,
        text=text,
        content_hash="0" * 64,
    )


def _sentence(
    text: str = "Led the platform team.",
    kind: str = "claim",
    verdict: str = "supported",
    drift_label: str = "supported",
    cited: list[uuid.UUID] | None = None,
) -> SentenceResult:
    return SentenceResult(
        index=1,
        text=text,
        kind=kind,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        drift_label=drift_label,  # type: ignore[arg-type]
        cited_span_ids=cited if cited is not None else [],
        evidence_note="Traces to the corpus.",
    )


class TestFramingIsNeverTouched:
    def test_a_framing_sentence_with_no_citation_is_left_completely_alone(self) -> None:
        """Framing is forced to `supported` by the prompt and is never checked against
        the corpus, so it cites nothing by design. Escalating it for that would flag
        every framing sentence in every document -- the failure that switches the
        tool off.
        """
        span = _span("Led the platform team of four engineers.")
        framing = _sentence(
            text="Having enjoyed the technical challenge, they stayed on.",
            kind="framing",
            drift_label="framing",
        )
        result = apply_rules(GateOutput(sentences=[framing]), [span])
        assert result.sentences[0] == framing

    def test_a_framing_sentence_citing_an_unknown_span_is_still_left_alone(self) -> None:
        framing = _sentence(kind="framing", drift_label="framing", cited=[uuid.uuid4()])
        result = apply_rules(GateOutput(sentences=[framing]), [_span("Anything.")])
        assert result.sentences[0] == framing


class TestUncitedSupport:
    def test_supported_with_no_citation_becomes_review(self) -> None:
        span = _span("Led the platform team.")
        result = apply_rules(GateOutput(sentences=[_sentence()]), [span])
        got = result.sentences[0]
        assert got.verdict == "review"
        assert got.rule_flags == ["uncited-support"]
        assert "cites no corpus span" in got.evidence_note

    def test_the_drift_label_is_not_changed(self) -> None:
        """The taxonomy is the model's classification. A heuristic does not relabel."""
        span = _span("Led the platform team.")
        result = apply_rules(GateOutput(sentences=[_sentence()]), [span])
        assert result.sentences[0].drift_label == "supported"

    def test_supported_with_a_real_citation_is_untouched(self) -> None:
        span = _span("Led the platform team.")
        sentence = _sentence(cited=[span.id])
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        assert result.sentences[0] == sentence

    def test_an_uncited_review_is_not_escalated(self) -> None:
        """`review` already asks for attention; there is nowhere to escalate it to,
        and `unsupported` is a verdict no rule may reach.
        """
        span = _span("Led the platform team.")
        sentence = _sentence(verdict="review", drift_label="ownership_inflation")
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        assert result.sentences[0] == sentence

    def test_an_uncited_unsupported_is_not_modified(self) -> None:
        span = _span("Led the platform team.")
        sentence = _sentence(verdict="unsupported", drift_label="invented_quantity")
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        assert result.sentences[0] == sentence


class TestUnknownCitation:
    def test_a_fabricated_span_id_escalates_a_supported_claim(self) -> None:
        span = _span("Led the platform team.")
        ghost = uuid.uuid4()
        sentence = _sentence(cited=[ghost])
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        got = result.sentences[0]
        assert got.verdict == "review"
        assert got.rule_flags == [f"unknown-citation:{ghost}"]
        assert str(ghost) in got.evidence_note

    def test_a_fabricated_id_is_flagged_on_a_review_sentence_without_moving_it(self) -> None:
        """A fabricated citation is worth recording wherever it appears, but the
        verdict contract still holds: only `supported` ever moves.
        """
        span = _span("Led the platform team.")
        ghost = uuid.uuid4()
        sentence = _sentence(verdict="review", drift_label="scope_inflation", cited=[ghost])
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        got = result.sentences[0]
        assert got.verdict == "review"
        assert got.rule_flags == [f"unknown-citation:{ghost}"]

    def test_a_mix_of_real_and_fabricated_ids_flags_only_the_fabricated_one(self) -> None:
        span = _span("Led the platform team.")
        ghost = uuid.uuid4()
        sentence = _sentence(cited=[span.id, ghost])
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        assert result.sentences[0].rule_flags == [f"unknown-citation:{ghost}"]

    def test_both_rules_can_fire_on_one_sentence(self) -> None:
        """An uncited `supported` claim cannot also cite a ghost, so the overlap case
        is a `supported` claim citing only fabricated ids: that is an unknown
        citation, not uncited support.
        """
        span = _span("Led the platform team.")
        ghost = uuid.uuid4()
        result = apply_rules(GateOutput(sentences=[_sentence(cited=[ghost])]), [span])
        flags = result.sentences[0].rule_flags
        assert flags == [f"unknown-citation:{ghost}"]
        assert "uncited-support" not in flags


class TestDegenerateInputs:
    def test_an_empty_corpus_does_not_raise(self) -> None:
        result = apply_rules(GateOutput(sentences=[_sentence()]), [])
        assert result.sentences[0].verdict == "review"

    def test_no_sentences_does_not_raise(self) -> None:
        assert apply_rules(GateOutput(sentences=[]), [_span("Anything.")]).sentences == []

    def test_existing_rule_flags_are_preserved_not_replaced(self) -> None:
        span = _span("Led the platform team.")
        sentence = _sentence().model_copy(update={"rule_flags": ["earlier-flag"]})
        result = apply_rules(GateOutput(sentences=[sentence]), [span])
        assert result.sentences[0].rule_flags == ["earlier-flag", "uncited-support"]


def test_apply_rules_does_not_mutate_its_input() -> None:
    span = _span("Led the platform team.")
    sentence = _sentence()
    output = GateOutput(sentences=[sentence])
    apply_rules(output, [span])
    assert output.sentences[0].verdict == "supported"
    assert output.sentences[0].rule_flags == []
