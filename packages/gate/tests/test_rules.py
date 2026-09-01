"""Unit tests for the deterministic rule tier (jfl_gate.rules.apply_rules).

Pure function, no model and no database -- every test builds a `GateOutput` and a
list of `Span` fixtures directly. The single most load-bearing test is the framing
one: a rule that touches framing is the failure CLAUDE.md calls out repeatedly as
the one that gets this tool switched off.
"""

from __future__ import annotations

import uuid

from jfl_core.models import Span
from jfl_gate.rules import apply_rules
from jfl_gate.schema import GateOutput, SentenceResult

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")

_BOUNDARY_SECTION = "Things stated explicitly as NOT true, or as boundaries to hold"


def _span(text: str, section_path: str = "Northwind Robotics") -> Span:
    return Span(
        id=uuid.uuid4(),
        user_id=USER,
        document_id=uuid.uuid4(),
        provenance="document",
        kind="bullet",
        section_path=section_path,
        ordinal=0,
        text=text,
        content_hash="0" * 64,
    )


def _sentence(
    text: str = "Led the platform team.",
    kind: str = "claim",
    verdict: str = "supported",
    drift_label: str = "supported",
    index: int = 1,
) -> SentenceResult:
    # `index` is required on SentenceResult (it is what the model returns instead of
    # echoing `text` -- see schema.py), but apply_rules never reads it: rules act on
    # `.text`, which these fixtures still set directly, exactly as check_text does
    # after its alignment check. The value here is arbitrary and untested by
    # anything in this file.
    return SentenceResult(
        index=index,
        text=text,
        kind=kind,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        drift_label=drift_label,  # type: ignore[arg-type]
        cited_span_ids=[],
        reason="Matches the corpus.",
    )


def _output(*sentences: SentenceResult) -> GateOutput:
    return GateOutput(sentences=list(sentences))


# --- the one that matters most ------------------------------------------------


def test_framing_sentence_is_left_completely_untouched() -> None:
    """A framing sentence containing both an unsourced number and a boundary term
    must not be flagged -- rules apply only to kind == "claim". Over-flagging
    framing is the failure CLAUDE.md says gets this tool switched off.
    """
    spans = [
        _span("Aria led a team of 6 engineers.", section_path="Aurora Systems"),
        _span(
            "Aria has not flown a kayelisk personally; that claim would be false.",
            section_path=_BOUNDARY_SECTION,
        ),
    ]
    sentence = _sentence(
        text="Having piloted 9999 kayelisk missions, Aria moved into management.",
        kind="framing",
        verdict="supported",
        drift_label="framing",
    )
    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0] is sentence
    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].drift_label == "framing"
    assert result.sentences[0].rule_flags == []


# --- rule 1: unsourced numbers --------------------------------------------------


def test_supported_claim_with_unsourced_number_becomes_review() -> None:
    spans = [_span("Aria led a team of 6 engineers.")]
    sentence = _sentence(text="Aria shipped the feature to 5000 users.")

    result = apply_rules(_output(sentence), spans)
    flagged = result.sentences[0]

    assert flagged.verdict == "review"
    assert flagged.drift_label == "supported"  # untouched: not the rule's to change
    assert flagged.rule_flags == ["unsourced-number"]
    assert "Matches the corpus." in flagged.reason
    assert "rule" in flagged.reason.lower()


def test_number_present_in_corpus_is_not_flagged() -> None:
    spans = [_span("Grew the team to 6 engineers.")]
    sentence = _sentence(text="Aria led a team of 6 engineers.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].rule_flags == []


def test_percent_sign_matches_a_bare_corpus_number() -> None:
    spans = [_span("Release velocity improved by 40.")]
    sentence = _sentence(text="Improved release velocity by 40%.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].rule_flags == []


def test_thousands_separator_matches_the_ungrouped_corpus_number() -> None:
    spans = [_span("Reached 1200 signups in the first month.")]
    sentence = _sentence(text="Reached 1,200 signups in the first month.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].rule_flags == []


def test_decimal_does_not_match_a_similar_looking_whole_number() -> None:
    spans = [_span("Cut latency by 35 percent.")]
    sentence = _sentence(text="Cut latency by 3.5 percent.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].verdict == "review"
    assert result.sentences[0].rule_flags == ["unsourced-number"]


# --- rule 2: boundary contact ---------------------------------------------------


def test_distinctive_boundary_term_is_flagged() -> None:
    # "kayelisk" is rare: it only ever appears in the boundary section, so it
    # survives the corpus-wide frequency cut and is distinctive.
    spans = [
        _span("Aria has not flown a kayelisk personally.", section_path=_BOUNDARY_SECTION),
        _span("Aria led backend delivery for the payments team.", section_path="Aurora Systems"),
        _span("Aria mentored two engineers on the payments team.", section_path="Aurora Systems"),
    ]
    sentence = _sentence(text="Aria flew the kayelisk on every mission.")

    result = apply_rules(_output(sentence), spans)
    flagged = result.sentences[0]

    assert flagged.verdict == "review"
    assert flagged.rule_flags == ["boundary-contact:kayelisk"]


def test_common_term_in_the_boundary_section_is_not_flagged() -> None:
    # "engineers" appears in the boundary span and in most non-boundary spans too,
    # so it must be excluded by the frequency cut -- a corpus-wide word is not
    # distinctive just because it also happens to appear in the boundary section.
    spans = [
        _span(
            "Aria has not led a team of engineers larger than ten.", section_path=_BOUNDARY_SECTION
        ),
        _span(
            "Aria mentored several engineers on the payments team.", section_path="Aurora Systems"
        ),
        _span("Aria hired three engineers for the platform team.", section_path="Aurora Systems"),
        _span("Aria onboarded new engineers every quarter.", section_path="Aurora Systems"),
        _span("Aria reviewed code written by other engineers.", section_path="Aurora Systems"),
        _span("Aria paired with engineers across two continents.", section_path="Aurora Systems"),
    ]
    sentence = _sentence(text="Aria led a team of twelve engineers at Aurora Systems.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].rule_flags == []


def test_boundary_section_match_is_case_insensitive_and_a_marker_substring() -> None:
    spans = [
        _span(
            "Aria has not operated a self-managed carrolite reactor.",
            section_path="Things Stated Explicitly As NOT True",  # different casing/heading text
        ),
        _span("Aria operated the payments pipeline for two years.", section_path="Aurora Systems"),
        _span("Aria operated the on-call rotation each quarter.", section_path="Aurora Systems"),
    ]
    sentence = _sentence(text="Aria personally operated the carrolite unit.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].rule_flags == ["boundary-contact:carrolite"]


# --- verdicts other than "supported" are never touched --------------------------


def test_already_unsupported_claim_is_not_modified() -> None:
    spans = [
        _span("Aria has not flown a kayelisk personally.", section_path=_BOUNDARY_SECTION),
    ]
    sentence = _sentence(
        text="Aria personally flew the kayelisk on 9999 missions.",
        verdict="unsupported",
        drift_label="invented_quantity",
    )

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0] is sentence
    assert result.sentences[0].verdict == "unsupported"
    assert result.sentences[0].rule_flags == []


def test_review_claim_is_not_modified() -> None:
    spans = [
        _span("Aria has not flown a kayelisk personally.", section_path=_BOUNDARY_SECTION),
    ]
    sentence = _sentence(
        text="Aria personally flew the kayelisk on 9999 missions.",
        verdict="review",
        drift_label="ownership_inflation",
    )

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0] is sentence
    assert result.sentences[0].verdict == "review"
    assert result.sentences[0].rule_flags == []


# --- edge cases ------------------------------------------------------------------


def test_empty_corpus_does_not_raise_and_leaves_a_numberless_claim_untouched() -> None:
    sentence = _sentence(text="Aria led the platform team at Aurora Systems.")

    result = apply_rules(_output(sentence), [])

    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].rule_flags == []


def test_empty_corpus_flags_any_number_in_a_claim() -> None:
    # No corpus numbers exist at all, so any number in a claim is, correctly,
    # unsourced -- this is "behaves sanely", not "behaves leniently".
    sentence = _sentence(text="Aria shipped the feature to 5000 users.")

    result = apply_rules(_output(sentence), [])

    assert result.sentences[0].verdict == "review"
    assert result.sentences[0].rule_flags == ["unsourced-number"]


def test_corpus_with_no_boundary_section_never_flags_boundary_contact() -> None:
    spans = [
        _span("Aria led backend delivery for the payments team.", section_path="Aurora Systems"),
    ]
    sentence = _sentence(text="Aria personally flew the kayelisk on every mission.")

    result = apply_rules(_output(sentence), spans)

    assert result.sentences[0].verdict == "supported"
    assert result.sentences[0].rule_flags == []


# --- immutability ------------------------------------------------------------------


def test_apply_rules_does_not_mutate_its_input() -> None:
    spans = [
        _span("Aria has not flown a kayelisk personally.", section_path=_BOUNDARY_SECTION),
    ]
    sentence = _sentence(text="Aria personally flew the kayelisk on 9999 missions.")
    output = _output(sentence)

    result = apply_rules(output, spans)

    assert result is not output
    assert output.sentences[0] is sentence
    assert sentence.verdict == "supported"
    assert sentence.rule_flags == []
    assert sentence.reason == "Matches the corpus."
