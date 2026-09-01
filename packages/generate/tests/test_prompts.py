"""Unit tests for prompt assembly: the extraction prompt, and the coverage
system/user split that prompt caching depends on -- the same property
jfl_gate/tests/test_prompt.py checks for the claim gate.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from jfl_core.models import Span
from jfl_generate.prompts import (
    COVERAGE_OUTPUT_SCHEMA,
    EXTRACT_OUTPUT_SCHEMA,
    build_coverage_system_prompt,
    build_coverage_user_message,
    build_extract_prompt,
)

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
_EXTRACT_SCHEMA = cast("dict[str, Any]", EXTRACT_OUTPUT_SCHEMA)
_COVERAGE_SCHEMA = cast("dict[str, Any]", COVERAGE_OUTPUT_SCHEMA)


def _span(text: str, section_path: str = "Northwind") -> Span:
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


# --- extraction ----------------------------------------------------------------


def test_extract_prompt_is_constant() -> None:
    """No corpus, no ad text baked in -- the ad is volatile and belongs in the
    user message, assembled by the caller.
    """
    assert build_extract_prompt() == build_extract_prompt()


def test_extract_prompt_mentions_splitting_compound_requirements() -> None:
    prompt = build_extract_prompt().lower()
    assert "atomic" in prompt or "separate" in prompt


def test_extract_output_schema_requires_every_field_and_forbids_extras() -> None:
    assert set(_EXTRACT_SCHEMA["required"]) == {"employer", "title", "location", "requirements"}
    assert _EXTRACT_SCHEMA["additionalProperties"] is False
    item_schema = _EXTRACT_SCHEMA["properties"]["requirements"]["items"]
    assert set(item_schema["required"]) == {"text", "necessity"}
    assert set(item_schema["properties"]["necessity"]["enum"]) == {
        "essential",
        "desirable",
        "unstated",
    }


# --- coverage --------------------------------------------------------------------


def test_coverage_system_prompt_includes_every_span_id_and_text() -> None:
    spans = [_span("Led the platform team"), _span("Shipped v2", section_path="Data")]
    prompt = build_coverage_system_prompt(spans)
    for span in spans:
        assert str(span.id) in prompt
        assert span.text in prompt


def test_coverage_system_prompt_on_empty_corpus_does_not_crash() -> None:
    prompt = build_coverage_system_prompt([])
    assert "empty" in prompt.lower()


def test_coverage_system_prompt_states_it_measures_the_corpus_not_the_candidate() -> None:
    """The one rule this prompt cannot afford to get wrong -- CLAUDE.md's decisions
    log: coverage is measured against the corpus, never against the candidate.
    """
    prompt = build_coverage_system_prompt([]).lower()
    assert "not a judgement of" in prompt or "not a statement" in prompt


def test_coverage_system_prompt_explains_absent_is_a_gap_not_a_shortcoming() -> None:
    prompt = build_coverage_system_prompt([]).lower()
    assert "gap" in prompt


def test_coverage_user_message_contains_the_requirements_under_check() -> None:
    requirements = ["5+ years of Python", "Led a team of 10"]
    message = build_coverage_user_message(requirements)
    for requirement in requirements:
        assert requirement in message


def test_volatile_requirements_are_never_baked_into_the_cached_system_prompt() -> None:
    """The system prompt (cached prefix) must not vary with the requirements under
    test -- if it did, every distinct job would silently bust the cache.
    """
    spans = [_span("Led the platform team")]
    system_a = build_coverage_system_prompt(spans)
    system_b = build_coverage_system_prompt(spans)
    assert system_a == system_b  # same corpus -> byte-identical system prompt

    for requirement in ["A distinctive requirement about rockets.", "A requirement about jam."]:
        assert requirement not in system_a


def test_coverage_output_schema_requires_every_field_and_forbids_extras() -> None:
    item_schema = _COVERAGE_SCHEMA["properties"]["results"]["items"]
    assert set(item_schema["required"]) == {"status", "cited_span_ids", "reason", "question"}
    assert item_schema["additionalProperties"] is False


def test_coverage_output_schema_status_enum_matches_the_taxonomy() -> None:
    item_schema = _COVERAGE_SCHEMA["properties"]["results"]["items"]
    assert set(item_schema["properties"]["status"]["enum"]) == {
        "evidenced",
        "partial",
        "absent",
        "contradicted",
    }
