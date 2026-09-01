"""Unit tests for prompt assembly: corpus formatting and the system/user split that
prompt caching depends on.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from jfl_core.models import Span
from jfl_gate.prompt import GATE_OUTPUT_SCHEMA, build_system_prompt, build_user_message

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
_SCHEMA = cast("dict[str, Any]", GATE_OUTPUT_SCHEMA)


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


def test_system_prompt_includes_every_span_id_and_text() -> None:
    spans = [_span("Led the platform team"), _span("Shipped v2", section_path="Data")]
    prompt = build_system_prompt(spans)
    for span in spans:
        assert str(span.id) in prompt
        assert span.text in prompt
        assert span.section_path is not None
        assert span.section_path in prompt


def test_system_prompt_on_empty_corpus_does_not_crash() -> None:
    prompt = build_system_prompt([])
    assert "empty" in prompt.lower()


def test_system_prompt_explains_framing_must_not_be_flagged() -> None:
    # This is the one rule the gate cannot afford to get wrong -- assert the prompt
    # actually states it, not just that the prompt is nonempty.
    prompt = build_system_prompt([])
    assert "framing" in prompt.lower()
    assert "never flag" in prompt.lower() or "do not over-flag" in prompt.lower()


def test_system_prompt_states_verdict_depends_on_corpus_not_wording() -> None:
    prompt = build_system_prompt([])
    assert "not of the sentence" in prompt.lower() or "not of the claim" in prompt.lower()


def test_system_prompt_lists_all_drift_labels_from_the_taxonomy() -> None:
    prompt = build_system_prompt([])
    for label in (
        "invented_quantity",
        "adjacency_substitution",
        "scope_inflation",
        "ownership_inflation",
        "outcome_attribution",
        "strategy_scope",
        "causality",
    ):
        assert label in prompt


def test_user_message_contains_the_sentences_under_test() -> None:
    sentences = ["Led the platform team.", "Wanting a new challenge, they moved teams."]
    message = build_user_message(sentences)
    for sentence in sentences:
        assert sentence in message


def test_volatile_sentences_are_never_baked_into_the_cached_system_prompt() -> None:
    """The system prompt (cached prefix) must not vary with the input text -- if it
    did, every distinct piece of text under test would silently bust the cache.
    """
    spans = [_span("Led the platform team")]
    sentences_a = ["A distinctive sentence about a spaceship launch."]
    sentences_b = ["A completely different sentence about baking bread."]

    system_a = build_system_prompt(spans)
    system_b = build_system_prompt(spans)
    assert system_a == system_b  # same corpus -> byte-identical system prompt

    for sentence in sentences_a + sentences_b:
        assert sentence not in system_a


def test_output_schema_requires_every_field_and_forbids_extras() -> None:
    item_schema = _SCHEMA["properties"]["sentences"]["items"]
    assert set(item_schema["required"]) == {
        "index",
        "kind",
        "verdict",
        "drift_label",
        "cited_span_ids",
        "reason",
    }
    assert item_schema["additionalProperties"] is False


def test_output_schema_enums_match_the_taxonomy() -> None:
    item_schema = _SCHEMA["properties"]["sentences"]["items"]
    assert set(item_schema["properties"]["verdict"]["enum"]) == {
        "supported",
        "review",
        "unsupported",
    }
    assert set(item_schema["properties"]["drift_label"]["enum"]) == {
        "supported",
        "invented_quantity",
        "adjacency_substitution",
        "scope_inflation",
        "ownership_inflation",
        "outcome_attribution",
        "strategy_scope",
        "causality",
        "framing",
    }
    assert set(item_schema["properties"]["kind"]["enum"]) == {"claim", "framing"}
