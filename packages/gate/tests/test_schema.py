"""Unit tests for response parsing into the pydantic output models."""

from __future__ import annotations

import uuid

import pytest
from jfl_gate.schema import GateOutput, SentenceResult
from pydantic import ValidationError

SPAN_ID = "0425d123-ed29-5a6a-a06d-d00267574046"


def test_valid_payload_parses() -> None:
    data = {
        "sentences": [
            {
                "text": "Led a team of 12 engineers.",
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [SPAN_ID],
                "reason": "Corpus documents leading a 12-person team.",
            }
        ]
    }
    result = GateOutput.model_validate(data)
    assert len(result.sentences) == 1
    assert result.sentences[0].cited_span_ids == [uuid.UUID(SPAN_ID)]


def test_empty_cited_span_ids_is_allowed() -> None:
    result = SentenceResult.model_validate(
        {
            "text": "Wanting more autonomy, they changed teams.",
            "kind": "framing",
            "verdict": "supported",
            "drift_label": "framing",
            "cited_span_ids": [],
            "reason": "Motivation is ungroundable framing.",
        }
    )
    assert result.cited_span_ids == []


def test_unknown_verdict_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "text": "x",
                "kind": "claim",
                "verdict": "maybe",  # not one of supported/review/unsupported
                "drift_label": "supported",
                "cited_span_ids": [],
                "reason": "x",
            }
        )


def test_unknown_drift_label_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "text": "x",
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "temporal_compression",  # rejected taxonomy category
                "cited_span_ids": [],
                "reason": "x",
            }
        )


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "text": "x",
                "kind": "opinion",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [],
                "reason": "x",
            }
        )


def test_non_uuid_cited_span_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "text": "x",
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": ["not-a-uuid"],
                "reason": "x",
            }
        )


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "text": "x",
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                # "cited_span_ids" missing
                "reason": "x",
            }
        )


def test_multiple_sentences_preserve_order() -> None:
    data = {
        "sentences": [
            {
                "text": "first",
                "kind": "framing",
                "verdict": "supported",
                "drift_label": "framing",
                "cited_span_ids": [],
                "reason": "r1",
            },
            {
                "text": "second",
                "kind": "claim",
                "verdict": "unsupported",
                "drift_label": "invented_quantity",
                "cited_span_ids": [],
                "reason": "r2",
            },
        ]
    }
    result = GateOutput.model_validate(data)
    assert [s.text for s in result.sentences] == ["first", "second"]
