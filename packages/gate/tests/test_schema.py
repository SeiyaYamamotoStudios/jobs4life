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
                "index": 1,
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [SPAN_ID],
                "evidence_note": "Corpus documents leading a 12-person team.",
            }
        ]
    }
    result = GateOutput.model_validate(data)
    assert len(result.sentences) == 1
    assert result.sentences[0].cited_span_ids == [uuid.UUID(SPAN_ID)]


def test_text_defaults_empty_since_the_model_is_never_asked_for_it() -> None:
    """The wire payload carries `index`, never `text` -- `text` is filled in later,
    by jfl_gate.gate.check_text, from the input sentence list. Parsed on its own
    (as here), it stays at its default.
    """
    result = SentenceResult.model_validate(
        {
            "index": 1,
            "kind": "framing",
            "verdict": "supported",
            "drift_label": "framing",
            "cited_span_ids": [],
            "evidence_note": "Motivation is ungroundable framing.",
        }
    )
    assert result.text == ""


def test_empty_cited_span_ids_is_allowed() -> None:
    result = SentenceResult.model_validate(
        {
            "index": 1,
            "kind": "framing",
            "verdict": "supported",
            "drift_label": "framing",
            "cited_span_ids": [],
            "evidence_note": "Motivation is ungroundable framing.",
        }
    )
    assert result.cited_span_ids == []


def test_unknown_verdict_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "index": 1,
                "kind": "claim",
                "verdict": "maybe",  # not one of supported/review/unsupported
                "drift_label": "supported",
                "cited_span_ids": [],
                "evidence_note": "x",
            }
        )


def test_unknown_drift_label_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "index": 1,
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "temporal_compression",  # rejected taxonomy category
                "cited_span_ids": [],
                "evidence_note": "x",
            }
        )


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "index": 1,
                "kind": "opinion",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [],
                "evidence_note": "x",
            }
        )


def test_non_uuid_cited_span_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "index": 1,
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": ["not-a-uuid"],
                "evidence_note": "x",
            }
        )


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "index": 1,
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                # "cited_span_ids" missing
                "evidence_note": "x",
            }
        )


def test_missing_index_is_rejected() -> None:
    """`index` is the field that replaced echoed text -- unlike `text`, it has no
    default, since a result the alignment check cannot place is exactly the failure
    mode the wire-format change must not introduce silently.
    """
    with pytest.raises(ValidationError):
        SentenceResult.model_validate(
            {
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [],
                "evidence_note": "x",
            }
        )


def test_multiple_sentences_preserve_order() -> None:
    data = {
        "sentences": [
            {
                "index": 1,
                "kind": "framing",
                "verdict": "supported",
                "drift_label": "framing",
                "cited_span_ids": [],
                "evidence_note": "r1",
            },
            {
                "index": 2,
                "kind": "claim",
                "verdict": "unsupported",
                "drift_label": "invented_quantity",
                "cited_span_ids": [],
                "evidence_note": "r2",
            },
        ]
    }
    result = GateOutput.model_validate(data)
    assert [s.index for s in result.sentences] == [1, 2]
