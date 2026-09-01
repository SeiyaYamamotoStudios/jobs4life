"""Unit tests for response parsing into the pydantic output models."""

from __future__ import annotations

import uuid

import pytest
from jfl_generate.schema import CoverageOutput, ExtractOutput, RequirementCoverageResult
from pydantic import ValidationError

SPAN_ID = "0425d123-ed29-5a6a-a06d-d00267574046"


class TestExtractOutput:
    def test_valid_payload_parses(self) -> None:
        data = {
            "employer": "Acme Corp",
            "title": "Senior Engineer",
            "location": "Remote",
            "requirements": [
                {"text": "5+ years of Python", "necessity": "essential"},
                {"text": "Kubernetes experience", "necessity": "desirable"},
            ],
        }
        result = ExtractOutput.model_validate(data)
        assert result.employer == "Acme Corp"
        assert len(result.requirements) == 2
        assert result.requirements[0].necessity == "essential"

    def test_blank_employer_title_location_normalise_to_none(self) -> None:
        result = ExtractOutput.model_validate(
            {"employer": "", "title": "", "location": "  ", "requirements": []}
        )
        assert result.employer is None
        assert result.title is None
        assert result.location is None

    def test_unknown_necessity_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractOutput.model_validate(
                {
                    "employer": "Acme",
                    "title": "Engineer",
                    "location": "",
                    "requirements": [{"text": "Python", "necessity": "mandatory"}],
                }
            )

    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractOutput.model_validate(
                {
                    "employer": "Acme",
                    "title": "Engineer",
                    # "location" missing
                    "requirements": [],
                }
            )

    def test_requirements_preserve_order(self) -> None:
        result = ExtractOutput.model_validate(
            {
                "employer": "",
                "title": "",
                "location": "",
                "requirements": [
                    {"text": "first", "necessity": "essential"},
                    {"text": "second", "necessity": "unstated"},
                ],
            }
        )
        assert [r.text for r in result.requirements] == ["first", "second"]


class TestCoverageOutput:
    def test_valid_payload_parses(self) -> None:
        result = RequirementCoverageResult.model_validate(
            {
                "status": "evidenced",
                "cited_span_ids": [SPAN_ID],
                "reason": "Corpus documents this directly.",
                "question": "",
            }
        )
        assert result.status == "evidenced"
        assert result.cited_span_ids == [uuid.UUID(SPAN_ID)]
        assert result.question is None

    def test_blank_question_normalises_to_none(self) -> None:
        result = RequirementCoverageResult.model_validate(
            {
                "status": "contradicted",
                "cited_span_ids": [],
                "reason": "Corpus states the opposite.",
                "question": "",
            }
        )
        assert result.question is None

    def test_non_blank_question_is_kept(self) -> None:
        result = RequirementCoverageResult.model_validate(
            {
                "status": "absent",
                "cited_span_ids": [],
                "reason": "Corpus is silent on this.",
                "question": "Have you worked with Kubernetes?",
            }
        )
        assert result.question == "Have you worked with Kubernetes?"

    def test_unknown_status_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequirementCoverageResult.model_validate(
                {
                    "status": "met",  # not evidenced/partial/absent/contradicted
                    "cited_span_ids": [],
                    "reason": "x",
                    "question": "",
                }
            )

    def test_non_uuid_cited_span_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequirementCoverageResult.model_validate(
                {
                    "status": "evidenced",
                    "cited_span_ids": ["not-a-uuid"],
                    "reason": "x",
                    "question": "",
                }
            )

    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequirementCoverageResult.model_validate(
                {
                    "status": "evidenced",
                    "cited_span_ids": [],
                    # "reason" missing
                    "question": "",
                }
            )

    def test_multiple_results_preserve_order(self) -> None:
        data = {
            "results": [
                {"status": "evidenced", "cited_span_ids": [], "reason": "r1", "question": ""},
                {"status": "absent", "cited_span_ids": [], "reason": "r2", "question": "q2"},
            ]
        }
        result = CoverageOutput.model_validate(data)
        assert [r.status for r in result.results] == ["evidenced", "absent"]
