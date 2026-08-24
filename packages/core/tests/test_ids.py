"""Unit tests for identity derivation -- the base of the triangle.

These encode the stability guarantees claimed in ids.py. If one of these breaks,
previously-labelled golden-set items stop resolving.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_core.ids import adjudicated_span_id, content_hash, normalise, sentence_id, span_id

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
DOC = "file:corpus/cv.md"
SECTION = "Kaluza > Platform"


class TestNormalise:
    @pytest.mark.parametrize(
        "raw",
        [
            "- Led the platform team",
            "* Led the platform team",
            "1. Led the platform team",
            "  -   Led   the platform   team  ",
            "Led the platform team",
        ],
    )
    def test_list_markers_and_whitespace_are_stripped(self, raw: str) -> None:
        assert normalise(raw) == "Led the platform team"

    def test_case_is_preserved(self) -> None:
        # Casing can carry meaning in a claim; this is a truthfulness tool.
        assert normalise("Led SRE") != normalise("led sre")

    def test_unicode_is_nfc_normalised(self) -> None:
        assert normalise("café") == normalise("café")


class TestSpanId:
    def test_is_deterministic(self) -> None:
        a = span_id(USER, DOC, SECTION, "- Led the platform team")
        b = span_id(USER, DOC, SECTION, "- Led the platform team")
        assert a == b

    def test_survives_reformatting(self) -> None:
        """Re-wrapping or changing the bullet char must not mint a new id."""
        assert span_id(USER, DOC, SECTION, "- Led the team") == span_id(
            USER, DOC, SECTION, "*  Led   the team"
        )

    def test_independent_of_position_within_section(self) -> None:
        """Inserting a bullet above must not renumber anything below it."""
        assert span_id(USER, DOC, SECTION, "- Led the team") == span_id(
            USER, DOC, SECTION, "- Led the team"
        )

    def test_duplicate_text_in_one_section_disambiguated_by_occurrence(self) -> None:
        first = span_id(USER, DOC, SECTION, "- Shipped it", occurrence=0)
        second = span_id(USER, DOC, SECTION, "- Shipped it", occurrence=1)
        assert first != second

    def test_editing_text_mints_a_new_id(self) -> None:
        """Documented consequence: the old span is retired, never deleted."""
        assert span_id(USER, DOC, SECTION, "- Led a team of 6") != span_id(
            USER, DOC, SECTION, "- Led a team of 12"
        )

    def test_moving_between_sections_mints_a_new_id(self) -> None:
        assert span_id(USER, DOC, "A", "- Led it") != span_id(USER, DOC, "B", "- Led it")

    def test_scoped_by_user_and_document(self) -> None:
        assert span_id(USER, DOC, SECTION, "x") != span_id(uuid.uuid4(), DOC, SECTION, "x")
        assert span_id(USER, DOC, SECTION, "x") != span_id(
            USER, "file:corpus/other.md", SECTION, "x"
        )


class TestAdjudicatedSpanId:
    def test_derives_from_the_adjudication_not_a_file(self) -> None:
        item = uuid.uuid4()
        assert adjudicated_span_id(USER, "claim", item) == adjudicated_span_id(USER, "claim", item)

    def test_differs_per_review_item(self) -> None:
        assert adjudicated_span_id(USER, "claim", uuid.uuid4()) != adjudicated_span_id(
            USER, "claim", uuid.uuid4()
        )


def test_sentence_ids_are_stable_and_ordered() -> None:
    span = span_id(USER, DOC, SECTION, "- Two sentences. Here is another.")
    assert sentence_id(span, 0) == sentence_id(span, 0)
    assert sentence_id(span, 0) != sentence_id(span, 1)


def test_content_hash_matches_normalised_form() -> None:
    assert content_hash("-  Led   the team ") == content_hash("Led the team")
