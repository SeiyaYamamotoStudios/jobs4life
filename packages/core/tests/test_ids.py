"""Unit tests for identity derivation -- the base of the triangle.

These encode the stability guarantees claimed in ids.py. If one of these breaks,
previously-labelled golden-set items stop resolving.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_core.ids import (
    adjudicated_span_id,
    adjudicated_span_id_from_answer,
    content_hash,
    gap_question_id,
    job_id,
    normalise,
    requirement_id,
    sentence_id,
    span_id,
)

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
DOC = "file:corpus/cv.md"
SECTION = "Northwind > Platform"


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


class TestJobId:
    def test_is_deterministic(self) -> None:
        ad = "Senior Engineer at Acme. Must know Python."
        assert job_id(USER, ad) == job_id(USER, ad)

    def test_re_pasting_the_same_ad_resolves_to_the_same_id(self) -> None:
        """Whitespace-only differences must not mint a new job -- content_hash
        normalises them away, same as a span's.
        """
        assert job_id(USER, "  Senior Engineer  ") == job_id(USER, "Senior Engineer")

    def test_different_ad_text_mints_a_different_id(self) -> None:
        assert job_id(USER, "Senior Engineer") != job_id(USER, "Staff Engineer")

    def test_scoped_by_user(self) -> None:
        assert job_id(USER, "Senior Engineer") != job_id(uuid.uuid4(), "Senior Engineer")


class TestRequirementId:
    def test_is_deterministic(self) -> None:
        job = uuid.uuid4()
        assert requirement_id(job, "5+ years of Python") == requirement_id(
            job, "5+ years of Python"
        )

    def test_scoped_by_job(self) -> None:
        assert requirement_id(uuid.uuid4(), "Python") != requirement_id(uuid.uuid4(), "Python")

    def test_different_text_mints_a_different_id(self) -> None:
        job = uuid.uuid4()
        assert requirement_id(job, "Python") != requirement_id(job, "Kubernetes")

    def test_unchanged_requirement_keeps_its_id_across_re_extraction(self) -> None:
        """The point of scoping by (job, text) rather than by ordinal: an ad
        edited to add a requirement must not renumber the ones already there.
        """
        job = uuid.uuid4()
        assert requirement_id(job, "Python") == requirement_id(job, "Python")


class TestGapQuestionId:
    def test_is_deterministic(self) -> None:
        requirement = uuid.uuid4()
        assert gap_question_id(requirement) == gap_question_id(requirement)

    def test_depends_only_on_the_requirement_not_the_question_text(self) -> None:
        """Re-running coverage must refresh one stable question per requirement,
        never accumulate near-duplicates -- so the id must NOT vary with the
        question text the model happens to generate this time.
        """
        requirement = uuid.uuid4()
        assert gap_question_id(requirement) == gap_question_id(requirement)

    def test_differs_per_requirement(self) -> None:
        assert gap_question_id(uuid.uuid4()) != gap_question_id(uuid.uuid4())


class TestAdjudicatedSpanIdFromAnswer:
    def test_is_deterministic(self) -> None:
        question = uuid.uuid4()
        assert adjudicated_span_id_from_answer(
            USER, "I led the migration", question
        ) == adjudicated_span_id_from_answer(USER, "I led the migration", question)

    def test_differs_per_question(self) -> None:
        assert adjudicated_span_id_from_answer(
            USER, "Yes", uuid.uuid4()
        ) != adjudicated_span_id_from_answer(USER, "Yes", uuid.uuid4())

    def test_differs_per_answer_text(self) -> None:
        question = uuid.uuid4()
        assert adjudicated_span_id_from_answer(
            USER, "Yes", question
        ) != adjudicated_span_id_from_answer(USER, "No", question)

    def test_never_collides_with_review_item_adjudication(self) -> None:
        """Two different write-back paths into the same NS_SPAN namespace -- the
        distinct 'gap_answer' key must keep them apart even given the same user,
        text, and (coincidentally equal) id.
        """
        shared_id = uuid.uuid4()
        assert adjudicated_span_id_from_answer(USER, "same text", shared_id) != adjudicated_span_id(
            USER, "same text", shared_id
        )
