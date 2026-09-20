"""Unit tests for the profile setup question definitions -- PLAN.md slice B3a.

No database, no model, no filesystem: this only checks the static table of
questions is internally consistent and agrees with the other two places its
closed set of keys is written down (`jfl_core.models` and `jfl_core.db.tables`).
"""

from __future__ import annotations

from typing import get_args

from jfl_core import models
from jfl_core.db import tables
from jfl_core.profile_questions import (
    CORPUS_QUESTION_KEYS,
    CORPUS_SECTIONS,
    DEFAULT_DISCIPLINE_CHOICES,
    DEFERRED_QUESTIONS,
    MAX_OBJECTIVES,
    OBJECTIVE_QUESTIONS,
    QUESTION_KEYS,
    QUESTIONS,
    QUESTIONS_BY_KEY,
    RULED_OUT_QUESTION,
    SECTION_ORDER,
    SECTION_TITLES,
    questions_in_section,
)


def test_question_keys_are_unique() -> None:
    keys = [q.key for q in QUESTIONS]
    assert len(keys) == len(set(keys))


def test_question_numbers_are_unique_and_in_scope() -> None:
    numbers = [q.number for q in QUESTIONS]
    assert len(numbers) == len(set(numbers))
    # Scope is 1-17; 10/11 (objectives) and 17 (ruled-out) are not simple keyed
    # questions and so are not in QUESTIONS -- see the module docstring. 15 and
    # 16 are, even though they also become corpus text. 18 is the CV upload and
    # produces no answer row at all.
    assert set(numbers) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16}


def test_questions_by_key_matches_questions() -> None:
    assert {q.key: q for q in QUESTIONS} == QUESTIONS_BY_KEY
    assert set(QUESTIONS_BY_KEY) == set(QUESTION_KEYS)


def test_every_question_belongs_to_a_titled_section() -> None:
    for question in QUESTIONS:
        assert question.section in SECTION_TITLES
        assert question.section in SECTION_ORDER


def test_section_order_has_no_duplicates_and_covers_every_titled_section() -> None:
    assert len(SECTION_ORDER) == len(set(SECTION_ORDER))
    assert set(SECTION_ORDER) == set(SECTION_TITLES)


def test_questions_in_section_returns_only_that_sections_questions() -> None:
    for section in SECTION_ORDER:
        found = questions_in_section(section)
        assert all(q.section == section for q in found)
    # Every question shows up in exactly the section it names.
    total = sum(len(questions_in_section(s)) for s in SECTION_ORDER)
    assert total == len(QUESTIONS)


def test_structured_questions_are_exactly_the_documented_four() -> None:
    structured_keys = {q.key for q in QUESTIONS if q.structured is not None}
    assert structured_keys == {"levels", "comp_floor", "contract_types", "disciplines"}


def test_question_keys_agree_with_the_models_literal_and_the_table_tuple() -> None:
    """The same agreement `test_value_lists_agree.py` checks for other closed
    sets, but sourced from the question table rather than hand-copied -- if a
    question is added here without updating the other two, this fails instead
    of the CHECK constraint failing at the first real INSERT.
    """
    assert set(QUESTION_KEYS) == set(get_args(models.ProfileQuestionKey))
    assert set(QUESTION_KEYS) == set(tables._PROFILE_QUESTION_KEYS)


def test_default_discipline_choices_are_eight_and_unique() -> None:
    values = [v for v, _ in DEFAULT_DISCIPLINE_CHOICES]
    assert len(values) == 8
    assert len(values) == len(set(values))


def test_objective_and_ruled_out_questions_are_outside_the_keyed_set() -> None:
    """10/11 and 17 are not simple keyed answers (see the module docstring) --
    their numbers must not collide with a QUESTIONS entry.
    """
    keyed_numbers = {q.number for q in QUESTIONS}
    assert OBJECTIVE_QUESTIONS.number_what not in keyed_numbers
    assert OBJECTIVE_QUESTIONS.number_evidence not in keyed_numbers
    assert RULED_OUT_QUESTION.number not in keyed_numbers
    assert MAX_OBJECTIVES == 4


def test_deferred_questions_are_18_only() -> None:
    """15 and 16 are built (they are the two answers that also become corpus
    text); 18 is answered by uploading a CV, not by typing into the page.
    """
    assert {n for n, _ in DEFERRED_QUESTIONS} == {18}


def test_only_15_and_16_reach_the_corpus() -> None:
    """The one place that distinction is written down. Every other answer is a
    preference about what the user wants; these two are claims about them, and
    a preference that leaked into the corpus would become evidence the tool
    then cites back at them.
    """
    assert set(CORPUS_QUESTION_KEYS) == {"depth_genuine", "recurring_gaps"}
    assert {QUESTIONS_BY_KEY[k].number for k in CORPUS_QUESTION_KEYS} == {15, 16}
    assert {q.key for q in QUESTIONS if q.to_corpus} == set(CORPUS_QUESTION_KEYS)


def test_every_corpus_question_has_a_section_to_land_in() -> None:
    assert set(CORPUS_SECTIONS) == set(CORPUS_QUESTION_KEYS)
    assert len(set(CORPUS_SECTIONS.values())) == len(CORPUS_SECTIONS)
