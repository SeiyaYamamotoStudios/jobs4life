"""Form parsing for `/profile`. No database, no network, no model.

The tier tests are the ones worth reading. A capability's depth is not typed in
-- it falls out of two or three questions about what the person did -- so the
mapping from answers to tier is the only place that decision is made, and it has
to be wrong in no direction: a missing answer must not round down to `absent`,
and an answered "no" must not be mistaken for a missing answer.

`test_the_screens_offer_exactly_the_stored_values` is this page's substitute for
a CHECK constraint. The design doc accepts openly that the value-list drift guard
cannot reach inside JSONB; this is what replaces it, and if it is ever deleted
the screens and the model are free to drift apart silently.
"""

from __future__ import annotations

import uuid

import pytest
from jfl_core.profile import (
    CAPABILITY_TIERS,
    CONSTRAINT_KINDS,
    INTERESTS,
    STANCES,
    Capability,
    CapabilityTier,
    Constraint,
)
from jfl_web.profile import (
    CONSTRAINT_FIELDS,
    INTEREST_CHOICES,
    MAX_ITEMS,
    STANCE_CHOICES,
    TIER_NAMES,
    TIER_QUESTIONS,
    FormTooLongError,
    InvalidAmountError,
    InvalidChoiceError,
    InvalidYearError,
    MissingStanceError,
    TooManyItemsError,
    answers_for_tier,
    checked_text,
    merge_capabilities,
    parse_amount,
    parse_capability,
    parse_constraints,
    parse_disciplines,
    parse_lines,
    parse_objectives,
    parse_stance,
    parse_tier_answer,
    parse_year,
    tier_from_answers,
)

# -- the screens and the model agree ------------------------------------------


def test_the_screens_offer_exactly_the_stored_values() -> None:
    assert {value for value, _ in STANCE_CHOICES} == set(STANCES)
    assert {value for value, _ in INTEREST_CHOICES} == set(INTERESTS)
    assert {field.kind for field in CONSTRAINT_FIELDS} == set(CONSTRAINT_KINDS)
    assert set(TIER_NAMES) == set(CAPABILITY_TIERS)


def test_every_tier_question_has_a_distinct_key() -> None:
    keys = [question.key for question in TIER_QUESTIONS]
    assert keys == ["hands_on", "production", "oversight"]


# -- tiers come from behaviour, never from a rating ---------------------------


@pytest.mark.parametrize(
    ("hands_on", "production", "oversight", "expected"),
    [
        (True, True, None, "production_depth"),
        (True, False, None, "working"),
        (False, None, True, "oversight_only"),
        (False, None, False, "absent"),
        # The irrelevant answer cannot change the outcome: someone who did the
        # work themselves is tiered on production, whatever they say about
        # reviewing others.
        (True, True, True, "production_depth"),
        (True, False, False, "working"),
    ],
)
def test_the_answers_decide_the_tier(
    hands_on: bool, production: bool | None, oversight: bool | None, expected: str
) -> None:
    assert tier_from_answers(hands_on, production, oversight) == expected


@pytest.mark.parametrize(
    ("hands_on", "production", "oversight"),
    [
        (None, None, None),
        (None, True, True),
        # Answered "I did it", but not whether it ran in production: the
        # question that decides between the top two tiers is unanswered, so
        # there is no tier. Rounding down to `working` would be a guess, and a
        # guess in the direction that looks modest is still a guess.
        (True, None, None),
        (False, None, None),
    ],
)
def test_an_unanswered_question_leaves_the_tier_unstated(
    hands_on: bool | None, production: bool | None, oversight: bool | None
) -> None:
    assert tier_from_answers(hands_on, production, oversight) is None


@pytest.mark.parametrize("tier", CAPABILITY_TIERS)
def test_a_saved_tier_reads_back_as_the_answers_that_produced_it(tier: CapabilityTier) -> None:
    answers = answers_for_tier(tier)
    assert tier_from_answers(**answers) == tier


def test_an_unstated_tier_reads_back_as_no_answers() -> None:
    assert answers_for_tier(None) == {"hands_on": None, "production": None, "oversight": None}


def test_a_tier_answer_the_page_never_offered_is_refused() -> None:
    assert parse_tier_answer("yes") is True
    assert parse_tier_answer("no") is False
    assert parse_tier_answer("") is None
    with pytest.raises(InvalidChoiceError):
        parse_tier_answer("sort of")


def test_a_capability_save_never_takes_evidence_from_the_form() -> None:
    """Evidence is a list of span ids and the browser does not get to assert
    one. The parsed row keeps the label, evidence and source it already had.
    """
    span = uuid.uuid4()
    existing = Capability(label="FX pricing", evidence=[span], source="cv_fact")
    updated = parse_capability(
        existing,
        hands_on="yes",
        production="yes",
        oversight="",
        interest="want_more",
        last_used="2024",
    )
    assert updated.tier == "production_depth"
    assert updated.interest == "want_more"
    assert updated.last_used == 2024
    assert updated.evidence == [span]
    assert updated.source == "cv_fact"
    assert updated.label == "FX pricing"


def test_depth_and_appetite_are_independent() -> None:
    """A capability can be deep and unwanted, which is exactly the case a single
    combined "skill" number would erase.
    """
    updated = parse_capability(
        Capability(label="On-call rota design"),
        hands_on="yes",
        production="yes",
        oversight="",
        interest="never_again",
        last_used="",
    )
    assert (updated.tier, updated.interest) == ("production_depth", "never_again")


def test_a_year_that_is_not_a_year_is_refused() -> None:
    assert parse_year("") is None
    assert parse_year("2019") == 2019
    with pytest.raises(InvalidYearError):
        parse_year("last summer")


# -- constraints ---------------------------------------------------------------


def test_a_constraint_needs_a_stance_before_it_is_stored() -> None:
    with pytest.raises(MissingStanceError):
        parse_constraints({"places-location": "London"})


def test_a_stance_the_page_never_offered_is_refused() -> None:
    with pytest.raises(InvalidChoiceError):
        parse_constraints({"stance-location": "maybe"})


def test_a_kind_with_nothing_said_about_it_is_simply_absent() -> None:
    assert parse_constraints({}) == []
    assert parse_stance("") is None


def test_locations_keep_the_order_they_were_given() -> None:
    constraints = parse_constraints(
        {"stance-location": "must", "places-location": "London\n\nBristol\n  Leeds  "}
    )
    assert constraints == [
        Constraint(kind="location", stance="must", value={"places": ["London", "Bristol", "Leeds"]})
    ]


def test_comp_carries_guaranteed_and_headline_separately() -> None:
    constraints = parse_constraints(
        {
            "stance-comp_floor": "must",
            "comp-guaranteed": "120,000",
            "comp-headline": "145000",
            "comp-ccy": "gbp",
            "note-comp_floor": "base + pension, ignoring equity",
        }
    )
    assert constraints[0].value == {"ccy": "GBP", "guaranteed": 120000, "headline": 145000}
    assert constraints[0].note == "base + pension, ignoring equity"


def test_one_comp_figure_without_the_other_is_kept_as_given() -> None:
    """Neither figure is derived from the other. A guaranteed floor with no
    headline stated is a complete answer, not half of one.
    """
    constraints = parse_constraints({"stance-comp_floor": "must", "comp-guaranteed": "120000"})
    assert constraints[0].value == {"ccy": "GBP", "guaranteed": 120000}


def test_a_comp_figure_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(InvalidAmountError):
        parse_amount("about a hundred grand")


def test_a_note_is_kept_exactly_as_written() -> None:
    constraints = parse_constraints(
        {"stance-categorical_no": "never", "note-categorical_no": "No gambling. Ever."}
    )
    assert constraints[0].note == "No gambling. Ever."


# -- lists, disciplines, objectives -------------------------------------------


def test_a_list_too_long_to_rank_is_refused_rather_than_trimmed() -> None:
    with pytest.raises(TooManyItemsError):
        parse_lines("\n".join(str(n) for n in range(MAX_ITEMS + 1)))


def test_text_over_the_limit_is_refused_rather_than_truncated() -> None:
    with pytest.raises(FormTooLongError):
        checked_text("x" * 50, limit=10)


def test_disciplines_keep_both_lists_and_their_order() -> None:
    disciplines = parse_disciplines("engineering management\nplatform", "frontend")
    assert disciplines.practises == ["engineering management", "platform"]
    assert disciplines.not_practised == ["frontend"]


def test_disciplines_serialise_with_the_documented_key() -> None:
    dumped = parse_disciplines("a", "b").model_dump(by_alias=True)
    assert dumped == {"practises": ["a"], "not": ["b"]}


def test_an_objective_keeps_the_slot_it_was_typed_into() -> None:
    objectives = parse_objectives([(1, "", ""), (2, "Bigger scope", "A department to run")])
    assert [o.rank for o in objectives] == [2]
    assert objectives[0].evidence_of_delivery == "A department to run"


def test_evidence_with_no_objective_is_dropped_with_it() -> None:
    assert parse_objectives([(1, "", "A department to run")]) == []


# -- seeding -------------------------------------------------------------------


def test_a_saved_row_always_beats_a_seed() -> None:
    """A seed is re-derived on every request. If it could overwrite a saved row
    it would silently undo the tier the user actually answered for.
    """
    saved = Capability(label="FX pricing", tier="working")
    seeded = Capability(label="fx pricing", source="cv_fact", evidence=[uuid.uuid4()])
    merged = merge_capabilities([saved], [seeded])
    assert len(merged) == 1
    assert merged[0].tier == "working"
    # ...but the seed's evidence is kept, because the tier is the user's and the
    # evidence is the corpus's.
    assert merged[0].evidence == seeded.evidence


def test_a_seed_with_no_saved_row_is_shown() -> None:
    seeded = Capability(label="Incident command", source="cv_fact")
    merged = merge_capabilities([], [seeded])
    assert [c.label for c in merged] == ["Incident command"]
    assert merged[0].tier is None
