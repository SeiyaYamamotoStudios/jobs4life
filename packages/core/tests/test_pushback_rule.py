"""The update rule, arithmetic first. No database, no model, no clock.

These are the tests that make the anti-flattery design a property rather than
an intention: the shrinkage, both caps, the asymmetry between a claim that
reduces what you assert and one that increases it, and the guarantee that
repeating yourself moves the number less rather than more.

The numbers below are written out rather than computed from the constants on
purpose. A test that recomputes the formula passes whatever the formula is,
which is exactly what a change to K or to a cap needs to be caught by.
"""

from __future__ import annotations

import math

import pytest
from jfl_core.pushback import (
    COULD_GET_OVERALL,
    KAPPA,
    MAX_DELTA_PER_PUSHBACK,
    MAX_DISPLACEMENT_PER_DIMENSION,
    WANT_OVERALL,
    Direction,
    DriftMeter,
    ProtectedTargetError,
    PushbackEffect,
    adjustment_for,
    decide,
    dimension_axis,
    displayed_score,
    observations,
    shrink,
    target_dimension,
    valid_dimension,
)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-6)


# -- shrinkage ---------------------------------------------------------------


def test_the_shrinkage_factor_is_kappa_over_kappa_plus_n() -> None:
    assert KAPPA == 5
    assert close(shrink(1.0, 0), 1.0)
    assert close(shrink(1.0, 1), 5 / 6)
    assert close(shrink(1.0, 2), 5 / 7)
    assert close(shrink(1.0, 10), 1 / 3)


def test_the_designs_worked_receipt_reproduces() -> None:
    """ "of a possible 1 point of movement, 0.5 used; shrunk because this is the
    second thing you've told me about company stage" -- the design's own
    example, which is 0.6 * 5/(5+1). It is the check that the formula
    implemented is the one the design section states, and not the empirical-Bayes
    form its section 2.1 describes, which would move the number *more* with each
    restatement.
    """
    assert close(shrink(0.6, 1), 0.5)


def test_more_prior_observations_means_less_movement_not_more() -> None:
    moves = [decide_preference(n=n).applied_delta for n in range(6)]
    assert moves == sorted(moves, reverse=True)
    assert moves[0] > moves[-1]


def decide_preference(
    *,
    n: int = 0,
    displacement: float = 0.0,
    asserted: float = 1.0,
    new_information: bool = True,
    direction: Direction = "up",
) -> PushbackEffect:
    return decide(
        kind="preference",
        dimension="constraint:workplace",
        direction=direction,
        asserted=asserted,
        prior_observations=n,
        displacement=displacement,
        new_information=new_information,
    )


# -- the two caps ------------------------------------------------------------


def test_no_single_pushback_moves_a_score_by_more_than_one_point() -> None:
    assert MAX_DELTA_PER_PUSHBACK == 1.0
    for asserted in (1.0, 2.0, 3.0, 40.0):
        effect = decide_preference(asserted=asserted)
        assert abs(effect.applied_delta) <= MAX_DELTA_PER_PUSHBACK


def test_the_cap_binds_after_shrinkage_not_before() -> None:
    """A big assertion late in the record is still shrunk; the cap is the floor
    under that, not a replacement for it.
    """
    late = decide_preference(asserted=3.0, n=25)
    assert close(late.after_shrinkage, 3.0 * 5 / 30)
    assert close(late.applied_delta, 0.5)


def test_a_dimension_stops_at_two_points_of_displacement() -> None:
    assert MAX_DISPLACEMENT_PER_DIMENSION == 2.0
    effect = decide_preference(n=2, displacement=1.8333333)
    assert close(effect.applied_delta, 2.0 - 1.8333333)
    assert effect.comparison_offered is True


def test_at_the_cap_nothing_moves_and_the_comparison_is_offered() -> None:
    effect = decide_preference(displacement=2.0)
    assert effect.applied_delta == 0.0
    assert effect.disposition == "recorded_only"
    assert effect.comparison_offered is True


def test_the_third_pushback_on_one_dimension_is_the_one_that_runs_out() -> None:
    """Three corrections of one point each on the same dimension: the first two
    land, the third hits the two-point limit and the tool stops arguing.
    """
    displacement = 0.0
    results = []
    for n in range(3):
        effect = decide_preference(n=n, displacement=displacement)
        displacement += effect.applied_delta
        results.append(effect)
    assert [r.comparison_offered for r in results] == [False, False, True]
    assert close(displacement, MAX_DISPLACEMENT_PER_DIMENSION)


def test_the_cap_is_on_displacement_from_baseline_so_the_other_way_is_free() -> None:
    """Having pushed a dimension up to its limit does not stop you pushing it
    back down. The cap is displacement from where the tool put it, not a budget
    of corrections.
    """
    effect = decide_preference(displacement=2.0, direction="down")
    assert effect.applied_delta < 0
    assert effect.comparison_offered is False


# -- repetition --------------------------------------------------------------


def test_a_restatement_contributes_nothing_and_says_so() -> None:
    effect = decide_preference(new_information=False)
    assert effect.applied_delta == 0.0
    assert effect.repetition is True
    assert effect.disposition == "recorded_only"


def test_a_restatement_still_counts_as_an_observation_for_the_next_one() -> None:
    """The mechanism by which saying it louder moves the number *less*: the
    restatement moves nothing itself and raises `n`, so the next correction
    carrying an actual new fact is shrunk harder than it would have been.
    """
    assert observations(prior_pushbacks=2) == 2
    first = decide_preference(n=1)
    after_a_restatement = decide_preference(n=2)
    assert after_a_restatement.applied_delta < first.applied_delta


def test_a_submitted_application_counts_for_three_observations() -> None:
    """Enacting a preference is better evidence of it than asserting one."""
    assert observations(prior_pushbacks=1, submitted_applications=2) == 7


# -- the asymmetry, which is the whole design --------------------------------


@pytest.mark.parametrize("asserted", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("n", [0, 1, 20])
@pytest.mark.parametrize("displacement", [-5.0, 0.0, 5.0])
def test_a_capability_claim_upward_never_moves_anything(
    asserted: float, n: int, displacement: float
) -> None:
    """No combination of inputs produces a non-zero delta for "you have
    underrated me". This is the ratchet, and it is closed by a return statement
    rather than by a threshold.
    """
    effect = decide(
        kind="capability",
        dimension=COULD_GET_OVERALL,
        direction="up",
        asserted=asserted,
        prior_observations=n,
        displacement=displacement,
        new_information=True,
    )
    assert effect.applied_delta == 0.0
    assert effect.disposition == "pending_evidence"
    assert effect.evidence_required is True


def test_a_capability_claim_downward_is_applied_in_full_immediately() -> None:
    effect = decide(
        kind="capability",
        dimension=COULD_GET_OVERALL,
        direction="down",
        asserted=3.0,
        prior_observations=50,
        displacement=0.0,
    )
    assert close(effect.applied_delta, -3.0)
    assert effect.disposition == "accepted"
    assert effect.evidence_required is False


def test_self_deprecation_is_not_shrunk_and_not_capped() -> None:
    """Down is free: a claim that reduces what you assert needs no grounding,
    which is the claim gate's philosophy pointed at a score.
    """
    effect = decide(
        kind="capability",
        dimension=COULD_GET_OVERALL,
        direction="down",
        asserted=3.0,
        prior_observations=0,
        displacement=-10.0,
    )
    assert close(effect.applied_delta, -3.0)
    assert effect.comparison_offered is False


def test_a_factual_objection_changes_nothing_about_the_person() -> None:
    effect = decide(
        kind="factual",
        dimension="constraint:workplace",
        direction="up",
        asserted=3.0,
        prior_observations=0,
        displacement=0.0,
    )
    assert effect.applied_delta == 0.0
    assert effect.disposition == "recorded_only"
    assert effect.evidence_required is False


# -- what may be named at all ------------------------------------------------


@pytest.mark.parametrize(
    "dimension",
    [
        "gate_verdict",
        "span:3f2a",
        "coverage:requirement-1",
        "eval_label",
        "unmeasured",
        "could_get_score",
        "",
        "profiles.data",
    ],
)
def test_nothing_protected_can_even_be_named(dimension: str) -> None:
    """The list of what pushback may change is an allowlist, so a claim-gate
    verdict, a span, a coverage status, an eval label and the "unmeasured"
    label are all refused by not being on it -- including ones nobody thought
    to forbid.
    """
    assert valid_dimension(dimension) is False
    with pytest.raises(ProtectedTargetError):
        target_dimension("preference", dimension)


def test_the_things_that_may_be_named_are_the_users_own() -> None:
    for dimension in (
        "constraint:workplace",
        "objective:1",
        "capability:0123456789abcdef",
        WANT_OVERALL,
        COULD_GET_OVERALL,
    ):
        assert valid_dimension(dimension) is True


def test_a_capability_claim_raised_from_the_want_panel_retargets() -> None:
    """A person typing under "do I want this" may well write a claim about
    their own depth. It is re-targeted rather than misapplied, and the
    re-target can only ever reach the axis where upward movement is impossible.
    """
    assert target_dimension("capability", "constraint:workplace") == COULD_GET_OVERALL
    assert target_dimension("preference", "capability:abc") == WANT_OVERALL
    assert target_dimension("preference", "objective:2") == "objective:2"


def test_dimension_axis_never_crosses() -> None:
    assert dimension_axis("constraint:comp_floor") == "want"
    assert dimension_axis("objective:1") == "want"
    assert dimension_axis(WANT_OVERALL) == "want"
    assert dimension_axis("capability:abc") == "get"
    assert dimension_axis(COULD_GET_OVERALL) == "get"


# -- what the panel shows ----------------------------------------------------


def test_a_correction_never_rewrites_the_stored_number() -> None:
    display = displayed_score(4, 0.8)
    assert display.stored == 4
    assert display.displayed == 5
    assert display.adjusted is True


def test_rounding_is_symmetric_about_the_stored_number() -> None:
    assert displayed_score(4, 0.5).displayed == 5
    assert displayed_score(4, -0.5).displayed == 3


def test_a_displayed_score_stays_inside_one_to_ten() -> None:
    assert displayed_score(10, 2.0).displayed == 10
    assert displayed_score(1, -2.0).displayed == 1


def test_an_already_sent_application_keeps_the_score_it_was_sent_under() -> None:
    display = displayed_score(4, 2.0, frozen=True)
    assert display.displayed == 4
    assert display.adjustment == 0.0
    assert display.frozen is True


def test_corrections_cannot_lift_a_number_over_a_broken_must_have() -> None:
    """The breach ceiling survives pushback. Otherwise two preference
    corrections would put a 4 above a sentence saying the job is on site five
    days a week, and the cap would have bought nothing.
    """
    display = displayed_score(2, 2.0, breach_ceiling=True)
    assert display.displayed == 2
    assert display.capped_by_breach is True


def test_the_breach_ceiling_bounds_the_correction_and_not_the_stored_number() -> None:
    """A run that recorded a breach has already been through the derivation
    that applies the ceiling. This layer bounds what a correction may add to
    what it stored; it does not re-derive it downwards.
    """
    display = displayed_score(3, 0.0, breach_ceiling=True)
    assert display.displayed == 3
    assert display.capped_by_breach is False


def test_an_override_is_shown_beside_the_number_never_instead_of_it() -> None:
    display = displayed_score(4, 0.0, override=9)
    assert display.stored == 4
    assert display.displayed == 4
    assert display.override == 9
    assert display.effective == 9


def test_an_empty_profile_has_no_number_and_a_correction_does_not_invent_one() -> None:
    display = displayed_score(None, 1.0)
    assert display.displayed is None
    assert display.stored is None


# -- the total cap across dimensions ----------------------------------------


def test_stacking_dimensions_cannot_beat_the_per_dimension_cap() -> None:
    displacements = {f"constraint:k{i}": 2.0 for i in range(8)}
    total = adjustment_for(displacements, list(displacements), axis="want")
    assert total == MAX_DISPLACEMENT_PER_DIMENSION


def test_only_the_dimensions_this_score_carries_count() -> None:
    displacements = {"constraint:workplace": 1.0, "constraint:notice": 1.0}
    total = adjustment_for(displacements, ["constraint:workplace"], axis="want")
    assert close(total, 1.0)


def test_a_capability_correction_cannot_leak_into_the_want_number() -> None:
    displacements = {COULD_GET_OVERALL: -2.0, WANT_OVERALL: 0.5}
    assert close(adjustment_for(displacements, [], axis="want"), 0.5)
    assert close(adjustment_for(displacements, [], axis="get"), -2.0)


# -- the drift meter ---------------------------------------------------------


def test_the_drift_meter_reads_the_way_the_design_writes_it() -> None:
    meter = DriftMeter(total=11, upward=10, downward=1, net=3.1)
    assert meter.sentence == "11 pushbacks, 10 upward, +3.1 net"
    assert meter.visible is True


def test_the_drift_meter_signs_a_downward_net() -> None:
    assert DriftMeter(total=2, upward=0, downward=2, net=-1.25).sentence == (
        "2 pushbacks, 0 upward, -1.2 net"
    )


def test_one_pushback_is_singular() -> None:
    assert DriftMeter(total=1, upward=1, downward=0, net=1.0).sentence == (
        "1 pushback, 1 upward, +1.0 net"
    )


def test_an_untouched_account_has_nothing_to_show() -> None:
    assert DriftMeter(total=0, upward=0, downward=0, net=0.0).visible is False
