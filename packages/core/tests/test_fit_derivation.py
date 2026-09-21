"""Where the "do I want this" number comes from.

`docs/profile-schema.md`: the model is never asked for this number. It gives a
four-word verdict per constraint and per objective and `jfl_core.fit` derives
the number from those, so it can never say something the verdicts listed under
it do not. These pin the properties that make that claim true.

No model, no database: this is arithmetic over verdicts.
"""

from __future__ import annotations

import pytest
from jfl_core.fit import BREACH_CEILING, MAX_SCORE, MIN_SCORE, want_it_basis
from jfl_core.models import ConstraintVerdict, FitVerdict, ObjectiveVerdict


def _constraint(stance: str, verdict: FitVerdict, kind: str = "location") -> ConstraintVerdict:
    return ConstraintVerdict.model_validate(
        {"kind": kind, "stance": stance, "label": kind, "verdict": verdict}
    )


def _objective(rank: int, verdict: FitVerdict) -> ObjectiveVerdict:
    return ObjectiveVerdict(rank=rank, objective=f"objective {rank}", verdict=verdict)


class TestTheNumber:
    def test_nothing_recorded_gives_no_number_at_all(self) -> None:
        """Not a 1. With nothing recorded there is nothing for the ad to be
        measured against, and a number would be a claim where there is only a
        silence.
        """
        basis = want_it_basis([], [])
        assert basis.score is None
        assert basis.items == 0

    def test_everything_evidenced_is_the_top_of_the_range(self) -> None:
        basis = want_it_basis([_constraint("must", "evidenced")], [_objective(1, "evidenced")])
        assert basis.score == MAX_SCORE

    def test_everything_silent_is_the_bottom_of_the_range(self) -> None:
        """Silence is not half-credit and it is not ignored. The number answers
        "how much of what you said matters does this ad actually evidence", and
        an ad that says nothing evidences nothing -- which is only honest
        because the counts are printed beside it.
        """
        basis = want_it_basis([_constraint("must", "silent")], [_objective(1, "silent")])
        assert basis.score == MIN_SCORE
        assert (basis.silent, basis.evidenced) == (2, 0)

    def test_partial_lands_between_the_two(self) -> None:
        low = want_it_basis([_constraint("must", "silent")], [])
        middle = want_it_basis([_constraint("must", "partial")], [])
        high = want_it_basis([_constraint("must", "evidenced")], [])
        assert low.score is not None and middle.score is not None and high.score is not None
        assert low.score < middle.score < high.score

    def test_a_must_outweighs_a_nice(self) -> None:
        """A negative or a hard preference is worth three of a soft one, which
        is what "must" means.
        """
        must_met = want_it_basis(
            [_constraint("must", "evidenced"), _constraint("nice", "silent", "comp_floor")], []
        )
        nice_met = want_it_basis(
            [_constraint("must", "silent"), _constraint("nice", "evidenced", "comp_floor")], []
        )
        assert must_met.score is not None and nice_met.score is not None
        assert must_met.score > nice_met.score

    def test_a_higher_ranked_objective_outweighs_a_lower_one(self) -> None:
        first = want_it_basis([], [_objective(1, "evidenced"), _objective(3, "silent")])
        third = want_it_basis([], [_objective(1, "silent"), _objective(3, "evidenced")])
        assert first.score is not None and third.score is not None
        assert first.score > third.score

    @pytest.mark.parametrize("stance", ["must", "never"])
    def test_a_breached_must_or_never_holds_the_number_down(self, stance: str) -> None:
        """Everything else about the job can be evidenced and the number still
        does not sit above the sentence saying the job is on site five days a
        week. The breach is also stated in plain words -- the cap is not a
        substitute for saying so.
        """
        verdicts = [
            _constraint(stance, "contradicted", "workplace"),
            _constraint("must", "evidenced", "location"),
            _constraint("must", "evidenced", "comp_floor"),
            _constraint("must", "evidenced", "contract"),
        ]
        basis = want_it_basis(verdicts, [])
        assert basis.score == BREACH_CEILING
        assert basis.breaches == 1
        assert basis.capped

    def test_a_contradicted_nice_is_a_disappointment_not_a_breach(self) -> None:
        basis = want_it_basis([_constraint("nice", "contradicted", "comp_floor")], [])
        assert basis.breaches == 0
        assert not basis.capped

    def test_every_derived_number_is_inside_the_range(self) -> None:
        for stance in ("must", "nice", "never"):
            for verdict in ("evidenced", "partial", "silent", "contradicted"):
                basis = want_it_basis([_constraint(stance, verdict)], [])
                assert basis.score is not None
                assert MIN_SCORE <= basis.score <= MAX_SCORE


class TestTheCounts:
    def test_the_counts_add_up_to_what_was_judged(self) -> None:
        constraints = [
            _constraint("must", "evidenced", "location"),
            _constraint("nice", "partial", "comp_floor"),
            _constraint("never", "contradicted", "categorical_no"),
        ]
        basis = want_it_basis(constraints, [_objective(1, "silent")])
        assert basis.items == 4
        assert (basis.evidenced, basis.partial, basis.silent, basis.contradicted) == (1, 1, 1, 1)


def test_nothing_here_takes_the_other_axis() -> None:
    """The two axes are never combined. `want_it_basis` is the only public
    function in the module and it cannot see "could I get this" at all.
    """
    import inspect

    from jfl_core import fit

    assert "could_get" not in inspect.getsource(fit)
