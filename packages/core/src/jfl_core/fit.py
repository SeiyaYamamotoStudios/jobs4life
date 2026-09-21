"""Where the "do I want this" number comes from, and nothing else.

`docs/profile-schema.md`, 2026-09-21: **"do I want this" is not a predicted-
satisfaction score.** Computed person-job fit predicts satisfaction at rho ~= .28
(against .61 when people rate fit themselves), people forecast their own job
satisfaction badly, and what does predict it is met expectations (.39). So the
model is never asked for this number. It is asked what the **ad** evidences
about each constraint and each objective the user recorded -- `evidenced`,
`partial`, `silent` or `contradicted` -- and the number falls out of those
verdicts here, in code, where it can be read and argued with.

Two consequences worth stating, because they look like defects until they are
read as the design:

* **Silence lowers the number.** It is not a middling half-credit and it is not
  ignored. The number answers "how much of what you said matters does this ad
  actually evidence", and an ad that says nothing evidences nothing. That is
  only honest if the silences are shown, which is why `WantItBasis` carries the
  counts and the panel prints them beside the number.
* **An empty profile has no number at all** -- `score` is None, not 1. With no
  constraints and no objectives recorded there is nothing for the ad to be
  measured against, and 1/10 would be a claim where there is only a silence.

This sits in `jfl_core` rather than beside the scoring call because three
callers need the same arithmetic: `jfl_generate.scoring` derives the number,
`jfl_web` prints the counts under it, and tests read both. `core` holds no HTTP
or framework types, and there is none here.

There is deliberately no function in this module that takes "could I get this".
The two axes are never combined -- CLAUDE.md's standing decision, and
`tests/test_scores_never_composited.py` says so mechanically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jfl_core.models import FIT_VERDICTS, ConstraintVerdict, FitVerdict, ObjectiveVerdict

MIN_SCORE = 1
MAX_SCORE = 10

# Re-exported so callers can reach the vocabulary and the arithmetic through
# one import. The words themselves live beside the Literal they belong to.
__all__ = [
    "BREACH_CEILING",
    "FIT_VERDICTS",
    "MAX_SCORE",
    "MIN_SCORE",
    "WantItBasis",
    "want_it_basis",
]

# What one verdict is worth, on 0..1. `silent` scores nothing because the ad
# evidences nothing, not because silence is bad; `contradicted` scores nothing
# because the ad evidences the opposite. They are told apart on the screen, in
# the counts, never by a middle number here.
_CREDIT: dict[FitVerdict, float] = {
    "evidenced": 1.0,
    "partial": 0.5,
    "silent": 0.0,
    "contradicted": 0.0,
}

# A `never` carries the same weight as a `must`: giving a negative preference
# equal standing with a positive one is the point of having three stances.
_STANCE_WEIGHT = {"must": 3, "never": 3, "nice": 1}

# Objectives are ranked, and the ranking is the user's own statement of what
# this move is for.
_RANK_WEIGHT = {1: 3, 2: 2}
_LOWER_RANK_WEIGHT = 1

# A `must` or `never` the ad contradicts holds the number down whatever else it
# evidences. It is still stated in plain words as a breach -- the cap is not a
# substitute for saying so, it is what stops a 7 sitting above a sentence
# explaining that the job is on site five days a week.
BREACH_CEILING = 2


def _objective_weight(rank: int) -> int:
    return _RANK_WEIGHT.get(rank, _LOWER_RANK_WEIGHT)


@dataclass(frozen=True, slots=True)
class WantItBasis:
    """The number, and every count it was derived from.

    Stored? Only `score` is. The counts are recomputed from the stored verdicts
    wherever they are printed, so the panel can never show a tally that
    disagrees with the verdicts listed under it.
    """

    score: int | None
    items: int
    evidenced: int
    partial: int
    silent: int
    contradicted: int
    # Of the contradictions, how many are on a `must` or a `never`. These are
    # the breaches, and they are what applies `BREACH_CEILING`.
    breaches: int

    @property
    def capped(self) -> bool:
        return self.breaches > 0


def want_it_basis(
    constraints: Sequence[ConstraintVerdict],
    objectives: Sequence[ObjectiveVerdict],
) -> WantItBasis:
    """Derive the number from the verdicts, and report what it was derived from.

    Weighted share of what the ad evidences: `must` and `never` count three,
    `nice` counts one, and an objective counts by its rank. `evidenced` is full
    credit, `partial` half, `silent` and `contradicted` none. The share maps
    onto 1-10, and any breached `must` or `never` caps it at `BREACH_CEILING`.
    """
    weights: list[int] = [_STANCE_WEIGHT.get(c.stance, 1) for c in constraints]
    weights += [_objective_weight(o.rank) for o in objectives]
    verdicts: list[FitVerdict] = [c.verdict for c in constraints]
    verdicts += [o.verdict for o in objectives]

    counts = {word: 0 for word in _CREDIT}
    for verdict in verdicts:
        counts[verdict] += 1
    breaches = sum(
        1 for c in constraints if c.verdict == "contradicted" and c.stance in ("must", "never")
    )

    if not weights or sum(weights) == 0:
        score = None
    else:
        earned = sum(w * _CREDIT[v] for w, v in zip(weights, verdicts, strict=True))
        share = earned / sum(weights)
        score = MIN_SCORE + round(share * (MAX_SCORE - MIN_SCORE))
        if breaches:
            score = min(score, BREACH_CEILING)

    return WantItBasis(
        score=score,
        items=len(weights),
        evidenced=counts["evidenced"],
        partial=counts["partial"],
        silent=counts["silent"],
        contradicted=counts["contradicted"],
        breaches=breaches,
    )
