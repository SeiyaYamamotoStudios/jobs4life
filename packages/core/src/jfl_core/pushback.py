"""What a disagreement with a score is allowed to change -- the arithmetic, pure.

Research and agreed design: `~/jobs4life-profile-research/feedback-loops.md`,
"The loop we should build". This module is that section's rule and nothing else:
no storage, no model, no HTTP, no clock. It takes what the user asserted and
what the record already holds, and returns what moves.

## Why the rule is shaped like this

A tool that agrees with you on request is the thing this project exists to
oppose. OpenAI's April 2025 GPT-4o rollout is the documented version of the
failure: a thumbs-up training signal made the model sycophantic, **offline evals
looked fine and A/B tests showed users liked it**, because a metric that
measures whether the user liked the answer cannot detect flattery. So the guards
here are structural rather than tuned, and the drift meter
(`jfl_core.storage.pushbacks`) is the separate number whose whole job is to
catch what no other metric can see.

Four rules, and the asymmetry between the middle two is the whole design:

* **Preference** ("do I want this") -- the user is the sole authority on their
  own wants, so it is accepted, but **shrunk**: `delta = d * K/(K + n)` with
  K = 5 and `n` the prior observations touching that dimension. One correction
  is one observation with high variance, and replacing a parameter with the
  value implied by one observation is a learning rate of 1.0. Capped at one
  point per pushback and at two points of total displacement per dimension.
* **Capability, downward** ("you have overrated me") -- applied in full,
  immediately, no evidence asked for and no cap. A claim that *reduces* what
  you assert needs no grounding; that is the claim gate's philosophy, and this
  is it pointed at a score.
* **Capability, upward** ("you have underrated my depth") -- **the number does
  not move, ever.** It is a claim about the world, and nobody in this loop is
  the authority on it. It opens an evidence question naming the exact fact, and
  the number changes when a confirmed corpus fact exists and the job is scored
  again. This is the ratchet, and `_capability` returns before any arithmetic
  runs so there is no code path that could produce a non-zero delta for it.
* **Factual objection** about the ad ("it does say remote") -- the fact gets
  fixed and the job re-scored. The profile is not involved at all.

## The formula, and a known inconsistency in the source

The design section states `dw = d * K/(K + n)` and its worked receipt confirms
it: "moved 2.1 -> 2.6 ... of a possible 1 point of movement, 0.5 used; shrunk
because this is the second thing you've told me about company stage" is exactly
`0.6 * 5/(5+1) = 0.5`. The same file's section 2.1 describes the standard
empirical-Bayes form `n/(n+K)` and says the first pushback moves one sixth --
which is the *other* formula, and would make each repetition move the number
*more*. That is the wrong direction for this product, so the design section's
form is what is implemented: **more prior noise on a dimension means less
movement per correction.** Recorded here rather than silently resolved.

## Repetition is not evidence

`n` counts distinct observations, and a restatement with no new fact increments
`n` while contributing no `d` -- so the second and third attempts move the
number *less*, not more, which is deliberate and is said out loud in the
receipt. Whether a restatement carries new information is a judgement on free
text and is classified by a model the user can correct; **an exact restatement
of text already recorded on the same dimension and direction is a repetition
whatever anyone says**, which is the part no prompt can be talked out of.

## What is never reachable from here

`decide` takes a dimension from a closed allowlist and refuses anything else.
There is deliberately no dimension naming a claim-gate verdict, a span, a
per-requirement coverage status, an eval label or the "unmeasured" label: they
are not on the list, and a list is refused-by-default rather than
forbidden-by-review. Nothing in this module imports a repository that could
reach them either -- see `tests/test_pushback_reaches_nothing_protected.py`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from jfl_core.fit import BREACH_CEILING, MAX_SCORE, MIN_SCORE

# How many observations the prior is worth. 5, from the design section; the
# number is a judgement, not a measurement, and it is the one knob here.
KAPPA = 5

# No single pushback moves a displayed score by more than this, whatever the
# user asserted and whatever the shrinkage leaves. Applied *after* shrinkage,
# because the design states it as a cap on the movement rather than on the
# assertion.
MAX_DELTA_PER_PUSHBACK = 1.0

# Total signed displacement one dimension may accumulate before the tool stops
# arguing about the number and asks for a comparison instead. Two points, from
# the design section.
MAX_DISPLACEMENT_PER_DIMENSION = 2.0

# The most any one score's displayed number may be moved by corrections, summed
# across every dimension it carries. Without it a user with eight dimensions
# could stack eight separate two-point displacements into sixteen points and the
# per-dimension cap would have bought nothing.
MAX_TOTAL_ADJUSTMENT = 2.0

# The most a single pushback may assert. The form offers 1, 2 or 3; this is the
# floor under that, so a hand-made request cannot assert 40 and land on the cap
# by a different route.
MAX_ASSERTED_POINTS = 3.0

# What one submitted application is worth as an observation, against one
# pushback's 1. The design's open questions propose "a submitted application
# counts 3, a score-and-discard counts 1, a pushback counts 1", on the
# stated-versus-revealed grounds in its section 1.4: enacting a preference is
# better evidence of it than asserting one. The per-dimension refinement (which
# dimensions a given application actually touched) is deliberately not built --
# it needs a structural query over stored JSONB verdicts, and the honest
# approximation is named here rather than hidden.
OBSERVATION_WEIGHT_SUBMITTED_APPLICATION = 3

Axis = Literal["want", "get"]
AXES: tuple[str, ...] = ("want", "get")

# What the disagreement is *about*. Three kinds, three different consequences.
PushbackKind = Literal["preference", "capability", "factual"]
PUSHBACK_KINDS: tuple[str, ...] = ("preference", "capability", "factual")

Direction = Literal["up", "down"]
DIRECTIONS: tuple[str, ...] = ("up", "down")

# `accepted` -- something moved. `recorded_only` -- the words are kept and
# nothing moved. `pending_evidence` -- nothing moved and there is a named fact
# that would move it. All three are first-class: "recorded but not accepted" is
# a dignified outcome, not a silent no-op.
Disposition = Literal["accepted", "recorded_only", "pending_evidence"]
DISPOSITIONS: tuple[str, ...] = ("accepted", "recorded_only", "pending_evidence")

# The only things a pushback may name. An allowlist, so anything added to the
# system later is refused here until somebody decides otherwise -- which is the
# safe direction for a list whose job is to keep the claim gate, the corpus and
# the coverage statuses out of reach.
WANT_DIMENSION_PREFIXES: tuple[str, ...] = ("constraint:", "objective:")
GET_DIMENSION_PREFIXES: tuple[str, ...] = ("capability:",)
WANT_OVERALL = "want_overall"
COULD_GET_OVERALL = "could_get_overall"


class ProtectedTargetError(ValueError):
    """A pushback named something it may never change.

    Raised rather than ignored. A pushback that quietly targets nothing looks
    to the user exactly like one that was applied, and this tool's whole claim
    is that it shows the distance between what happened and what is claimed.
    """


def valid_dimension(dimension: str) -> bool:
    """Whether this names something a pushback is allowed to move."""
    if dimension in (WANT_OVERALL, COULD_GET_OVERALL):
        return True
    return dimension.startswith(WANT_DIMENSION_PREFIXES + GET_DIMENSION_PREFIXES)


def dimension_axis(dimension: str) -> Axis:
    """Which score a displacement on this dimension is displayed against."""
    if dimension == COULD_GET_OVERALL or dimension.startswith(GET_DIMENSION_PREFIXES):
        return "get"
    return "want"


def target_dimension(kind: PushbackKind, dimension: str) -> str:
    """Where this pushback's displacement actually lands.

    A person typing under the "do I want this" panel may well write a claim
    about their own depth, and forcing them to file it under the right heading
    first would be the friction that gets a tool abandoned. So the panel decides
    what they are *looking at* and the classification decides what it *is*: a
    capability claim raised from the want panel re-targets to the capability
    axis, and a preference raised from the capability panel re-targets to the
    want axis. The re-targeting is shown in the receipt, never silent.

    It opens no hole. The only re-target towards the capability axis carries a
    capability classification, and a capability claim can only ever move a
    number downwards -- upwards it moves nothing at all, by `_capability`.
    """
    if not valid_dimension(dimension):
        raise ProtectedTargetError(f"{dimension!r} is not something a pushback may change")
    axis = dimension_axis(dimension)
    if kind == "capability" and axis == "want":
        return COULD_GET_OVERALL
    if kind == "preference" and axis == "get":
        return WANT_OVERALL
    return dimension


def observations(*, prior_pushbacks: int, submitted_applications: int = 0) -> int:
    """`n` for the shrinkage denominator: distinct observations on a dimension.

    Pushbacks count one each -- including the restatements that contribute no
    `d`, which is what makes saying it again move the number less. Submitted
    applications count three each, because enacting a preference is better
    evidence of it than asserting one.
    """
    return max(prior_pushbacks, 0) + OBSERVATION_WEIGHT_SUBMITTED_APPLICATION * max(
        submitted_applications, 0
    )


def shrink(asserted: float, n: int) -> float:
    """`d * K/(K + n)` -- the one line that is most of the anti-overfitting story."""
    if asserted <= 0:
        return 0.0
    return asserted * KAPPA / (KAPPA + max(n, 0))


@dataclass(frozen=True, slots=True)
class PushbackEffect:
    """Exactly what one pushback did, in the numbers the receipt quotes.

    Every field is reported to the user. "Nothing moved" is stated with the
    same precision as "0.83 of the 1 point available moved", because a loop the
    user cannot audit is a loop they are right not to trust.
    """

    target: str
    disposition: Disposition
    # Signed, in points of displayed score. Zero for every capability-upward
    # and every factual pushback, always.
    applied_delta: float
    # What the user asserted, what shrinkage left of it, and what survived the
    # two caps -- so the receipt can say which guard actually bound.
    asserted: float
    after_shrinkage: float
    prior_observations: int
    # True when the per-dimension displacement cap is what stopped this, which
    # is when the tool stops arguing about the number and offers a comparison.
    comparison_offered: bool = False
    # True when this restated something already on the record with no new fact.
    repetition: bool = False
    # Set for capability-upward: the number changes when a confirmed corpus
    # fact exists, and this names the fact that would do it.
    evidence_required: bool = False

    @property
    def moved(self) -> bool:
        return self.applied_delta != 0.0


def decide(
    *,
    kind: PushbackKind,
    dimension: str,
    direction: Direction,
    asserted: float,
    prior_observations: int,
    displacement: float,
    new_information: bool = True,
) -> PushbackEffect:
    """What this pushback changes. Pure, total, and the only place that decides.

    `displacement` is the signed displacement this dimension has already
    accumulated, which the caller reads from the append-only pushback log --
    there is no stored weight to drift, so "the profile" cannot quietly move at
    all. `new_information` is the model's judgement, correctable by the user;
    the caller is responsible for forcing it False on an exact restatement.
    """
    target = target_dimension(kind, dimension)
    asserted = max(0.0, min(float(asserted), MAX_ASSERTED_POINTS))

    if kind == "factual":
        # About the ad, not about the person and not about their preferences.
        # The fix is to correct the job record and score it again; the profile
        # is not involved, so nothing here moves.
        return PushbackEffect(
            target=target,
            disposition="recorded_only",
            applied_delta=0.0,
            asserted=asserted,
            after_shrinkage=0.0,
            prior_observations=prior_observations,
        )

    if kind == "capability":
        return _capability(
            target=target,
            direction=direction,
            asserted=asserted,
            prior_observations=prior_observations,
        )

    return _preference(
        target=target,
        direction=direction,
        asserted=asserted,
        prior_observations=prior_observations,
        displacement=displacement,
        new_information=new_information,
    )


def _capability(
    *,
    target: str,
    direction: Direction,
    asserted: float,
    prior_observations: int,
) -> PushbackEffect:
    """The asymmetric bar: down is free, up costs a fact.

    Upward returns before any arithmetic. There is no shrinkage to tune, no cap
    to raise and no branch below this one that could produce a non-zero delta
    for it -- which is the difference between a guard and a setting.
    """
    if direction == "up":
        return PushbackEffect(
            target=target,
            disposition="pending_evidence",
            applied_delta=0.0,
            asserted=asserted,
            after_shrinkage=0.0,
            prior_observations=prior_observations,
            evidence_required=True,
        )
    # Downward: in full, immediately, uncapped and unshrunk. Someone telling us
    # we have overrated them is reducing what they assert, and this tool never
    # makes that expensive.
    return PushbackEffect(
        target=target,
        disposition="accepted" if asserted > 0 else "recorded_only",
        applied_delta=-asserted,
        asserted=asserted,
        after_shrinkage=asserted,
        prior_observations=prior_observations,
    )


def _preference(
    *,
    target: str,
    direction: Direction,
    asserted: float,
    prior_observations: int,
    displacement: float,
    new_information: bool,
) -> PushbackEffect:
    """Accepted, shrunk, and capped twice."""
    repetition = not new_information
    # A restatement contributes no `d`. It still counted as an observation when
    # the caller computed `prior_observations`, which is what makes the *next*
    # one move less.
    effective = 0.0 if repetition else asserted
    after_shrinkage = min(shrink(effective, prior_observations), MAX_DELTA_PER_PUSHBACK)

    sign = 1.0 if direction == "up" else -1.0
    headroom = MAX_DISPLACEMENT_PER_DIMENSION - sign * displacement
    headroom = max(headroom, 0.0)
    magnitude = min(after_shrinkage, headroom)
    applied = sign * magnitude

    # The cap bound: either it trimmed the movement, or there was no headroom
    # left to trim. Either way the answer is to stop arguing about the number.
    comparison_offered = after_shrinkage > 0 and magnitude < after_shrinkage - 1e-9

    return PushbackEffect(
        target=target,
        disposition="accepted" if applied != 0.0 else "recorded_only",
        applied_delta=applied,
        asserted=asserted,
        after_shrinkage=after_shrinkage,
        prior_observations=prior_observations,
        comparison_offered=comparison_offered,
        repetition=repetition,
    )


# -- what the panel actually shows -------------------------------------------


@dataclass(frozen=True, slots=True)
class DisplayedScore:
    """One axis's number as the panel prints it, with its provenance attached.

    The stored run is never rewritten. A correction is a labelled layer over
    the number the tool produced, computed here from the append-only log, so
    "what the tool said" and "what you moved it to" are both always readable
    -- and a pushback can no more edit `application_scores` than it can edit
    the corpus.
    """

    stored: int | None
    adjustment: float
    displayed: int | None
    # Set only by an explicit local override, which is scoped to one
    # application, labelled wherever it appears, and feeds nothing.
    override: int | None = None
    # True once this application has been sent: the score it was sent under is
    # the audit trail and corrections do not reach back into it.
    frozen: bool = False
    # True when a breached must-have or never held the number down after the
    # correction, so no amount of preference pushback erases a broken gate.
    capped_by_breach: bool = False

    @property
    def adjusted(self) -> bool:
        return (
            self.displayed is not None and self.stored is not None and self.displayed != self.stored
        )

    @property
    def effective(self) -> int | None:
        """What the user is actually looking at: the override if they set one."""
        return self.override if self.override is not None else self.displayed


def adjustment_for(
    displacements: Mapping[str, float],
    dimensions: Iterable[str],
    *,
    axis: Axis,
) -> float:
    """Total correction applying to one axis of one score, clamped.

    Only dimensions this score actually carries count -- a displacement on a
    constraint the ad was never judged against has nothing to move here -- and
    the axis is honoured, so a capability correction can never leak into the
    want number or the other way round.
    """
    wanted = {d for d in dimensions if valid_dimension(d) and dimension_axis(d) == axis}
    overall = WANT_OVERALL if axis == "want" else COULD_GET_OVERALL
    wanted.add(overall)
    total = sum(value for key, value in displacements.items() if key in wanted)
    return max(-MAX_TOTAL_ADJUSTMENT, min(MAX_TOTAL_ADJUSTMENT, total))


def _round_half_away(value: float) -> int:
    """0.5 rounds away from zero, so +0.5 and -0.5 are symmetric.

    Python's banker's rounding would make a +0.5 correction round to an even
    number and a -0.5 one round the other way, which is a visible asymmetry in
    a feature whose entire subject is asymmetry.
    """
    return int(value + (0.5 if value >= 0 else -0.5))


def _moved(stored: int, adjustment: float) -> int:
    """The stored number plus the rounded correction -- never the rounded sum.

    Rounding the sum would make the answer depend on the stored number's parity
    and on which side of it the correction fell: +0.5 on a 4 would show a move
    and -0.5 on the same 4 would not, which is a thumb on the scale in exactly
    the direction this feature exists to resist. Rounding the correction itself
    is symmetric by construction.
    """
    return stored + _round_half_away(adjustment)


def displayed_score(
    stored: int | None,
    adjustment: float,
    *,
    override: int | None = None,
    frozen: bool = False,
    breach_ceiling: bool = False,
) -> DisplayedScore:
    """Apply a correction to a stored number for display, and say so.

    `frozen` wins over everything: an already-sent application keeps the score
    it was sent under, because the record of what the tool told you at the time
    is the audit trail. `breach_ceiling` holds a want number at
    `jfl_core.fit.BREACH_CEILING` when the ad breaks a must-have or a never --
    otherwise a couple of preference pushbacks would lift a 2 to a 4 over a
    sentence saying the job is on site five days a week.
    """
    if stored is None:
        return DisplayedScore(
            stored=None, adjustment=0.0, displayed=None, override=override, frozen=frozen
        )
    if frozen:
        return DisplayedScore(
            stored=stored, adjustment=0.0, displayed=stored, override=override, frozen=True
        )
    moved = max(MIN_SCORE, min(MAX_SCORE, _moved(stored, adjustment)))
    capped = False
    if breach_ceiling:
        # The ceiling bounds the *correction*, never the stored number. A run
        # that recorded a breach has already been through
        # `jfl_core.fit.want_it_basis`, so lowering what it stored here would
        # be this layer second-guessing the pipeline rather than bounding the
        # user -- and the one thing a correction must not do is make a broken
        # must-have disappear.
        ceiling = max(BREACH_CEILING, stored)
        if moved > ceiling:
            moved = ceiling
            capped = True
    return DisplayedScore(
        stored=stored,
        adjustment=adjustment,
        displayed=moved,
        override=override,
        capped_by_breach=capped,
    )


@dataclass(frozen=True, slots=True)
class DriftMeter:
    """The number whose only job is to catch what no other metric can see.

    OpenAI's post-mortem on the GPT-4o sycophancy rollout is the argument for
    it: their offline evals looked fine and their A/B tests showed users liked
    it, because every metric they watched was one flattery scores well on. So
    this counts corrections, counts how many of them pushed upwards, and sums
    the signed movement -- and it is shown on the screen where a person is
    about to push back, not on a dashboard nobody opens.

    `upward` counts what was *asserted*, not what was applied: a run of upward
    pushbacks that the guards refused is exactly the thing worth seeing.
    """

    total: int
    upward: int
    downward: int
    net: float
    # How many asked for a number to go up and were told the corpus decides.
    pending_evidence: int = 0

    @property
    def visible(self) -> bool:
        return self.total > 0

    @property
    def sentence(self) -> str:
        """The meter in the design's own words: "11 pushbacks, 10 upward, +3.1 net"."""
        pushbacks = "pushback" if self.total == 1 else "pushbacks"
        return f"{self.total} {pushbacks}, {self.upward} upward, {self.net:+.1f} net"


__all__ = [
    "AXES",
    "COULD_GET_OVERALL",
    "DIRECTIONS",
    "DISPOSITIONS",
    "GET_DIMENSION_PREFIXES",
    "KAPPA",
    "MAX_ASSERTED_POINTS",
    "MAX_DELTA_PER_PUSHBACK",
    "MAX_DISPLACEMENT_PER_DIMENSION",
    "MAX_TOTAL_ADJUSTMENT",
    "OBSERVATION_WEIGHT_SUBMITTED_APPLICATION",
    "PUSHBACK_KINDS",
    "WANT_DIMENSION_PREFIXES",
    "WANT_OVERALL",
    "Axis",
    "Direction",
    "Disposition",
    "DisplayedScore",
    "DriftMeter",
    "ProtectedTargetError",
    "PushbackEffect",
    "PushbackKind",
    "adjustment_for",
    "decide",
    "dimension_axis",
    "displayed_score",
    "observations",
    "shrink",
    "target_dimension",
    "valid_dimension",
]
