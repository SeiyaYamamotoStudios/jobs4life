"""Aggregates the claim gate's verdicts against the golden set's expected labels.

The headline metric is **over-claim rate**, not accuracy -- CLAUDE.md, "Project":
"Headline metric is over-claim rate: claims passed as grounded that were not. Not
accuracy." This module exists specifically so nobody can quietly collapse that,
and its sibling, into one number later. Both are computed here, always together:

  * **over-claim** -- the gate returned "supported" for an item whose ground truth
    was "unsupported" or "review". This is the error with a real cost: the gate
    let a claim through as grounded when it was not. It is the failure mode the
    whole project exists to catch.
  * **over-flag** -- the gate returned "unsupported" for an item whose ground
    truth was "supported". This is the error that gets the tool switched off --
    the drift taxonomy's "do not over-flag framing" rule and most of the gate's
    prompt exist mainly to keep this number down.

A gate that flags every claim as unsupported scores a perfect (zero) over-claim
rate and is worthless -- it has just refused to ever say anything is grounded. A
gate that passes every claim as supported scores a perfect (zero) over-flag rate
and is worse than worthless: it is, functionally, the over-claiming tool this
project exists to oppose, wearing this project's name. **Never average, subtract,
or otherwise fold these two rates into one score** -- either one alone can be
driven to zero by a gate with no judgement at all, and only the pair together
rules that out. Report both, next to each other, every time.

`framing_rate` is tracked here too but deliberately excluded from both of the
above: FEVER claims are all factual assertions by construction (that is what the
dataset *is*), so any item the gate calls `kind="framing"` is not evidence the
gate correctly spotted framing -- it is evidence something is off in the gate's
prompt or this dataset's fit to it. It is diagnostic, never a headline number.

`framing_over_claim_rate` gets its own number, separate from all of the above, for
a reason specific to how the gate is built rather than to this dataset: framing is
the **only path in the gate with no check anywhere**. Per the prompt contract
(`jfl_gate/prompt.py`), a sentence classified `kind="framing"` is forced to
`verdict="supported"` unconditionally -- it is never compared against the corpus.
And the deterministic rule tier is explicitly forbidden from touching framing
sentences (`jfl_gate/rules.py`: "Framing is never touched"). So a claim that gets
misclassified as framing is not one wrong verdict among several possible wrong
verdicts on that item -- it is a claim for which every layer of defence this
system has was structurally bypassed before any of them ran. An over-claim that
happened *because* the gate called it "review" or "unsupported" incorrectly at
least reflects a judgement that was actually made and checked; an over-claim that
happened because the gate called it "framing" reflects no judgement being applied
at all. That is a qualitatively worse failure than an ordinary over-claim, and
folding it back into the general `over_claim_rate` (recoverable only by manually
cross-referencing `framing_rate` against `over_claim_rate` after the fact) would
hide the one number that says how often the system's failure mode with no defence
in depth actually fires. It gets its own count and its own rate for that reason.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["supported", "review", "unsupported"]
SentenceKind = Literal["claim", "framing"]


@dataclass(frozen=True)
class ItemResult:
    """One golden item's outcome: what should have happened (`expected`, from the
    golden set), and what the gate actually did (`actual`, `kind`). `actual` is
    `None` when the gate call itself failed (a `GateError`) rather than returning
    a verdict -- that is a harness failure, not a drift judgement, and is counted
    separately rather than folded into either rate.
    """

    item_id: str
    expected: Verdict
    actual: Verdict | None
    kind: SentenceKind | None = None
    error: str | None = None


def is_over_claim(expected: Verdict, actual: Verdict) -> bool:
    """The gate passed a claim as grounded that the golden set says was not."""
    return actual == "supported" and expected in ("unsupported", "review")


def is_over_flag(expected: Verdict, actual: Verdict) -> bool:
    """The gate flagged a claim the golden set says was actually grounded."""
    return actual == "unsupported" and expected == "supported"


@dataclass(frozen=True)
class ScoreSummary:
    n_items: int
    n_scored: int  # items where the gate returned a verdict
    n_errors: int  # items where the gate call itself failed (excluded from every rate below)

    over_claim_count: int
    over_claim_denominator: int  # scored items expected "unsupported" or "review"
    # None, not 0.0, when the denominator is 0: the rate is undefined on this slice,
    # not zero -- a real zero and "no data" must never look the same on a plot.
    over_claim_rate: float | None

    over_flag_count: int
    over_flag_denominator: int  # scored items expected "supported"
    over_flag_rate: float | None

    # (expected, actual) -> count, scored items only. The raw material for both
    # rates above and for a per-expected-label breakdown (group by the first
    # element) -- kept as one structure rather than three so they can never drift
    # out of sync with each other.
    confusion: dict[tuple[Verdict, Verdict], int]

    framing_count: int  # scored items the gate classified kind="framing"
    framing_rate: float | None  # framing_count / n_scored -- diagnostic, never headline

    # Scored items classified kind="framing" that were ALSO an over-claim (expected
    # "unsupported" or "review", forced to verdict "supported" by the framing
    # contract). Denominator is n_scored, matching framing_rate, not
    # over_claim_denominator -- this answers "how often does a framing
    # misclassification silently pass an ungrounded claim", not "of the claims that
    # should have failed, how many did framing let through" (a fair question too,
    # but a different one, and mixing the two denominators would make this number
    # incomparable to framing_rate right above it). See the module docstring for
    # why this gets its own number rather than being read off framing_rate and
    # over_claim_rate together.
    framing_over_claim_count: int
    framing_over_claim_rate: float | None


def per_expected_breakdown(
    confusion: dict[tuple[Verdict, Verdict], int],
) -> dict[Verdict, dict[Verdict, int]]:
    """`confusion`, regrouped by expected label -- expected -> {actual: count}."""
    breakdown: dict[Verdict, dict[Verdict, int]] = defaultdict(dict)
    for (expected, actual), count in confusion.items():
        breakdown[expected][actual] = count
    return dict(breakdown)


def aggregate(results: Sequence[ItemResult]) -> ScoreSummary:
    """Reduce a list of per-item outcomes to the headline pair plus supporting detail.

    Pure and side-effect free: no I/O, no Inspect types in or out, so the
    arithmetic here is testable against hand-built `ItemResult` lists without a
    gate call, a golden set, or an Inspect task in sight.
    """
    n_items = len(results)
    # Pairing each result with its narrowed (non-None) `actual` here, once, is what
    # lets every loop below see a plain `Verdict` instead of `Verdict | None` --
    # mypy cannot carry a filter like `r.actual is not None` through a later
    # `for r in scored` on its own.
    scored: list[tuple[ItemResult, Verdict]] = [
        (r, r.actual) for r in results if r.actual is not None
    ]
    n_scored = len(scored)
    n_errors = n_items - n_scored

    over_claim_count = 0
    over_claim_denominator = 0
    over_flag_count = 0
    over_flag_denominator = 0
    framing_count = 0
    framing_over_claim_count = 0
    confusion: Counter[tuple[Verdict, Verdict]] = Counter()

    for r, actual in scored:
        item_is_over_claim = is_over_claim(r.expected, actual)

        if item_is_over_claim:
            over_claim_count += 1
        if r.expected in ("unsupported", "review"):
            over_claim_denominator += 1

        if is_over_flag(r.expected, actual):
            over_flag_count += 1
        if r.expected == "supported":
            over_flag_denominator += 1

        confusion[(r.expected, actual)] += 1
        if r.kind == "framing":
            framing_count += 1
            if item_is_over_claim:
                framing_over_claim_count += 1

    over_claim_rate = over_claim_count / over_claim_denominator if over_claim_denominator else None
    over_flag_rate = over_flag_count / over_flag_denominator if over_flag_denominator else None
    framing_rate = framing_count / n_scored if n_scored else None
    framing_over_claim_rate = framing_over_claim_count / n_scored if n_scored else None

    return ScoreSummary(
        n_items=n_items,
        n_scored=n_scored,
        n_errors=n_errors,
        over_claim_count=over_claim_count,
        over_claim_denominator=over_claim_denominator,
        over_claim_rate=over_claim_rate,
        over_flag_count=over_flag_count,
        over_flag_denominator=over_flag_denominator,
        over_flag_rate=over_flag_rate,
        confusion=dict(confusion),
        framing_count=framing_count,
        framing_rate=framing_rate,
        framing_over_claim_count=framing_over_claim_count,
        framing_over_claim_rate=framing_over_claim_rate,
    )
