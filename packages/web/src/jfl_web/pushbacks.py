"""What the pushback loop says on screen -- wording and view shapes, no storage.

The arithmetic is `jfl_core.pushback` and the log is
`jfl_core.storage.pushbacks`. This module turns what those two hold into
sentences, and it holds the one piece of copy that matters most: **the
receipt**, which says what changed, what did not, and what would.

Three rules about the wording, each of which follows from a decision rather
than from taste:

* **"Recorded but not accepted" is a dignified outcome, never a silent no-op.**
  A user whose correction moved nothing must be told that in the same number of
  words as one whose correction moved something, with the reason and the way
  forward. A loop that quietly ignores people is worse than one that argues
  with them.
* **The size of the effect is always stated relative to what was asserted.**
  "0.8 of the 1 point available" is the honest form; "applied" on its own is
  not, because it hides the shrinkage that is the whole anti-flattery design.
* **The refusals are explained, not merely enacted.** Being told "the number
  stayed at 4 because the corpus has no record of it, and here is the sentence
  that would change that" is the product. Being told "no" is not.

Nothing here is named `reason` -- CLAUDE.md's 2026-09-02 decision. A model's
one-line account of a classification is a `classification_note`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from jfl_core.models import ApplicationDetail, ApplicationScore, Pushback, ScoreOverride
from jfl_core.pushback import (
    COULD_GET_OVERALL,
    MAX_DELTA_PER_PUSHBACK,
    MAX_DISPLACEMENT_PER_DIMENSION,
    WANT_OVERALL,
    Axis,
    DisplayedScore,
    DriftMeter,
    adjustment_for,
    displayed_score,
)

# The headline. Shown wherever a score is, not on a page nobody opens: the
# failure this number exists to catch is invisible to every other metric, which
# is exactly what happened to OpenAI's GPT-4o rollout -- their offline evals
# looked fine and their A/B tests showed users liked it.
DRIFT_METER_HEADING = "What your corrections have done"

DRIFT_METER_NOTE = (
    "Shown because this is the one thing no other number here can catch: a tool that "
    "agrees with you whenever you ask looks good on every measure except this one. "
    "Upward counts what you asked for, not what was applied."
)

DRIFT_METER_NONE = "You have not corrected any score yet."

# On the button. Not "disagree" and not "correct" -- "push back" is what the
# design calls it and it promises the right thing: your words are recorded,
# and what they change depends on what kind of statement they are.
PUSHBACK_HEADING = "Push back on this"

PUSHBACK_INTRO = (
    "Say what is wrong with this, in your own words. It is recorded exactly as you "
    "write it, with the number and the sentence you were shown, whether or not it "
    "changes anything."
)

# Asked because people are far more reliable at relative judgements than
# absolute ones -- so this deliberately does not ask what the number should be,
# only roughly how far out it is, and the cap means the exact figure matters
# very little.
ASSERTED_POINTS_LABEL = "Roughly how far out is it?"
ASSERTED_POINTS_CHOICES: tuple[tuple[str, str], ...] = (
    ("1", "A little"),
    ("2", "Quite a lot"),
    ("3", "A long way"),
)

DIRECTION_LABEL = "Which way?"
DIRECTION_CHOICES: tuple[tuple[str, str], ...] = (
    ("up", "Too low"),
    ("down", "Too high"),
)

CLASSIFY_HEADING = "What kind of statement is this?"

CLASSIFY_INTRO = (
    "The three kinds do sharply different things, so this is yours to confirm before "
    "anything happens. Nothing has been applied yet."
)

CLASSIFY_FAILED = (
    "The classifier could not be reached, which costs you one click and nothing else: "
    "pick the kind yourself."
)

KIND_WORDING: dict[str, str] = {
    "preference": "About what I want",
    "capability": "About what I can do",
    "factual": "About what the ad says",
}

KIND_CONSEQUENCE: dict[str, str] = {
    "preference": (
        "You are the only authority on what you want, so this is accepted -- shrunk, "
        "because one correction is one observation, and capped."
    ),
    "capability": (
        "Downward this is applied in full and immediately. Upward the number does not "
        "move: it is a claim about the world, and it changes once you have confirmed the "
        "fact, not when you assert it."
    ),
    "factual": (
        "This is about the ad, not about you. Nothing on your profile moves; the fix is "
        "to read the ad again and score it again."
    ),
}

NEW_INFORMATION_LABEL = "Does this say something you have not already told me here?"

NEW_INFORMATION_NOTE = (
    "Repeating a point counts as another observation and contributes nothing to the "
    "number, so saying it again moves it less rather than more. That is deliberate."
)

# The receipt's three headings, in the design's own words.
RECEIPT_CHANGED = "What changed now"
RECEIPT_UNCHANGED = "What did not change"
RECEIPT_WOULD = "What would change it"

# Said once, plainly, wherever a pushback is offered. This is the list the
# feature is structurally unable to touch -- not a policy, a shape: there is no
# dimension naming any of them, and nothing on the pushback path imports a
# repository that could reach them.
NEVER_CHANGED = (
    "what the claim gate says about any sentence, or anything downstream of it",
    "whether a fact is one you have confirmed",
    "any requirement's coverage status -- those follow from the facts you confirmed",
    "the golden set, the eval labels or the measured over-claim rate",
    'the "unmeasured" label on these two scores',
    "the score an application was already sent under",
)

NEVER_CHANGED_HEADING = "What pushing back cannot change, ever"

# When the per-dimension cap binds. The repair is comparative rather than
# scalar: a comparison produces a constraint ("this job over that one") instead
# of an unanchored number, and it is both cheaper for the user and better data.
COMPARISON_HEADING = "Let us do this differently"

COMPARISON_INTRO = (
    "You have moved this as far as corrections go -- two points is the limit, and "
    "arguing about the number is the weakest way to tell me what you want anyway. "
    "One comparison is worth several: which of these two would you rather have?"
)

COMPARISON_NONE = (
    "There is nothing to compare this against yet -- score a second job and this "
    "becomes a question worth asking."
)

# The escape hatch, offered honestly.
OVERRIDE_HEADING = "Set the number yourself"

OVERRIDE_NOTE = (
    "This application only. It is shown as an override wherever it appears, it does "
    "not change any other job's score, it does not touch your corpus, and it does not "
    "change what the claim gate will say about a CV bullet that claims the same thing."
)

OVERRIDE_LABEL = "Your number, not the tool's"

SENT_NOTE = (
    "You have already applied for this one, so it keeps the score it was sent under. "
    "Your pushback is still recorded and still counts towards other jobs."
)

ADJUSTED_NOTE = "moved by your corrections"

EVIDENCE_HEADING = "The sentence that would move this"

EVIDENCE_NOTE = (
    "In your own words, and recorded exactly as you write them. No model goes anywhere "
    "near this text. I will not write it for you."
)

EVIDENCE_RECORDED = (
    "Recorded, and it now counts as a fact you have confirmed. The number moves when this "
    "job is scored again against it -- not before, because the number is a statement "
    "about what you have confirmed."
)

# The corpus section an answer to a pushback's evidence question lands under.
# The heading is part of a span's id, so it is written down once, here, and
# changing it would retire every statement already recorded under it.
EVIDENCE_SECTION = "Answered questions"


def drift_meter_sentence(meter: DriftMeter) -> str:
    return meter.sentence if meter.visible else DRIFT_METER_NONE


def was_sent(detail: ApplicationDetail) -> bool:
    """Whether this application has actually been sent.

    Read from the timeline rather than from the current status, because the
    status moves on: an application now at `rejected` was sent, and one moved
    back to `interested` by a correction was not. A score an application was
    sent under is the record of what the tool told you at the time, and
    corrections do not reach back into it.
    """
    if any(event.to_status == "applied" for event in detail.events):
        return True
    return detail.application.status in ("screening", "interviewing", "offer", "rejected")


def dimension_options(score: ApplicationScore, axis: Axis) -> list[tuple[str, str]]:
    """(key, label) for everything on this score a pushback may name.

    Drawn from the score itself, so the list is the user's own constraints,
    objectives and claims rather than a vocabulary invented here -- and so
    there is no way to name a claim-gate verdict, a span or a coverage status,
    because none of them is on it.
    """
    if axis == "want":
        options = [(WANT_OVERALL, "the number overall")]
        for constraint in score.constraint_verdicts:
            options.append((f"constraint:{constraint.kind}", constraint.label or constraint.kind))
        for objective in score.objective_verdicts:
            options.append(
                (
                    f"objective:{objective.rank}",
                    objective.objective or f"objective {objective.rank}",
                )
            )
        return options
    options = [(COULD_GET_OVERALL, "the number overall")]
    seen: set[str] = set()
    for lever in score.levers:
        key = f"capability:{_slug(lever.fact_text)}"
        if key in seen:
            continue
        seen.add(key)
        options.append((key, lever.fact_text))
    return options


def _slug(text: str) -> str:
    """A stable key for a capability drawn from a lever's own words.

    Content-derived, the same rule span ids and capability keys follow: the
    position of a lever in a list changes between runs, and a displacement
    keyed by position would follow the wrong claim the next time the job was
    scored.
    """
    folded = " ".join(text.split()).casefold()
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()[:16]


def dimension_label(dimension: str, options: list[tuple[str, str]]) -> str:
    """The words for a dimension, taken from the options the page offered.

    Falls back to the key itself: a score re-run may no longer carry a
    constraint an older pushback named, and showing the key is better than
    showing nothing or than inventing a label the user never wrote.
    """
    for key, label in options:
        if key == dimension:
            return label
    if dimension == WANT_OVERALL:
        return 'the "do I want this" number'
    if dimension == COULD_GET_OVERALL:
        return 'the "could I get this" number'
    return dimension.split(":", 1)[-1] or dimension


def evidence_question(label: str) -> str:
    """The exact fact that would move a capability claim, named.

    The list of what counts comes straight from the drift taxonomy's
    `ownership_inflation` row -- decisions, budget, on-call, headcount -- which
    is the evidence that claim shape requires. Naming it is what makes the
    refusal actionable rather than merely a refusal.
    """
    subject = label.strip() or "this"
    return (
        f"What did {subject} actually involve -- what decisions were yours, what "
        "budget, what on-call, how many people?"
    )


def axis_displays(
    score: ApplicationScore | None,
    displacements: dict[str, float],
    overrides: dict[str, ScoreOverride],
    *,
    sent: bool,
) -> dict[str, DisplayedScore]:
    """Both axes as the panel prints them -- the tool's number, the correction
    over it, and the override beside it. Two entries, never a third.

    The stored run is not rewritten by any of this. A correction is a labelled
    layer computed here from the append-only log, so "what the tool said" and
    "what you moved it to" are both always readable, and a pushback can no more
    edit `application_scores` than it can edit the corpus.
    """
    if score is None:
        return {}
    want_dims = [key for key, _ in dimension_options(score, "want")]
    get_dims = [key for key, _ in dimension_options(score, "get")]
    return {
        "want": displayed_score(
            score.want_it_score,
            adjustment_for(displacements, want_dims, axis="want"),
            override=overrides["want"].value if "want" in overrides else None,
            frozen=sent,
            # A breached must-have or never holds the number down whatever the
            # corrections say. Without this, two preference pushbacks would lift
            # a 2 to a 4 over a sentence explaining that the job is on site five
            # days a week, and the cap would have bought nothing.
            breach_ceiling=bool(score.hard_gate_breaches),
        ),
        "get": displayed_score(
            score.could_get_score,
            adjustment_for(displacements, get_dims, axis="get"),
            override=overrides["get"].value if "get" in overrides else None,
            frozen=sent,
        ),
    }


@dataclass(frozen=True, slots=True)
class Receipt:
    """What one pushback did, in three headings. Honest when the answer is
    "nothing changed", which is most of the time and by design.
    """

    changed: str
    unchanged: str
    would_change: str = ""
    comparison_offered: bool = False
    evidence_required: bool = False


def receipt(pushback: Pushback, *, label: str) -> Receipt:
    """Build the receipt for an applied pushback."""
    effect = pushback.effect
    asserted = float(effect.get("asserted", pushback.asserted_points) or 0)
    after = float(effect.get("after_shrinkage", 0) or 0)
    delta = pushback.applied_delta or 0.0
    n = effect.get("prior_observations", 0)
    repetition = bool(effect.get("repetition"))
    comparison = bool(effect.get("comparison_offered"))

    if pushback.disposition == "pending_evidence":
        return Receipt(
            changed=("This score is marked disputed and your words are attached to it, dated."),
            unchanged=(
                f"The number stayed at {pushback.shown_score}. It is a statement about what "
                f"you have confirmed about {label}, not a judgement about you, and nothing "
                "you say about yourself can move it on its own."
            ),
            would_change=(
                "One sentence in your own words, recorded word for word as a fact about "
                "you. Score this job again afterwards and the number is recomputed from it."
            ),
            evidence_required=True,
        )

    if pushback.classification == "factual":
        return Receipt(
            changed="Your objection is recorded against this score, in your words.",
            unchanged=(
                "Nothing on your profile moved. This is about what the ad says, so it is "
                "not evidence about what you want or what you can do."
            ),
            would_change=(
                "Read the ad again and score it again -- the number is computed from the "
                "ad, so correcting the ad is what corrects the number."
            ),
        )

    if repetition:
        return Receipt(
            changed=(f"Recorded, and counted: this dimension now has {n} observations behind it."),
            unchanged=(
                f"The number stayed at {pushback.shown_score}. You have made this point "
                "before without a new fact, and a restatement contributes nothing to the "
                "number -- it only makes the next correction move less. That is deliberate."
            ),
            would_change=(
                "Something you have not told me yet about this. A new fact moves it; the "
                "same one said more firmly does not."
            ),
            comparison_offered=comparison,
        )

    if delta == 0.0 and comparison:
        return Receipt(
            changed="Recorded against this score, in your words.",
            unchanged=(
                f"The number stayed at {pushback.shown_score}. Corrections on this dimension "
                f"have already moved it the full {MAX_DISPLACEMENT_PER_DIMENSION:g} points "
                "they are allowed to."
            ),
            would_change=(
                "A comparison rather than an argument about the number: pick which of two "
                "jobs you would rather have and I learn more from that than from another "
                "point."
            ),
            comparison_offered=True,
        )

    if delta == 0.0:
        return Receipt(
            changed="Recorded against this score, in your words.",
            unchanged=f"The number stayed at {pushback.shown_score}.",
        )

    available = min(after, MAX_DELTA_PER_PUSHBACK)
    changed = (
        f"Applied: {label} moved {delta:+.2f} of a possible "
        f"{MAX_DELTA_PER_PUSHBACK:g} point, on the {asserted:g} you asserted"
    )
    if n:
        changed += f" -- shrunk, because this dimension already had {n} observations behind it"
    changed += "."
    unchanged = (
        f"The stored score is still {pushback.shown_score}: corrections are shown as a "
        "labelled layer over what the tool produced, never written back over it."
    )
    if available > abs(delta) + 1e-9:
        unchanged += (
            f" {available - abs(delta):.2f} of the movement was held back by the "
            f"{MAX_DISPLACEMENT_PER_DIMENSION:g}-point limit on one dimension."
        )
    return Receipt(changed=changed, unchanged=unchanged, comparison_offered=comparison)


@dataclass(frozen=True, slots=True)
class AxisView:
    """One axis of one score, as the panel prints it: the tool's number, the
    correction layered over it, and the override if the user set one. Three
    values, never collapsed into one, for the same reason the two axes are
    never collapsed into one.
    """

    axis: str
    label: str
    display: DisplayedScore
    override: ScoreOverride | None = None
    options: list[tuple[str, str]] | None = None


__all__ = [
    "ADJUSTED_NOTE",
    "ASSERTED_POINTS_CHOICES",
    "ASSERTED_POINTS_LABEL",
    "CLASSIFY_FAILED",
    "CLASSIFY_HEADING",
    "CLASSIFY_INTRO",
    "COMPARISON_HEADING",
    "COMPARISON_INTRO",
    "COMPARISON_NONE",
    "DIRECTION_CHOICES",
    "DIRECTION_LABEL",
    "DRIFT_METER_HEADING",
    "DRIFT_METER_NONE",
    "DRIFT_METER_NOTE",
    "EVIDENCE_HEADING",
    "EVIDENCE_NOTE",
    "EVIDENCE_RECORDED",
    "EVIDENCE_SECTION",
    "KIND_CONSEQUENCE",
    "KIND_WORDING",
    "NEVER_CHANGED",
    "NEVER_CHANGED_HEADING",
    "NEW_INFORMATION_LABEL",
    "NEW_INFORMATION_NOTE",
    "OVERRIDE_HEADING",
    "OVERRIDE_LABEL",
    "OVERRIDE_NOTE",
    "PUSHBACK_HEADING",
    "PUSHBACK_INTRO",
    "RECEIPT_CHANGED",
    "RECEIPT_UNCHANGED",
    "RECEIPT_WOULD",
    "SENT_NOTE",
    "AxisView",
    "axis_displays",
    "Receipt",
    "dimension_label",
    "dimension_options",
    "drift_meter_sentence",
    "evidence_question",
    "receipt",
    "was_sent",
]
