"""What the pushback box says on screen -- wording and view shapes, no storage.

The arithmetic is `jfl_core.pushback` and the log is
`jfl_core.storage.pushbacks`. This module turns what those two hold into
sentences a person reads once and understands.

The owner's complaint that shaped it: *"The push back facility from the scoring
is too complicated. I think it should just be a description that someone types
into a box. Currently it's complex -- I am not sure what's changing as a
result."* Two failures, and this module answers both:

* **One box.** No kind to pick, no direction, no "how far out". The words are
  read (a cheap model call, in the worker), the reading is applied at once, and
  the card says what was taken. "Not what I meant" undoes it and lets the user
  pick the right reading from a short list -- the only place the kinds appear,
  and in words a person would use.
* **A result you cannot miss.** The card leads with the number before -> after,
  or "stays at" when nothing moved, and then says why in one sentence: small
  steps by design, the fact that would move it, or that it was about the ad.

Rules about the wording, each from a decision rather than taste:

* **"Nothing moved" gets as many words as "something moved."** A loop that
  quietly ignores people is worse than one that argues with them.
* **No internal words on screen.** Nothing here says shrinkage, dimension,
  classification, preference, capability or corpus; `test_pushback_wording.py`
  sweeps the rendered markup for them.
* **Never claim an action that did not happen.** A factual objection about the
  ad is *not* fed back into scoring yet, so the card says so and offers the
  re-score button rather than saying the ad was re-read with the correction.

Nothing here is named `reason` -- CLAUDE.md's 2026-09-02 decision.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from jfl_core.fit import BREACH_CEILING, MAX_SCORE, MIN_SCORE
from jfl_core.models import ApplicationDetail, ApplicationScore, Pushback, ScoreOverride
from jfl_core.pushback import (
    COULD_GET_OVERALL,
    MAX_DISPLACEMENT_PER_DIMENSION,
    MAX_TOTAL_ADJUSTMENT,
    WANT_OVERALL,
    Axis,
    Direction,
    DisplayedScore,
    DriftMeter,
    PushbackKind,
    adjustment_for,
    dimension_axis,
    displayed_score,
)

from jfl_web.scores import COULD_GET_LABEL, WANT_IT_LABEL

# -- the box -----------------------------------------------------------------

BOX_LABEL = "Disagree with this? Tell us why."
BOX_BUTTON = "Send"


# -- the readings: the only place the kinds appear, in a person's words -------


@dataclass(frozen=True, slots=True)
class Reading:
    """One way of reading what someone typed, in their words and in ours.

    `choice` is how the person would say it (the "Not what I meant" list);
    `meaning` is the card's first line, how we say we took it. `direction` is
    None for the ad: an objection about the ad keeps whichever way the words
    pushed, and moves nothing either way.
    """

    key: str
    kind: PushbackKind
    direction: Direction | None
    choice: str
    meaning: str


READINGS: tuple[Reading, ...] = (
    Reading(
        "want_more",
        "preference",
        "up",
        "I'd want this more than you scored",
        "You'd take roles like this more readily than we scored.",
    ),
    Reading(
        "want_less",
        "preference",
        "down",
        "I'd want this less than you scored",
        "You'd be less keen on roles like this than we scored.",
    ),
    Reading(
        "fit_more",
        "capability",
        "up",
        "I'm a stronger fit than you scored",
        "You think you're a stronger fit than we scored.",
    ),
    Reading(
        "fit_less",
        "capability",
        "down",
        "You've overrated my fit",
        "You think we've overrated your fit.",
    ),
    Reading(
        "ad_misread",
        "factual",
        None,
        "You've misread the ad",
        "You think we've misread the ad.",
    ),
)

_READINGS_BY_KEY = {reading.key: reading for reading in READINGS}


def reading_by_key(key: str) -> Reading | None:
    return _READINGS_BY_KEY.get(key)


def reading_of(pushback: Pushback) -> Reading | None:
    """Which reading was applied to this row, if any."""
    for reading in READINGS:
        if reading.kind != pushback.classification:
            continue
        if reading.direction is None or reading.direction == pushback.asserted_direction:
            return reading
    return None


# -- the card's sentences ----------------------------------------------------

NOT_MEANT = "Not what I meant"
NOT_MEANT_INTRO = "This undoes it. Which is closer?"
JUST_UNDO = "Just undo it"

READING_NOW = "Reading what you wrote… nothing has changed yet."
READING_FAILED = (
    "We couldn't work out what you meant, so nothing has changed yet. Which is closest?"
)
UNDONE = "Undone. That correction no longer counts for anything."

EVERY_JOB = "This counts for every job you score, not just this one."
SMALL_STEPS = (
    "One correction moves it a little; repeated ones move it less, so a single mood "
    "can't rewrite your profile."
)
TRIMMED = (
    f"It stopped there: corrections can move a score {MAX_DISPLACEMENT_PER_DIMENSION:g} "
    "points at most."
)
IN_FULL = "Taken as you said it, in full: telling us we've overrated you needs no proof."
REPEATED = (
    "You've made this point before, and saying it again doesn't move it. Something new would."
)
REPEATED_FIT = "You've said this before. Saying it again can't move it; the fact behind it can."
AT_THE_LIMIT = (
    f"Your corrections have already moved this as far as they can -- "
    f"{MAX_DISPLACEMENT_PER_DIMENSION:g} points. Comparing two jobs tells us more than "
    "another nudge:"
)
AT_THE_LIMIT_LINK = "pick one to compare it with"
TOTAL_LIMIT = (
    f"Corrections have already moved this score as far as they can here -- "
    f"{MAX_TOTAL_ADJUSTMENT:g} points. Yours still counts for other jobs."
)
BREACH_HOLDS = (
    "A must-have this ad breaks holds it here, whatever your corrections say. Yours "
    "still counts for other jobs."
)
SENT_HOLDS = (
    "You've already applied, so this one keeps the score it was sent under. Your "
    "correction still counts for other jobs."
)
NEEDS_THE_FACT = (
    "Your score won't move on your word alone -- confirm the fact behind it and it will:"
)
NEEDS_THE_FACT_LINK = "answer this"
FACT_CONFIRMED = "You've confirmed the fact behind it. Score this job again and it counts."
ABOUT_THE_AD = "Nothing about you changed: this is about the ad, so your profile is untouched."
AD_NOT_REREAD = (
    "We can't yet re-read the ad with your correction in mind: scoring again reads the "
    "ad as it was pasted. Your note stays with this score."
)
NOTHING_MOVED = "Kept, in your words. Nothing moved."

# The question a "stronger fit" claim opens. The list of what counts is the
# drift taxonomy's `ownership_inflation` row -- decisions, budget, on-call,
# headcount -- which is the evidence that claim shape requires.
EVIDENCE_QUESTION = (
    "What's the fact behind it? Say what you did, where, and what was yours -- the "
    "decisions, the budget, the on-call, how many people."
)

EVIDENCE_NOTE = (
    "In your own words, and kept exactly as you write them. Nothing rewrites this text, "
    "and we won't write it for you."
)

# The section an answer to a "stronger fit" question lands under. The heading is
# part of a span's id, so it is written down once, here, and changing it would
# retire every statement already recorded under it.
EVIDENCE_SECTION = "Answered questions"


# -- the drift sentence ------------------------------------------------------

# When the one quiet sentence becomes the loud panel. See `drift_line`.
DRIFT_LOUD_MIN_UPWARD = 3
DRIFT_LOUD_UPWARD_SHARE = 2 / 3

DRIFT_LINK = "See everything you've said"
DRIFT_LOUD = (
    "Most of that asks for higher scores. A tool that agrees whenever it's asked stops "
    "telling you anything, so it's worth a look."
)

# What pushing back is structurally unable to touch. Plain words for the list
# `jfl_core.pushback` enforces by allowlist: no dimension names any of these,
# and nothing on the pushback path imports a repository that could reach them.
NEVER_CHANGED_HEADING = "What disagreeing can never change"
NEVER_CHANGED = (
    "what we say about any sentence in your CV or cover letter",
    "which facts about you are confirmed",
    "how well your confirmed facts cover a job's requirements",
    "how we measure this tool's own accuracy",
    'the "unmeasured" label on these two scores',
    "the score an application was already sent under",
)

# The escape hatch, offered honestly.
OVERRIDE_HEADING = "Set the number yourself"

OVERRIDE_NOTE = (
    "This application only. It is shown as an override wherever it appears, it does "
    "not change any other job's score, it does not touch the facts you confirmed, and "
    "it does not change what we say about a CV line that claims the same thing."
)

OVERRIDE_LABEL = "Your number, not the tool's"

SENT_NOTE = (
    "You have already applied for this one, so it keeps the score it was sent under. "
    "Your corrections still count towards other jobs."
)

ADJUSTED_NOTE = "moved by your corrections"


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


# -- the card ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Change:
    """One number, before and after. `after` None means it stayed put."""

    label: str
    before: str
    after: str | None = None
    # What the big number above now shows, when that is the rounded `after`.
    shown_as: int | None = None


CardState = Literal["reading", "failed", "applied", "undone"]


@dataclass(frozen=True, slots=True)
class Card:
    """What one pushback did, written for a person. Built by `card()`."""

    state: CardState
    meaning: str = ""
    change: Change | None = None
    lines: tuple[str, ...] = ()
    # A sentence that ends in a link: ("lead", "link words", "href").
    link: tuple[str, str, str] | None = None
    rescore: bool = False
    reading_key: str = ""
    # The readings offered under "Not what I meant" (or when reading failed).
    other_readings: tuple[Reading, ...] = field(default_factory=tuple)
    can_undo: bool = False


def _fmt(value: float) -> str:
    """5 -> "5", 5.83 -> "5.8". One decimal is as precise as this deserves."""
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _axis_label(axis: Axis) -> str:
    return WANT_IT_LABEL if axis == "want" else COULD_GET_LABEL


def _stored(score: ApplicationScore, axis: Axis) -> int | None:
    return score.want_it_score if axis == "want" else score.could_get_score


def _exact(stored: int, adjustment: float, ceiling: int | None) -> float:
    value = min(float(MAX_SCORE), max(float(MIN_SCORE), stored + adjustment))
    if ceiling is not None:
        value = min(value, float(ceiling))
    return value


def _before_after(
    pushback: Pushback, score: ApplicationScore, log: list[Pushback]
) -> tuple[Axis, int, float, float, int | None, bool] | None:
    """This score's number just before this correction and just after it.

    Computed from the log, never stored: the profile before a correction is the
    sum of everything applied ahead of it, and after is that plus this one.
    Returns (axis, stored, before, after, shown_as, held_by_breach), or None
    when the correction is not in the log (withdrawn) or the axis has no number.
    """
    axis = dimension_axis(pushback.target_dimension or pushback.dimension)
    stored = _stored(score, axis)
    if stored is None:
        return None
    before: dict[str, float] = defaultdict(float)
    for row in log:
        if row.id == pushback.id:
            break
        before[row.target_dimension] += row.applied_delta or 0.0
    else:
        return None
    after = dict(before)
    after[pushback.target_dimension] = after.get(pushback.target_dimension, 0.0) + (
        pushback.applied_delta or 0.0
    )
    dims = [key for key, _ in dimension_options(score, axis)]
    breach = axis == "want" and bool(score.hard_gate_breaches)
    ceiling = max(BREACH_CEILING, stored) if breach else None
    adj_before = adjustment_for(before, dims, axis=axis)
    adj_after = adjustment_for(after, dims, axis=axis)
    value_before = _exact(stored, adj_before, ceiling)
    value_after = _exact(stored, adj_after, ceiling)
    shown = displayed_score(stored, adj_after, breach_ceiling=breach).displayed
    held = breach and stored + adj_after > float(ceiling or MAX_SCORE)
    return axis, stored, value_before, value_after, shown, held


def card(
    pushback: Pushback,
    *,
    score: ApplicationScore | None,
    log: list[Pushback],
    sent: bool,
) -> Card:
    """The one card the panel shows for the latest pushback.

    `log` is `PostgresPushbackRepository.applied_log()`: every correction that
    still counts, in the order applied. `score` is the score on screen now, so
    the before -> after is about the number the person is looking at.
    """
    reading = reading_of(pushback)
    others = tuple(r for r in READINGS if reading is None or r.key != reading.key)

    if pushback.withdrawn:
        return Card(
            state="undone",
            meaning=reading.meaning if reading else "",
            lines=(UNDONE,),
            reading_key=reading.key if reading else "",
        )
    if pushback.status != "applied":
        if pushback.error_code is not None:
            return Card(state="failed", lines=(READING_FAILED,), other_readings=READINGS)
        return Card(state="reading", lines=(READING_NOW,))

    base = {
        "state": "applied",
        "meaning": reading.meaning if reading else "",
        "reading_key": reading.key if reading else "",
        "other_readings": others,
        "can_undo": True,
    }
    effect = pushback.effect
    delta = pushback.applied_delta or 0.0
    kind = pushback.classification
    everywhere = pushback.target_dimension in (WANT_OVERALL, COULD_GET_OVERALL)

    if kind == "factual":
        return Card(**base, lines=(ABOUT_THE_AD, AD_NOT_REREAD), rescore=True)  # type: ignore[arg-type]

    numbers = _before_after(pushback, score, log) if score is not None else None
    axis = dimension_axis(pushback.target_dimension or pushback.dimension)
    label = _axis_label(axis)

    def stays() -> Change | None:
        if score is None:
            return None
        stored = _stored(score, axis)
        if stored is None:
            return None
        if sent:
            return Change(label=label, before=_fmt(stored))
        if numbers is not None:
            return Change(label=label, before=_fmt(numbers[3]))
        return Change(label=label, before=_fmt(stored))

    if pushback.disposition == "pending_evidence":
        lines: list[str] = []
        if effect.get("repetition"):
            lines.append(REPEATED_FIT)
        if pushback.resulting_span_id is not None:
            lines.append(FACT_CONFIRMED)
            return Card(**base, change=stays(), lines=tuple(lines), rescore=True)  # type: ignore[arg-type]
        return Card(
            **base,  # type: ignore[arg-type]
            change=stays(),
            lines=tuple(lines),
            link=(NEEDS_THE_FACT, NEEDS_THE_FACT_LINK, f"/pushbacks/{pushback.id}/evidence"),
        )

    if delta == 0.0:
        if effect.get("repetition"):
            return Card(**base, change=stays(), lines=(REPEATED,))  # type: ignore[arg-type]
        if effect.get("comparison_offered"):
            return Card(
                **base,  # type: ignore[arg-type]
                change=stays(),
                link=(AT_THE_LIMIT, AT_THE_LIMIT_LINK, "/applications"),
            )
        return Card(**base, change=stays(), lines=(NOTHING_MOVED,))  # type: ignore[arg-type]

    if sent:
        return Card(**base, change=stays(), lines=(SENT_HOLDS,))  # type: ignore[arg-type]

    if numbers is None:
        # No number on this axis to show a move against; say what it did.
        lines = [EVERY_JOB] if everywhere else []
        return Card(**base, lines=tuple(lines))  # type: ignore[arg-type]

    _, _, before, after, shown, held = numbers
    if abs(after - before) < 1e-9:
        return Card(
            **base,  # type: ignore[arg-type]
            change=Change(label=label, before=_fmt(before)),
            lines=(BREACH_HOLDS if held else TOTAL_LIMIT,),
        )

    change = Change(
        label=label,
        before=_fmt(before),
        after=_fmt(after),
        shown_as=shown if shown is not None and abs(shown - after) > 1e-9 else None,
    )
    lines = []
    if kind == "capability":
        lines.append(IN_FULL)
    else:
        asserted = float(effect.get("asserted", pushback.asserted_points) or 0)
        after_shrinkage = float(effect.get("after_shrinkage", 0) or 0)
        if after_shrinkage < asserted - 1e-9 or abs(delta) < asserted - 1e-9:
            lines.append(SMALL_STEPS)
        if effect.get("comparison_offered"):
            lines.append(TRIMMED)
    if everywhere:
        lines.append(EVERY_JOB)
    if held:
        lines.append(BREACH_HOLDS)
    return Card(**base, change=change, lines=tuple(lines))  # type: ignore[arg-type]


# -- one line per correction, for the history and the /pushbacks log ----------


def _amount(value: float) -> str:
    value = abs(value)
    if abs(value - 1.0) < 1e-9:
        return "1 point"
    if value < 1.0:
        return f"{_fmt(value)} of a point"
    return f"{_fmt(value)} points"


def history_line(pushback: Pushback) -> str:
    """What one correction changed, in one sentence."""
    if pushback.withdrawn:
        return "Undone -- it no longer counts."
    if pushback.status != "applied":
        if pushback.error_code is not None:
            return "Not read yet -- nothing changed."
        return "Still reading it."
    if pushback.classification == "factual":
        return "Changed nothing about you: it was about the ad."
    if pushback.disposition == "pending_evidence":
        if pushback.resulting_span_id is not None:
            return "You confirmed the fact behind it; it counts once the job is scored again."
        return "Changed nothing yet: waiting on the fact behind it."
    delta = pushback.applied_delta or 0.0
    if delta:
        axis = dimension_axis(pushback.target_dimension or pushback.dimension)
        way = "up" if delta > 0 else "down"
        line = f"Moved “{_axis_label(axis)}” {way} by {_amount(delta)}"
        if pushback.target_dimension in (WANT_OVERALL, COULD_GET_OVERALL):
            line += ", for every job"
        return line + "."
    if pushback.effect.get("repetition"):
        return "Changed nothing: you'd said it before."
    if pushback.effect.get("comparison_offered"):
        return "Changed nothing: corrections had already moved it as far as they can."
    return "Changed nothing."


# -- the drift sentence --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DriftLine:
    text: str
    loud: bool


def _times(count: int) -> str:
    if count == 1:
        return "once"
    if count == 2:
        return "twice"
    return f"{count} times"


def _which_way(meter: DriftMeter) -> str:
    if meter.upward == meter.total:
        return "all upward" if meter.total > 1 else "upward"
    if meter.downward == meter.total:
        return "all downward" if meter.total > 1 else "downward"
    if meter.upward > meter.downward:
        return "mostly upward"
    if meter.downward > meter.upward:
        return "mostly downward"
    return "as often up as down"


def drift_line(meter: DriftMeter) -> DriftLine | None:
    """One quiet sentence under the score, loud only past a threshold.

    Loud when **at least three** corrections have asked for a higher number
    **and at least two thirds** of all of them did, or when the net movement
    has reached the most corrections may move any one score (two points).

    Why those: three is where the research's worked example stops treating an
    upward push as a one-off and names the disagreement as standing; two
    thirds separates a one-sided run from ordinary calibration in both
    directions, which is what an honest user's corrections look like; and a
    net of two points means every score on screen may already be sitting at
    the limit of what corrections can do -- the point at which a flattering
    drift has stopped being hypothetical. Below all three, the sentence is
    there for anyone who looks and does not shout at anyone who does not.
    """
    if meter.total <= 0:
        return None
    text = f"You've pushed back {_times(meter.total)} on this profile, {_which_way(meter)}."
    lopsided = (
        meter.upward >= DRIFT_LOUD_MIN_UPWARD
        and meter.upward >= DRIFT_LOUD_UPWARD_SHARE * meter.total
    )
    loud = lopsided or meter.net >= MAX_TOTAL_ADJUSTMENT - 1e-9
    return DriftLine(text=text, loud=loud)


__all__ = [
    "ADJUSTED_NOTE",
    "BOX_BUTTON",
    "BOX_LABEL",
    "DRIFT_LINK",
    "DRIFT_LOUD",
    "DRIFT_LOUD_MIN_UPWARD",
    "DRIFT_LOUD_UPWARD_SHARE",
    "EVIDENCE_NOTE",
    "EVIDENCE_QUESTION",
    "EVIDENCE_SECTION",
    "JUST_UNDO",
    "NEVER_CHANGED",
    "NEVER_CHANGED_HEADING",
    "NOT_MEANT",
    "NOT_MEANT_INTRO",
    "OVERRIDE_HEADING",
    "OVERRIDE_LABEL",
    "OVERRIDE_NOTE",
    "READINGS",
    "SENT_NOTE",
    "Card",
    "Change",
    "DriftLine",
    "Reading",
    "axis_displays",
    "card",
    "dimension_label",
    "dimension_options",
    "drift_line",
    "history_line",
    "reading_by_key",
    "reading_of",
    "was_sent",
]
