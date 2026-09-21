"""Display and form-parsing for `/profile` -- `docs/profile-schema.md`.

No SQL and no storage here; storage is `jfl_core.storage.profiles`. This module
turns submitted form fields into the models that module stores, and holds the
words the screen puts on the page -- the stance names, the tier questions, the
interest options -- so that "what the screens offer" is one list rather than
strings scattered through a template.

That matters more here than it usually would. The design doc is explicit that
the value-list drift guard cannot reach inside JSONB, so the agreement between
what the page offers and what the model accepts is held by
`packages/web/tests/test_profile.py` instead of by a CHECK constraint. Both
sides of that agreement are in this file and in `jfl_core.profile`.

**Text is never truncated, only rejected** -- the same rule the rest of the app
follows. A silently shortened answer is not what the user wrote, and this
project's whole claim is measuring distance from what someone actually said.

**Nothing here guesses.** A field left blank produces no value at all, never a
default: a constraint with no stance chosen is rejected rather than stored as a
"nice to have", and a capability whose behavioural questions are unanswered
keeps `tier=None` and reads as "not stated".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from jfl_core.profile import (
    CAPABILITY_INTERESTS,
    CAPABILITY_TIERS,
    CONSTRAINT_KINDS,
    CONSTRAINT_STANCES,
    MAX_OBJECTIVES,
    Capability,
    CapabilityInterest,
    CapabilityTier,
    Constraint,
    ConstraintKind,
    ConstraintStance,
    Disciplines,
    Objective,
    comp_value,
    location_value,
    text_value,
)

# Generous limits. They bound a row, not what anyone is allowed to say.
MAX_NOTE = 2000
MAX_TEXT = 2000
MAX_LABEL = 200
MAX_SELF_ASSESSMENT = 4000
MAX_OBJECTIVE_TEXT = 2000
# Ordered lists (locations, disciplines) are ranked by hand, and a ranked list
# nobody can hold in their head is not ranked. The cap is a nudge, not a rule
# about the world, so it rejects rather than trims.
MAX_ITEMS = 20

DEFAULT_CURRENCY = "GBP"


class FormTooLongError(ValueError):
    """A submitted text field exceeds its limit. Rejected, never truncated."""


class TooManyItemsError(ValueError):
    """A ranked list came back longer than `MAX_ITEMS`."""


class InvalidChoiceError(ValueError):
    """A submitted value is not one of the choices the page offers.

    Raised rather than ignored: a stance of "maybe" means the form was not the
    one we rendered, and quietly storing something else in its place is how a
    profile ends up saying what the user did not.
    """


class MissingStanceError(ValueError):
    """A constraint carries a value or a note but no must/nice/never.

    Not an oversight to paper over: "I said London" does not say whether London
    is a must or a preference, and the difference is the entire reason to record
    it. So the save is refused and the page says which one needs answering.
    """


class InvalidAmountError(ValueError):
    """A comp figure is not a whole, non-negative number."""


class InvalidYearError(ValueError):
    """A "last used" year is not a plausible four-digit year."""


# -- the words on the page ----------------------------------------------------

STANCE_CHOICES: tuple[tuple[ConstraintStance, str], ...] = (
    ("must", "Must have"),
    ("nice", "Nice to have"),
    ("never", "Never"),
)

ConstraintValueKind = Literal["places", "comp", "text"]


@dataclass(frozen=True, slots=True)
class ConstraintField:
    kind: ConstraintKind
    title: str
    prompt: str
    value_kind: ConstraintValueKind
    placeholder: str = ""


CONSTRAINT_FIELDS: tuple[ConstraintField, ...] = (
    ConstraintField(
        kind="location",
        title="Location",
        prompt="Where you would work from, best first.",
        value_kind="places",
        placeholder="London\nBristol\nanywhere within 90 minutes of Reading",
    ),
    ConstraintField(
        kind="workplace",
        title="Workplace",
        prompt="Remote, hybrid, on-site -- and how much of it.",
        value_kind="text",
        placeholder="Remote, or hybrid at no more than one day a fortnight",
    ),
    ConstraintField(
        kind="level_floor",
        title="Level",
        prompt="The lowest level you would take, and what you would not step back to.",
        value_kind="text",
        placeholder="Engineering manager or above; not back to individual contributor",
    ),
    ConstraintField(
        kind="comp_floor",
        title="Compensation",
        prompt="What you would not go below.",
        value_kind="comp",
    ),
    ConstraintField(
        kind="contract",
        title="Contract",
        prompt="Permanent, fixed term, contract -- and which you will not take.",
        value_kind="text",
        placeholder="Permanent only",
    ),
    ConstraintField(
        kind="right_to_work",
        title="Right to work",
        prompt="Where you can work without sponsorship.",
        value_kind="text",
        placeholder="UK; no sponsorship needed",
    ),
    ConstraintField(
        kind="notice",
        title="Notice",
        prompt="What you owe your current employer, and how firm it is.",
        value_kind="text",
        placeholder="Three months, negotiable to two",
    ),
    ConstraintField(
        kind="categorical_no",
        title="Categorical no",
        prompt="What you will not do at all, whatever else is on offer.",
        value_kind="text",
        placeholder="Gambling; defence; anything requiring weekly flights",
    ),
)

CONSTRAINT_FIELDS_BY_KIND: dict[str, ConstraintField] = {f.kind: f for f in CONSTRAINT_FIELDS}

# One line of copy, on the page, beside the two comp boxes. It is the whole
# reason the two boxes exist.
COMP_COPY = (
    "Guaranteed and headline are asked separately because a headline number is "
    "not an offer: it is base plus a bonus that may not pay and equity that may "
    "not vest. The guaranteed figure is the one a decision can rest on."
)


@dataclass(frozen=True, slots=True)
class TierQuestion:
    """One behavioural question. Answered yes or no, or left alone.

    `decides_when` names the answer to the *first* question that makes this one
    matter, so the page can say plainly which of the three is doing the work
    rather than asking three questions and explaining none of them.
    """

    key: str
    wording: str
    decides_when: str


# Behavioural, not a self-rating: "rate your Kubernetes 1-5" measures confidence,
# which is not what a CV claim needs to be measured against. Each of these asks
# about something that either happened or did not.
TIER_QUESTIONS: tuple[TierQuestion, ...] = (
    TierQuestion(
        key="hands_on",
        wording="Did you do this work yourself, hands-on?",
        decides_when="always",
    ),
    TierQuestion(
        key="production",
        wording=(
            "Did you operate it in production -- on call for it, or the person "
            "answering when it broke?"
        ),
        decides_when="if you did it yourself",
    ),
    TierQuestion(
        key="oversight",
        wording="Did you review or direct other people's work on it without doing it yourself?",
        decides_when="if you did not do it yourself",
    ),
)

TIER_ANSWERS: tuple[str, ...] = ("yes", "no")

TIER_NAMES: dict[CapabilityTier, str] = {
    "production_depth": "Production depth",
    "working": "Working",
    "oversight_only": "Oversight only",
    "absent": "Absent",
}

TIER_DESCRIPTIONS: dict[CapabilityTier, str] = {
    "production_depth": "You did it yourself and carried it in production.",
    "working": "You did it yourself, without carrying it in production.",
    "oversight_only": "You directed or reviewed it; someone else did it.",
    "absent": "On these answers, neither -- and that is a fine thing to record.",
}

INTEREST_CHOICES: tuple[tuple[CapabilityInterest, str], ...] = (
    ("want_more", "Want more of it"),
    ("happy_to", "Happy to keep doing it"),
    ("rather_not", "Would rather not"),
    ("never_again", "Not again"),
)


# -- parsing ------------------------------------------------------------------


def checked_text(value: str, limit: int = MAX_TEXT) -> str:
    if len(value) > limit:
        raise FormTooLongError(f"That text is longer than {limit} characters.")
    return value.strip()


def parse_lines(value: str, limit: int = MAX_ITEMS) -> list[str]:
    """One item per line, in the order given -- that order is the ranking.

    Blank lines are dropped rather than kept as empty items; a list longer than
    `limit` is rejected rather than trimmed, because trimming would silently
    delete the items the user ranked last.
    """
    if len(value) > MAX_TEXT * 2:
        raise FormTooLongError(f"That list is longer than {MAX_TEXT * 2} characters.")
    items = [line.strip() for line in value.splitlines()]
    kept = [item for item in items if item]
    if len(kept) > limit:
        raise TooManyItemsError(f"That is more than {limit} entries -- keep the list rankable.")
    return kept


def parse_stance(value: str) -> ConstraintStance | None:
    """None where nothing was chosen. Never a default."""
    stripped = value.strip()
    if not stripped:
        return None
    if stripped not in CONSTRAINT_STANCES:
        raise InvalidChoiceError(f"{stripped!r} is not must, nice or never.")
    return stripped


def parse_interest(value: str) -> CapabilityInterest | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped not in CAPABILITY_INTERESTS:
        raise InvalidChoiceError(f"{stripped!r} is not one of the interest options.")
    return stripped


def parse_amount(value: str) -> int | None:
    """A whole, non-negative number of currency units, or None if left blank."""
    stripped = value.strip().replace(",", "").replace(" ", "")
    if not stripped:
        return None
    if not stripped.isdigit():
        raise InvalidAmountError("That figure is not a whole number.")
    return int(stripped)


def parse_year(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    if not stripped.isdigit() or not 1900 <= int(stripped) <= 2100:
        raise InvalidYearError("That is not a four-digit year.")
    return int(stripped)


def parse_tier_answer(value: str) -> bool | None:
    """yes / no / unanswered. Anything else is a form we did not render."""
    stripped = value.strip()
    if not stripped:
        return None
    if stripped not in TIER_ANSWERS:
        raise InvalidChoiceError(f"{stripped!r} is not yes or no.")
    return stripped == "yes"


def tier_from_answers(
    hands_on: bool | None, production: bool | None, oversight: bool | None
) -> CapabilityTier | None:
    """The tier those three answers imply, or None while it is still unanswered.

    The mapping, and nothing else is inferred:

      * did it, ran it in production          -> production_depth
      * did it, did not run it in production  -> working
      * did not do it, oversaw others         -> oversight_only
      * did not do it, did not oversee it     -> absent

    The unasked question never matters: whether someone reviewed others' work is
    irrelevant once they say they did the work themselves, and whether they were
    on call is irrelevant once they say they did not do it. So a row answered
    only where it counts still produces a tier, and a row missing the answer that
    counts stays `None` -- "not stated", never a guess at the lower value.
    """
    if hands_on is None:
        return None
    if hands_on:
        if production is None:
            return None
        return "production_depth" if production else "working"
    if oversight is None:
        return None
    return "oversight_only" if oversight else "absent"


def answers_for_tier(tier: CapabilityTier | None) -> dict[str, bool | None]:
    """The saved tier read back as the answers that produced it, so the form
    re-renders what the user actually ticked.

    The irrelevant question comes back `None` rather than guessed: someone with
    production depth was never asked whether they also reviewed others' work, and
    showing an answer there would be the page inventing one.
    """
    if tier is None:
        return {"hands_on": None, "production": None, "oversight": None}
    mapping: dict[str, dict[str, bool | None]] = {
        "production_depth": {"hands_on": True, "production": True, "oversight": None},
        "working": {"hands_on": True, "production": False, "oversight": None},
        "oversight_only": {"hands_on": False, "production": None, "oversight": True},
        "absent": {"hands_on": False, "production": None, "oversight": False},
    }
    return mapping[tier]


def parse_constraints(form: Mapping[str, str]) -> list[Constraint]:
    """The whole constraints section, in the page's order.

    A kind with nothing said about it produces no constraint at all -- not a row
    saying "not stated", which would be indistinguishable from an answer once
    something read it back.
    """
    constraints: list[Constraint] = []
    for field in CONSTRAINT_FIELDS:
        stance = parse_stance(form.get(f"stance-{field.kind}", ""))
        note = checked_text(form.get(f"note-{field.kind}", ""), MAX_NOTE)
        value = _constraint_value(field, form)
        if stance is None:
            if value or note:
                raise MissingStanceError(
                    f"Say whether {field.title.lower()} is a must, a nice-to-have or a never "
                    "-- it is recorded with a stance or not at all."
                )
            continue
        constraints.append(Constraint(kind=field.kind, stance=stance, value=value, note=note))
    return constraints


def _constraint_value(field: ConstraintField, form: Mapping[str, str]) -> dict[str, object]:
    if field.value_kind == "places":
        places = parse_lines(form.get(f"places-{field.kind}", ""))
        return location_value(places) if places else {}
    if field.value_kind == "comp":
        guaranteed = parse_amount(form.get("comp-guaranteed", ""))
        headline = parse_amount(form.get("comp-headline", ""))
        currency = checked_text(form.get("comp-ccy", ""), 8).upper() or DEFAULT_CURRENCY
        if guaranteed is None and headline is None:
            return {}
        return comp_value(guaranteed, headline, currency)
    return text_value(checked_text(form.get(f"text-{field.kind}", ""), MAX_TEXT))


def parse_capability(
    existing: Capability,
    *,
    hands_on: str,
    production: str,
    oversight: str,
    interest: str,
    last_used: str,
) -> Capability:
    """One capability row as submitted, on top of the row as it stands.

    `label`, `evidence` and `source` come from `existing` and never from the
    form: evidence is a list of span ids, and a span id arriving from a browser
    is a claim about grounding that the browser does not get to make.
    """
    tier = tier_from_answers(
        parse_tier_answer(hands_on),
        parse_tier_answer(production),
        parse_tier_answer(oversight),
    )
    return existing.model_copy(
        update={
            "tier": tier,
            "interest": parse_interest(interest),
            "last_used": parse_year(last_used),
        }
    )


def parse_disciplines(practises: str, not_this: str) -> Disciplines:
    return Disciplines(practises=parse_lines(practises), **{"not": parse_lines(not_this)})


def parse_objectives(slots: Sequence[tuple[int, str, str]]) -> list[Objective]:
    """Up to four ranked slots. A slot with no text is simply absent.

    The rank is the slot the user put it in, so slots do not shuffle underneath
    them when an earlier one is cleared. Evidence with no objective text is
    dropped with it -- "what would show a role delivers this" is meaningless
    without the this.
    """
    objectives: list[Objective] = []
    for rank, text, evidence in slots:
        if rank < 1 or rank > MAX_OBJECTIVES:
            raise InvalidChoiceError(f"{rank} is not an objective slot.")
        body = checked_text(text, MAX_OBJECTIVE_TEXT)
        if not body:
            continue
        objectives.append(
            Objective(
                rank=rank,
                text=body,
                evidence_of_delivery=checked_text(evidence, MAX_OBJECTIVE_TEXT),
            )
        )
    return objectives


def merge_capabilities(
    saved: Sequence[Capability], seeded: Sequence[Capability]
) -> list[Capability]:
    """What the page shows: every saved row, plus any seed the user has not
    touched yet.

    Saved wins on a collision, always. A seed is a proposal from a confirmed CV
    fact, and re-deriving it must never overwrite the tier a person actually
    answered for -- nor drop the evidence it arrived with, which is why a saved
    row with no evidence of its own inherits the seed's.
    """
    by_key = {capability.key: capability for capability in saved}
    for proposal in seeded:
        current = by_key.get(proposal.key)
        if current is None:
            by_key[proposal.key] = proposal
        elif not current.evidence and proposal.evidence:
            by_key[proposal.key] = current.model_copy(update={"evidence": proposal.evidence})
    return list(by_key.values())


# Re-exported so the template context has one import site for the page's words.
__all__ = [
    "CAPABILITY_TIERS",
    "COMP_COPY",
    "CONSTRAINT_FIELDS",
    "CONSTRAINT_FIELDS_BY_KIND",
    "CONSTRAINT_KINDS",
    "INTEREST_CHOICES",
    "MAX_ITEMS",
    "MAX_LABEL",
    "MAX_NOTE",
    "MAX_OBJECTIVE_TEXT",
    "MAX_SELF_ASSESSMENT",
    "MAX_TEXT",
    "STANCE_CHOICES",
    "TIER_DESCRIPTIONS",
    "TIER_NAMES",
    "TIER_QUESTIONS",
    "FormTooLongError",
    "InvalidAmountError",
    "InvalidChoiceError",
    "InvalidYearError",
    "MissingStanceError",
    "TooManyItemsError",
    "answers_for_tier",
    "checked_text",
    "merge_capabilities",
    "parse_amount",
    "parse_capability",
    "parse_constraints",
    "parse_disciplines",
    "parse_interest",
    "parse_lines",
    "parse_objectives",
    "parse_stance",
    "parse_tier_answer",
    "parse_year",
    "tier_from_answers",
]
