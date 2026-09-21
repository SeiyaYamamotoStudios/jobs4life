"""The profile: one row, five sections -- `docs/profile-schema.md`, 2026-09-21.

**PLACEHOLDER.** The storage layer for this shape is being written in parallel;
this module is the minimum that lets `/profile` be built and tested against a
real table, and is meant to be replaced wholesale by that work. Everything the
web layer needs from it is in the four names at the bottom of this docstring, so
the swap is an import change rather than a rewrite.

One Pydantic model is the only write path into `profiles.data`, which is the
cost the design doc accepts openly: the value-list drift guard
(`packages/core/tests/test_value_lists_agree.py`) compares a `Literal` against a
tuple against a CHECK constraint, and none of that reaches inside JSONB. So the
guard here is this model plus
`packages/web/tests/test_profile.py::test_the_screens_offer_exactly_the_stored_values`,
which asserts that what the screens offer is exactly what the model accepts.
Weaker than a CHECK, deliberately, and the price of a shape we expect to change.

Two rules from the design doc are enforced in the types rather than in prose:

* **A tier is optional.** `tier is None` means the user has not answered the
  behavioural questions for that capability, and the page says "not stated".
  Nothing infers a tier from a CV, a job title or another capability.
* **A capability with no evidence is a claim, not a fact.** `evidence` holds
  span ids; an empty list is rendered as "claimed, not yet evidenced" and must
  never be presented as grounding.

`Constraint.value` is a plain dict rather than a per-kind model, the same choice
`ProfileAnswer.structured` made and for the same reason: three kinds have three
genuinely different shapes and a model unifying them would have every field
optional, which validates nothing. The shapes are built by `location_value`,
`comp_value` and `text_value` below, so there is one place that knows them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jfl_core.ids import fold
from jfl_core.models import CandidateFact

SCHEMA_VERSION = 1

# -- closed value sets --------------------------------------------------------
# `must / nice / never`: a negative preference gets equal standing with a
# positive one, rather than being a missing positive.
ConstraintStance = Literal["must", "nice", "never"]

ConstraintKind = Literal[
    "location",
    "workplace",
    "level_floor",
    "comp_floor",
    "contract",
    "right_to_work",
    "notice",
    "categorical_no",
]

# Depth, in observable terms. Set by behavioural questions, never by a
# self-rating -- see `jfl_web.profile.TIER_QUESTIONS` for the exact wording the
# screen asks and `tier_from_answers` for the mapping.
CapabilityTier = Literal["production_depth", "working", "oversight_only", "absent"]

# Appetite, which is a different question from depth and is asked separately.
CapabilityInterest = Literal["want_more", "happy_to", "rather_not", "never_again"]

CapabilitySource = Literal["cv_fact", "user"]

CONSTRAINT_KINDS: tuple[ConstraintKind, ...] = (
    "location",
    "workplace",
    "level_floor",
    "comp_floor",
    "contract",
    "right_to_work",
    "notice",
    "categorical_no",
)

CONSTRAINT_STANCES: tuple[ConstraintStance, ...] = ("must", "nice", "never")
CAPABILITY_TIERS: tuple[CapabilityTier, ...] = (
    "production_depth",
    "working",
    "oversight_only",
    "absent",
)
CAPABILITY_INTERESTS: tuple[CapabilityInterest, ...] = (
    "want_more",
    "happy_to",
    "rather_not",
    "never_again",
)

MAX_OBJECTIVES = 4


def capability_key(label: str) -> str:
    """A stable, URL-safe id for a capability row.

    Derived from the folded label, so the same capability proposed twice from
    two CVs is one row, and so a row's save URL does not change when the label
    is re-cased. Hex rather than a slug because a label may be any text at all.
    """
    return hashlib.sha256(fold(label).encode("utf-8")).hexdigest()[:16]


def location_value(places: Sequence[str]) -> dict[str, Any]:
    """Locations are an **ordered list**, not a relocate boolean: first choice
    first. A single "will you relocate" flag loses the ordering, which is the
    part that decides anything.
    """
    return {"places": [p for p in (place.strip() for place in places) if p]}


def comp_value(guaranteed: int | None, headline: int | None, ccy: str) -> dict[str, Any]:
    """Guaranteed and headline are carried **separately**, because a headline
    number is not an offer. Either may be absent; neither is derived from the
    other.
    """
    value: dict[str, Any] = {"ccy": ccy}
    if guaranteed is not None:
        value["guaranteed"] = guaranteed
    if headline is not None:
        value["headline"] = headline
    return value


def text_value(text: str) -> dict[str, Any]:
    """Every other constraint kind: the user's own words, stored verbatim."""
    stripped = text.strip()
    return {"text": stripped} if stripped else {}


class Constraint(BaseModel):
    """One constraint of one kind, with the stance the user gave it.

    `stance` is required. A value typed with no stance chosen is rejected at the
    form rather than stored with a guessed stance -- "I said London" does not
    say whether London is a must or a preference, and the difference is the
    whole point of recording it.
    """

    kind: ConstraintKind
    stance: ConstraintStance
    value: dict[str, Any] = Field(default_factory=dict)
    # The user's own words about this constraint, stored verbatim. Never a
    # model's paraphrase.
    note: str = ""


class Capability(BaseModel):
    """Something the user can do, at a stated depth, with a separate appetite.

    `tier` and `interest` are independent axes and are never combined: what you
    are good at and what you want to keep doing are different questions, and a
    single "skill level" number answers neither.
    """

    label: str
    tier: CapabilityTier | None = None
    interest: CapabilityInterest | None = None
    last_used: int | None = None
    # Span ids. Empty means "claimed, not yet evidenced" -- a claim, not a fact.
    evidence: list[uuid.UUID] = Field(default_factory=list)
    source: CapabilitySource = "user"

    @property
    def key(self) -> str:
        return capability_key(self.label)

    @property
    def evidenced(self) -> bool:
        return bool(self.evidence)


class Disciplines(BaseModel):
    """What you practise, ranked, and an explicit "not this".

    The negative list is not decoration: without it the same job title at two
    employers reads as the same job, and a filter has nothing to rule out.
    """

    model_config = ConfigDict(populate_by_name=True)

    practises: list[str] = Field(default_factory=list)
    # Serialised as "not", per the design doc's JSON. `not` is a keyword, so the
    # attribute carries the trailing underscore and the alias carries the name.
    not_: list[str] = Field(default_factory=list, alias="not")


class Objective(BaseModel):
    """One objective, its rank, and what would show a role delivers it.

    **Ranked, never weighted.** A weight invites blending, and blending hides
    the trade-off that is the only reason to look at objectives at all.
    """

    rank: int = Field(ge=1, le=MAX_OBJECTIVES)
    text: str
    evidence_of_delivery: str = ""


class SelfAssessment(BaseModel):
    """Questions 15 and 16 -- the only part of the profile that is a claim about
    the person rather than a preference, and therefore the only part that also
    becomes corpus text, verbatim, through the one existing write path
    (`jfl_core.storage.user_corpus`).

    Held here as well as in the corpus on purpose, and the two hold different
    things: the corpus holds the citable span, this holds what the user typed
    and when, so a profile version stays a readable record of what they believed
    about themselves in March.
    """

    depth_genuine: str = ""
    recurring_gaps: str = ""


class Profile(BaseModel):
    """The whole profile. Absent is absent: an empty section means "not stated",
    never a default and never an inference.
    """

    constraints: list[Constraint] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    disciplines: Disciplines = Field(default_factory=Disciplines)
    objectives: list[Objective] = Field(default_factory=list)
    self_assessment: SelfAssessment = Field(default_factory=SelfAssessment)

    def constraint(self, kind: str) -> Constraint | None:
        return next((c for c in self.constraints if c.kind == kind), None)

    def capability(self, key: str) -> Capability | None:
        return next((c for c in self.capabilities if c.key == key), None)

    def objective(self, rank: int) -> Objective | None:
        return next((o for o in self.objectives if o.rank == rank), None)

    def is_empty(self) -> bool:
        return self == Profile()


class ProfileVersion(BaseModel):
    """One saved row. The table is append-only, so a version is never edited --
    which is what makes "what did I believe in March" free to answer.
    """

    id: uuid.UUID
    schema_version: int = SCHEMA_VERSION
    created_at: dt.datetime
    profile: Profile


def seed_capabilities_from_facts(facts: Iterable[CandidateFact]) -> list[Capability]:
    """Capability rows proposed from **confirmed** CV facts.

    PLACEHOLDER for the sibling's seeding function, and knowingly crude: one row
    per confirmed fact, labelled with the fact's opening clause. It is the right
    shape, not the right wording -- what matters here is that a seeded row
    arrives with `tier=None` (the user still answers the behavioural questions),
    with `source="cv_fact"`, and carrying the span its evidence actually is.

    Only confirmed facts are used. A proposed fact is a CV's claim, and seeding
    from it would put a capability in front of the user that they never said was
    true.
    """
    seeded: dict[str, Capability] = {}
    for fact in facts:
        if fact.state != "confirmed":
            continue
        label = _capability_label(fact.corpus_text)
        if not label:
            continue
        key = capability_key(label)
        existing = seeded.get(key)
        if existing is None:
            seeded[key] = Capability(
                label=label,
                source="cv_fact",
                evidence=[fact.span_id] if fact.span_id else [],
            )
        elif fact.span_id and fact.span_id not in existing.evidence:
            existing.evidence.append(fact.span_id)
    return list(seeded.values())


_LABEL_LIMIT = 80


def _capability_label(text: str) -> str:
    """The fact's opening clause, trimmed to something that reads as a row
    label. Cut on a word boundary rather than mid-word: a label is shown to the
    user and is the thing they tier.
    """
    first = text.strip().split(" -- ")[0].split(". ")[0].strip().rstrip(".")
    if len(first) <= _LABEL_LIMIT:
        return first
    cut = first[:_LABEL_LIMIT].rsplit(" ", 1)[0]
    return f"{cut}..."


__all__ = [
    "CAPABILITY_INTERESTS",
    "CAPABILITY_TIERS",
    "CONSTRAINT_KINDS",
    "CONSTRAINT_STANCES",
    "MAX_OBJECTIVES",
    "SCHEMA_VERSION",
    "Capability",
    "CapabilityInterest",
    "CapabilitySource",
    "CapabilityTier",
    "Constraint",
    "ConstraintKind",
    "ConstraintStance",
    "Disciplines",
    "Objective",
    "Profile",
    "ProfileVersion",
    "SelfAssessment",
    "capability_key",
    "comp_value",
    "location_value",
    "seed_capabilities_from_facts",
    "text_value",
]
