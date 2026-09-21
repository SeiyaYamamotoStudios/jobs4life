"""The profile: one row, five sections -- `docs/profile-schema.md`, 2026-09-21.

Replaces the eighteen free-text questions of PLAN.md B3a (`profile_answers`,
`profile_objectives`, `profile_ruled_out`). One denormalised JSONB row per save,
append-only, latest wins: one read serves scoring, drafting and the filter.

**This module is the only write path into `profiles.data`.** JSONB carries no
CHECK constraint, so every guarantee the rest of this schema gets from the
database has to be got here instead -- which is why the models below forbid
unknown fields and why every closed set is a `Literal` exported beside a tuple
of the same values. The design names the cost explicitly: this is weaker than a
CHECK, it was accepted deliberately (owner, 2026-09-21), and it is the price of
a shape we expect to change while we learn what belongs in it. The two tests
that hold it up are `packages/core/tests/test_profile_model.py` (the Literals
and the round trip) and the value-list test that pairs each `Literal` with the
tuple the screens offer, so the web layer cannot invent a fifth tier.

## Five sections, not the design's four

`constraints`, `capabilities`, `disciplines` and `objectives` are the design's.
`self_assessment` is added here, and it is not a preference like the other four:
it holds profile questions 15 and 16 -- where your depth is genuine, and the
gaps that keep coming up -- which are claims *about the person*. They were the
only two of the eighteen questions that also became corpus text, and they keep
doing exactly that, through the one write path
(`jfl_core.storage.user_corpus` over `jfl_core.corpus_source`). The text lives
in the profile so the screen has somewhere to read it back from; the *fact*
lives in the corpus as a `provenance='document'` span, because that is what the
claim gate can cite. `CORPUS_SECTIONS` below is what ties the two together, and
its headings are unchanged from `profile_questions.CORPUS_SECTIONS` on purpose:
a span's id derives from its section and its content, so renaming a heading
would retire every statement already recorded under it and mint a new one.

## What is deliberately not here

No model call touches anything in this module, and nothing here is inferred. A
section the user has not filled in is empty and reports "not stated"; it is
never defaulted and never guessed at.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jfl_core.models import CandidateFact

# Bumped when a stored shape stops being readable by the model below. Written on
# every row so a later reader can tell what it is looking at rather than guess.
SCHEMA_VERSION = 1


# -- closed sets -------------------------------------------------------------
#
# Each `Literal` is what the code may produce; the tuple beside it is what the
# screens offer, and a test asserts they hold the same values. There is no third
# copy in a CHECK constraint, because this is JSONB -- see the module docstring.

# Hired's one good idea: a negative preference gets equal standing with a
# positive one. "never" is a statement, not the absence of a "must".
Stance = Literal["must", "nice", "never"]
STANCES: tuple[str, ...] = ("must", "nice", "never")

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
CONSTRAINT_KINDS: tuple[str, ...] = (
    "location",
    "workplace",
    "level_floor",
    "comp_floor",
    "contract",
    "right_to_work",
    "notice",
    "categorical_no",
)

# Our scale in our words, anchored in what someone has actually done rather than
# in a self-rating -- the shape SFIA arrived at, never SFIA's text or its name
# (licensed against commercial use, and IT-only).
CapabilityTier = Literal["production_depth", "working", "oversight_only", "absent"]
CAPABILITY_TIERS: tuple[str, ...] = (
    "production_depth",
    "working",
    "oversight_only",
    "absent",
)

# A separate axis from tier, as O*NET rates level and importance separately:
# what you are good at and what you want to keep doing are different questions.
# The design names `want_more`; the other three are ours, and mirror `Stance`'s
# rule that "I do not want to do this again" is a preference worth stating
# rather than a low score on a positive one.
Interest = Literal["want_more", "happy_to", "rather_not", "never_again"]
INTERESTS: tuple[str, ...] = ("want_more", "happy_to", "rather_not", "never_again")

# Where a capability row came from. `cv_fact` rows are proposed from confirmed
# candidate facts and arrive untiered; `user` rows the user added themselves.
CapabilitySource = Literal["cv_fact", "user"]
CAPABILITY_SOURCES: tuple[str, ...] = ("cv_fact", "user")

MAX_OBJECTIVES = 4


# -- the sections ------------------------------------------------------------


class _Section(BaseModel):
    """Unknown fields are rejected, not ignored.

    Pydantic's default is to drop what it does not recognise, which in a store
    with no CHECK constraint means a typo'd key is accepted, silently discarded
    on the next save, and reported to nobody.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Constraint(_Section):
    """One hard or soft requirement, in the user's words plus an optional
    machine-readable value.

    `value` is kind-specific and deliberately untyped: locations are an ordered
    list (SEEK's shape -- a ranking, not a relocate boolean), comp carries
    `guaranteed` and `headline` separately because a headline number is not an
    offer, and several kinds carry nothing at all. A gate that reads `value`
    reads a documented shape per kind; a Pydantic model here would have to
    unify eight of them.

    `note` is the user's own words and is never rewritten.
    """

    kind: ConstraintKind
    stance: Stance
    value: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class Capability(_Section):
    """Something the user can do, at a stated depth, with the evidence for it.

    **A tier with no evidence is a claim, not a fact** -- the same status a CV
    line has before confirmation. `evidence` holds corpus span ids, which is
    what makes a capability citable rather than merely asserted; `tier` is None
    until the user sets it, so a row proposed from a CV never arrives carrying a
    depth nobody chose.
    """

    label: str
    tier: CapabilityTier | None = None
    interest: Interest | None = None
    # Year, not a date: "when did you last do this" is answered to the year, and
    # a month would be false precision.
    last_used: int | None = None
    evidence: list[uuid.UUID] = Field(default_factory=list)
    source: CapabilitySource = "user"

    @field_validator("label")
    @classmethod
    def _label_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a capability needs a label")
        return value


class Disciplines(_Section):
    """What the user practises, as distinct from what employers called them --
    plus an explicit "not this" list, without which the same job title at two
    employers reads as the same job. `practises` is ranked, best first.

    The JSON key is `not`, which is a Python keyword, hence the alias.
    """

    practises: list[str] = Field(default_factory=list)
    not_practised: list[str] = Field(default_factory=list, alias="not")


class Objective(_Section):
    """One thing this move is for, ranked, with what would show a role delivers
    it. Scored separately and never blended: Theory of Work Adjustment split
    satisfaction from satisfactoriness in 1969 and blending them hides the
    trade-off that is the whole reason to look.
    """

    rank: Annotated[int, Field(ge=1, le=MAX_OBJECTIVES)]
    text: str = ""
    evidence_of_delivery: str = ""


class SelfAssessment(_Section):
    """Profile questions 15 and 16 -- the two answers that are claims about the
    person rather than preferences, and therefore also become corpus text.

    Stored here verbatim so the screen can show the user what they wrote; stored
    in the corpus as spans so the claim gate can cite them. One write path for
    the corpus half (`CORPUS_SECTIONS`, `jfl_core.storage.profile.save_profile`),
    never two -- two mechanisms for one fact is how one sentence ends up with
    two span ids that the gate reads as two pieces of evidence.
    """

    depth_genuine: str = ""
    recurring_gaps: str = ""


class Profile(_Section):
    """One user's profile, whole. Every section is optional and an empty one
    reports "not stated" -- never a default, never an inference.
    """

    constraints: list[Constraint] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    disciplines: Disciplines = Field(default_factory=Disciplines)
    objectives: list[Objective] = Field(default_factory=list)
    self_assessment: SelfAssessment = Field(default_factory=SelfAssessment)

    @model_validator(mode="after")
    def _objective_ranks_are_distinct(self) -> Profile:
        """Up to four objectives, each at its own rank. Two objectives sharing
        a rank would make "objective 2's verdict" ambiguous in a scoring result
        keyed by rank, which is exactly the merge the design forbids.
        """
        if len(self.objectives) > MAX_OBJECTIVES:
            raise ValueError(f"at most {MAX_OBJECTIVES} objectives")
        ranks = [o.rank for o in self.objectives]
        if len(set(ranks)) != len(ranks):
            raise ValueError("two objectives share a rank")
        return self

    def as_json(self) -> dict[str, Any]:
        """What goes into `profiles.data`: JSON-safe, aliased keys, nothing
        dropped. The repository is the only caller; it is here so that no other
        one has to remember `by_alias=True`.
        """
        return self.model_dump(mode="json", by_alias=True)

    @property
    def is_empty(self) -> bool:
        """True for a user who has filled nothing in. `current()` returns this
        rather than None, so a caller never has to ask whether a profile exists
        before reading it.
        """
        return self == Profile()


class ProfileVersion(BaseModel):
    """One saved row. History is append-only, so a version is immutable and
    has no `updated_at`.
    """

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    created_at: dt.datetime
    data: Profile


# -- the corpus half of the self-assessment ----------------------------------

# Which `self_assessment` field lands under which corpus `## ` heading. The
# headings are unchanged from the retired `profile_questions.CORPUS_SECTIONS`
# because a span id derives from its section path and its content: renaming one
# would retire every statement already recorded under it and mint a new span for
# the same sentence.
CORPUS_SECTIONS: dict[str, str] = {
    "depth_genuine": "Depth and exposure",
    "recurring_gaps": "Recurring gaps",
}


def self_assessment_corpus_lines(profile: Profile) -> dict[str, list[str]]:
    """Corpus section heading -> the lines that section should now hold.

    A section holds exactly one live statement, so re-answering replaces rather
    than adds and a cleared box empties the section -- an answer the user
    deleted must not go on being cited at them. Blank text yields an empty list
    for that heading rather than dropping it, because "clear this section" and
    "leave this section alone" have to be distinguishable to the caller.

    Pure, so the mapping is testable without a database. The write itself is
    `jfl_core.storage.profile.save_profile`.
    """
    assessment = profile.self_assessment
    lines: dict[str, list[str]] = {}
    for field_name, heading in CORPUS_SECTIONS.items():
        text_value = str(getattr(assessment, field_name, "") or "").strip()
        lines[heading] = [text_value] if text_value else []
    return lines


# -- seeding capabilities from confirmed CV facts ----------------------------


def _label_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def propose_capabilities(
    facts: Sequence[CandidateFact],
    *,
    existing: Sequence[Capability] = (),
) -> list[Capability]:
    """Capability rows proposed from confirmed candidate facts, grouped by role.

    Only `confirmed` facts carrying a `span_id` are used. An unconfirmed fact is
    a CV's claim, not the user's -- letting one seed a capability would put a
    row on the profile that nothing in the corpus evidences, which is the
    precise failure the 2026-09-18 decision exists to prevent.

    One row per role, labelled with the role and carrying every confirmed span
    under it as `evidence`, `source="cv_fact"` and `tier=None`: the user sets
    the depth, and until they do the row is untiered rather than guessed at.
    Roles the profile already has a capability for (by label, whitespace and
    case folded) are left alone, so re-running this never overwrites a tier the
    user chose.

    Pure: takes facts, returns rows, touches nothing. The repository-shaped
    caller is `jfl_core.storage.profile.propose_capabilities_from_facts`.
    """
    taken = {_label_key(c.label) for c in existing}
    order: list[str] = []
    by_role: dict[str, list[CandidateFact]] = {}
    for fact in facts:
        if fact.state != "confirmed" or fact.span_id is None:
            continue
        if not fact.role_label.strip():
            continue
        if _label_key(fact.role_label) in taken:
            continue
        if fact.role_key not in by_role:
            by_role[fact.role_key] = []
            order.append(fact.role_key)
        by_role[fact.role_key].append(fact)

    proposed: list[Capability] = []
    for role_key in order:
        group = by_role[role_key]
        # The earliest fact's spelling wins, exactly as `RoleGroup` picks a
        # label: one spelling has to, and the stable choice beats the clever one.
        label = group[0].role_label.strip()
        evidence: list[uuid.UUID] = []
        for fact in group:
            if fact.span_id is not None and fact.span_id not in evidence:
                evidence.append(fact.span_id)
        proposed.append(Capability(label=label, tier=None, evidence=evidence, source="cv_fact"))
    return proposed


__all__ = [
    "CAPABILITY_SOURCES",
    "CAPABILITY_TIERS",
    "CONSTRAINT_KINDS",
    "CORPUS_SECTIONS",
    "INTERESTS",
    "MAX_OBJECTIVES",
    "SCHEMA_VERSION",
    "STANCES",
    "Capability",
    "CapabilitySource",
    "CapabilityTier",
    "Constraint",
    "ConstraintKind",
    "Disciplines",
    "Interest",
    "Objective",
    "Profile",
    "ProfileVersion",
    "SelfAssessment",
    "Stance",
    "propose_capabilities",
    "self_assessment_corpus_lines",
]
