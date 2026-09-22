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
import hashlib
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from jfl_core.models import CandidateFact

# Bumped when a stored shape stops being readable by the model below. Written on
# every row so a later reader can tell what it is looking at rather than guess.
SCHEMA_VERSION = 1


# -- closed sets -------------------------------------------------------------
#
# Each `Literal` is what the code may produce; the tuple beside it is what the
# screens offer, and a test asserts they hold the same values. Written out
# rather than derived with `get_args`, which would make that test assert
# nothing. There is no third copy in a CHECK constraint, because this is JSONB
# -- see the module docstring.

# Hired's one good idea: a negative preference gets equal standing with a
# positive one. "never" is a statement, not the absence of a "must".
Stance = Literal["must", "nice", "never"]
STANCES: tuple[Stance, ...] = ("must", "nice", "never")

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

# Our scale in our words, anchored in what someone has actually done rather than
# in a self-rating -- the shape SFIA arrived at, never SFIA's text or its name
# (licensed against commercial use, and IT-only).
CapabilityTier = Literal["production_depth", "working", "oversight_only", "absent"]
CAPABILITY_TIERS: tuple[CapabilityTier, ...] = (
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
INTERESTS: tuple[Interest, ...] = ("want_more", "happy_to", "rather_not", "never_again")

# Where a capability row came from. `cv_fact` rows are proposed from confirmed
# candidate facts and arrive untiered; `clustered` rows are the same facts
# grouped across roles by one cheap model call, which is what makes "FX pricing
# platforms" a row rather than one row per employer; `user` rows the user added
# themselves -- and a row they renamed keeps the source it arrived with, because
# where it came from is a fact about its history, not about its wording.
CapabilitySource = Literal["cv_fact", "clustered", "user"]
CAPABILITY_SOURCES: tuple[CapabilitySource, ...] = ("cv_fact", "clustered", "user")

MAX_OBJECTIVES = 4


# -- the sections ------------------------------------------------------------


class _Section(BaseModel):
    """Unknown fields are rejected, not ignored.

    Pydantic's default is to drop what it does not recognise, which in a store
    with no CHECK constraint means a typo'd key is accepted, silently discarded
    on the next save, and reported to nobody.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def capability_key(label: str) -> str:
    """A stable, URL-safe id for a capability row, derived from its label.

    There is no id column to key a row by -- the whole profile is one JSONB
    document -- and the screens need something to name a row in a form action
    and an anchor. Insertion order will not do: a row's position changes when
    an earlier one is removed, and a form posted against a stale page would
    then tier the wrong capability.

    So the key is content-derived, the same rule span ids follow. Whitespace is
    collapsed and case folded first, so "FX pricing" and "fx  pricing" are one
    row and never two; the digest is what makes it safe in a path, which a raw
    label is not -- "CI/CD" would otherwise split the route.
    """
    folded = " ".join(label.split()).casefold()
    return hashlib.sha256(folded.encode("utf-8")).hexdigest()[:16]


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


# The three `value` shapes, built here rather than spelled out wherever a form
# is parsed. `value` is untyped by design (see `Constraint`), which makes one
# builder per shape the only place the shape is actually written down -- and a
# reader of a stored constraint has one module to look in.


def location_value(places: Sequence[str]) -> dict[str, Any]:
    """An **ordered** list, best first -- SEEK's shape. Deliberately not a
    `relocate` boolean: "London, then Bristol, then remote anywhere" is a
    ranking, and a boolean cannot hold it.
    """
    return {"places": list(places)}


def comp_value(guaranteed: int | None, headline: int | None, ccy: str = "GBP") -> dict[str, Any]:
    """Guaranteed and headline **separately**, because a headline number is not
    an offer: it is base plus a bonus that may not pay and equity that may not
    vest. Neither figure is derived from the other, and a figure not given is
    absent rather than zero.
    """
    value: dict[str, Any] = {"ccy": ccy}
    if guaranteed is not None:
        value["guaranteed"] = guaranteed
    if headline is not None:
        value["headline"] = headline
    return value


def text_value(text: str) -> dict[str, Any]:
    """The kinds that carry a sentence rather than a structure. Empty text
    produces no value at all, never `{"text": ""}` -- a blank that reads back
    as an answer is indistinguishable from one.
    """
    return {"text": text} if text else {}


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

    @property
    def key(self) -> str:
        """This row's stable id -- see `capability_key`. Not stored: it is a
        function of the label, so it cannot drift out of line with it.
        """
        return capability_key(self.label)

    @property
    def has_evidence(self) -> bool:
        """Whether the corpus can back this row. A tier with no evidence is a
        **claim**, the same status a CV line has before confirmation, and the
        two are told apart everywhere they are read: scoring treats an
        unevidenced capability as a lever, never as evidence.
        """
        return bool(self.evidence)


class Disciplines(_Section):
    """What the user practises, as distinct from what employers called them --
    plus an explicit "not this" list, without which the same job title at two
    employers reads as the same job. `practises` is ranked, best first.

    The JSON key is `not`, which is a Python keyword, hence the aliases. They
    are split rather than given as one `alias=`: a bare `alias` renames the
    constructor argument too, so every caller would have to write
    `**{"not": [...]}` and a type checker could not see the field at all.
    `AliasChoices` keeps `not_practised=` working in Python while `"not"` is
    what is read from and written to `profiles.data`.
    """

    practises: list[str] = Field(default_factory=list)
    not_practised: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("not", "not_practised"),
        serialization_alias="not",
    )


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

    def constraint(self, kind: str) -> Constraint | None:
        """This user's constraint of that kind, or None if they have not stated
        one. Lists are short (eight kinds at most), so the scan is honest and a
        lookup dict would only be a second place for the list to be wrong.

        None is a real answer here and the screens render it as "not stated" --
        never as a blank field that reads like an empty one.
        """
        return next((c for c in self.constraints if c.kind == kind), None)

    def objective(self, rank: int) -> Objective | None:
        """The objective at that rank, or None for an empty slot. Rank is the
        slot the user typed it into, so clearing rank 1 leaves rank 2 where it
        was rather than shuffling it up underneath them.
        """
        return next((o for o in self.objectives if o.rank == rank), None)

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
        answer = str(getattr(assessment, field_name, "") or "").strip()
        lines[heading] = [answer] if answer else []
    return lines


# -- seeding capabilities from confirmed CV facts ----------------------------


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

    **A fact already cited by a saved capability produces no seed.** A role is
    not a capability -- "FX pricing platforms" spans several of them -- so the
    grouping worth showing comes from `jfl_generate.capabilities`, and these
    per-role rows are the free fallback for facts nothing on the profile
    accounts for yet. Once a fact's span is somebody's evidence, proposing it
    again under its employer's name would put the same material on the page
    twice.

    Pure: takes facts, returns rows, touches nothing. The repository-shaped
    caller is `jfl_core.storage.profile.propose_capabilities_from_facts`.
    """
    taken = {capability_key(c.label) for c in existing}
    evidenced = {span_id for capability in existing for span_id in capability.evidence}
    order: list[str] = []
    by_role: dict[str, list[CandidateFact]] = {}
    for fact in facts:
        if fact.state != "confirmed" or fact.span_id is None:
            continue
        if not fact.role_label.strip():
            continue
        if capability_key(fact.role_label) in taken:
            continue
        if fact.span_id in evidenced:
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


# -- choosing what one clustering call is given ------------------------------

# How many confirmed facts go into one grouping call. A user with thirty-three
# CVs can confirm several hundred facts, and the whole lot in one call is a
# long prompt, a long output, and a model asked to hold more in its head than
# it can group well.
#
# Chosen as a **cap with the remainder left for the next run**, not truncation
# and not batching by role. Truncation would silently drop somebody's material,
# which is the failure this project exists to measure. Batching by role would
# defeat the point of the call, since a capability is exactly the thing that
# spans roles. A cap leaves the rest visible, named on the screen, and picked
# up by the next run -- which is progressive because accepting a proposal makes
# its facts evidenced, and an evidenced fact is not sent again.
MAX_FACTS_PER_CLUSTER_CALL = 120


def facts_to_cluster(
    facts: Sequence[CandidateFact],
    *,
    existing: Sequence[Capability] = (),
    limit: int = MAX_FACTS_PER_CLUSTER_CALL,
) -> tuple[list[CandidateFact], list[CandidateFact]]:
    """(what one call is given, what did not fit) -- both, never a silent drop.

    Only `confirmed` facts carrying a `span_id` are eligible, for the reason
    `propose_capabilities` gives: an unconfirmed fact is a CV's claim, not the
    user's, and grouping one would put a capability on the profile that nothing
    in the corpus evidences.

    A fact whose span is already cited by a saved capability is skipped
    entirely -- it is accounted for, and re-proposing it would ask the user the
    same question twice. That is also what makes running this again cover new
    ground rather than repeat itself.

    Order is the caller's, which for
    `PostgresCandidateFactRepository.list_facts` is role then CV order -- so
    the overflow is the tail of the CV rather than an arbitrary slice.
    """
    evidenced = {span_id for capability in existing for span_id in capability.evidence}
    eligible = [
        fact
        for fact in facts
        if fact.state == "confirmed" and fact.span_id is not None and fact.span_id not in evidenced
    ]
    return eligible[:limit], eligible[limit:]


__all__ = [
    "CAPABILITY_SOURCES",
    "MAX_FACTS_PER_CLUSTER_CALL",
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
    "capability_key",
    "comp_value",
    "facts_to_cluster",
    "location_value",
    "propose_capabilities",
    "self_assessment_corpus_lines",
    "text_value",
]
