"""Pydantic models for generation's structured-output calls.

Field-for-field, these mirror the JSON schemas in `prompts.py`: those JSON
schemas constrain what the model can return over the wire, these models give
callers typed access to the parsed result. Kept in sync by hand, same
discipline as `jfl_gate.schema` -- there is only one of each, so generating
one from the other would be one abstraction for one caller.
"""

from __future__ import annotations

import uuid
from typing import Literal

from jfl_core.models import FitVerdict
from pydantic import BaseModel, Field, field_validator

Necessity = Literal["essential", "desirable", "unstated"]


def _blank_to_none(value: str | None) -> str | None:
    """The model returns "" rather than a nullable JSON type for "the ad doesn't
    say" -- normalised to None here so callers get an ordinary optional field.
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


class ExtractedRequirement(BaseModel):
    text: str
    necessity: Necessity


class ExtractOutput(BaseModel):
    employer: str | None
    title: str | None
    location: str | None
    requirements: list[ExtractedRequirement]

    @field_validator("employer", "title", "location", mode="before")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        return _blank_to_none(value)


CoverageStatus = Literal["evidenced", "partial", "absent", "contradicted"]


class RequirementCoverageResult(BaseModel):
    status: CoverageStatus
    cited_span_ids: list[uuid.UUID]
    evidence_note: str
    # Non-empty only when status is "absent" or "partial" -- see prompts.py.
    question: str | None = None

    @field_validator("question", mode="before")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        return _blank_to_none(value)


class CoverageOutput(BaseModel):
    results: list[RequirementCoverageResult]


class DraftOutput(BaseModel):
    # "" for no title -- see DRAFT_OUTPUT_SCHEMA in prompts.py for why it is separate.
    title: str
    draft: str


class AssessAnswerOutput(BaseModel):
    """The wire shape of `assess_answer`'s response -- never a property named
    `reason`, see CLAUDE.md's 2026-09-02 decision. `gaps` is "" when there is
    nothing worth flagging.
    """

    assessment: str
    gaps: str = ""

    @field_validator("gaps", mode="before")
    @classmethod
    def _normalise(cls, value: str | None) -> str:
        return "" if value is None else value


class DraftAnswerOutput(BaseModel):
    draft: str


class SuggestedTitleItem(BaseModel):
    """The wire shape one suggested title comes back as. Sanitised into
    `jfl_core.models.SuggestedTitle` by `jfl_generate.titles.suggest_titles`
    before anything downstream sees it -- this type exists only to give the
    parsed response typed access, same split as every other model here.
    """

    title: str
    gloss: str


class TitleSuggestionsOutput(BaseModel):
    titles: list[SuggestedTitleItem]


# -- slice B4: two scores for one application --------------------------------
#
# Field-for-field with SCORE_OUTPUT_SCHEMA in prompts.py. There is deliberately
# no composite field and no property named `reason` -- see that schema's comment
# and CLAUDE.md's standing decisions.
#
# And, since 2026-09-21, no `want_it_score` either: the model gives a four-word
# verdict per constraint and per objective, and `jfl_core.fit.want_it_basis`
# derives the number from those. See `docs/profile-schema.md`.


class ConstraintVerdictItem(BaseModel):
    """`index` is 1-based into the numbered constraints in the user message.
    `jfl_generate.scoring` resolves it back to the user's own words, so a
    verdict can never restate what they said their constraint was.
    """

    index: int
    verdict: FitVerdict
    note: str = ""


class ObjectiveVerdictItem(BaseModel):
    rank: int
    verdict: FitVerdict
    note: str = ""


class LeverItem(BaseModel):
    """`claim_index` is 1-based into the numbered unevidenced claims in the
    user message -- a capability the user tiered but never evidenced, or an
    unconfirmed fact from their CV. `jfl_generate.scoring` resolves it back to
    the stored claim's own words, so a lever can never paraphrase it.
    """

    claim_index: int
    would_move_to: int
    note: str


class ScoreOutput(BaseModel):
    # Bounded here rather than in the JSON schema: the wire schema stays to the
    # plain types the API's structured output takes, and an out-of-range number
    # becomes an ordinary parse failure with a `runs` row, not a stored score
    # the CHECK constraint would reject at INSERT.
    could_get_score: int = Field(ge=1, le=10)
    could_get_assessment: str
    want_it_assessment: str
    constraint_verdicts: list[ConstraintVerdictItem]
    objective_verdicts: list[ObjectiveVerdictItem]
    levers: list[LeverItem]


class CvFactItem(BaseModel):
    """One candidate fact as it comes back over the wire. Sanitised into
    `jfl_core.models.ProposedFact` by `jfl_generate.cv_facts.to_proposed_facts`
    before anything downstream sees it -- same split as `SuggestedTitleItem`.
    """

    role_label: str
    source_line: str
    fact_text: str
    # "" from the model means "no question needed" -- see prompts.py.
    probe: str | None = None

    @field_validator("probe", mode="before")
    @classmethod
    def _normalise(cls, value: str | None) -> str | None:
        return _blank_to_none(value)


class CvFactsOutput(BaseModel):
    facts: list[CvFactItem]
