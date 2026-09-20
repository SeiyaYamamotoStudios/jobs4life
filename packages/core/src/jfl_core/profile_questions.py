"""Profile setup question definitions -- PLAN.md slice B3a.

Scoring needs to know what the user wants and what they will not accept; this
module is the single source of truth for the questions that capture it. **Keys
never change once shipped** -- they are what a stored `profile_answers.question_key`
means, forever, so renaming one here silently orphans every answer already given
under the old name.

Scope: questions 1-17. Question 18 is a CV upload to the sent-document store and
is not modelled here -- it is the CV onboarding flow, not a profile answer.

**Questions 15 and 16 are not preferences.** Every other question records what
the user wants; those two record what is true about them -- where their depth is
genuine and where it is exposure only, and the gaps that keep coming up. They
are therefore the only answers on the page that also become corpus text, stored
verbatim through the same path a confirmed CV fact takes -- one write path, not
two (`jfl_core.corpus_source`, reached through
`jfl_core.storage.user_corpus`). `CORPUS_QUESTION_KEYS` below is the closed set,
so "which answers reach the corpus" is one named constant rather than a
condition repeated at each call site.

Every question is optional; a skipped one is simply absent, never defaulted or
inferred (see `jfl_core.storage.profile`). Four of them -- 3, 4, 5, 9 -- also
offer an OPTIONAL structured value alongside the free text, never instead of
it, because a deterministic gate will eventually want a machine-readable value
where the PLAN table implies one (levels, comp floor, contract types,
disciplines). Questions 10/11 (objectives) and 17 (ruled-out decisions) are not
simple keyed text answers and have their own tables -- see `jfl_core.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# -- sections: how the profile page groups questions, in assessment order -----
# (PLAN.md B3a: hard gates -> discipline -> coverage (built) -> objectives ->
# trajectory -> signals about the place -> tells). Ruled-out decisions are their
# own section, last, since they are not part of that assessment order.
ProfileSection = Literal[
    "hard_gates",
    "discipline",
    "objectives",
    "trajectory",
    "place",
    "tells",
    "depth_and_gaps",
    "ruled_out",
]

SECTION_TITLES: dict[ProfileSection, str] = {
    "hard_gates": "Hard gates",
    "discipline": "Discipline",
    "objectives": "Objectives",
    "trajectory": "Trajectory",
    "place": "Signals about the place",
    "tells": "Tells",
    "depth_and_gaps": "Depth and gaps",
    "ruled_out": "Ruled out",
}

# Rendered once, on the page, next to the sections above -- not a section itself.
SECTION_ORDER: tuple[ProfileSection, ...] = (
    "hard_gates",
    "discipline",
    "objectives",
    "trajectory",
    "place",
    "tells",
    "depth_and_gaps",
    "ruled_out",
)

StructuredKind = Literal["levels", "comp_floor", "contract_types", "disciplines"]


@dataclass(frozen=True, slots=True)
class ProfileQuestion:
    """One question from PLAN.md's B3a table.

    `key` is the stable identifier stored in `profile_answers.question_key` --
    never the question `number`, which is only this table's row order and is
    free to be renumbered if the plan changes. `stage` is PLAN's "Consumed by"
    column, kept for documentation and for showing the user why a question is
    being asked. `structured` names which optional structured value (if any)
    this question also offers; see `jfl_core.storage.profile`.
    """

    key: str
    number: int
    wording: str
    stage: str
    section: ProfileSection
    structured: StructuredKind | None = None
    # True for the two questions whose answers are claims about the person
    # rather than preferences, and therefore also become corpus text. See the
    # module docstring and `CORPUS_QUESTION_KEYS`.
    to_corpus: bool = False


QUESTIONS: tuple[ProfileQuestion, ...] = (
    ProfileQuestion(
        key="location_commute",
        number=1,
        wording="Where are you based, and how far / how often will you travel to an office?",
        stage="gate: location, commute",
        section="hard_gates",
    ),
    ProfileQuestion(
        key="workplace_arrangements",
        number=2,
        wording="Which working arrangements will you consider?",
        stage="gate: workplace presets",
        section="hard_gates",
        # Deliberately no structured value here -- see the module docstring in
        # jfl_web.routes.profile for why this question reuses /jobs's saved
        # workplace preset (job_filters.workplace_mode) instead of duplicating it.
    ),
    ProfileQuestion(
        key="levels",
        number=3,
        wording="Which levels — IC, EM, above EM?",
        stage="gate: level",
        section="hard_gates",
        structured="levels",
    ),
    ProfileQuestion(
        key="comp_floor",
        number=4,
        wording=(
            "Lowest total package you would accept, and what your current one is made of "
            "(base, bonus, equity, pension, car)?"
        ),
        stage="gate: comp floor; objective: comp",
        section="hard_gates",
        structured="comp_floor",
    ),
    ProfileQuestion(
        key="contract_types",
        number=5,
        wording="Contract types — permanent, contract (inside/outside IR35), fixed-term?",
        stage="gate: contract",
        section="hard_gates",
        structured="contract_types",
    ),
    ProfileQuestion(
        key="notice_period",
        number=6,
        wording="Notice period, and earliest or preferred start date?",
        stage="gate, timing",
        section="hard_gates",
    ),
    ProfileQuestion(
        key="right_to_work",
        number=7,
        wording="Right to work, sponsorship, security clearance held?",
        stage="gate",
        section="hard_gates",
    ),
    ProfileQuestion(
        key="categorical_no",
        number=8,
        wording="Anything you categorically will not do, or requirements you know you do not meet?",
        stage="gate: categorical",
        section="hard_gates",
    ),
    ProfileQuestion(
        key="disciplines",
        number=9,
        wording="Disciplines targeted, and not (tickboxes plus free text)",
        stage="discipline match",
        section="discipline",
        structured="disciplines",
    ),
    ProfileQuestion(
        key="trajectory",
        number=12,
        wording="Where do you want to be in two years, and how long do you expect to stay?",
        stage="trajectory",
        section="trajectory",
    ),
    ProfileQuestion(
        key="employer_deal_breakers",
        number=13,
        wording=("Employer deal-breakers: ownership (PE), funding stage, recent layoffs, sectors"),
        stage="signals about the place",
        section="place",
    ),
    ProfileQuestion(
        key="warning_signs",
        number=14,
        wording="Warning signs in job ads you have learned to distrust",
        stage="tells",
        section="tells",
    ),
    ProfileQuestion(
        key="depth_genuine",
        number=15,
        wording="Where is your depth genuine, and where is it exposure only?",
        stage="corpus, verbatim",
        section="depth_and_gaps",
        to_corpus=True,
    ),
    ProfileQuestion(
        key="recurring_gaps",
        number=16,
        wording="Gaps that keep coming up in roles you want",
        stage="corpus, verbatim; coverage, education planning",
        section="depth_and_gaps",
        to_corpus=True,
    ),
)

QUESTIONS_BY_KEY: dict[str, ProfileQuestion] = {q.key: q for q in QUESTIONS}

# The closed set of `profile_answers.question_key` values -- mirrored in
# jfl_core.models.ProfileQuestionKey (Literal) and in the migration's CHECK
# constraint. See packages/core/tests/test_value_lists_agree.py.
QUESTION_KEYS: tuple[str, ...] = tuple(q.key for q in QUESTIONS)

# The answers that are claims about the person, not preferences, and therefore
# also become corpus text. Derived from the questions themselves so there is one
# place to say it -- a caller asking "does this answer reach the corpus?" never
# gets to answer from a second list.
CORPUS_QUESTION_KEYS: tuple[str, ...] = tuple(q.key for q in QUESTIONS if q.to_corpus)

# The corpus section each of those answers is filed under. A section holds
# exactly one live span per question -- re-answering replaces it and retires
# what it replaced, so a superseded statement about the user can never keep
# grounding a claim. See `PostgresUserCorpusRepository.replace_section`.
CORPUS_SECTIONS: dict[str, str] = {
    "depth_genuine": "Depth and exposure",
    "recurring_gaps": "Recurring gaps",
}


def questions_in_section(section: ProfileSection) -> tuple[ProfileQuestion, ...]:
    return tuple(q for q in QUESTIONS if q.section == section)


# -- structured value choices --------------------------------------------------
# Each is a tuple of (stored value, display label). Stored values are the closed
# sets a later deterministic gate can match on; display labels are what the page
# shows.

LEVEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("ic", "Individual contributor"),
    ("em", "Engineering manager"),
    ("above_em", "Above EM"),
)

CONTRACT_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    ("permanent", "Permanent"),
    ("contract_inside_ir35", "Contract — inside IR35"),
    ("contract_outside_ir35", "Contract — outside IR35"),
    ("fixed_term", "Fixed-term"),
)

# The owner's own disciplines, observed in use (PLAN.md B3a). Editable per user:
# a user's free-form addition is stored alongside these in the same structured
# list, never in a second place -- see jfl_web.routes.profile.
DEFAULT_DISCIPLINE_CHOICES: tuple[tuple[str, str], ...] = (
    ("platform_infra_engineering", "Platform / infra engineering"),
    ("engineering_management", "Engineering management"),
    ("ml_modelling", "ML modelling"),
    ("data_engineering", "Data engineering"),
    ("consulting_practice_leadership", "Consulting practice leadership"),
    ("frontend", "Frontend"),
    ("embedded", "Embedded"),
    ("programme_management", "Programme management"),
)

DEFAULT_COMP_CURRENCY = "GBP"

# A short, closed list -- enough for a UK-based owner's likely applications
# without pretending to be exhaustive. Not a CHECK constraint: it is a select
# option list on the page, and `structured` is free-form JSON, so an unlisted
# currency a user types is stored, just not offered as a tick.
COMMON_CURRENCIES: tuple[str, ...] = ("GBP", "USD", "EUR", "CHF")

MAX_OBJECTIVES = 4


@dataclass(frozen=True, slots=True)
class ObjectiveQuestions:
    """Questions 10 and 11 -- not simple keyed answers (see the module
    docstring): up to four objectives, each a pair of free-text fields. Kept
    here, not in `QUESTIONS`, so `QUESTION_KEYS` stays exactly the set of
    `profile_answers.question_key` values.
    """

    number_what: int = 10
    wording_what: str = "What is this move for?"
    number_evidence: int = 11
    wording_evidence: str = "What would show a role delivers it?"
    stage: str = "objectives"


OBJECTIVE_QUESTIONS = ObjectiveQuestions()


@dataclass(frozen=True, slots=True)
class RuledOutQuestion:
    """Question 17 -- also not a simple keyed answer: a dated, kept list, not a
    single latest value. See `jfl_core.models.ProfileRuledOut`.
    """

    number: int = 17
    wording: str = (
        "Anything ruled out that you do not want reopened — dated, flagged if it reappears"
    )
    stage: str = "ruled-out decisions"


RULED_OUT_QUESTION = RuledOutQuestion()

# Question 18 is answered by uploading CVs, not by typing into this page, so it
# is named here only so the profile page can point at the upload rather than
# leave a numbered gap. Not a `ProfileQuestion`: it produces no
# `profile_answers` row at all.
DEFERRED_QUESTIONS: tuple[tuple[int, str], ...] = ((18, "Your current CV"),)

__all__ = [
    "COMMON_CURRENCIES",
    "CONTRACT_TYPE_CHOICES",
    "CORPUS_QUESTION_KEYS",
    "CORPUS_SECTIONS",
    "DEFAULT_COMP_CURRENCY",
    "DEFAULT_DISCIPLINE_CHOICES",
    "DEFERRED_QUESTIONS",
    "LEVEL_CHOICES",
    "MAX_OBJECTIVES",
    "OBJECTIVE_QUESTIONS",
    "QUESTIONS",
    "QUESTIONS_BY_KEY",
    "QUESTION_KEYS",
    "RULED_OUT_QUESTION",
    "ObjectiveQuestions",
    "ProfileQuestion",
    "ProfileSection",
    "RuledOutQuestion",
    "SECTION_ORDER",
    "SECTION_TITLES",
    "StructuredKind",
    "questions_in_section",
]
