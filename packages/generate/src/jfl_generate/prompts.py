"""Prompt assembly for generation's model calls.

Kept short and close to default model judgement on purpose -- see CLAUDE.md,
"How to develop the model-facing parts": no elaborate scaffolding up front,
tailor from observed behaviour once there is behaviour to observe. Corpus
rendering is reused from `jfl_gate.prompt.format_corpus` rather than
duplicated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from anthropic.types import TextBlockParam
from jfl_core.models import (
    DraftKind,
    Job,
    JobRequirement,
    ProfileAnswer,
    ProfileObjective,
    ProfileRuledOut,
    RequirementCoverage,
    Span,
)
from jfl_core.profile_questions import (
    OBJECTIVE_QUESTIONS,
    QUESTIONS,
    RULED_OUT_QUESTION,
    ProfileQuestion,
)
from jfl_gate.prompt import format_corpus

# Kept in exact correspondence with jfl_generate.schema.ExtractOutput.
EXTRACT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "employer": {"type": "string"},  # "" if the ad does not say
        "title": {"type": "string"},
        "location": {"type": "string"},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "necessity": {
                        "type": "string",
                        "enum": ["essential", "desirable", "unstated"],
                    },
                },
                "required": ["text", "necessity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["employer", "title", "location", "requirements"],
    "additionalProperties": False,
}

_EXTRACT_INSTRUCTIONS = """\
You are extracting structured requirements from a pasted job advertisement for jobs4life, \
a tool that measures how well a candidate's own record evidences a role's requirements. \
Read the job ad in the next message and return:

- employer, title, and location exactly as the ad states them ("" if the ad does not say)
- one entry per requirement, split into atomic, independently checkable items -- "5+ years \
of Python and Kubernetes" becomes two separate requirements, not one. Keep each \
requirement's wording close to the ad's own phrasing rather than rewriting it.
- necessity: "essential" if the ad states or clearly implies the requirement is required, \
"desirable" if it is framed as a plus or nice-to-have, "unstated" if the ad does not \
distinguish

Only extract requirements the ad actually states. Skip boilerplate that carries no \
checkable requirement: benefits, company description, equal-opportunity notices.
"""


def build_extract_prompt() -> str:
    """Constant: the ad text is volatile and belongs in the user message, assembled
    by the caller, not here.
    """
    return _EXTRACT_INSTRUCTIONS


# Kept in exact correspondence with jfl_generate.schema.CoverageOutput.
COVERAGE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["evidenced", "partial", "absent", "contradicted"],
                    },
                    "cited_span_ids": {"type": "array", "items": {"type": "string"}},
                    # Named `evidence_note`, not `reason`. Live-API bisection on
                    # 2026-09-02 found the schema property named exactly `reason`,
                    # combined with a long labelling system prompt, tripped the
                    # API's reverse-engineering/duplication classifier on every
                    # call (stop_reason "refusal", category "reasoning_extraction")
                    # -- dropping or renaming the property alone made the refusal
                    # go away. See jfl_gate.prompt for the same finding. Do not
                    # rename this back to `reason`.
                    "evidence_note": {"type": "string"},
                    "question": {"type": "string"},  # "" for evidenced/contradicted
                },
                "required": ["status", "cited_span_ids", "evidence_note", "question"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}

_COVERAGE_INSTRUCTIONS = """\
You are checking, for jobs4life, what a candidate's corpus can evidence for each \
requirement of a job. This is a report of what the corpus documents, NOT a judgement of \
whether the candidate is good enough -- do not let how strong a candidate "should" look \
bias the status; check only what is actually written below.

For each requirement, return one of these statuses:
  - evidenced: the corpus directly documents this
  - partial: the corpus documents something adjacent or weaker -- related experience, but \
not the specific thing claimed
  - absent: the corpus is silent on this. This is a gap in the record, NOT a statement \
that the candidate lacks the skill -- write the evidence_note as a gap, never as a \
shortcoming
  - contradicted: the corpus states something that rules this requirement out (watch for a \
section of explicit boundaries or things stated as not true, where the corpus has one)

Cite only span IDs that appear in the corpus below; never invent one. cited_span_ids may be \
empty, especially for absent or contradicted.

For any requirement that is absent or partial, write a short question the candidate could \
answer in a sentence or two about their own actual experience. Ask, don't lead -- "Have you \
done X?", not "Describe how you excelled at X." Leave question as "" for evidenced or \
contradicted.

## Corpus

{corpus}
"""


def build_coverage_system_prompt(spans: Sequence[Span]) -> str:
    return _COVERAGE_INSTRUCTIONS.format(corpus=format_corpus(spans))


def build_coverage_system_blocks(
    spans: Sequence[Span], *, cache: Literal["instructions", "corpus"]
) -> list[TextBlockParam]:
    """Two cacheable `system` blocks -- instructions first, corpus second -- instead
    of the one block `build_coverage_system_prompt` returns. See
    `jfl_gate.prompt.build_system_blocks`'s docstring for the full rationale
    (prompt caching is a prefix match; this is the same split for coverage's
    template). `cache="corpus"` is today's product path, byte-identical to the
    single-string prompt; `cache="instructions"` is for the eval harness, where
    many tiny per-item corpora would otherwise rewrite the cache from byte zero
    every call.
    """
    return _split_system_blocks(_COVERAGE_INSTRUCTIONS, format_corpus(spans), cache=cache)


def build_coverage_user_message(requirements: Sequence[str]) -> str:
    """The volatile half of the request -- requirements go in `messages`, never in
    `system`, so a byte change here never invalidates the cached corpus prefix.
    """
    numbered = "\n".join(f"{i + 1}. {requirement}" for i, requirement in enumerate(requirements))
    return (
        f"Check corpus coverage for each of the following {len(requirements)} requirements, "
        f"in order:\n\n{numbered}"
    )


# Kept in exact correspondence with jfl_generate.schema.DraftOutput.
#
# `title` is separate from `draft` so a heading for the whole document never reaches
# the claim gate as a sentence: `jfl_generate.draft` renders it as the document's
# lone markdown h1, which `jfl_gate.gate.split_units` sets aside unchecked. Before
# this, a title naming the target role and employer was checked as a claim to hold
# that role. "" means no title.
DRAFT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "draft": {"type": "string"},
    },
    "required": ["title", "draft"],
    "additionalProperties": False,
}

_KIND_INSTRUCTIONS: dict[DraftKind, str] = {
    "cv_bullets": (
        "Write CV bullet points for this job application: concise, achievement-focused "
        "bullets a candidate could add to their CV, one per line."
    ),
    "cover_letter": (
        "Write a cover letter for this job application: a few short paragraphs in the "
        "candidate's own voice, addressed to the employer."
    ),
}

# Deliberately short -- see CLAUDE.md, "How to develop the model-facing parts": no
# elaborate scaffolding up front, tailor from observed behaviour once there is
# behaviour to observe. This asks the model to genuinely try to be accurate; it does
# not try to prompt drift away (dishonest -- the claim gate exists to measure whatever
# residual drift a genuine attempt still produces) and it does not induce drift either
# (equally dishonest, the opposite direction).
_DRAFT_INSTRUCTIONS = """\
You are drafting application material for jobs4life, a tool that measures the distance \
between what a candidate's corpus documents and what is claimed on their behalf. \
{kind_instructions}

If the document has a title, return it in title rather than as a line of draft; \
otherwise return "" for title.

Select and emphasise what the job's requirements call for. Ground every factual claim in \
the corpus below, and do not assert anything the corpus does not support -- where the \
corpus is silent or only partial on a requirement, either omit the claim or write around \
it rather than inventing evidence to fill the gap.

## Corpus

{corpus}
"""


def build_draft_system_prompt(spans: Sequence[Span], kind: DraftKind) -> str:
    return _DRAFT_INSTRUCTIONS.format(
        kind_instructions=_KIND_INSTRUCTIONS[kind], corpus=format_corpus(spans)
    )


def build_draft_system_blocks(
    spans: Sequence[Span], kind: DraftKind, *, cache: Literal["instructions", "corpus"]
) -> list[TextBlockParam]:
    """Two cacheable `system` blocks -- instructions first, corpus second -- instead
    of the one block `build_draft_system_prompt` returns. See
    `jfl_gate.prompt.build_system_blocks`'s docstring for the full rationale.
    `cache="corpus"` is today's product path, byte-identical to the single-string
    prompt; `cache="instructions"` is for the eval harness's many tiny, mutually
    distinct per-item corpora.
    """
    template = _DRAFT_INSTRUCTIONS.format(
        kind_instructions=_KIND_INSTRUCTIONS[kind], corpus="{corpus}"
    )
    return _split_system_blocks(template, format_corpus(spans), cache=cache)


########################################################################
# Application questions -- two equal paths, "check my answer" and "draft one
# for me". See CLAUDE.md's 2026-09-18 decision and NEXT.md's task 4. The claim
# gate itself (`jfl_gate.gate.check_text`) is reused unchanged for both paths;
# what lives here is the *assessment* call (how well the answer answers the
# question, separate from grounding) and the draft call.
########################################################################


def _format_question_requirements(requirements: Sequence[JobRequirement]) -> str:
    """Numbered, necessity-tagged requirement lines, or a plain statement that
    none are known yet -- an application's ad may not have been extracted, and
    both calls below must still produce something useful rather than fail.
    """
    if not requirements:
        return "(no requirements extracted from this job's ad yet)"
    return "\n".join(f"{i}. [{r.necessity}] {r.text}" for i, r in enumerate(requirements, start=1))


def _format_question_job(job: Job | None) -> str:
    if job is None:
        return "(no job ad linked to this application yet)"
    bits = [job.title or "(unknown title)", "at", job.employer or "(unknown employer)"]
    return " ".join(bits)


# Kept in exact correspondence with jfl_generate.schema.AssessAnswerOutput.
ASSESS_ANSWER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        # Never `reason` -- see CLAUDE.md's 2026-09-02 decision: a schema
        # property named `reason`, combined with a labelling system prompt,
        # has tripped the API's reverse-engineering/duplication classifier
        # before.
        "gaps": {"type": "string"},  # "" if nothing worth flagging
    },
    "required": ["assessment", "gaps"],
    "additionalProperties": False,
}

_ASSESS_ANSWER_INSTRUCTIONS = """\
You are assessing, for jobs4life, how well a candidate's own answer actually answers one \
application question for one role. A separate check already compares the answer's claims \
against the candidate's corpus -- that is grounding, and it is not your job. Judge only \
whether the answer addresses the question, given what this role is asking for.

The current date and time is {now}.

Role: {job}

Requirements this role states:
{requirements}

Question: {question}

Candidate's answer:
{answer}

Return:
  - assessment: a short paragraph on how well the answer addresses the question for this \
role -- what it covers, and how what it covers connects to the role's requirements.
  - gaps: what the answer leaves out, and what a reader would still ask. "" if there is \
nothing worth flagging.
"""


def build_assess_answer_prompt(
    *,
    job: Job | None,
    requirements: Sequence[JobRequirement],
    question_text: str,
    answer_text: str,
    now: datetime,
) -> str:
    """Constant per call -- no corpus, so no system/message cache split (this
    call never touches the candidate's corpus at all; grounding is the claim
    gate's job). `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision:
    every model call is told what time it is), not read here.
    """
    return _ASSESS_ANSWER_INSTRUCTIONS.format(
        now=now.isoformat(),
        job=_format_question_job(job),
        requirements=_format_question_requirements(requirements),
        question=question_text,
        answer=answer_text,
    )


# Kept in exact correspondence with jfl_generate.schema.DraftAnswerOutput. No
# separate `title` field -- unlike `DRAFT_OUTPUT_SCHEMA`, an application
# question's answer is never a whole document with a heading of its own.
DRAFT_ANSWER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"draft": {"type": "string"}},
    "required": ["draft"],
    "additionalProperties": False,
}

# Deliberately short -- see CLAUDE.md, "How to develop the model-facing parts".
_DRAFT_ANSWER_INSTRUCTIONS = """\
You are drafting an answer to one application question for jobs4life, a tool that measures \
the distance between what a candidate's corpus documents and what is claimed on their \
behalf. Write a short, direct answer in the candidate's own voice -- a few sentences to a \
short paragraph, not a full letter.

Ground every factual claim in the corpus below, and do not assert anything the corpus does \
not support -- where the corpus is silent or only partial on something relevant, either \
omit the claim or write around it rather than inventing evidence to fill the gap.

The current date and time is {now}.

Role: {job}

Requirements this role states:
{requirements}

Question: {question}

## Corpus

{corpus}
"""


def build_draft_answer_system_blocks(
    spans: Sequence[Span],
    *,
    job: Job | None,
    requirements: Sequence[JobRequirement],
    question_text: str,
    now: datetime,
    cache: Literal["instructions", "corpus"],
) -> list[TextBlockParam]:
    """Two cacheable `system` blocks -- instructions (with the job and
    question folded in, since both are volatile per call but far smaller than
    the corpus) first, corpus second. See
    `jfl_gate.prompt.build_system_blocks`'s docstring for the full rationale.

    The job, its requirements and the question text go in `system` rather than
    `messages` here (unlike `build_draft_user_message`'s job/requirements,
    which are large enough and change often enough to belong in the volatile
    half) because this call has no other user message to carry them in --
    keeping one block cache-broken per call is simpler than inventing a user
    message whose only content is "draft it now" plus a duplicate of the
    question.
    """
    template = _DRAFT_ANSWER_INSTRUCTIONS.format(
        now=now.isoformat(),
        job=_format_question_job(job),
        requirements=_format_question_requirements(requirements),
        question=question_text,
        corpus="{corpus}",
    )
    return _split_system_blocks(template, format_corpus(spans), cache=cache)


def build_draft_answer_user_message() -> str:
    """The volatile half of the request. Everything that actually varies (the
    job, its requirements, the question) is already in the system blocks above
    -- see `build_draft_answer_system_blocks`'s docstring for why -- so this is
    a fixed instruction, not a template.
    """
    return "Write the answer now."


########################################################################
# Slice C7a: suggested title expansions. A short, cheap, standalone call --
# no corpus, no system/message split to cache. See CLAUDE.md's "How to
# develop the model-facing parts" (near-default judgement, no elaborate
# scaffolding) and PLAN.md's C7a.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.TitleSuggestionsOutput.
TITLE_SUGGESTIONS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    # Never `reason` -- see CLAUDE.md's 2026-09-02 decision and
                    # PLAN.md's C7a: a schema property named `reason`, combined
                    # with a labelling system prompt, has tripped the API's
                    # reverse-engineering/duplication classifier before.
                    "gloss": {"type": "string"},
                },
                "required": ["title", "gloss"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["titles"],
    "additionalProperties": False,
}

_TITLE_SUGGESTION_INSTRUCTIONS = """\
You are suggesting adjacent job titles for jobs4life, a tool that helps someone search \
job boards by title. The user has just added "{phrase}" to the title-matching phrases \
on a saved job filter.

Suggest other job titles that are the same role under a different name, or a close \
abbreviation or expansion of it -- the kind of title variation different employers or \
job boards use for what is really the same role. A few titles for adjacent-but-different \
roles are fine too, where a reasonable person searching on "{phrase}" would plausibly want \
them included as well.

Abbreviations are ambiguous out of context -- "SEM" could mean "senior engineering \
manager" or "search engine marketing" -- so use what is already known about this person \
to judge which reading fits:

- other title phrases already in their filter: {other_includes}
- title phrases they have excluded from their filter: {excludes}
- titles of roles they are currently tracking as applications: {application_titles}

The current date and time is {now}.

Return up to about 10 titles, each with a very short gloss noting anything worth \
flagging -- for instance that it is a step up, a step down, or a different \
specialisation, rather than a plain equivalent. Do not repeat "{phrase}" itself.
"""


def build_title_suggestion_prompt(
    phrase: str,
    *,
    other_includes: Sequence[str],
    excludes: Sequence[str],
    application_titles: Sequence[str],
    now: datetime,
) -> str:
    return _TITLE_SUGGESTION_INSTRUCTIONS.format(
        phrase=phrase,
        other_includes=", ".join(other_includes) or "(none)",
        excludes=", ".join(excludes) or "(none)",
        application_titles=", ".join(application_titles) or "(none)",
        now=now.isoformat(),
    )


def _split_system_blocks(
    template: str, corpus_text: str, *, cache: Literal["instructions", "corpus"]
) -> list[TextBlockParam]:
    """Partition `template` on its `{corpus}` placeholder into an instructions
    block and a corpus block, so concatenating the two blocks' text is always
    byte-identical to `template.format(corpus=corpus_text)` by construction --
    there is no second copy of any template to drift out of sync.
    """
    prefix, _, suffix = template.partition("{corpus}")
    blocks: list[TextBlockParam] = [
        {"type": "text", "text": prefix},
        {"type": "text", "text": corpus_text + suffix},
    ]
    cache_index = 0 if cache == "instructions" else 1
    blocks[cache_index]["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_draft_user_message(
    job: Job, requirements: Sequence[JobRequirement], coverage: Sequence[RequirementCoverage]
) -> str:
    """The volatile half of the request -- goes in `messages`, never in `system`, so
    a byte change here never invalidates the cached corpus prefix. Requirements are
    paired with their latest corpus-coverage verdict (see
    `JobRepository.latest_coverage`) so the model knows, before it writes anything,
    which requirements the corpus can actually back.
    """
    coverage_by_requirement = {c.requirement_id: c for c in coverage}
    lines = [f"Job: {job.title or '(unknown title)'} at {job.employer or '(unknown employer)'}"]
    if job.location:
        lines.append(f"Location: {job.location}")
    lines.append("")
    lines.append("Job ad:")
    lines.append(job.raw_text)
    lines.append("")
    lines.append("Requirements and current corpus coverage:")
    for i, requirement in enumerate(requirements, start=1):
        coverage_row = coverage_by_requirement.get(requirement.id)
        status = coverage_row.status if coverage_row else "unknown"
        evidence_note = coverage_row.evidence_note if coverage_row else "(no coverage recorded)"
        lines.append(f"{i}. [{requirement.necessity}] {requirement.text}")
        lines.append(f"   coverage: {status} -- {evidence_note}")
    lines.append("")
    lines.append("Write the draft now.")
    return "\n".join(lines)


########################################################################
# Slice B4: two scores for one application. One model call, no corpus in
# the prompt -- "could I get this" is judged from the requirements and the
# corpus coverage verdicts already recorded for the job, which is what
# keeps it grounded on confirmed corpus facts only (coverage is computed
# against spans, and spans are the confirmed corpus). See CLAUDE.md's
# 2026-09-18 decision and PLAN.md B4.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.ScoreOutput.
#
# There is deliberately no third number in this schema. CLAUDE.md's standing
# decision: "do I want this" and "could I get this" are reported separately and
# never averaged, so a composite has nowhere to be returned to.
#
# Nothing here is named `reason` -- see CLAUDE.md's 2026-09-02 decision, where a
# labelling prompt plus a schema demanding a label and a `reason` per item
# tripped the API's reverse-engineering classifier on every call. The paragraph
# behind each number is an `assessment`.
SCORE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "could_get_score": {"type": "integer"},
        "could_get_assessment": {"type": "string"},
        "want_it_score": {"type": "integer"},
        "want_it_assessment": {"type": "string"},
        "objective_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ordinal": {"type": "integer"},
                    "verdict": {"type": "string"},
                },
                "required": ["ordinal", "verdict"],
                "additionalProperties": False,
            },
        },
        "hard_gate_breaches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gate": {"type": "string"},
                    "breach": {"type": "string"},
                },
                "required": ["gate", "breach"],
                "additionalProperties": False,
            },
        },
        "levers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # The index of an unconfirmed claim in the user message's
                    # numbered list -- never the claim's text. The stored fact's
                    # own words are copied back by `jfl_generate.scoring`, so a
                    # lever can never quietly paraphrase what the user's CV said.
                    "fact_index": {"type": "integer"},
                    "would_move_to": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["fact_index", "would_move_to", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "could_get_score",
        "could_get_assessment",
        "want_it_score",
        "want_it_assessment",
        "objective_verdicts",
        "hard_gate_breaches",
        "levers",
    ],
    "additionalProperties": False,
}

NOT_STATED = "not stated"

_SCORE_INSTRUCTIONS = """\
You are scoring one job for jobs4life, a tool that measures the distance between what a \
person's own record actually evidences and what is being claimed, and shows the number \
rather than flattering anyone.

Score the job on two SEPARATE axes, each an integer from 1 to 10. Never combine, average \
or reconcile them, and never return a third number. A role this person would love and \
will not get, and one they would dislike and would walk into, must land on different \
numbers on different axes: the disagreement between the two is the useful signal, and \
merging them destroys it.

**could_get_score (1-10)** -- how far this person's recorded evidence covers what the job \
asks for. Judge it ONLY from the requirements and the corpus coverage verdicts given in \
the next message. Coverage is a report of what their corpus documents, never a judgement \
of the person: "evidenced" means the corpus documents it, "partial" means something \
adjacent, "absent" means the corpus is SILENT -- a gap in the record, not a shortcoming \
-- and "contradicted" means the corpus rules it out. Weigh essential requirements more \
than desirable ones. Do not credit anything coverage does not evidence, and do not credit \
the unconfirmed CV claims listed in that message: those are handled by `levers` below.

**want_it_score (1-10)** -- how well this job matches what this person has said they \
want. Judge it against their profile answers, in this order: hard gates, discipline, each \
objective separately, trajectory, signals about the employer, tells. Where an answer is \
given as "{not_stated}", say so in your assessment and do NOT guess what they would have \
said. An unanswered question narrows what you can conclude; it never raises or lowers the \
score by assumption.

**could_get_assessment** and **want_it_assessment** are one paragraph each, in plain \
words, explaining that number: what drove it, and what is unknown. Write about the record \
and the job, not about the person's worth.

**hard_gate_breaches** -- one entry for every hard gate this ad breaks: location or \
commute, working arrangement, the lowest package they would accept, contract type, notice \
or start date, right to work or clearance, or anything they said they categorically will \
not do. `gate` names which one; `breach` states in plain words what the ad does about it \
("the ad is on-site in Manchester five days a week; they will travel to an office at most \
one day a week"). State every breach even though the number already reflects it -- a gate \
the ad breaks must never be folded silently into a score. Never invent one from a question \
answered "{not_stated}", and return an empty list if the ad breaks none.

**objective_verdicts** -- one short verdict per objective listed, keyed by its `ordinal`, \
each judged on its own evidence. Never merge two objectives into one verdict. Give a \
verdict for every objective listed and for none that is not.

**levers** -- only about the numbered unconfirmed CV claims in the next message. Those are \
things this person's own CVs claim that they have not confirmed, so they are not evidence \
and did not count towards could_get_score. Where confirming one would raise that score, \
return its `fact_index`, the score it would move to (`would_move_to`, 1-10, higher than \
could_get_score), and a `note` naming which requirement it would cover. Return an empty \
list when none of them would change anything.

Do not soften either number to be encouraging, and do not deflate it to look rigorous. \
The whole product is that these numbers are honest.
"""


def build_score_system_prompt() -> str:
    """Constant: everything volatile -- the job, the coverage verdicts, the
    profile answers, the unconfirmed claims and the current time -- goes in the
    user message, so this block can be cached across every scoring call.
    """
    return _SCORE_INSTRUCTIONS.format(not_stated=NOT_STATED)


def build_score_system_blocks() -> list[TextBlockParam]:
    """One cacheable `system` block. Unlike coverage and drafting there is no
    corpus here to split off -- see the section comment above for why this call
    reads coverage verdicts rather than the corpus itself.
    """
    return [
        {
            "type": "text",
            "text": build_score_system_prompt(),
            "cache_control": {"type": "ephemeral"},
        }
    ]


@dataclass(frozen=True, slots=True)
class ProposedFactView:
    """The three fields scoring needs off a CV-derived candidate fact.

    A structural view rather than an import of the candidate-facts model: that
    module is being built alongside this slice (see
    `jfl_core.storage.candidate_facts`), and nothing here should have to change
    when the real one lands.
    """

    fact_text: str
    role_label: str = ""
    source_line: str = ""


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    """Everything one scoring call is told, assembled by the caller.

    `answers` is `PostgresProfileRepository.get_current_answers()` -- a key
    absent from it means the question was never answered, which is rendered
    "not stated" and never guessed at (PLAN.md B3a). `objectives` are the
    in-use slots, each carried separately. `proposed_facts` are the
    **unconfirmed** CV claims: they are not evidence and appear in the prompt
    only so the model can name which of them would move the first number.
    """

    job: Job
    requirements: Sequence[JobRequirement]
    coverage: Sequence[RequirementCoverage]
    answers: Mapping[str, ProfileAnswer]
    objectives: Sequence[ProfileObjective] = ()
    ruled_out: Sequence[ProfileRuledOut] = ()
    proposed_facts: Sequence[ProposedFactView] = ()
    now: datetime | None = None


def unanswered_questions(answers: Mapping[str, ProfileAnswer]) -> list[ProfileQuestion]:
    """The profile questions this user has not answered, in assessment order.

    An answer whose text is blank and whose structured value is empty counts as
    unanswered: the storage layer keeps such a row to record that a previous
    answer was cleared, and a cleared answer is not an answer.
    """
    unanswered: list[ProfileQuestion] = []
    for question in QUESTIONS:
        answer = answers.get(question.key)
        if answer is None or (not answer.answer_text.strip() and not answer.structured):
            unanswered.append(question)
    return unanswered


def _answer_line(question: ProfileQuestion, answer: ProfileAnswer | None) -> list[str]:
    if answer is None or (not answer.answer_text.strip() and not answer.structured):
        return [f"- {question.wording}", f"  {NOT_STATED}"]
    lines = [f"- {question.wording}", f"  {answer.answer_text.strip() or NOT_STATED}"]
    if answer.structured:
        lines.append(f"  (structured: {json.dumps(answer.structured, sort_keys=True)})")
    return lines


def build_score_user_message(inputs: ScoreInputs) -> str:
    """The volatile half of the request -- goes in `messages`, never in
    `system`, so a byte change here never invalidates the cached instructions.

    Told what time it is, per CLAUDE.md's 2026-09-07 decision: a judgement about
    notice periods, start dates or how long a search has been running is
    guessing without it.
    """
    job = inputs.job
    lines: list[str] = []
    if inputs.now is not None:
        lines.append(f"The current date and time is {inputs.now.isoformat()}.")
        lines.append("")
    lines.append(f"Job: {job.title or '(unknown title)'} at {job.employer or '(unknown employer)'}")
    lines.append(f"Location as the ad states it: {job.location or NOT_STATED}")
    lines.append("")

    lines.append("## The job ad")
    lines.append(job.raw_text)
    lines.append("")

    lines.append("## Requirements, and what this person's corpus can evidence")
    if inputs.requirements:
        coverage_by_requirement = {c.requirement_id: c for c in inputs.coverage}
        for i, requirement in enumerate(inputs.requirements, start=1):
            row = coverage_by_requirement.get(requirement.id)
            status = row.status if row else "not checked"
            note = row.evidence_note if row else "(no coverage recorded)"
            lines.append(f"{i}. [{requirement.necessity}] {requirement.text}")
            lines.append(f"   coverage: {status} -- {note}")
    else:
        lines.append("(none extracted)")
    lines.append("")

    lines.append("## What this person has said they want, in their own words")
    for question in QUESTIONS:
        lines.extend(_answer_line(question, inputs.answers.get(question.key)))
    lines.append("")

    lines.append("## Objectives for this move, each to be judged on its own")
    if inputs.objectives:
        for objective in inputs.objectives:
            lines.append(f"- ordinal {objective.ordinal}")
            lines.append(
                f"  {OBJECTIVE_QUESTIONS.wording_what} "
                f"{objective.objective_text.strip() or NOT_STATED}"
            )
            lines.append(
                f"  {OBJECTIVE_QUESTIONS.wording_evidence} "
                f"{objective.evidence_text.strip() or NOT_STATED}"
            )
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append(f"## {RULED_OUT_QUESTION.wording}")
    live_ruled_out = [r for r in inputs.ruled_out if r.reopened_at is None]
    if live_ruled_out:
        for entry in live_ruled_out:
            lines.append(f"- {entry.decision_text} (recorded {entry.recorded_at.date()})")
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append("## Unconfirmed claims from this person's own CVs -- NOT evidence")
    lines.append(
        "These have not been confirmed, so they did not count towards could_get_score. "
        "Cite one by its number in `levers` if confirming it would raise that score."
    )
    if inputs.proposed_facts:
        for i, fact in enumerate(inputs.proposed_facts, start=1):
            label = f" [{fact.role_label}]" if fact.role_label else ""
            lines.append(f"{i}.{label} {fact.fact_text}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("Score this job on both axes now, separately.")
    return "\n".join(lines)
