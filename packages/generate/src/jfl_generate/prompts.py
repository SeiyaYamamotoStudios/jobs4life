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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from anthropic.types import TextBlockParam
from jfl_core.models import (
    FIT_VERDICTS,
    DraftKind,
    Job,
    JobRequirement,
    NotStated,
    RequirementCoverage,
    Span,
)
from jfl_core.profile import Capability, Constraint, Profile
from jfl_core.pushback import DIRECTIONS, PUSHBACK_KINDS
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


########################################################################
# Capability clustering. The same shape as the title call above: short,
# cheap, standalone, no corpus and no system/message split to cache. The
# facts are volatile and go in the user message; only the instructions are
# constant.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.CapabilityClusterOutput.
CAPABILITY_CLUSTER_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    # The short ids from the user message ("f1", "f2"), never a
                    # uuid: a uuid is expensive to emit and easy to corrupt a
                    # character of, and a corrupted uuid is indistinguishable
                    # from a real one. A corrupted "f420" is not, and
                    # `jfl_generate.capabilities` drops anything it did not
                    # send.
                    "fact_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "fact_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["capabilities"],
    "additionalProperties": False,
}

# No property named `reason`, and no free-text note per item either -- a long
# labelling prompt plus a schema demanding a label and a note per item is the
# exact shape that tripped the API's reverse-engineering classifier on
# 2026-09-02. A label and the ids it covers is all this call needs.
_CAPABILITY_CLUSTER_INSTRUCTIONS = """\
You are grouping one person's confirmed career facts into capabilities for jobs4life, \
a tool that measures how well someone's own record evidences a role's requirements.

A capability is something this person can do. It usually spans several facts and \
several employers -- "FX pricing platforms", "hiring engineering managers", "incident \
command". It is not a job title, not an employer, and not one fact restated.

Each fact in the next message is numbered with an id like f1, and shows the role it was \
recorded under. Return a list of capabilities, each with:

- label: the capability in this person's own vocabulary, taken from the words they \
actually used. A few words at most. Never a job title, a seniority, or an employer's name.
- fact_ids: the ids of the facts that capability covers.

Rules:

- Use only ids that appear in the next message. Do not invent one.
- A fact belongs to at most one capability. Where two would fit, choose the better one.
- Group across roles: the same capability practised at two employers is one entry, not two.
- Do not merge two genuinely different things to make the list shorter.
- A fact that fits no capability worth naming is left out. Leaving it out is better than \
inventing a label for it -- the person is shown what you did not place, and nothing is lost.
- Return at most {max_capabilities} capabilities.

The current date and time is {now}.
"""


def build_capability_cluster_prompt(*, max_capabilities: int, now: datetime) -> str:
    """Constant but for the ceiling and the clock: the facts are volatile and
    belong in the user message, assembled by `jfl_generate.capabilities`.
    """
    return _CAPABILITY_CLUSTER_INSTRUCTIONS.format(
        max_capabilities=max_capabilities, now=now.isoformat()
    )


########################################################################
# Profile suggestions from uploaded CVs. Same shape as the two calls
# above: short, cheap, standalone, no corpus and no system/message split
# to cache. The CVs are volatile and go in the user message; only the
# instructions are constant.
#
# A CV states claims about the world -- those become candidate facts and
# are confirmed one at a time. It also states plain *settings*, and this
# is the call that reads those off.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.ProfileSuggestionsOutput.
PROFILE_SUGGESTIONS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # An open string rather than a JSON enum: an unknown kind is
                    # dropped by `jfl_generate.profile_suggestions.to_proposals`
                    # against a whitelist, which is a guarantee in code rather
                    # than a constraint the wire format might or might not hold.
                    "kind": {"type": "string"},
                    "value": {"type": "string"},
                    # The CV's own words. Never `reason` and never a rationale
                    # -- see CLAUDE.md's 2026-09-02 decision. This is a quote,
                    # and a quote that is not in the CV gets the whole
                    # suggestion dropped.
                    "source_line": {"type": "string"},
                },
                "required": ["kind", "value", "source_line"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}

_PROFILE_SUGGESTIONS_INSTRUCTIONS = """\
You are reading one person's own CVs for jobs4life, a tool that measures how well \
someone's record evidences a role's requirements. They uploaded these CVs themselves. \
Read off the plain settings the CVs state, so they can confirm each one with a click \
instead of typing it out again.

Return a list of suggestions. Each carries a kind, a value, and the line or phrase from \
the CV it came from -- copied out exactly, so the person can see what made you say it.

The four kinds, and nothing else:

- discipline: something this person practises, in the CV's own vocabulary -- \
"engineering management", "platform engineering". What they actually do, as distinct \
from what an employer called them. One suggestion per discipline.
- not_discipline: a discipline the CV says in words they do not practise, or have \
stopped practising. Only where it is written down. Never because something is missing \
from the CV: silence is not a statement.
- location: somewhere this person has actually worked, named in the CV. One suggestion \
per place, most recent first.
- level: the seniority the CVs describe, written as an observation about what they have \
been doing -- "has been operating at engineering-manager level". At most one, and never \
phrased as a requirement, a floor or a minimum. What they will accept next is their \
choice, not something a CV states.

Rules:

- Use only those four kinds. Anything else is discarded unread.
- Do not suggest pay, contract type, right to work, notice period, a remote or hybrid \
preference, or anything they would refuse. A CV records what someone has done; it does \
not state what they now require, and a guess would put a requirement on their profile \
that they never made.
- source_line must be copied from the CV exactly as written. If you cannot point at a \
line, leave the suggestion out.
- Do not repeat a suggestion. Prefer the person's own wording over a tidier phrase.
- Return at most {max_suggestions} suggestions.

The current date and time is {now}.
"""


def build_profile_suggestions_prompt(*, max_suggestions: int, now: datetime) -> str:
    """Constant but for the ceiling and the clock: the CVs are volatile and
    belong in the user message, assembled by `jfl_generate.profile_suggestions`.
    """
    return _PROFILE_SUGGESTIONS_INSTRUCTIONS.format(
        max_suggestions=max_suggestions, now=now.isoformat()
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


# The ceiling on what a draft may claim -- `docs/profile-schema.md`, "How it
# constrains generation": say `working` and no generated CV says "deep
# expertise". This is the thesis applied to our own output, and it is cheaper
# than catching the same over-claim at the gate. It is **not** enforcement --
# the claim gate remains the backstop, and nothing inspects a finished draft to
# see whether the instruction was obeyed.
#
# It goes in the volatile user message rather than the instructions for the
# ordinary reason: a user's capability list in `system` would invalidate the
# cached corpus prefix on every call.
_DRAFT_CEILING_HEADING = (
    "The depth this person confirmed, which is the ceiling on what you may claim"
)

_DRAFT_CEILING_RULE = (
    "Do not write a claim above the depth listed here. Working level is not deep "
    "expertise; oversight only is not hands-on. A capability that is not listed has "
    "no confirmed depth, so do not characterise its depth at all."
)


def _draft_capability_ceiling(capabilities: Sequence[Capability]) -> list[str]:
    """A capability the user never tiered is left out entirely, not listed as
    "not stated": the rule below says an unlisted capability has no confirmed
    depth, which is exactly what an untiered row is. Listing it with a blank
    depth would invite the model to pick one.
    """
    lines = [f"## {_DRAFT_CEILING_HEADING}"]
    listed = [c for c in capabilities if c.tier is not None and c.tier != "absent"]
    if listed:
        for capability in listed:
            lines.append(f"- {capability.label}: {tier_wording(capability.tier)}")
    else:
        lines.append("(none confirmed)")
    for capability in capabilities:
        if capability.tier == "absent":
            lines.append(
                f"- {capability.label}: this person says they do NOT have this. Never claim it."
            )
    lines.append(_DRAFT_CEILING_RULE)
    lines.append("")
    return lines


def build_draft_user_message(
    job: Job,
    requirements: Sequence[JobRequirement],
    coverage: Sequence[RequirementCoverage],
    capabilities: Sequence[Capability] = (),
) -> str:
    """The volatile half of the request -- goes in `messages`, never in `system`, so
    a byte change here never invalidates the cached corpus prefix. Requirements are
    paired with their latest corpus-coverage verdict (see
    `JobRepository.latest_coverage`) so the model knows, before it writes anything,
    which requirements the corpus can actually back.
    """
    lines = _draft_job_lines(job, requirements, coverage)
    lines.extend(_draft_capability_ceiling(capabilities))
    lines.append("Write the draft now.")
    return "\n".join(lines)


def _draft_job_lines(
    job: Job,
    requirements: Sequence[JobRequirement],
    coverage: Sequence[RequirementCoverage],
) -> list[str]:
    """The job, its ad, and each requirement with its latest corpus coverage --
    shared by the bullets-only draft and the complete CV, so both calls read
    the job the same way.
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
    return lines


########################################################################
# The complete CV (`jfl_generate.cv_document`). One call writes the parts of a
# CV that are claims -- summary, skills, a descriptor and bullets per role --
# against a skeleton of roles read deterministically from the corpus
# (`jfl_core.cv_skeleton`). The model is given the roles by number and returns
# them by number: it never writes a title, an employer or a date, so none of
# those can drift.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.CvDocumentOutput. No
# property is named `reason` (CLAUDE.md, 2026-09-02), and no role carries a
# title, employer or date field for the model to fill.
CV_DOCUMENT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "array", "items": {"type": "string"}},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["label", "text"],
                "additionalProperties": False,
            },
        },
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # The role's number in the user message's list -- never its
                    # title or employer.
                    "index": {"type": "integer"},
                    "descriptor": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "descriptor", "bullets"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "skills", "roles"],
    "additionalProperties": False,
}

# Short and near-default, like _DRAFT_INSTRUCTIONS -- CLAUDE.md, "How to develop
# the model-facing parts". The length guidance is the owner's own CVs': two
# pages usually, three at most; about twenty bullets, weighted to recent roles;
# bullets of roughly 25-40 words.
_CV_DOCUMENT_INSTRUCTIONS = """\
You are writing a complete CV for jobs4life, a tool that measures the distance between \
what a candidate's corpus documents and what is claimed on their behalf. The next message \
gives the job and the candidate's roles, numbered, each with the confirmed facts filed \
under it. Each role's title, employer and dates are already fixed; do not write them.

The CV speaks as the candidate, in the first person. The corpus is written about them in the \
third person; the CV is not. The summary uses "I" ("I lead a distributed team…"). Skill \
lines and bullets start with a verb and leave the "I" implied ("Led…", "Built…", \
"Introduced…"). Never refer to the candidate by name, as "he", "she" or "they", or as "the \
candidate".

Return:
- summary: one to three short paragraphs, in the candidate's own voice, on why they fit \
this job
- skills: six to eight entries, each a short label and one line on what they bring there
- roles: for each role you write about, its number, a one-line descriptor of what the \
employer does, and bullets chosen and compressed from that role's facts toward this job's \
requirements

Aim for two pages, three at most: about twenty bullets in total, weighted to the recent \
roles, each roughly 25 to 40 words. An older role may have one bullet or none.

Ground every factual claim in the corpus below, and do not assert anything the corpus does \
not support -- where the corpus is silent or only partial on a requirement, omit the claim \
rather than invent evidence to fill the gap. The descriptor is about the employer, not the \
candidate: write it only from what the corpus says about the employer, or return "" for it.

## Corpus

{corpus}
"""


def build_cv_document_system_blocks(
    spans: Sequence[Span], *, cache: Literal["instructions", "corpus"] = "corpus"
) -> list[TextBlockParam]:
    """Instructions then corpus, the corpus block cached -- the same split as
    `build_draft_system_blocks`. The clock goes in the user message, not here:
    a timestamp in `system` would invalidate the cached corpus on every call.
    """
    return _split_system_blocks(_CV_DOCUMENT_INSTRUCTIONS, format_corpus(spans), cache=cache)


@dataclass(frozen=True)
class CvPromptRole:
    """One skeleton role as the prompt shows it. `jfl_core.cv_skeleton.SkeletonRole`
    carries more (span ids); this is only what the model reads."""

    title: str
    employer: str
    dates: str
    facts: Sequence[str] = ()


def build_cv_document_user_message(
    job: Job,
    requirements: Sequence[JobRequirement],
    coverage: Sequence[RequirementCoverage],
    roles: Sequence[CvPromptRole],
    *,
    boundaries: Sequence[str] = (),
    capabilities: Sequence[Capability] = (),
    now: datetime,
) -> str:
    """The volatile half: the clock, the job, the numbered roles with their
    facts, what the corpus says is NOT true, and the depth ceiling.
    """
    lines = [f"The current date and time is {now.isoformat()}.", ""]
    lines.extend(_draft_job_lines(job, requirements, coverage))
    lines.append("## The candidate's roles, most recent first")
    if not roles:
        lines.append("(no roles recorded)")
    for i, role in enumerate(roles, start=1):
        heading = " -- ".join(part for part in (role.title, role.employer) if part)
        lines.append(f"Role {i}: {heading or '(untitled)'} ({role.dates or 'dates not stated'})")
        if role.facts:
            lines.extend(f"- {fact}" for fact in role.facts)
        else:
            lines.append("- (no confirmed facts filed under this role)")
    lines.append("")
    if boundaries:
        lines.append("## Stated as NOT true -- never claim any of these")
        lines.extend(f"- {line}" for line in boundaries)
        lines.append("")
    lines.extend(_draft_capability_ceiling(capabilities))
    lines.append("Write the CV now.")
    return "\n".join(lines)


########################################################################
# Slice B4, rebuilt on the 2026-09-21 profile: two scores for one
# application. One model call, no corpus in the prompt -- "could I get
# this" is judged from the requirements, the corpus coverage verdicts
# already recorded for the job, and the capabilities the user has
# *evidenced*, which is what keeps it grounded on confirmed corpus facts
# only (coverage is computed against spans, and spans are the confirmed
# corpus).
#
# **The model is not asked for a "do I want this" number.** See
# `docs/profile-schema.md`: computed person-job fit predicts satisfaction
# at rho ~= .28 and people forecast their own job satisfaction badly, so
# the honest product is a list of what the ad evidences and what it is
# silent on. The model returns a four-word verdict per constraint and per
# objective; `jfl_core.fit` derives the number from those, where it can
# never say something the verdicts do not.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.ScoreOutput.
#
# There is deliberately no third number in this schema -- and, since
# 2026-09-21, only one number at all. CLAUDE.md's standing decision: "do I want
# this" and "could I get this" are reported separately and never averaged, so a
# composite has nowhere to be returned to.
#
# Nothing here is named `reason` -- see CLAUDE.md's 2026-09-02 decision, where a
# labelling prompt plus a schema demanding a label and a `reason` per item
# tripped the API's reverse-engineering classifier on every call. The sentences
# behind a number are an `assessment`; the sentence behind a verdict is a `note`.
SCORE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "could_get_score": {"type": "integer"},
        "could_get_assessment": {"type": "string"},
        # No `want_it_score`. It is derived from the verdicts below by
        # `jfl_core.fit.want_it_basis`, so a number and the verdicts under it
        # cannot disagree.
        "want_it_assessment": {"type": "string"},
        "constraint_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # The constraint's number in the user message's list --
                    # never its text, so a verdict cannot quietly restate what
                    # the user said their constraint was.
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(FIT_VERDICTS)},
                    "note": {"type": "string"},
                },
                "required": ["index", "verdict", "note"],
                "additionalProperties": False,
            },
        },
        "objective_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "verdict": {"type": "string", "enum": list(FIT_VERDICTS)},
                    "note": {"type": "string"},
                },
                "required": ["rank", "verdict", "note"],
                "additionalProperties": False,
            },
        },
        "levers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # The index of an unevidenced claim in the user message's
                    # numbered list -- never the claim's text. The stored
                    # claim's own words are copied back by
                    # `jfl_generate.scoring`, so a lever can never quietly
                    # paraphrase what the user's CV or profile said.
                    "claim_index": {"type": "integer"},
                    "would_move_to": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["claim_index", "would_move_to", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "could_get_score",
        "could_get_assessment",
        "want_it_assessment",
        "constraint_verdicts",
        "objective_verdicts",
        "levers",
    ],
    "additionalProperties": False,
}

NOT_STATED = "not stated"

# No `.format()` on this one: it is a constant, which is what makes the cached
# system block byte-identical across every scoring call any user makes.
_SCORE_INSTRUCTIONS = """\
You are scoring one job for jobs4life, a tool that measures the distance between what a \
person's own record actually evidences and what is being claimed, and shows the number \
rather than flattering anyone.

There are two SEPARATE axes. Never combine, average or reconcile them, and never return a \
third number. A role this person would love and will not get, and one they would dislike \
and would walk into, must land on different numbers on different axes: the disagreement \
between the two is the useful signal, and merging them destroys it.

**could_get_score (1-10)** -- how far this person's recorded evidence covers what the job \
asks for. Judge it ONLY from the requirements, the corpus coverage verdicts, and the \
capabilities shown as evidenced in the next message. Coverage is a report of what their \
corpus documents, never a judgement of the person: "evidenced" means the corpus documents \
it, "partial" means something adjacent, "absent" means the corpus is SILENT -- a gap in \
the record, not a shortcoming -- and "contradicted" means the corpus rules it out. Weigh \
essential requirements more than desirable ones.

A requirement met by a capability at production depth with corpus evidence behind it is \
not the same as one met by a working-level claim with nothing behind it. Only an \
evidenced capability counts as evidence. Credit nothing from the "CLAIMED, NOT EVIDENCE" \
section -- those are handled by `levers` below.

**could_get_assessment** -- ONE OR TWO SENTENCES in plain words: what drove that number, \
and what is unknown. Not a paragraph. The per-requirement detail is already on the page, \
and repeating it there wastes the reader's attention. Write about the record and the job, \
never about the person's worth.

**You are not asked for a "do I want this" number.** That number is computed from the \
verdicts you give below, so that it can never say something your own verdicts do not. \
Report instead, for every constraint and every objective listed, what THE AD evidences, \
using exactly one of these four words:

- `evidenced` -- the ad states something that satisfies it.
- `partial` -- the ad points that way without stating it.
- `silent` -- the ad does not say. This is the most useful verdict you can give and it is \
never a failure: a silence is a question to ask at interview. Never fill one in from what \
employers usually do, from the sector, or from the job title.
- `contradicted` -- the ad states something that breaks it.

The verdict is always about the person's constraint, never about the thing itself. For a \
constraint they hold as `never`, `evidenced` means the ad rules that thing out and \
`contradicted` means the ad does it.

**constraint_verdicts** -- one entry per numbered constraint, keyed by its `index`, for \
every one of them including the ones you mark `silent`. `note` is one short sentence \
naming or quoting what in the ad decided it, or saying plainly that the ad does not \
mention it. Never invent a breach out of a silence.

**objective_verdicts** -- one entry per numbered objective, keyed by its `rank`, each \
judged on its own evidence. Never merge two objectives into one verdict, and never give a \
verdict for an objective that is not listed.

**want_it_assessment** -- ONE OR TWO SENTENCES on what the ad evidences of what this \
person said matters, and what it is silent on. State no number: you are not given one.

**levers** -- only about the numbered claims in the "CLAIMED, NOT EVIDENCE" section. Those \
are things this person's own CVs or profile claim that no confirmed evidence stands \
behind, so they are not evidence and did not count towards could_get_score. Where \
confirming one would raise that score, return its `claim_index`, the score it would move \
to (`would_move_to`, 1-10, higher than could_get_score), and a `note` naming which \
requirement it would cover. Return an empty list when none of them would change anything.

Do not soften either judgement to be encouraging, and do not deflate it to look rigorous. \
The whole product is that these are honest.
"""


def build_score_system_prompt() -> str:
    """Constant: everything volatile -- the job, the coverage verdicts, the
    profile, the unevidenced claims and the current time -- goes in the user
    message, so this block can be cached across every scoring call.
    """
    return _SCORE_INSTRUCTIONS


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

    `profile` is `PostgresProfileRepository.current()`. An empty section means
    the user has not filled it in, which is rendered "not stated" and never
    guessed at. `proposed_facts` are the **unconfirmed** CV claims: like a
    capability the user tiered but never evidenced, they are not evidence and
    appear in the prompt only so the model can name which of them would move
    the first number.
    """

    job: Job
    requirements: Sequence[JobRequirement]
    coverage: Sequence[RequirementCoverage]
    profile: Profile = field(default_factory=Profile)
    proposed_facts: Sequence[ProposedFactView] = ()
    now: datetime | None = None


# How each tier reads on the page and in the prompt. Our scale in our words --
# see `jfl_core.profile.CapabilityTier`.
TIER_WORDING: Mapping[str, str] = {
    "production_depth": "production depth",
    "working": "working",
    "oversight_only": "oversight only",
    "absent": "absent",
}

# `tier` is None until the user answers the behavioural questions, and that is
# a real state rather than a missing one -- a row proposed from a CV arrives
# untiered and must never be described as having a depth nobody chose.
NO_TIER = "not stated"


def tier_wording(tier: str | None) -> str:
    return NO_TIER if tier is None else TIER_WORDING.get(tier, tier)


_KIND_WORDING: Mapping[str, str] = {
    "location": "location",
    "workplace": "working arrangement",
    "level_floor": "level",
    "comp_floor": "lowest package",
    "contract": "contract type",
    "right_to_work": "right to work",
    "notice": "notice or start date",
    "categorical_no": "categorically will not do",
}

# Which profile sections a score reports as "not stated" when they are empty.
# Not a guess and not a default: an empty section narrows what the score can
# conclude and never moves it by assumption.
_SECTION_WORDING: Mapping[str, str] = {
    "constraints": "What you must have, would like, and will not accept",
    "capabilities": "What you can do, and at what depth",
    "disciplines": "What you practise, and what you do not",
    "objectives": "What this move is for",
}


def constraint_label(constraint: Constraint) -> str:
    """One constraint in the user's own words, as both the prompt and the
    stored verdict label it. The value is rendered as the JSON the user's own
    form wrote -- a comp floor carries guaranteed and headline separately, and
    flattening it to one number here would assert something they did not.
    """
    parts = [_KIND_WORDING.get(constraint.kind, constraint.kind.replace("_", " "))]
    if constraint.value:
        parts.append(json.dumps(constraint.value, sort_keys=True))
    note = constraint.note.strip()
    if note:
        parts.append(note)
    return " -- ".join(parts)


@dataclass(frozen=True, slots=True)
class ClaimedItem:
    """One thing that is claimed and not evidenced, whatever claimed it.

    A capability the user tiered but never backed with a corpus span, and an
    unconfirmed fact from their own CV, are the same status -- a claim -- and
    the prompt numbers them in one list so a lever can point at either. The
    order is fixed here, and `jfl_generate.scoring` resolves an index back
    against the same list, so the model never gets to supply the text.
    """

    text: str
    role_label: str = ""
    claim_kind: str = "cv_fact"
    tier: str = ""


def claimed_items(inputs: ScoreInputs) -> list[ClaimedItem]:
    """Capabilities with no evidence behind them, then unconfirmed CV facts.

    A capability tiered `absent` is not a claim -- the user is saying they do
    not have it -- so it is never a lever.
    """
    items = [
        ClaimedItem(
            text=capability.label,
            claim_kind="capability",
            tier=tier_wording(capability.tier),
        )
        for capability in inputs.profile.capabilities
        if capability.tier != "absent" and not capability.has_evidence
    ]
    items += [
        ClaimedItem(text=fact.fact_text, role_label=fact.role_label)
        for fact in inputs.proposed_facts
    ]
    return items


def not_stated_sections(profile: Profile) -> list[NotStated]:
    """The profile sections this user has not filled in, in assessment order."""
    filled = {
        "constraints": bool(profile.constraints),
        "capabilities": bool(profile.capabilities),
        "disciplines": bool(profile.disciplines.practises or profile.disciplines.not_practised),
        "objectives": bool(profile.objectives),
    }
    return [
        NotStated(question_key=key, wording=wording)
        for key, wording in _SECTION_WORDING.items()
        if not filled[key]
    ]


def _capability_line(capability: Capability) -> str:
    bits = [f"[{tier_wording(capability.tier)}]", capability.label]
    if capability.last_used is not None:
        bits.append(f"(last used {capability.last_used})")
    return " ".join(bits)


def build_score_user_message(inputs: ScoreInputs) -> str:
    """The volatile half of the request -- goes in `messages`, never in
    `system`, so a byte change here never invalidates the cached instructions.

    Told what time it is, per CLAUDE.md's 2026-09-07 decision: a judgement about
    notice periods, start dates or how long a search has been running is
    guessing without it.
    """
    job = inputs.job
    profile = inputs.profile
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

    lines.append("## Capabilities with corpus evidence behind them -- these ARE evidence")
    evidenced = [c for c in profile.capabilities if c.has_evidence and c.tier != "absent"]
    if evidenced:
        for capability in evidenced:
            lines.append(f"- {_capability_line(capability)}")
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append("## What this person says they do NOT have")
    absent = [c.label for c in profile.capabilities if c.tier == "absent"]
    absent += list(profile.disciplines.not_practised)
    if absent:
        for label in absent:
            lines.append(f"- {label}")
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append("## What this person practises, in their own words")
    if profile.disciplines.practises:
        for discipline in profile.disciplines.practises:
            lines.append(f"- {discipline}")
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append("## Constraints -- give a verdict for every one of these, by index")
    if profile.constraints:
        for i, constraint in enumerate(profile.constraints, start=1):
            lines.append(f"{i}. [{constraint.stance}] {constraint_label(constraint)}")
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append("## Objectives -- give a verdict for every one of these, by rank")
    if profile.objectives:
        for objective in profile.objectives:
            lines.append(f"- rank {objective.rank}")
            lines.append(f"  what this move is for: {objective.text.strip() or NOT_STATED}")
            lines.append(
                "  what would show a role delivers it: "
                f"{objective.evidence_of_delivery.strip() or NOT_STATED}"
            )
    else:
        lines.append(NOT_STATED)
    lines.append("")

    lines.append("## CLAIMED, NOT EVIDENCE")
    lines.append(
        "Their own CVs and profile claim these and nothing confirmed stands behind them, "
        "so they did not count towards could_get_score. Cite one by its number in `levers` "
        "if confirming it would raise that score."
    )
    claimed = claimed_items(inputs)
    if claimed:
        for i, item in enumerate(claimed, start=1):
            if item.claim_kind == "capability":
                lines.append(f"{i}. [claimed at {item.tier}, no evidence] {item.text}")
            else:
                label = f" [{item.role_label}]" if item.role_label else ""
                lines.append(f"{i}.{label} {item.text}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("Score this job now: one number for could_get_score, and a verdict each.")
    return "\n".join(lines)


# -- CV intake (slice B6) ----------------------------------------------------
#
# Kept in exact correspondence with jfl_generate.schema.CvFactsOutput.
CV_FACTS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role_label": {"type": "string"},
                    "source_line": {"type": "string"},
                    "fact_text": {"type": "string"},
                    # A question to put to the person, not a justification of
                    # the label above it -- and never named `reason`. See
                    # CLAUDE.md's 2026-09-02 decision: a long labelling prompt
                    # plus a schema demanding a label and a reason per item
                    # reads to the API as a distillation harvest, and every call
                    # comes back refused.
                    "probe": {"type": "string"},  # "" when no question is needed
                },
                "required": ["role_label", "source_line", "fact_text", "probe"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

_CV_FACTS_INSTRUCTIONS = """\
You are reading someone's CV for jobs4life, a tool that measures the distance between what \
a person's record can evidence and what they claim. Treat this CV as a set of claims the \
person once made about themselves, NOT as evidence: nothing in it counts until they \
confirm it. Your job is to turn it into small, separate facts they can confirm, edit or \
reject one at a time.

The current date and time is {now}. CV dates are often relative ("2021-present", "last \
year"), so use it to read them rather than guessing how much time has passed.

Return one entry per fact. For each one:

- role_label: the role the fact belongs to, written the way the CV writes it -- employer, \
title and dates, for instance "Acme Ltd -- Engineering Manager, 2021-2024". Use the same \
wording for every fact from the same role. For a fact belonging to no particular role \
(education, a certification, a personal project) use the CV's own heading for that section.
- source_line: the CV's own words that the fact came from, copied exactly. Nothing added, \
reworded, corrected or tidied -- this is quoted back to the person, and a line they did not \
write is worse than no line at all.
- fact_text: one short statement of the fact in plain words. One fact per entry: "led a team \
of eight and owned the payments platform" is two entries, not one. Never make the claim \
stronger than the line does, and never add a number, a scope or an outcome the line does \
not state.
- probe: a single short question to put to the person, or "" for no question. Write a probe \
whenever the fact carries a number of any kind, a team or budget size, or the words led, \
owned, drove or delivered -- those are the shapes where a CV most often says more than the \
person would say out loud. Ask what actually happened ("How many people reported to you?", \
"What did owning it involve -- decisions, budget, on-call?"). Never ask a leading question \
that invites a better answer than the truth.

Skip anything that is not a checkable fact about this person: contact details, the summary \
paragraph's self-description ("a passionate engineer"), referees, and bare lists of \
technologies with no claim attached.
"""


def build_cv_facts_prompt(*, now: datetime) -> str:
    """Constant apart from the clock: the CV is volatile and belongs in the user
    message, assembled by the caller.
    """
    return _CV_FACTS_INSTRUCTIONS.format(now=now.isoformat())


########################################################################
# Pushback classification. The same shape as the title call above: short,
# cheap, standalone, no corpus -- this call classifies one disagreement with a
# score, it does not weigh it. See jfl_generate.pushback.classify_pushback and
# jfl_core.pushback's module docstring for the loop this feeds.
########################################################################

# Kept in exact correspondence with jfl_generate.schema.PushbackClassificationOutput.
PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(PUSHBACK_KINDS)},
        # Which way the person thinks the number should go. The one-box form
        # asks for their words and nothing else, so this is read from them.
        "direction": {"type": "string", "enum": list(DIRECTIONS)},
        "new_information": {"type": "boolean"},
        # Never `reason` -- see CLAUDE.md's 2026-09-02 decision and
        # PLAN.md's C7a: a schema property named `reason`, combined with a
        # labelling system prompt, has tripped the API's
        # reverse-engineering/duplication classifier before. This is a note
        # *to the user* about what kind of statement this is, never the
        # model's reasoning about how it decided.
        "classification_note": {"type": "string"},
    },
    "required": ["kind", "direction", "new_information", "classification_note"],
    "additionalProperties": False,
}

# No free-text "reason" field and no instruction to justify the pick at length --
# see the schema comment above. The note field is described as being for the
# user, not for showing working.
_PUSHBACK_CLASSIFICATION_INSTRUCTIONS = """\
You are reading one person's disagreement with a job score jobs4life showed them, for a \
tool whose whole claim is measuring the distance between what someone can evidence and what \
they assert -- applied here to a score instead of a CV bullet. They typed their disagreement \
into a single box under two numbers, so read their words and decide which of three kinds it \
is and which way they think the score should go. The three have sharply different \
consequences, so pick the kind that is actually true of the sentence rather than the one \
that sounds most agreeable.

- "preference" -- a statement about what the person WANTS. Folded into the "do I want this" \
number, by a small and shrinking amount.
- "capability" -- a statement about what the person CAN DO or HAS DONE. If they are saying \
the tool rated them too high, the "could I get this" number moves down. If they are saying \
the tool rated them too low, the number moves NOTHING -- a claim of greater capability needs \
evidence, not agreement, so this opens a question instead of taking their word for it.
- "factual" -- a statement about the JOB AD itself, not about the person -- disputing what \
it says rather than what they want or can do. Nothing about the person moves.

Decide `direction`: "up" if they think the relevant number is too low, "down" if they think \
it is too high. For a factual objection, the direction the corrected reading of the ad would \
move the score.

The two numbers they were shown:
- "could I get this": {could_get_score} out of 10 -- "{could_get_explanation}"
- "do I want this": {want_score} out of 10 -- "{want_explanation}"

Their words: "{user_text}"

{earlier_section}

Decide `new_information`: whether this pushback states a fact the earlier ones did not \
already state. Restating the same point more forcefully, or adding emphasis with no new \
fact, is NOT new information -- it is the same claim said again, and saying it again should \
not count as saying more.

Do not rewrite, tidy or improve the person's words anywhere in your answer. \
`classification_note` is a short note to them about what kind of statement you read this as, \
never a paraphrase of what they said and never your reasoning about how you decided.

The current date and time is {now}.
"""


def build_pushback_classification_prompt(
    *,
    user_text: str,
    could_get_score: int | None,
    could_get_explanation: str,
    want_score: int | None,
    want_explanation: str,
    earlier_texts: Sequence[str],
    now: datetime,
) -> str:
    if earlier_texts:
        bullets = "\n".join(f'- "{text}"' for text in earlier_texts)
        earlier_section = f"Earlier things they have said about their scores:\n{bullets}"
    else:
        earlier_section = "They have said nothing else about their scores before."
    return _PUSHBACK_CLASSIFICATION_INSTRUCTIONS.format(
        could_get_score=could_get_score if could_get_score is not None else "unscored",
        could_get_explanation=could_get_explanation or "(none)",
        want_score=want_score if want_score is not None else "unscored",
        want_explanation=want_explanation or "(none)",
        user_text=user_text,
        earlier_section=earlier_section,
        now=now.isoformat(),
    )
