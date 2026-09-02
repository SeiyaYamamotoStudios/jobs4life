"""Prompt assembly for generation's model calls.

Kept short and close to default model judgement on purpose -- see CLAUDE.md,
"How to develop the model-facing parts": no elaborate scaffolding up front,
tailor from observed behaviour once there is behaviour to observe. Corpus
rendering is reused from `jfl_gate.prompt.format_corpus` rather than
duplicated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from anthropic.types import TextBlockParam
from jfl_core.models import DraftKind, Job, JobRequirement, RequirementCoverage, Span
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
You are extracting structured requirements from a pasted job advertisement for job4life, \
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
                    "reason": {"type": "string"},
                    "question": {"type": "string"},  # "" for evidenced/contradicted
                },
                "required": ["status", "cited_span_ids", "reason", "question"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}

_COVERAGE_INSTRUCTIONS = """\
You are checking, for job4life, what a candidate's corpus can evidence for each \
requirement of a job. This is a report of what the corpus documents, NOT a judgement of \
whether the candidate is good enough -- do not let how strong a candidate "should" look \
bias the status; check only what is actually written below.

For each requirement, return one of these statuses:
  - evidenced: the corpus directly documents this
  - partial: the corpus documents something adjacent or weaker -- related experience, but \
not the specific thing claimed
  - absent: the corpus is silent on this. This is a gap in the record, NOT a statement \
that the candidate lacks the skill -- write the reason as a gap, never as a shortcoming
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
DRAFT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "draft": {"type": "string"},
    },
    "required": ["draft"],
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
You are drafting application material for job4life, a tool that measures the distance \
between what a candidate's corpus documents and what is claimed on their behalf. \
{kind_instructions}

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
        reason = coverage_row.reason if coverage_row else "(no coverage recorded)"
        lines.append(f"{i}. [{requirement.necessity}] {requirement.text}")
        lines.append(f"   coverage: {status} -- {reason}")
    lines.append("")
    lines.append("Write the draft now.")
    return "\n".join(lines)
