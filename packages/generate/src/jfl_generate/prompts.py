"""Prompt assembly for generation's two model calls.

Kept short and close to default model judgement on purpose -- see CLAUDE.md,
"How to develop the model-facing parts": no elaborate scaffolding up front,
tailor from observed behaviour once there is behaviour to observe. Corpus
rendering is reused from `jfl_gate.prompt.format_corpus` rather than
duplicated.
"""

from __future__ import annotations

from collections.abc import Sequence

from jfl_core.models import Span
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


def build_coverage_user_message(requirements: Sequence[str]) -> str:
    """The volatile half of the request -- requirements go in `messages`, never in
    `system`, so a byte change here never invalidates the cached corpus prefix.
    """
    numbered = "\n".join(f"{i + 1}. {requirement}" for i, requirement in enumerate(requirements))
    return (
        f"Check corpus coverage for each of the following {len(requirements)} requirements, "
        f"in order:\n\n{numbered}"
    )
