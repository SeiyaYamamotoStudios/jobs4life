"""Prompt assembly for the baseline gate.

Everything here is placeholder quality except the instruction text itself, which is
where the actual care went (see CLAUDE.md: "do not over-flag framing" is the single
rule that decides whether this tool survives contact with a real user). Nothing in
this module is volatile -- instructions and corpus both belong in the cached system
block. The sentences under test go in the user message, assembled by the caller, so
this module never needs to know what varies per call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from anthropic.types import TextBlockParam
from jfl_core.models import Span

# Kept in exact correspondence with jfl_gate.schema.GateOutput. See that module's
# docstring for why these are hand-kept in sync rather than generated from each other.
#
# Two deliberate exceptions, both populated after parsing rather than asked of the
# model: `SentenceResult.rule_flags`, filled in by the deterministic rule tier
# (jfl_gate.rules.apply_rules), and `SentenceResult.text`, filled in by
# jfl_gate.gate.check_text from the input sentence list once `_check_alignment` has
# confirmed `index` below lines up with it. Asking the model to echo the sentence
# text back (it already has the numbered list in the user message) cost output
# tokens for no informational gain -- `index` is what replaced it.
GATE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "kind": {"type": "string", "enum": ["claim", "framing"]},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "review", "unsupported"],
                    },
                    "drift_label": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "invented_quantity",
                            "adjacency_substitution",
                            "scope_inflation",
                            "ownership_inflation",
                            "outcome_attribution",
                            "strategy_scope",
                            "causality",
                            "framing",
                        ],
                    },
                    "cited_span_ids": {"type": "array", "items": {"type": "string"}},
                    # Named `evidence_note`, not `reason`. Live-API bisection on
                    # 2026-09-02 found the schema property named exactly `reason`,
                    # combined with this module's labelling prompt, tripped the
                    # API's reverse-engineering/duplication classifier on every
                    # call (stop_reason "refusal", category "reasoning_extraction")
                    # -- dropping or renaming the property alone made the refusal
                    # go away. Do not rename this back to `reason`.
                    "evidence_note": {"type": "string"},
                },
                "required": [
                    "index",
                    "kind",
                    "verdict",
                    "drift_label",
                    "cited_span_ids",
                    "evidence_note",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sentences"],
    "additionalProperties": False,
}

_INSTRUCTIONS = """\
You are the grounding gate for job4life, a tool whose entire value is measuring the \
distance between what a candidate's corpus documents and what generated text claims on \
their behalf. You will be given sentences from a piece of AI-generated job-application \
text. For each one, decide whether it is a factual claim that must trace to the corpus \
below, or narrative framing that carries no factual content to check, then return a \
verdict.

## The single most important rule: do not over-flag framing

Framing is connective tissue: why someone made a choice, what they were weighing, the \
sequence events happened in, what motivated a move. It is ungroundable BY NATURE -- no \
corpus entry could ever confirm or deny "because the team was struggling to keep up" or \
"having enjoyed the technical challenge." Flagging framing as unsupported is the single \
failure that gets this tool switched off: a person reads it as being accused of lying \
about their own motivations, for a sentence that was never a factual claim in the first \
place.

Framing is about the *person*: their motivation, their sequence, what they were \
weighing. A flat assertion about the world is not framing, even when it names no \
person and the corpus says nothing about it. "Dark matter is theoretical" is a \
checkable claim that happens to lack evidence, not connective tissue. If a sentence \
asserts something that could in principle be confirmed or denied, classify it as a \
claim and let the verdict record that the corpus is silent. The absence of \
candidate-shaped content is not evidence of framing.

Sentences like these are framing. Their kind is "framing", and a framing sentence's \
verdict is ALWAYS "supported" and its drift_label is ALWAYS "framing" -- framing is \
never checked against the corpus:
  - "After the acquisition, priorities shifted toward reliability work."
  - "Wanting to work closer to the infrastructure layer, they moved into platform \
engineering."
  - "The team had grown quickly, and process hadn't kept pace."
  - "This was the first project where they had full ownership of the roadmap."
  - "Having shipped that migration, the next challenge was the on-call rotation."

Contrast with claims: sentences asserting a fact that must trace to the corpus.
  - "Led a team of 12 engineers." -- a fact about scope
  - "Reduced p99 latency by 40%." -- a fact about outcome, with a number
  - "Owned the FX pricing platform end to end." -- a fact about ownership

A sentence can carry both in one clause ("Wanting more ownership, they took over the \
pricing platform"). Classify by the factual content: here "took over the pricing \
platform" is the claim to check; "wanting more ownership" is framing folded into the \
same sentence and must not, on its own, cause a flag.

## Verdict is a function of the corpus, not of the sentence's wording

The identical sentence can be supported against one corpus and drift against another. \
"Owned the FX pricing platform" is supported if the corpus describes what that \
ownership entailed -- decisions made, budget, on-call, headcount, or similar. The same \
sentence is ownership_inflation if the corpus only shows the person worked ON the \
platform with no evidence of ownership. Do not pattern-match on strong verbs like \
"led," "owned," or "drove" -- check what the corpus actually documents for that person, \
that scope, that outcome.

## Drift labels

Hard fails -- always verdict "unsupported", because no plausible missing corpus entry \
would rescue them:
  - invented_quantity: a number that appears nowhere in the corpus
  - adjacency_substitution: the corpus describes a different or lesser role or action \
than the one claimed (corpus says "reviewed", claim says "built") -- this is a \
contradiction, not a gap

Evidence-dependent -- always verdict "review" when the corpus does not support them, \
never "unsupported": the shape of the claim is legitimate, and the corpus may simply be \
missing the specific evidence needed.
  - scope_inflation: the claimed boundary (squad, team, department) is not backed
  - ownership_inflation: what ownership entailed (decisions, budget, on-call, \
headcount) is not backed
  - outcome_attribution: the link between the person's work and a stated metric or \
outcome is not backed
  - strategy_scope: the claimed strategic scope (for the team, the org, the function) \
is not backed
  - causality: the person's role in a causal chain is not backed by hard evidence

Not drift -- never flag these as anything other than what they are:
  - framing: ungroundable by nature; verdict is always "supported"
  - supported: the claim traces cleanly to the corpus

## Output

Return exactly one result per input sentence, in the same order the sentences were \
given -- one result object per sentence number below, in ascending order, none \
skipped, none repeated. For each: its index (the sentence's number from the list \
below), kind, verdict, drift_label, the corpus span IDs (if any) that support the \
verdict, and one short sentence as an evidence_note explaining the verdict. \
cited_span_ids may be empty -- for framing, or for a claim with no support in the \
corpus at all. Use only the exact span IDs given in the corpus below; never invent one.

## Corpus

{corpus}
"""


def format_corpus(spans: Sequence[Span]) -> str:
    """One line per span: its citable ID, its section, and its text.

    Section breadcrumbs are included because "scope_inflation" and "strategy_scope"
    verdicts often turn on which team or org a bullet sits under.
    """
    if not spans:
        return "(the corpus is empty -- no spans have been ingested for this user yet)"
    lines = []
    for span in spans:
        section = span.section_path or "(no section)"
        lines.append(f"[{span.id}] ({section}) {span.text}")
    return "\n".join(lines)


def build_system_prompt(spans: Sequence[Span]) -> str:
    return _INSTRUCTIONS.format(corpus=format_corpus(spans))


def build_system_blocks(
    spans: Sequence[Span], *, cache: Literal["instructions", "corpus"]
) -> list[TextBlockParam]:
    """Two cacheable `system` blocks -- instructions first, corpus second -- instead
    of the one block `build_system_prompt` returns. Prompt caching is a prefix
    match: with one block, a per-call corpus interpolated into the same string as
    the instructions means every distinct corpus rewrites the cache from byte
    zero. Splitting the two apart lets the caller put the cache breakpoint on
    whichever block is actually stable for its call pattern.

    `cache="corpus"` (today's product path) puts the breakpoint on the corpus
    block, so the cached prefix is instructions+corpus -- byte-for-byte what
    `build_system_prompt` returns as one block; see
    test_system_blocks_with_cache_corpus_is_byte_identical_to_the_single_string_prompt.

    `cache="instructions"` puts the breakpoint on the instructions block instead,
    so the ~1,500-token instructions cache once across many distinct per-item
    corpora (the eval harness's shape: 210 tiny, mutually distinct corpora) and
    each corpus then bills as ordinary, uncached input. Anthropic's minimum
    cacheable prefix is 1024 tokens on claude-sonnet-5 (512 on claude-opus-5);
    the instructions block is comfortably above both floors, so this option
    really does cache -- do not trim the instructions below that floor without
    re-checking this comment.

    The split is a plain partition on the `{corpus}` placeholder in
    `_INSTRUCTIONS`, so concatenating the two blocks' text is *always*
    byte-identical to `build_system_prompt`'s output by construction, not by
    convention -- there is no second copy of the template to drift out of sync.
    """
    corpus_text = format_corpus(spans)
    prefix, _, suffix = _INSTRUCTIONS.partition("{corpus}")
    blocks: list[TextBlockParam] = [
        {"type": "text", "text": prefix},
        {"type": "text", "text": corpus_text + suffix},
    ]
    cache_index = 0 if cache == "instructions" else 1
    blocks[cache_index]["cache_control"] = {"type": "ephemeral"}
    return blocks


def build_user_message(sentences: Sequence[str]) -> str:
    """The volatile half of the request -- goes in `messages`, never in `system`,
    so a byte change here never invalidates the cached corpus prefix.
    """
    numbered = "\n".join(f"{i + 1}. {sentence}" for i, sentence in enumerate(sentences))
    return f"Classify each of the following {len(sentences)} sentences:\n\n{numbered}"
