"""The deterministic rule tier: runs after the model, never before it.

No model call, no I/O, no added cost -- this is post-processing over an
already-parsed `GateOutput`. What it buys is a guarantee no prompt can give: a
rule either fires or it does not, with no sampling variance.

**A deterministic rule can prove contact, never crossing.** So every rule here can
only move a sentence from `supported` to `review` -- never to `unsupported`, never a
downgrade of an existing `review`/`unsupported`, never an upgrade to `supported`.
And a rule never touches `drift_label`: the taxonomy is the model's classification,
not a heuristic's. What fired is recorded in `SentenceResult.rule_flags` instead.

Rules apply only to `kind == "claim"` sentences. Framing is never touched -- see
CLAUDE.md's repeated point that over-flagging framing is the failure that gets this
tool switched off.

## The rules are definitional, and that is the whole design

Both rules below check something that is *true or false by inspection*, not
something inferred from word statistics. A cited span id either exists in the
corpus or it does not. A `supported` verdict either carries a citation or it does
not. Neither can produce a false positive, because neither is estimating anything.

## Two heuristic rules were tried, measured, and deleted -- do not reintroduce them

An earlier version of this module carried `unsourced-number` (flag a claim whose
numerals appear nowhere in the corpus) and `boundary-contact` (flag a claim sharing
a "distinctive" term with the corpus's stated-boundaries section). Both were removed
on 2026-09-01 after measurement on real material:

  * Across three real gate runs the boundary rule escalated 16, 9 and 14 claims from
    `supported` to `review` while `drift_label` stayed `supported` -- the model had
    traced them cleanly and the heuristic overrode it, on roughly 22% of all claims.
  * It matched the corpus owner's **own name**, because boundaries are written as
    "<name> is not ...", so the name sits in that section, is rare elsewhere, and
    therefore scored as distinctive. Every document containing the author's name was
    escalated.
  * It matched `hold`, a word from the section heading "boundaries to hold" -- the
    heading tokenised into boundary terms.
  * It matched ordinary CV vocabulary: `systems`, `team`, `code`, `experience`.
  * Swept over 32 real CVs, boundary variants fired on between 1.1% and 54.7% of
    sentences depending on threshold, and the best-scoring unigram variant's top
    terms were employer names.
  * `unsourced-number` fired on 7/68 real claims and escalated none of them; every
    hit was a year, a version string, or a phone number. A separate analysis over
    2,347 units found `invented_quantity` never fired once -- every number traced.

The failure was not a threshold that needed tuning. Unigram overlap cannot separate
"he does not have X" from "his depth *is* in X", because a boundary span states both
halves. If a boundary check is ever attempted again, it needs to read the negation,
which is judgement -- which is the model's job, and the model already has the whole
corpus including that section in its context.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from jfl_core.models import Span

from jfl_gate.schema import GateOutput, SentenceResult


def _flags_for(sentence: SentenceResult, corpus_span_ids: frozenset[uuid.UUID]) -> list[str]:
    """Definitional checks only. Order is stable so `rule_flags` is comparable."""
    flags: list[str] = []

    # A cited id the corpus does not contain. The prompt tells the model to use only
    # the ids given to it and never to invent one, so a miss here is a fabricated
    # citation -- the verdict rests on evidence that does not exist.
    unknown = sorted(str(c) for c in sentence.cited_span_ids if c not in corpus_span_ids)
    # A "citation" that is not even a uuid names no span at all, so it is the
    # same definitional miss -- set aside at parse time (`partition_citation_ids`)
    # rather than failing the whole check, and flagged here like any other.
    unknown.extend(sentence.unparseable_citations)
    flags.extend(f"unknown-citation:{c}" for c in unknown)

    # `supported` means, per the prompt, that the claim traces cleanly to the corpus.
    # With no citation there is nothing to check that against, so the support is
    # unverifiable rather than wrong -- which is exactly what `review` is for.
    if sentence.verdict == "supported" and not sentence.cited_span_ids:
        flags.append("uncited-support")

    return flags


def _reason_clause(flags: Sequence[str]) -> str:
    unknown = [f.split(":", 1)[1] for f in flags if f.startswith("unknown-citation:")]
    clauses = []
    if unknown:
        clauses.append(f"rule: cited span id(s) not in the corpus ({', '.join(unknown)})")
    if "uncited-support" in flags:
        clauses.append("rule: marked supported but cites no corpus span")
    return "; ".join(clauses)


def apply_rules(output: GateOutput, spans: Sequence[Span]) -> GateOutput:
    """Run the deterministic rule tier over an already-parsed `GateOutput`.

    Pure and side-effect free: returns a new `GateOutput` rather than mutating
    `output` or anything reachable from it.

    Flags are recorded on every claim sentence, so a fabricated citation on an
    already-flagged sentence is still visible. The **verdict** moves only for a
    claim currently at `supported` -- see the module docstring for why that is the
    only direction a rule may ever move one.
    """
    corpus_span_ids = frozenset(span.id for span in spans)

    sentences: list[SentenceResult] = []
    for sentence in output.sentences:
        if sentence.kind != "claim":
            sentences.append(sentence)
            continue

        flags = _flags_for(sentence, corpus_span_ids)
        if not flags:
            sentences.append(sentence)
            continue

        update: dict[str, object] = {
            "rule_flags": [*sentence.rule_flags, *flags],
            "evidence_note": f"{sentence.evidence_note} {_reason_clause(flags)}".strip(),
        }
        if sentence.verdict == "supported":
            update["verdict"] = "review"
        sentences.append(sentence.model_copy(update=update))

    return GateOutput(sentences=sentences)
