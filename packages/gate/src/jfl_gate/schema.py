"""Pydantic models for the gate's structured output.

Field-for-field, this mirrors `GATE_OUTPUT_SCHEMA` in prompt.py: that JSON schema
constrains what the model can return over the wire, these models give the CLI and
tests typed access to the parsed result. Keep the two in sync by hand -- there is
only one of each, so generating one from the other would be one abstraction for
one caller.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

# The drift taxonomy from CLAUDE.md, verbatim. `supported` and `framing` are not
# drift; the rest are the confirmed labels -- do not add to this list without the
# same evidence CLAUDE.md requires.
DriftLabel = Literal[
    "supported",
    "invented_quantity",
    "adjacency_substitution",
    "scope_inflation",
    "ownership_inflation",
    "outcome_attribution",
    "strategy_scope",
    "causality",
    "framing",
]

SentenceKind = Literal["claim", "framing"]
Verdict = Literal["supported", "review", "unsupported"]


class SentenceResult(BaseModel):
    # 1-based position in the input sentence list -- matches the numbering
    # `build_user_message` gives the model. The model returns this instead of
    # echoing the sentence text (a cost lever: output tokens dominate the gate's
    # bill, and echoing ~150 sentences back verbatim was a large slice of that).
    # `jfl_gate.gate._check_alignment` is what makes this safe: it is the
    # deterministic check that a misaligned or dropped/reordered result is
    # caught loudly instead of silently attaching a verdict to the wrong
    # sentence -- see that function's docstring.
    index: int
    kind: SentenceKind
    verdict: Verdict
    drift_label: DriftLabel
    cited_span_ids: list[uuid.UUID]
    # Named `evidence_note`, not `reason` -- see prompt.py's GATE_OUTPUT_SCHEMA
    # for why (live-API classifier false positive, found 2026-09-02).
    evidence_note: str
    # Populated after parsing -- never by the model -- so both defaults below
    # let the model's parsed JSON, which includes neither key, still validate
    # unchanged. See prompt.py's GATE_OUTPUT_SCHEMA for the other half of this
    # note.
    #
    # rule_flags: filled in by the deterministic rule tier (jfl_gate.rules).
    rule_flags: list[str] = Field(default_factory=list)
    # text: filled in by jfl_gate.gate.check_text from the input sentence list,
    # once `_check_alignment` has confirmed `index` lines up 1:1 with it. Kept
    # on the model (rather than dropped) because callers (the CLI, evals) still
    # want the sentence text alongside its verdict; it is only the wire format
    # to the model that stopped carrying it.
    text: str = ""


class GateOutput(BaseModel):
    sentences: list[SentenceResult]
