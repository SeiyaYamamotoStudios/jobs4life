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

from pydantic import BaseModel

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
    text: str
    kind: SentenceKind
    verdict: Verdict
    drift_label: DriftLabel
    cited_span_ids: list[uuid.UUID]
    reason: str


class GateOutput(BaseModel):
    sentences: list[SentenceResult]
