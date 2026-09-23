"""Pydantic models for the gate's structured output.

Field-for-field, this mirrors `GATE_OUTPUT_SCHEMA` in prompt.py: that JSON schema
constrains what the model can return over the wire, these models give the CLI and
tests typed access to the parsed result. Keep the two in sync by hand -- there is
only one of each, so generating one from the other would be one abstraction for
one caller.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# How much of an unparseable citation is kept for display. The value came from
# the model, not the user, and it is quoted into an evidence note -- a bound
# keeps a runaway string from becoming the note.
_MAX_UNPARSEABLE_CHARS = 64


def partition_citation_ids(raw: object) -> tuple[list[uuid.UUID], list[str]]:
    """Split a model's `cited_span_ids` into well-formed uuids and the rest.

    **One malformed citation must never void a whole result.** Strict parsing
    used to reject the entire response when a single id was not a uuid -- a
    production coverage run failed on `results.4.cited_span_ids.1` and every
    other requirement's verdict went with it. A citation is definitional: it
    names a span that exists or it does not, and a string that is not even a
    uuid names nothing. So it is set aside here, per id, and the caller decides
    what that absence means for the verdict it was attached to.

    Order is preserved in both lists. A non-list is treated as no citations at
    all rather than guessed at.
    """
    valid: list[uuid.UUID] = []
    unparseable: list[str] = []
    if not isinstance(raw, list):
        return valid, unparseable
    for item in raw:
        if isinstance(item, uuid.UUID):
            valid.append(item)
            continue
        if isinstance(item, str):
            try:
                valid.append(uuid.UUID(item))
                continue
            except ValueError:
                pass
        unparseable.append(str(item)[:_MAX_UNPARSEABLE_CHARS])
    return valid, unparseable


def split_unparseable_citations(data: Any) -> Any:
    """A `model_validator(mode="before")` body shared by every result type that
    carries `cited_span_ids`: move anything that is not a uuid out of that list
    and into `unparseable_citations`, where it is counted rather than fatal.
    """
    if not isinstance(data, dict) or "cited_span_ids" not in data:
        return data
    valid, unparseable = partition_citation_ids(data["cited_span_ids"])
    if not unparseable:
        return data
    return {
        **data,
        "cited_span_ids": valid,
        "unparseable_citations": [*data.get("unparseable_citations", []), *unparseable],
    }


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

# "claim" and "framing" are the model's call. "title" never is: GATE_OUTPUT_SCHEMA
# does not offer it, and `check_text` rejects a model result carrying it. It is
# assigned only by the splitter, to a document's title (see
# `jfl_gate.gate.split_units`), which is never sent to the model and so has no
# verdict and no drift label -- None, never a default "supported", because
# nothing checked it.
SentenceKind = Literal["claim", "framing", "title"]
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
    #
    # Once `check_text` returns, `index` is the unit's 1-based position in the whole
    # document, title included. With no title in the document -- every FEVER item,
    # every plain-text CV -- that is exactly the numbering the model was given.
    index: int
    kind: SentenceKind
    # None only for kind="title"; see SentenceKind.
    verdict: Verdict | None
    drift_label: DriftLabel | None
    cited_span_ids: list[uuid.UUID]
    # Anything the model put in `cited_span_ids` that is not a uuid, set aside
    # at parse time instead of failing the whole check -- see
    # `partition_citation_ids`. Never supplied by the model (the wire schema has
    # no such property); `jfl_gate.rules` treats each one exactly as it treats a
    # well-formed id the corpus does not contain.
    unparseable_citations: list[str] = Field(default_factory=list)
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

    @model_validator(mode="before")
    @classmethod
    def _set_aside_unparseable_citations(cls, data: Any) -> Any:
        return split_unparseable_citations(data)


class GateOutput(BaseModel):
    sentences: list[SentenceResult]
