"""CV intake: one model call per CV, proposing facts the user then confirms.

Fixed control flow, one call, no corpus -- not agentic, see CLAUDE.md's "What
is agentic, and what is not". Mirrors `jfl_generate.extract` closely: same
client construction, same exception ladder, exactly one `runs` row on every
path including failure.

**The CV is input, never grounding.** It is read out of the sent-document store
and what comes back is a list of *candidate* facts, in `proposed` state, which
ground nothing until the user confirms them individually. Grounding on CVs
would make every later CV "supported" and switch the over-claim measurement off
silently (CLAUDE.md, 2026-09-18).

**Probes are guaranteed, not requested.** The prompt asks for a one-line
question wherever a fact carries a number, a team size, or led/owned/drove/
delivered. The model mostly complies; `to_proposed_facts` makes it certain, by
fitting a plain question to any such fact that came back without one. Same
reasoning as the deterministic rule tier in the claim gate: a prompt cannot
give a guarantee and a rule can, and the shapes in question are exactly the
scope, ownership and outcome inflation the drift taxonomy names.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.ids import fact_fingerprint, role_key
from jfl_core.models import ProposedFact, RunRecord
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import CV_FACTS_OUTPUT_SCHEMA, build_cv_facts_prompt
from jfl_generate.schema import CvFactItem, CvFactsOutput

# A CV runs to a few thousand tokens; the output is one short object per fact
# and a long CV can carry a hundred of them. Generous, and the truncation path
# below is what catches a CV that still does not fit.
MAX_TOKENS = 16_000

# Ceilings, applied after the call. A value past one of these is not truncated
# into a half-sentence -- the entry is dropped, because a cut fact is a
# different fact and this one is about to be shown to its subject as their own
# words.
MAX_ROLE_CHARS = 200
MAX_SOURCE_LINE_CHARS = 1_000
MAX_FACT_CHARS = 400
MAX_PROBE_CHARS = 300
# More facts than any real CV holds. A hard ceiling on what a misbehaving
# response could otherwise write into the confirmation screen.
MAX_FACTS = 300

# The shapes that must carry a probe. Deliberately narrow and observed, not
# invented: a digit anywhere (a headcount, a percentage, a revenue figure, a
# number of years), and the four verbs the drift taxonomy's scope, ownership
# and outcome categories are written in. Common inflections are included
# because a CV writes "leading" and "has owned" as readily as "led".
_HAS_NUMBER = re.compile(r"\d")
_OWNERSHIP = re.compile(
    r"\b(?:lead|leads|leading|led|own|owns|owning|owned|drive|drives|driving|drove|driven"
    r"|deliver|delivers|delivering|delivered)\b",
    re.IGNORECASE,
)

_NUMBER_PROBE = "What is that number exactly, and where does it come from?"
_OWNERSHIP_PROBE = (
    "What did that actually involve on your part -- decisions, budget, headcount, on-call?"
)

Outcome = Literal["ok", "error", "refused", "skipped"]


def needs_probe(fact_text: str) -> bool:
    """Whether this fact's shape requires a question before it can be trusted."""
    return bool(_HAS_NUMBER.search(fact_text) or _OWNERSHIP.search(fact_text))


def _default_probe(fact_text: str) -> str:
    return _NUMBER_PROBE if _HAS_NUMBER.search(fact_text) else _OWNERSHIP_PROBE


def _clean(value: str | None, limit: int) -> str | None:
    """Trim and collapse whitespace; None if empty or past `limit`."""
    if value is None:
        return None
    text = " ".join(value.split())
    if not text or len(text) > limit:
        return None
    return text


def to_proposed_facts(
    items: Sequence[CvFactItem], *, sent_document_id: uuid.UUID | None = None
) -> list[ProposedFact]:
    """Turn what the model returned into rows ready for `add_proposed`.

    Drops entries missing a role, a source line or a fact; fits a probe to any
    fact whose shape needs one and did not get one; and deduplicates within
    this one response by the same fingerprint the table is unique on, so a
    model that says the same thing twice does not produce a row that conflicts
    with itself mid-insert.
    """
    seen: set[str] = set()
    proposed: list[ProposedFact] = []
    for item in items:
        role_label = _clean(item.role_label, MAX_ROLE_CHARS)
        source_line = _clean(item.source_line, MAX_SOURCE_LINE_CHARS)
        fact_text = _clean(item.fact_text, MAX_FACT_CHARS)
        if not role_label or not source_line or not fact_text:
            continue
        probe = _clean(item.probe, MAX_PROBE_CHARS)
        if probe is None and needs_probe(fact_text):
            probe = _default_probe(fact_text)
        fingerprint = fact_fingerprint(role_label, fact_text)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        proposed.append(
            ProposedFact(
                sent_document_id=sent_document_id,
                role_label=role_label,
                role_key=role_key(role_label),
                source_line=source_line,
                fact_text=fact_text,
                probe=probe,
                fingerprint=fingerprint,
                ordinal=len(proposed),
            )
        )
        if len(proposed) >= MAX_FACTS:
            break
    return proposed


def extract_cv_facts(
    ctx: RequestContext,
    run_repo: RunRepository,
    *,
    cv_text: str,
    now: datetime,
) -> list[CvFactItem]:
    """Read one CV and propose candidate facts from it. Always writes exactly
    one `runs` row -- on success, on an API error, and on a refusal alike --
    before returning or raising.

    `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision: every model
    call is told what time it is), not read here, so one handler attempt and
    its test share one timestamp.
    """
    if not cv_text.strip():
        raise GenerateError("no text found in the CV")

    # An explicit key from the context wins; with no key, hand the SDK a bare
    # client so it resolves an `ant auth login` OAuth profile -- see
    # jfl_gate.gate.check_text for the full rationale.
    client = (
        anthropic.Anthropic(api_key=ctx.anthropic_api_key)
        if ctx.anthropic_api_key
        else anthropic.Anthropic()
    )

    started_at = datetime.now(UTC)
    clock_start = time.monotonic()

    outcome: Outcome = "ok"
    error_text: str | None = None
    response: anthropic.types.Message | None = None
    try:
        response = client.messages.create(
            model=ctx.model,
            max_tokens=MAX_TOKENS,
            # No corpus, no cache: the instructions are constant but small, and
            # the CV itself is what varies.
            system=build_cv_facts_prompt(now=now),
            messages=[{"role": "user", "content": cv_text}],
            output_config={"format": {"type": "json_schema", "schema": CV_FACTS_OUTPUT_SCHEMA}},
        )
    # Most-specific-first: RateLimitError/AuthenticationError/etc. are themselves
    # APIStatusError subclasses, so the broad catch must come last.
    except anthropic.RateLimitError as e:
        outcome, error_text = "error", f"rate_limited: {e}"
    except anthropic.AuthenticationError as e:
        outcome, error_text = "error", f"authentication_error: {e}"
    except anthropic.PermissionDeniedError as e:
        outcome, error_text = "error", f"permission_denied: {e}"
    except anthropic.NotFoundError as e:
        outcome, error_text = "error", f"not_found: {e}"
    except anthropic.BadRequestError as e:
        outcome, error_text = "error", f"bad_request: {e}"
    except anthropic.APIStatusError as e:
        outcome, error_text = "error", f"api_status_{e.status_code}: {e}"
    except anthropic.APIConnectionError as e:
        outcome, error_text = "error", f"connection_error: {e}"

    latency_ms = int((time.monotonic() - clock_start) * 1000)

    def record(
        outcome: Outcome,
        error: str | None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        cost_usd: object = None,
    ) -> None:
        run_repo.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="generate",
                stage="extract_cv_facts",
                model=ctx.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost_usd,  # type: ignore[arg-type]
                latency_ms=latency_ms,
                outcome=outcome,
                error=error,
                started_at=started_at,
            )
        )

    if response is None:
        record(outcome, error_text)
        raise GenerateError(error_text)

    usage = response.usage
    tokens_in = usage.input_tokens
    tokens_out = usage.output_tokens
    cache_read_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0
    cost_usd = compute_cost_usd(
        ctx.model, tokens_in, tokens_out, cache_read_tokens, cache_write_tokens
    )

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        record(
            "refused",
            f"refusal: {category}",
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise GenerateError(f"model refused to respond: {category}")

    if response.stop_reason == "max_tokens":
        record(
            "error",
            f"truncated: output hit max_tokens ({MAX_TOKENS})",
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise GenerateError(
            f"model output was truncated at max_tokens ({MAX_TOKENS}); "
            "the CV is too long for one call"
        )

    text_block = next((b for b in response.content if isinstance(b, TextBlock)), None)
    if text_block is None:
        record(
            "error",
            "no text block in response",
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise GenerateError("model response had no text content block")

    try:
        parsed = CvFactsOutput.model_validate(json.loads(text_block.text))
    except (json.JSONDecodeError, ValueError) as e:
        record(
            "error",
            f"parse_error: {e}",
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise GenerateError(f"could not parse structured output: {e}") from e

    record(
        "ok",
        None,
        tokens_in,
        tokens_out,
        cache_read_tokens,
        cache_write_tokens,
        cost_usd,
    )
    return parsed.facts
