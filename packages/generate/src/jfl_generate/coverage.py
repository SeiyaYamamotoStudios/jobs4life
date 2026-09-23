"""Coverage checking: fixed control flow, one model call, corpus cached.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." Mirrors
`jfl_gate.gate.check_text` closely: the whole corpus goes in a cached system
block exactly as the claim gate does it, requirements go in the volatile user
message, and one call returns a verdict per requirement. Coverage is measured
against the corpus, never against the candidate -- see CLAUDE.md's decisions
log.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import MODEL_EFFORT, RequestContext
from jfl_core.models import RunRecord
from jfl_core.repositories import GroundingRepository, RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    COVERAGE_OUTPUT_SCHEMA,
    build_coverage_system_blocks,
    build_coverage_user_message,
)
from jfl_generate.schema import CoverageOutput, CoverageStatus, RequirementCoverageResult

MAX_TOKENS = 16000

Outcome = Literal["ok", "error", "refused", "skipped"]

# What a status becomes when every citation behind it was dropped. One step
# down, never to `contradicted`: a citation that could not be checked is an
# absence of evidence, not evidence of the opposite -- and reading silence as
# contradiction is the error CLAUDE.md's 2026-09-05 entry calls the
# worst-tempered one available. `evidenced` keeps credit for the model having
# found *something* relevant (it is `partial`, a gap question away from being
# checked properly); `partial` resting on nothing checkable is `absent`, which
# is exactly the corpus-silence status that asks the gap question.
_DOWNGRADE: dict[CoverageStatus, CoverageStatus] = {
    "evidenced": "partial",
    "partial": "absent",
}

_DOWNGRADE_NOTE = (
    "[The corpus citation given for this could not be checked -- it named no span "
    "in your confirmed facts -- so it is recorded as {status} rather than {was}.]"
)


def drop_unverifiable_citations(
    output: CoverageOutput, corpus_span_ids: frozenset[uuid.UUID]
) -> CoverageOutput:
    """Drop every citation that names no span in the corpus, per id, and
    downgrade a status that is left resting on nothing.

    **Definitional, the same rule `jfl_gate.rules` applies to the claim gate:**
    a cited span id either exists in the user's corpus or it does not. Nothing
    here estimates anything, so nothing here can be a false positive. A
    malformed id (set aside at parse time) and a well-formed id the corpus does
    not contain are the same miss and are treated identically.

    The downgrade fires only when dropping is what emptied the list. An
    `evidenced` result the model returned with no citations at all is left as
    the model gave it -- that is a different question (the prompt's own
    contract), and changing it here would move coverage results on runs that
    never had a bad citation.

    Pure: returns a new `CoverageOutput` with the two counts filled in.
    """
    dropped_total = 0
    downgraded = 0
    results: list[RequirementCoverageResult] = []
    for item in output.results:
        kept = [c for c in item.cited_span_ids if c in corpus_span_ids]
        dropped = len(item.cited_span_ids) - len(kept) + len(item.unparseable_citations)
        if dropped == 0:
            results.append(item)
            continue
        dropped_total += dropped
        update: dict[str, object] = {"cited_span_ids": kept, "unparseable_citations": []}
        new_status = _DOWNGRADE.get(item.status)
        if not kept and new_status is not None:
            downgraded += 1
            update["status"] = new_status
            note = _DOWNGRADE_NOTE.format(status=new_status, was=item.status)
            update["evidence_note"] = f"{item.evidence_note} {note}".strip()
        results.append(item.model_copy(update=update))
    return CoverageOutput(
        results=results, dropped_citations=dropped_total, downgraded_requirements=downgraded
    )


def check_coverage(
    ctx: RequestContext,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    requirements: Sequence[str],
) -> CoverageOutput:
    """Check what the corpus can evidence for each requirement, in order. Always
    writes exactly one `runs` row -- on success, on an API error, and on a
    refusal alike -- before returning or raising. A result count that does not
    match the requirement count is treated as a parse failure: there is no safe
    way to guess which requirement a stray or missing result belongs to.
    """
    if not requirements:
        raise GenerateError("no requirements to check")

    # All non-retired spans, both provenances -- same call the claim gate makes.
    spans = grounding_repo.all_spans(ctx.user_id)
    system_blocks = build_coverage_system_blocks(spans, cache="corpus")
    user_message = build_coverage_user_message(requirements)

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
            # Stable prefix (instructions + corpus) in `system`, cached; the
            # requirements under check go in `messages` below, never here -- any
            # byte of volatile content here would invalidate the cache.
            system=system_blocks,
            messages=[{"role": "user", "content": user_message}],
            # Effort pinned, not left to the model default: Opus 5 ran at `high` by
            # default and Opus 5.5 would drop to `medium` -- see
            # jfl_core.context.MODEL_EFFORT.
            output_config={
                "format": {"type": "json_schema", "schema": COVERAGE_OUTPUT_SCHEMA},
                "effort": MODEL_EFFORT,
            },
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
        attributes: dict[str, object] | None = None,
    ) -> None:
        run_repo.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="generate",
                stage="coverage",
                model=ctx.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                cost_usd=cost_usd,  # type: ignore[arg-type]
                latency_ms=latency_ms,
                outcome=outcome,
                error=error,
                attributes=attributes,
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
            "the document is too long for one call"
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
        result = CoverageOutput.model_validate(json.loads(text_block.text))
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

    if len(result.results) != len(requirements):
        record(
            "error",
            f"expected {len(requirements)} results, got {len(result.results)}",
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise GenerateError(
            f"expected {len(requirements)} coverage results, got {len(result.results)}"
        )

    # One bad citation used to void the whole run (production, 2026-09-23:
    # `results.4.cited_span_ids.1 Input should be a valid UUID`). Now it is
    # dropped, per id, and counted on this run's row so it is visible rather
    # than silent.
    result = drop_unverifiable_citations(result, frozenset(span.id for span in spans))

    record(
        "ok",
        None,
        tokens_in,
        tokens_out,
        cache_read_tokens,
        cache_write_tokens,
        cost_usd,
        attributes={
            "dropped_citations": result.dropped_citations,
            "downgraded_requirements": result.downgraded_requirements,
        },
    )
    return result
