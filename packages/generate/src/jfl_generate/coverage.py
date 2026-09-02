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
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.models import RunRecord
from jfl_core.repositories import GroundingRepository, RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    COVERAGE_OUTPUT_SCHEMA,
    build_coverage_system_blocks,
    build_coverage_user_message,
)
from jfl_generate.schema import CoverageOutput

MAX_TOKENS = 16000

Outcome = Literal["ok", "error", "refused", "skipped"]


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
            output_config={"format": {"type": "json_schema", "schema": COVERAGE_OUTPUT_SCHEMA}},
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

    record(
        "ok",
        None,
        tokens_in,
        tokens_out,
        cache_read_tokens,
        cache_write_tokens,
        cost_usd,
    )
    return result
