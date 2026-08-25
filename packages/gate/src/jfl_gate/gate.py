"""The baseline gate: fixed control flow, one model call.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." The steps are:
split input text into sentences (reusing the ingestion sentence splitter), load the
whole corpus for the user, one call to Claude with the corpus cached and the
sentences volatile, parse the structured response, record exactly one `runs` row.
No retrieval, no verifier, no second call, no loop.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.ingest.parser import split_sentences
from jfl_core.models import RunRecord
from jfl_core.repositories import GroundingRepository, RunRepository

from jfl_gate.pricing import MODEL, compute_cost_usd
from jfl_gate.prompt import GATE_OUTPUT_SCHEMA, build_system_prompt, build_user_message
from jfl_gate.schema import GateOutput

MAX_TOKENS = 16000

Outcome = Literal["ok", "error", "refused", "skipped"]


class GateError(RuntimeError):
    """Raised when the gate cannot produce a verdict: bad config, an API error, or a
    refusal. A `runs` row recording the failure has already been written by the time
    this is raised -- the caller (the CLI) just needs to report it.
    """


def _sentences_from_text(text: str) -> list[str]:
    return [text[start:end] for start, end in split_sentences(text)]


def check_text(
    ctx: RequestContext,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    text: str,
) -> GateOutput:
    """Run the baseline gate over `text`. Always writes exactly one `runs` row --
    on success, on an API error, and on a refusal alike -- before returning or
    raising.
    """
    if not ctx.anthropic_api_key:
        raise GateError("no Anthropic API key in this RequestContext -- set ANTHROPIC_API_KEY")

    sentences = _sentences_from_text(text)
    if not sentences:
        raise GateError("no sentences found in the input text")

    # All non-retired spans, both provenances: `all_spans` defaults to excluding
    # retired ones, and does not filter by provenance at all.
    spans = grounding_repo.all_spans(ctx.user_id)
    system_prompt = build_system_prompt(spans)
    user_message = build_user_message(sentences)

    client = anthropic.Anthropic(api_key=ctx.anthropic_api_key)

    started_at = datetime.now(UTC)
    clock_start = time.monotonic()

    outcome: Outcome = "ok"
    error_text: str | None = None
    response: anthropic.types.Message | None = None
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # Stable prefix (instructions + corpus) in `system`, cached; the
            # sentences under test go in `messages` below, never in this block --
            # any byte of volatile content here would invalidate the cache.
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": GATE_OUTPUT_SCHEMA}},
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
                component="gate",
                stage="baseline",
                model=MODEL,
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
        raise GateError(error_text)

    usage = response.usage
    tokens_in = usage.input_tokens
    tokens_out = usage.output_tokens
    cache_read_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0
    cost_usd = compute_cost_usd(tokens_in, tokens_out, cache_read_tokens, cache_write_tokens)

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
        raise GateError(f"model refused to respond: {category}")

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
        raise GateError("model response had no text content block")

    try:
        result = GateOutput.model_validate(json.loads(text_block.text))
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
        raise GateError(f"could not parse structured output: {e}") from e

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
