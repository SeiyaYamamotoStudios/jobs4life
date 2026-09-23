"""Suggested title expansions -- slice C7a. Fixed control flow, one cheap model
call, no corpus. Mirrors `jfl_generate.extract.extract_requirements`'s shape
closely: same client construction, same exception ladder, exactly one `runs`
row on every path.

**Always `claude-haiku-4-5`, never `ctx.model`.** This is the second, cheaper
model CLAUDE.md's 2026-09-05 decision log says stays selectable per call site
-- the product model decision that entry made is about the generation and gate
calls; suggesting adjacent job titles is a classification-shaped task that does
not need the product model, and the owner named the model explicitly for this
call (PLAN.md's C7a).

Not agentic -- see CLAUDE.md, "What is agentic, and what is not."
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
from jfl_core.models import RunRecord, SuggestedTitle
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd
from jfl_intake.filtering import already_covered, parse_terms

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import TITLE_SUGGESTIONS_OUTPUT_SCHEMA, build_title_suggestion_prompt
from jfl_generate.schema import SuggestedTitleItem, TitleSuggestionsOutput

# Not `jfl_gate.pricing.MODEL` and not `ctx.model` -- see the module docstring.
MODEL = "claude-haiku-4-5"

MAX_TOKENS = 1024

# A suggested title longer than this is dropped rather than truncated -- a cut
# title is a different, wrong title, and this call is cheap enough to just ask
# again if the model produces nonsense. Matches `jfl_web.jobads.MAX_TITLE_CHARS`
# in spirit; kept separate because the two have no reason to move together.
MAX_TITLE_CHARS = 80

# Comfortably above what the prompt asks for ("about 10"); a hard ceiling on
# what a misbehaving response could otherwise hand back to the panel.
MAX_SUGGESTIONS = 10

Outcome = Literal["ok", "error", "refused", "skipped"]


def suggest_titles(
    ctx: RequestContext,
    run_repo: RunRepository,
    *,
    phrase: str,
    other_includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
    application_titles: Sequence[str] = (),
    now: datetime,
) -> list[SuggestedTitle]:
    """Adjacent titles for one phrase just added to a saved filter's title
    includes. Always writes exactly one `runs` row -- on success, on an API
    error, and on a refusal alike -- before returning or raising.

    `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision: every model
    call is told what time it is), not read here, so one handler attempt and
    its test share one timestamp.
    """
    phrase = phrase.strip()
    if not phrase:
        raise GenerateError("no phrase given")

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
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_title_suggestion_prompt(
                phrase,
                other_includes=other_includes,
                excludes=excludes,
                application_titles=application_titles,
                now=now,
            ),
            messages=[{"role": "user", "content": f'Suggest titles adjacent to "{phrase}".'}],
            output_config={
                "format": {"type": "json_schema", "schema": TITLE_SUGGESTIONS_OUTPUT_SCHEMA}
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
    ) -> None:
        run_repo.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="generate",
                stage="suggest_titles",
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
        raise GenerateError(error_text)

    usage = response.usage
    tokens_in = usage.input_tokens
    tokens_out = usage.output_tokens
    cache_read_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0
    cost_usd = compute_cost_usd(MODEL, tokens_in, tokens_out, cache_read_tokens, cache_write_tokens)

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
        raise GenerateError(f"model output was truncated at max_tokens ({MAX_TOKENS})")

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
        parsed = TitleSuggestionsOutput.model_validate(json.loads(text_block.text))
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
    return _sanitise(parsed.titles, phrase, other_includes)


def _excluded_keys(phrase: str, other_includes: Sequence[str]) -> tuple[frozenset[str], ...]:
    """The word-set keys (`jfl_intake.filtering.parse_terms`) already in the
    filter -- the phrase just typed, and every other include phrase already
    saved. A suggestion one of these already covers (`already_covered`: its
    words are a superset of an existing key's) would not widen anything, so it
    is dropped rather than offered as if it were new.
    """
    keys: list[frozenset[str]] = []
    for text in (phrase, *other_includes):
        keys.extend(k for k in parse_terms(text) if k not in keys)
    return tuple(keys)


def _sanitise(
    raw: Sequence[SuggestedTitleItem], phrase: str, other_includes: Sequence[str]
) -> list[SuggestedTitle]:
    """Trim, drop empties and anything implausibly long, replace commas (a
    comma would silently split one suggested title into two filter
    alternatives -- see `jfl_intake.filtering.parse_terms`), and drop anything
    the filter already matches -- by the filter's own rule, all of an
    alternative's words present, so "Senior Engineering Manager" is dropped
    when "engineering manager" is there -- and exact repeats of earlier items
    in this same response.
    """
    excluded = _excluded_keys(phrase, other_includes)
    seen: set[frozenset[str]] = set()
    result: list[SuggestedTitle] = []
    for item in raw:
        title = " ".join(item.title.replace(",", " ").split())
        if not title or len(title) > MAX_TITLE_CHARS:
            continue
        keys = parse_terms(title)
        if not keys:
            continue
        key = keys[0]
        if already_covered(key, excluded) or key in seen:
            continue
        seen.add(key)
        gloss = " ".join(item.gloss.replace(",", " ").split())
        result.append(SuggestedTitle(title=title, gloss=gloss))
        if len(result) >= MAX_SUGGESTIONS:
            break
    return result
