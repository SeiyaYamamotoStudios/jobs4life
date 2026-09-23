"""`classify_pushback`: a cheap model call that proposes what kind of
disagreement a score pushback is. Mirrors `jfl_generate.titles.suggest_titles`
closely -- same client construction, same most-specific-first exception
ladder, exactly one `runs` row on every path.

**Always `claude-haiku-4-5`, never `ctx.model`.** Same reasoning as
`titles.py`: classifying free text into one of three kinds is a
classification-shaped task, not the product model's job, and CLAUDE.md's
2026-09-05 decision log leaves this second, cheaper model selectable per call
site rather than tied to the product-model decision it made.

**This call reads; the rule decides.** The pushback box is one textarea, so
this call reads three things out of the words: what kind of statement it is,
which way it pushes, and whether it says anything new. The worker applies that
reading straight away and the screen shows, in plain words, what was taken and
what changed, with a one-click "Not what I meant" that undoes it and re-applies
under the reading the user picks. Undo-after instead of confirm-before -- and
what makes that safe is not this call but `jfl_core.pushback.decide`: no
reading, right or wrong, can move "could I get this" upward.

**Never asked to rewrite the user's words.** The prompt says so and the
sanitiser only collapses whitespace and caps length -- it does not paraphrase.
Tidying a person's own account of a disagreement into neater prose is the same
mistake CLAUDE.md's 2026-09-01 "a gap answer is stored verbatim" decision
already ruled out for corpus answers, applied here to a different kind of
verbatim text.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not."
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.models import RunRecord
from jfl_core.pushback import DIRECTIONS, PUSHBACK_KINDS, Direction, PushbackKind
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA,
    build_pushback_classification_prompt,
)
from jfl_generate.schema import PushbackClassificationOutput

# Not `jfl_gate.pricing.MODEL` and not `ctx.model` -- see the module docstring.
MODEL = "claude-haiku-4-5"

MAX_TOKENS = 512

# `classification_note` is a one-line note to the user, not an essay -- a note
# longer than this is trimmed rather than sent to the screen whole. Simple
# truncation is fine even mid-word: this call is cheap enough that a
# misbehaving response is not worth being clever about.
MAX_NOTE_CHARS = 200

Outcome = Literal["ok", "error", "refused", "skipped"]


@dataclass(frozen=True, slots=True)
class PushbackClassification:
    """One proposed classification, sanitised and ready to show the user for
    confirmation. Nothing here has been applied -- see the module docstring.
    """

    kind: PushbackKind
    direction: Direction
    new_information: bool
    note: str


def classify_pushback(
    ctx: RequestContext,
    run_repo: RunRepository,
    *,
    user_text: str,
    could_get_score: int | None,
    could_get_explanation: str,
    want_score: int | None,
    want_explanation: str,
    earlier_texts: Sequence[str] = (),
    now: datetime,
) -> PushbackClassification:
    """Propose a classification for one pushback. Always writes exactly one
    `runs` row -- on success, on an API error, and on a refusal alike --
    before returning or raising.

    Both numbers and their sentences are given, because the box sits under
    both and the words may be about either. `earlier_texts` is this user's
    recent earlier pushbacks, given so `new_information` can be judged against
    what has already been said rather than guessed at.

    `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision: every model
    call is told what time it is), not read here, so one handler attempt and
    its test share one timestamp.
    """
    user_text = user_text.strip()
    if not user_text:
        raise GenerateError("no user_text given")

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
            system=build_pushback_classification_prompt(
                user_text=user_text,
                could_get_score=could_get_score,
                could_get_explanation=could_get_explanation,
                want_score=want_score,
                want_explanation=want_explanation,
                earlier_texts=earlier_texts,
                now=now,
            ),
            messages=[{"role": "user", "content": "Classify this pushback."}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA,
                }
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
                stage="classify_pushback",
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
        parsed = PushbackClassificationOutput.model_validate(json.loads(text_block.text))
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
    return _sanitise(parsed)


def _sanitise(parsed: PushbackClassificationOutput) -> PushbackClassification:
    """Fold whitespace and cap the note; fall back an unrecognised `kind` to
    the one that is always safe.

    `"factual"` is that fallback, deliberately, not `"preference"` and not
    `"capability"`. `"capability"` downward applies in full and uncapped with
    no evidence asked for (`jfl_core.pushback._capability`), so a fabricated or
    malformed kind must never be able to reach it -- and `"factual"` is the one
    of the three that moves nothing at all, whichever direction the user
    asserted, so the worst a bad model answer can do here is show the user a
    reading that changed nothing, with "Not what I meant" beside it.

    A missing or unrecognised direction falls back the same way: a kind with no
    trustworthy direction becomes `"factual"`, and the direction becomes `"up"`
    -- which moves nothing on a factual reading and counts as an upward push on
    the drift meter, the conservative side for a meter that exists to catch
    flattery.
    """
    raw_kind = parsed.kind
    kind: PushbackKind = cast(PushbackKind, raw_kind) if raw_kind in PUSHBACK_KINDS else "factual"
    direction: Direction
    if parsed.direction in DIRECTIONS:
        direction = cast(Direction, parsed.direction)
    else:
        kind, direction = "factual", "up"
    note = " ".join(parsed.classification_note.split())
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS].rstrip()
    return PushbackClassification(
        kind=kind, direction=direction, new_information=parsed.new_information, note=note
    )
