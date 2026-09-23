"""Application questions -- two equal paths, fixed control flow, no loop.

See CLAUDE.md's 2026-09-18 decision ("check my answer" / "draft one for me"
side by side, the page advises, it never prescribes) and NEXT.md's task 4. Not
agentic -- see CLAUDE.md, "What is agentic, and what is not."

    "check my answer":
        question + job + requirements + the user's own answer text
            -> [model call: assess]        -> a paragraph on fit, and gaps
        answer text + whole corpus
            -> [model call: the claim gate] -> per-sentence verdicts

    "draft one for me":
        question + job + requirements + whole corpus
            -> [model call: draft]          -> answer text
        answer text + whole corpus
            -> [model call: the claim gate] -> per-sentence verdicts

Both calls in each path share `ctx.trace_id`, so one attempt's total cost is
one query: `SELECT sum(cost_usd) FROM runs WHERE trace_id = <trace_id>`, same
convention as `jfl_generate.draft`.

The assessment call is deliberately never asked to also judge a *drafted*
answer -- see `jfl_core.db.tables.application_question_answers`'s docstring
for why that would be circular (the model assessing its own draft against the
question it was just given). A draft is judged by the claim gate, the same as
any other generated text.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock, TextBlockParam
from jfl_core.context import MODEL_EFFORT, RequestContext
from jfl_core.models import Job, JobRequirement, RunRecord
from jfl_core.repositories import GroundingRepository, RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    ASSESS_ANSWER_OUTPUT_SCHEMA,
    DRAFT_ANSWER_OUTPUT_SCHEMA,
    build_assess_answer_prompt,
    build_draft_answer_system_blocks,
    build_draft_answer_user_message,
)
from jfl_generate.schema import AssessAnswerOutput, DraftAnswerOutput

# A paragraph plus a short "what it leaves out" -- generous headroom over what
# either call actually needs, not a measured ceiling (see jfl_gate.gate.MAX_TOKENS's
# docstring for why headroom is free: billing is on tokens actually generated).
ASSESS_MAX_TOKENS = 2000
DRAFT_ANSWER_MAX_TOKENS = 2000

Outcome = Literal["ok", "error", "refused", "skipped"]


def _client(ctx: RequestContext) -> anthropic.Anthropic:
    """An explicit key from the context wins; with no key, hand the SDK a bare
    client so it resolves an `ant auth login` OAuth profile -- see
    `jfl_gate.gate.check_text` for the full rationale.
    """
    return (
        anthropic.Anthropic(api_key=ctx.anthropic_api_key)
        if ctx.anthropic_api_key
        else anthropic.Anthropic()
    )


def _call(
    *,
    ctx: RequestContext,
    run_repo: RunRepository,
    stage: str,
    model: str,
    max_tokens: int,
    system: str | list[TextBlockParam],
    user_message: str,
    schema: dict[str, object],
) -> dict[str, object]:
    """The machinery shared by both calls in this module: client construction,
    the same most-specific-first exception ladder as every other model call in
    this codebase, and exactly one `runs` row on every path -- success, an API
    error, and a refusal alike -- before returning or raising `GenerateError`.

    Returns the parsed JSON body; callers validate it against their own
    pydantic model, same split `jfl_generate.titles.suggest_titles` uses.
    """
    client = _client(ctx)
    started_at = datetime.now(UTC)
    clock_start = time.monotonic()

    outcome: Outcome = "ok"
    error_text: str | None = None
    response: anthropic.types.Message | None = None
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
            # Effort pinned, not left to the model default: Opus 5 ran at `high` by
            # default and Opus 5.5 would drop to `medium` -- see
            # jfl_core.context.MODEL_EFFORT.
            output_config={
                "format": {"type": "json_schema", "schema": schema},
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
    ) -> None:
        run_repo.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="generate",
                stage=stage,
                model=model,
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
    cost_usd = compute_cost_usd(model, tokens_in, tokens_out, cache_read_tokens, cache_write_tokens)

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
            f"truncated: output hit max_tokens ({max_tokens})",
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise GenerateError(f"model output was truncated at max_tokens ({max_tokens})")

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
        parsed = json.loads(text_block.text)
    except json.JSONDecodeError as e:
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
    return parsed  # type: ignore[no-any-return]


def assess_answer(
    ctx: RequestContext,
    run_repo: RunRepository,
    *,
    question_text: str,
    answer_text: str,
    job: Job | None,
    requirements: list[JobRequirement],
    now: datetime,
) -> AssessAnswerOutput:
    """How well the candidate's own answer actually answers `question_text`
    for `job`. Separate from grounding -- the claim gate does that, over the
    same `answer_text`, in its own call. Always writes exactly one `runs` row.

    `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision), not read
    here, so one handler attempt and its test share one timestamp.
    """
    answer_text = answer_text.strip()
    if not answer_text:
        raise GenerateError("no answer text given")
    parsed = _call(
        ctx=ctx,
        run_repo=run_repo,
        stage="assess_answer",
        model=ctx.model,
        max_tokens=ASSESS_MAX_TOKENS,
        system=build_assess_answer_prompt(
            job=job,
            requirements=requirements,
            question_text=question_text,
            answer_text=answer_text,
            now=now,
        ),
        user_message="Assess this answer now.",
        schema=ASSESS_ANSWER_OUTPUT_SCHEMA,
    )
    try:
        return AssessAnswerOutput.model_validate(parsed)
    except ValueError as e:
        raise GenerateError(f"could not parse structured output: {e}") from e


def draft_application_answer(
    ctx: RequestContext,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    *,
    question_text: str,
    job: Job | None,
    requirements: list[JobRequirement],
    now: datetime,
) -> str:
    """A short, corpus-grounded answer to `question_text`, drafted for `job`.
    Always writes exactly one `runs` row for the draft call before returning
    or raising; automatic gating is the caller's job (mirrors
    `jfl_generate.draft.generate_draft`'s split of concerns: this returns text,
    the caller runs `jfl_gate.gate.check_text` over it, exactly as generation's
    other draft path does).

    Grounding is the corpus only -- `sent_documents`/`sent_spans` are never
    read here (see CLAUDE.md's architectural constraints and "Generated
    documents influence form, never truth").
    """
    spans = grounding_repo.all_spans(ctx.user_id)
    system_blocks = build_draft_answer_system_blocks(
        spans,
        job=job,
        requirements=requirements,
        question_text=question_text,
        now=now,
        cache="corpus",
    )
    parsed = _call(
        ctx=ctx,
        run_repo=run_repo,
        stage="draft_answer",
        model=ctx.model,
        max_tokens=DRAFT_ANSWER_MAX_TOKENS,
        system=system_blocks,
        user_message=build_draft_answer_user_message(),
        schema=DRAFT_ANSWER_OUTPUT_SCHEMA,
    )
    try:
        parsed_output = DraftAnswerOutput.model_validate(parsed)
    except ValueError as e:
        raise GenerateError(f"could not parse structured output: {e}") from e
    draft_text = parsed_output.draft.strip()
    if not draft_text:
        raise GenerateError("model returned an empty draft")
    return draft_text
