"""Two scores for one job: fixed control flow, one model call -- PLAN.md B4.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." Mirrors
`jfl_generate.coverage.check_coverage`'s shape closely: same client
construction, same exception ladder, exactly one `runs` row on every path
including failure.

**Two axes, 1-10 each, never composited.** CLAUDE.md's standing decision, and
the reason there is no function in this module that takes both numbers: a role
the person would love and will not get, and one they would dislike and would
walk into, must never land on the same number. Nothing downstream averages
them either, and `tests/test_scores_never_composited.py` says so mechanically.

**Only confirmed corpus facts count as evidence.** "Could I get this" is judged
from the job's requirements and the corpus coverage verdicts already recorded
for it -- coverage is computed against spans, and spans are the confirmed
corpus, so the unconfirmed CV claims this call is also shown can only ever
appear as *levers*: "your CVs claim X; confirm it and this moves from 5 to 7".
Their own words are copied back from the stored fact, never taken from the
model's paraphrase.

**It ships unmeasured, and says so on screen.** There is no golden set for fit
and inventing one would be the synthetic-data prohibition in a new coat. The
measured number this project publishes is the over-claim rate, which is a
different number about a different thing.

Nothing here is named `reason` -- see CLAUDE.md's 2026-09-02 decision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.models import (
    HardGateBreach,
    NotStated,
    ObjectiveVerdict,
    RunRecord,
    ScoreLever,
)
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    SCORE_OUTPUT_SCHEMA,
    ScoreInputs,
    build_score_system_blocks,
    build_score_user_message,
    unfilled_sections,
)
from jfl_generate.schema import ScoreOutput

# Two paragraphs, a handful of short verdicts and a few levers. Generous
# against what the schema can hold, and small against a gate call's 16k.
MAX_TOKENS = 4096

MIN_SCORE = 1
MAX_SCORE = 10

Outcome = Literal["ok", "error", "refused", "skipped"]


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One scoring run's answer, in core types, ready to store.

    Two numbers and two paragraphs, deliberately not reducible to one: there is
    no `overall`, no `average`, and no method here that would produce one.
    """

    could_get_score: int
    could_get_assessment: str
    want_it_score: int
    want_it_assessment: str
    objective_verdicts: list[ObjectiveVerdict] = field(default_factory=list)
    hard_gate_breaches: list[HardGateBreach] = field(default_factory=list)
    levers: list[ScoreLever] = field(default_factory=list)
    # The profile questions the user has not answered, computed here rather
    # than by the model, and reported as "not stated" on the page. PLAN.md B3a:
    # a skipped question is never guessed at.
    not_stated: list[NotStated] = field(default_factory=list)


def score_application(
    ctx: RequestContext,
    run_repo: RunRepository,
    inputs: ScoreInputs,
) -> ScoreResult:
    """Score one job on both axes. Always writes exactly one `runs` row -- on
    success, on an API error, and on a refusal alike -- before returning or
    raising.

    Raises `GenerateError` when there is nothing to score against: a job with no
    extracted requirements gives "could I get this" nothing to be measured
    against, and guessing a number from an unread ad is exactly the dishonesty
    this tool exists to oppose.
    """
    if not inputs.requirements:
        raise GenerateError("job has no requirements to score against")

    system_blocks = build_score_system_blocks()
    user_message = build_score_user_message(inputs)

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
            # Stable prefix (the instructions) in `system`, cached; everything
            # volatile -- the job, the coverage verdicts, the profile answers,
            # the unconfirmed claims, the current time -- in `messages`.
            system=system_blocks,
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": SCORE_OUTPUT_SCHEMA}},
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
                stage="score",
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
        parsed = ScoreOutput.model_validate(json.loads(text_block.text))
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
    return build_result(parsed, inputs)


def build_result(parsed: ScoreOutput, inputs: ScoreInputs) -> ScoreResult:
    """Turn one parsed response into stored types, keeping the user's own words.

    Pure, and separated from the call so it can be tested without a client. The
    two numbers are carried through untouched and are never combined.
    """
    return ScoreResult(
        could_get_score=parsed.could_get_score,
        could_get_assessment=parsed.could_get_assessment.strip(),
        want_it_score=parsed.want_it_score,
        want_it_assessment=parsed.want_it_assessment.strip(),
        objective_verdicts=_objective_verdicts(parsed, inputs),
        hard_gate_breaches=[
            HardGateBreach(gate=b.gate.strip(), breach=b.breach.strip())
            for b in parsed.hard_gate_breaches
            if b.breach.strip()
        ],
        levers=_levers(parsed, inputs),
        not_stated=[
            NotStated(question_key=name, wording=wording)
            for name, wording in unfilled_sections(inputs.profile)
        ],
    )


def _objective_verdicts(parsed: ScoreOutput, inputs: ScoreInputs) -> list[ObjectiveVerdict]:
    """One verdict per objective the user actually listed, in ordinal order.

    A verdict for an ordinal the user has no objective in is dropped rather
    than shown: the page would otherwise attribute an objective to them that
    they never wrote. The objective's own text comes from the stored profile,
    not from the response. `ordinal` here is the objective's `rank`, which is
    what the stored result has always keyed verdicts by.
    """
    by_ordinal = {o.rank: o for o in inputs.profile.objectives}
    seen: set[int] = set()
    verdicts: list[ObjectiveVerdict] = []
    for item in parsed.objective_verdicts:
        objective = by_ordinal.get(item.ordinal)
        if objective is None or item.ordinal in seen:
            continue
        seen.add(item.ordinal)
        verdicts.append(
            ObjectiveVerdict(
                ordinal=item.ordinal,
                objective=objective.text.strip(),
                verdict=item.verdict.strip(),
            )
        )
    return sorted(verdicts, key=lambda v: v.ordinal)


def _levers(parsed: ScoreOutput, inputs: ScoreInputs) -> list[ScoreLever]:
    """Resolve each lever's `fact_index` back to the stored fact's own words.

    An index naming no fact is dropped -- there is nothing honest to show for
    it. `would_move_to` is kept only when it is a real move: inside 1-10 and
    above the score it would move from. Otherwise the lever still appears, with
    its note and without a number, because "confirming this would cover
    requirement X" is worth saying even when the model's arithmetic is not.
    """
    facts = list(inputs.proposed_facts)
    seen: set[int] = set()
    levers: list[ScoreLever] = []
    for item in parsed.levers:
        index = item.fact_index
        if not (1 <= index <= len(facts)) or index in seen:
            continue
        seen.add(index)
        fact = facts[index - 1]
        # A real move is inside 1-10 AND above the score it moves from;
        # anything else is not arithmetic worth showing the user.
        real_move = (
            MIN_SCORE <= item.would_move_to <= MAX_SCORE
            and item.would_move_to > parsed.could_get_score
        )
        levers.append(
            ScoreLever(
                fact_text=fact.fact_text,
                role_label=fact.role_label,
                would_move_to=item.would_move_to if real_move else None,
                note=item.note.strip(),
            )
        )
    return levers
