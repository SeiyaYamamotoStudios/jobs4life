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

**Only one of the two numbers is the model's.** "Do I want this" is not asked
for: the model gives a four-word verdict -- evidenced / partial / silent /
contradicted -- on every constraint and every objective the user recorded, and
`jfl_core.fit.want_it_basis` derives the number from those. See
`docs/profile-schema.md`: computed person-job fit predicts satisfaction at
rho ~= .28, so a number asked for directly would be a forecast nobody can make.
The judgement stays the model's, item by item; only the arithmetic over its own
verdicts is ours, which is what stops the number saying something the verdicts
listed under it do not. **The silences are the product**: a `must` the ad is
silent on is a question to ask at interview, and the panel reads that way.

**Only confirmed corpus facts count as evidence.** "Could I get this" is judged
from the job's requirements, the corpus coverage verdicts already recorded for
it, and the profile capabilities that carry corpus evidence -- coverage is
computed against spans, and spans are the confirmed corpus. A capability the
user tiered but never evidenced is a claim, the same status a CV line has
before confirmation, so it and the unconfirmed CV facts can only ever appear as
*levers*: "your profile claims X at working level; confirm it and this moves
from 5 to 7". Their own words are copied back from the stored claim, never
taken from the model's paraphrase.

**It ships unmeasured, and says so on screen.** There is no golden set for fit
and inventing one would be the synthetic-data prohibition in a new coat. The
measured number this project publishes is the over-claim rate, which is a
different number about a different thing.

Nothing here is named `reason` -- see CLAUDE.md's 2026-09-02 decision.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import MODEL_EFFORT, RequestContext
from jfl_core.fit import want_it_basis
from jfl_core.models import (
    ConstraintVerdict,
    FitVerdict,
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
    claimed_items,
    constraint_label,
    not_stated_sections,
)
from jfl_generate.schema import ScoreOutput

# Two short assessments, a one-line verdict per constraint and objective, and a
# few levers. Generous against what the schema can hold, and small against a
# gate call's 16k.
MAX_TOKENS = 4096

MIN_SCORE = 1
MAX_SCORE = 10

Outcome = Literal["ok", "error", "refused", "skipped"]


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One scoring run's answer, in core types, ready to store.

    Two numbers and two short assessments, deliberately not reducible to one:
    there is no `overall`, no third field, and no method here that would
    produce one.

    `want_it_score` is None when the profile records no constraints and no
    objectives. There is then nothing for the ad to be measured against, and
    a 1 would be a claim where there is only a silence.
    """

    could_get_score: int
    could_get_assessment: str
    want_it_score: int | None
    want_it_assessment: str
    constraint_verdicts: list[ConstraintVerdict] = field(default_factory=list)
    objective_verdicts: list[ObjectiveVerdict] = field(default_factory=list)
    hard_gate_breaches: list[HardGateBreach] = field(default_factory=list)
    levers: list[ScoreLever] = field(default_factory=list)
    # The profile sections the user has not filled in, computed here rather
    # than by the model, and reported as "not stated" on the page. An empty
    # section is never guessed at.
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
            # Effort pinned, not left to the model default: Opus 5 ran at `high` by
            # default and Opus 5.5 would drop to `medium` -- see
            # jfl_core.context.MODEL_EFFORT.
            output_config={
                "format": {"type": "json_schema", "schema": SCORE_OUTPUT_SCHEMA},
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

    Pure, and separated from the call so it can be tested without a client.
    "Could I get this" is carried through untouched; "do I want this" is
    derived here from the verdicts below and from nothing else. The two are
    never combined.
    """
    constraint_verdicts = _constraint_verdicts(parsed, inputs)
    objective_verdicts = _objective_verdicts(parsed, inputs)
    basis = want_it_basis(constraint_verdicts, objective_verdicts)
    return ScoreResult(
        could_get_score=parsed.could_get_score,
        could_get_assessment=parsed.could_get_assessment.strip(),
        want_it_score=basis.score,
        want_it_assessment=parsed.want_it_assessment.strip(),
        constraint_verdicts=constraint_verdicts,
        objective_verdicts=objective_verdicts,
        hard_gate_breaches=_breaches(constraint_verdicts),
        levers=_levers(parsed, inputs),
        not_stated=not_stated_sections(inputs.profile),
    )


def _constraint_verdicts(parsed: ScoreOutput, inputs: ScoreInputs) -> list[ConstraintVerdict]:
    """One verdict per constraint the user actually recorded, in their order.

    A verdict for an index naming no constraint is dropped rather than shown:
    the page would otherwise attribute a constraint to them that they never
    wrote, and the derived number would count it. A constraint the model gave
    no verdict for reads `silent`, which is the honest default -- it is what
    the panel says when nothing decided it either way.
    """
    constraints = list(inputs.profile.constraints)
    notes: dict[int, tuple[FitVerdict, str]] = {}
    for item in parsed.constraint_verdicts:
        if 1 <= item.index <= len(constraints) and item.index not in notes:
            notes[item.index] = (item.verdict, item.note.strip())
    verdicts = []
    for i, constraint in enumerate(constraints, start=1):
        verdict, note = notes.get(i, ("silent", ""))
        verdicts.append(
            ConstraintVerdict(
                kind=constraint.kind,
                stance=constraint.stance,
                label=constraint_label(constraint),
                verdict=verdict,
                note=note,
            )
        )
    return verdicts


def _objective_verdicts(parsed: ScoreOutput, inputs: ScoreInputs) -> list[ObjectiveVerdict]:
    """One verdict per objective the user actually listed, in rank order.

    Same rule as constraints: a verdict for a rank they have no objective at is
    dropped, and an objective the model skipped reads `silent`. The objective's
    own text comes from the stored profile, not from the response.
    """
    objectives = sorted(inputs.profile.objectives, key=lambda o: o.rank)
    notes: dict[int, tuple[FitVerdict, str]] = {}
    ranks = {o.rank for o in objectives}
    for item in parsed.objective_verdicts:
        if item.rank in ranks and item.rank not in notes:
            notes[item.rank] = (item.verdict, item.note.strip())
    verdicts = []
    for objective in objectives:
        verdict, note = notes.get(objective.rank, ("silent", ""))
        verdicts.append(
            ObjectiveVerdict(
                rank=objective.rank,
                objective=objective.text.strip(),
                verdict=verdict,
                note=note,
            )
        )
    return verdicts


def _breaches(verdicts: Sequence[ConstraintVerdict]) -> list[HardGateBreach]:
    """The `must` and `never` constraints the ad contradicts, in plain words.

    Derived from the verdicts rather than asked for separately, so the panel
    can never show a breach the verdicts do not carry -- and a `nice` the ad
    contradicts is a disappointment, not a breach, so it is not listed here.
    """
    return [
        HardGateBreach(gate=v.label, breach=v.note)
        for v in verdicts
        if v.verdict == "contradicted" and v.stance in ("must", "never")
    ]


def _levers(parsed: ScoreOutput, inputs: ScoreInputs) -> list[ScoreLever]:
    """Resolve each lever's `claim_index` back to the stored claim's own words.

    An index naming no claim is dropped -- there is nothing honest to show for
    it. `would_move_to` is kept only when it is a real move: inside 1-10 and
    above the score it would move from. Otherwise the lever still appears, with
    its note and without a number, because "confirming this would cover
    requirement X" is worth saying even when the model's arithmetic is not.
    """
    claims = claimed_items(inputs)
    seen: set[int] = set()
    levers: list[ScoreLever] = []
    for item in parsed.levers:
        index = item.claim_index
        if not (1 <= index <= len(claims)) or index in seen:
            continue
        seen.add(index)
        claim = claims[index - 1]
        # A real move is inside 1-10 AND above the score it moves from;
        # anything else is not arithmetic worth showing the user.
        real_move = (
            MIN_SCORE <= item.would_move_to <= MAX_SCORE
            and item.would_move_to > parsed.could_get_score
        )
        levers.append(
            ScoreLever(
                fact_text=claim.text,
                role_label=claim.role_label,
                would_move_to=item.would_move_to if real_move else None,
                note=item.note.strip(),
                claim_kind=claim.claim_kind,
                tier=claim.tier,
            )
        )
    return levers
