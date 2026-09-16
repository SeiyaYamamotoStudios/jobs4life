"""Draft generation behind the claim gate: fixed control flow, two model calls.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." Slice 2b-core,
autonomous mode only (2b-full's interactive gaps-first mode is out of scope):

    job + requirements + latest coverage + whole corpus
        -> [model call 1: draft]          -> draft text
    draft text + whole corpus
        -> [model call 2: the claim gate] -> per-sentence verdicts

The first call mirrors `coverage.py`'s model-call machinery closely: same client
construction, the same most-specific-first exception ladder, the same refusal and
parse-failure handling, and exactly one `runs` row per call on every path. The
second call is `jfl_gate.gate.check_text` itself, called automatically -- see
CLAUDE.md's decisions log, "The claim gate runs automatically on generated text."
Both calls share `ctx.trace_id`, so a draft's total cost is one query:
`SELECT sum(cost_usd) FROM runs WHERE trace_id = <drafts.trace_id>`.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.models import Draft, DraftKind, RunRecord
from jfl_core.repositories import GroundingRepository, JobRepository, RunRepository
from jfl_gate.gate import check_text
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    DRAFT_OUTPUT_SCHEMA,
    build_draft_system_blocks,
    build_draft_user_message,
)
from jfl_generate.schema import DraftOutput

# A CV bullet list or cover letter is a few hundred words at most -- far short of
# coverage's 16000 (which scales with requirement count). Generous headroom, not a
# measured ceiling.
MAX_TOKENS = 8000

Outcome = Literal["ok", "error", "refused", "skipped"]


def compose_draft_text(title: str, body: str) -> str:
    """The stored and gated draft text: the model's title, if any, as a markdown h1
    above the body.

    An h1 is how `jfl_gate.gate.split_units` knows a title from a claim -- it sets
    a lone h1 aside as not checked. The title is flattened to one line, and any "#"
    it starts with is dropped, so it is always exactly one h1 line. If the body
    itself also carries an h1, there are two and neither is a title: both are
    checked, the same as before titles were separated at all.
    """
    title_line = " ".join(title.split()).lstrip("#").strip()
    if not title_line:
        return body
    return f"# {title_line}\n\n{body}"


def generate_draft(
    ctx: RequestContext,
    job_repo: JobRepository,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    job_id: uuid.UUID,
    kind: DraftKind,
) -> Draft:
    """Generate a draft for `job_id`, then run the claim gate on it automatically.
    Always writes exactly one `runs` row for the draft call -- on success, on an
    API error, and on a refusal alike -- before returning or raising; the gate
    pass records its own row the same way (see `check_text`).

    Requires coverage to already be recorded for this job (`jfl job coverage
    JOB_ID`) -- silently running coverage here would be a second, unbudgeted
    model call the caller never asked for.

    A flagged draft is still returned: the claim gate informs, it never blocks
    (see CLAUDE.md, "How the claim gate behaves"). Nothing here inspects the
    gate's verdicts to decide whether to persist or return the draft.
    """
    found = job_repo.get_job(ctx.user_id, job_id)
    if found is None:
        raise GenerateError(f"no job {job_id} for this user")
    job, requirements = found
    if not requirements:
        raise GenerateError("job has no requirements to draft against")

    coverage = job_repo.latest_coverage(ctx.user_id, job_id)
    if not coverage:
        raise GenerateError(
            f"no coverage recorded for job {job_id} -- run `jfl job coverage {job_id}` first"
        )

    # All non-retired spans, both provenances -- same call the claim gate makes.
    # Grounding input is the corpus only: sent_documents/sent_spans are never read
    # here (see CLAUDE.md's architectural constraints and the decisions log,
    # "Generated documents influence form, never truth").
    spans = grounding_repo.all_spans(ctx.user_id)
    system_blocks = build_draft_system_blocks(spans, kind, cache="corpus")
    user_message = build_draft_user_message(job, requirements, coverage)

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
            # Stable prefix (instructions + corpus) in `system`, cached; the job,
            # requirements and coverage under draft go in `messages` below, never
            # here -- any byte of volatile content here would invalidate the cache.
            system=system_blocks,
            messages=[{"role": "user", "content": user_message}],
            output_config={"format": {"type": "json_schema", "schema": DRAFT_OUTPUT_SCHEMA}},
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
                stage="draft",
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
        parsed = DraftOutput.model_validate(json.loads(text_block.text))
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

    # The claim gate runs automatically on generated text -- see CLAUDE.md's
    # decisions log. This writes its own `runs` row (component="gate",
    # stage="baseline") sharing ctx.trace_id with the draft call above. A
    # GateError here (an API failure, not a flagged claim) propagates uncaught:
    # the draft call's `runs` row has already committed, and the caller (the CLI)
    # reports the failure the same way it reports any other generation error.
    text = compose_draft_text(parsed.title, parsed.draft)
    gate_output = check_text(ctx, grounding_repo, run_repo, text)

    draft = Draft(
        user_id=ctx.user_id,
        job_id=job_id,
        kind=kind,
        text=text,
        gate_result=gate_output.model_dump(mode="json"),
        trace_id=ctx.trace_id,
    )
    job_repo.record_draft(draft)
    return draft
