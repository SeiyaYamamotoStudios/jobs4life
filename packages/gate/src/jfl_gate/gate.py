"""The baseline gate: fixed control flow, one model call.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." The steps are:
split input text into blocks and then sentences within each block (see
`split_blocks` below), load the whole corpus for the user, one call to Claude
with the corpus cached and the sentences volatile, parse the structured
response, record exactly one `runs` row. No retrieval, no verifier, no second
call, no loop.
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
from jfl_core.ingest.parser import split_sentences
from jfl_core.models import RunRecord
from jfl_core.repositories import GroundingRepository, RunRepository

from jfl_gate.input import BULLET_START
from jfl_gate.pricing import compute_cost_usd
from jfl_gate.prompt import GATE_OUTPUT_SCHEMA, build_system_blocks, build_user_message
from jfl_gate.rules import apply_rules
from jfl_gate.schema import GateOutput, SentenceResult

# One result object per input sentence, each with a drift label, cited span IDs and a
# reason, so output still scales with document length even without echoed sentence
# text: a whole CV runs to ~150 sentences and blew through 16000, truncating the JSON
# mid-string. That surfaced as a parse error, which named the wrong cause entirely --
# hence the explicit max_tokens check below.
#
# A ceiling this high forces streaming: the SDK refuses a non-streaming request whose
# max_tokens implies a possible >10-minute response. Billing is on tokens actually
# generated, so the headroom costs nothing when the document is short.
MAX_TOKENS = 64000

# Left at the SDK default. Measured low/medium/high on one real CV (85 sentences,
# see analysis/gate-runs/A_index_low.json, B_index_medium.json, C_index_high.json):
# framing was never wrongly flagged at any level (the one hard disqualifier), and
# output tokens only fell 15% at low / 2% at medium vs high -- a small saving next
# to what the index-vs-text wire format below already buys. Against that: at low
# effort, 2 of 85 claim sentences ("led the X team for Y") were misclassified as
# *framing* -- not merely a verdict change, but exempting a real claim from
# grounding entirely, which is a more dangerous failure than any verdict shift.
# Medium showed one such case; a second high-effort run (different wire format,
# same effort) also showed two, so this reads as ordinary sampling noise on
# borderline sentences rather than something `effort` reliably fixes -- but with
# one document and one call per level, that can't be told apart from a real
# effort effect, and the token savings on the table are too small to bet on the
# distinction. Not taking this lever; revisit with a larger sample if the
# doc-length ceiling (MAX_TOKENS) ever forces the question.
EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = "high"

Outcome = Literal["ok", "error", "refused", "skipped"]


class GateError(RuntimeError):
    """Raised when the gate cannot produce a verdict: bad config, an API error, or a
    refusal. A `runs` row recording the failure has already been written by the time
    this is raised -- the caller (the CLI) just needs to report it.
    """


def split_blocks(text: str) -> list[str]:
    """Group text into blocks: bullets or paragraphs.

    Mirrors the block model in jfl_core.ingest.parser -- a blank line ends a
    block, and a bullet marker always starts a new one, even directly below
    another bullet -- minus headings and section tracking, which checked text
    (unlike the corpus) has no use for: every block here is just a unit to
    sentence-split and send to the model. Text handed in from `read_input`'s
    PDF path already has every real break expressed as a blank line, so this
    only needs blank lines and bullet markers to recover the same blocks;
    plain hand-written or pasted text (never touched by that PDF pass) relies
    on the same two signals, exactly as corpus markdown does.
    """
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        bullet_match = BULLET_START.match(line)
        if bullet_match:
            flush()
            current.append(line[bullet_match.end() :])
            continue
        current.append(line)
    flush()

    return blocks


def sentences_from_text(text: str) -> list[str]:
    """One unit per sentence *within a block* -- never across blocks, so a
    heading, a bullet, and the next bullet down can never be fused into one
    claim the way flat, block-blind sentence-splitting fused them before.

    Public, like jfl_core.ingest.parser.split_sentences, so this can be tested
    directly instead of only through `check_text`, which needs a live (or
    mocked) model call to exercise at all.
    """
    return [
        block[start:end] for block in split_blocks(text) for start, end in split_sentences(block)
    ]


def _check_alignment(sentences: Sequence[str], results: Sequence[SentenceResult]) -> None:
    """Guard the removal of `SentenceResult.text` from the wire format (see
    GATE_OUTPUT_SCHEMA and SentenceResult's docstrings): the model now returns a
    1-based `index` per sentence instead of echoing its text back, which is cheaper
    but removes the one thing that used to make misalignment visible -- a dropped,
    duplicated, or reordered sentence used to show up as a result whose echoed text
    didn't match anything, or as a wrong count. `index` gives up none of that: this
    check demands the *exact* sequence 1..N, in that order, so a right-count-wrong-
    order response (e.g. two results swapped) is caught exactly as a missing or
    duplicated one would be. Failing anything less than that would risk attaching a
    verdict to the wrong sentence, which in a truthfulness tool is worse than an
    error the caller has to look at.
    """
    expected = list(range(1, len(sentences) + 1))
    actual = [r.index for r in results]
    if actual != expected:
        raise GateError(
            "model output misaligned with input sentences: "
            f"expected indices {expected}, got {actual}"
        )


def check_text(
    ctx: RequestContext,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    text: str,
    *,
    shared_corpus: bool = True,
) -> GateOutput:
    """Run the baseline gate over `text`. Always writes exactly one `runs` row --
    on success, on an API error, and on a refusal alike -- before returning or
    raising.

    `shared_corpus` describes the workload, and only moves the cache breakpoint.
    True (the default, and the real product) means many calls run against one
    user's corpus, so the corpus belongs inside the cached prefix. False means
    every call carries a *different* corpus -- the eval harness, where each
    golden item has its own few-hundred-token evidence set -- and caching the
    corpus writes an entry that is never read. There the breakpoint goes on the
    instructions instead, which are byte-identical across items: written once,
    read thereafter, with the per-item corpus billed as ordinary input.

    The rendered prompt text is the same either way, so this cannot change a
    verdict -- see the byte-identity tests in `tests/test_prompt.py`.
    """
    sentences = sentences_from_text(text)
    if not sentences:
        raise GateError("no sentences found in the input text")

    # All non-retired spans, both provenances: `all_spans` defaults to excluding
    # retired ones, and does not filter by provenance at all.
    spans = grounding_repo.all_spans(ctx.user_id)
    system_blocks = build_system_blocks(spans, cache="corpus" if shared_corpus else "instructions")
    user_message = build_user_message(sentences)

    # An explicit key from the context wins. With no key, hand the SDK a bare
    # client so it resolves an `ant auth login` OAuth profile from disk -- the
    # same profile resolution Claude Code uses. This is not a module-level
    # environment read: the context still decides, it just has the option of
    # deciding "use whatever ambient credential this machine is logged in with",
    # which is what makes local development work without minting a static key.
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
        # Streamed, then collapsed back to a single Message: nothing here consumes
        # partial output, but a request with MAX_TOKENS this high is rejected outright
        # unless it streams.
        with client.messages.stream(
            model=ctx.model,
            max_tokens=MAX_TOKENS,
            # Stable prefix (instructions + corpus) in `system`, cached; the
            # sentences under test go in `messages` below, never in this block --
            # any byte of volatile content here would invalidate the cache.
            system=system_blocks,
            messages=[{"role": "user", "content": user_message}],
            output_config={
                "format": {"type": "json_schema", "schema": GATE_OUTPUT_SCHEMA},
                "effort": EFFORT,
            },
        ) as stream:
            response = stream.get_final_message()
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
                component="gate",
                stage="baseline",
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
        raise GateError(error_text)

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
        raise GateError(f"model refused to respond: {category}")

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
        raise GateError(
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

    try:
        _check_alignment(sentences, result.sentences)
    except GateError as e:
        record(
            "error",
            str(e),
            tokens_in,
            tokens_out,
            cache_read_tokens,
            cache_write_tokens,
            cost_usd,
        )
        raise

    # Attach each sentence's own text now that alignment is confirmed -- the model
    # was never asked for it (see SentenceResult.text's docstring).
    result = GateOutput(
        sentences=[
            sentence.model_copy(update={"text": sentences[sentence.index - 1]})
            for sentence in result.sentences
        ]
    )

    # The rule tier makes no model call and adds no latency worth measuring, so it
    # runs here, after the one parse that can fail, and before the one `runs` row
    # this function writes on success -- never a second row of its own.
    result = apply_rules(result, spans)
    rule_escalations = sum(1 for sentence in result.sentences if sentence.rule_flags)

    record(
        "ok",
        None,
        tokens_in,
        tokens_out,
        cache_read_tokens,
        cache_write_tokens,
        cost_usd,
        attributes={"rule_escalations": rule_escalations},
    )
    return result
