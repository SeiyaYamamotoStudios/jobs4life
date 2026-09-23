"""The baseline gate: fixed control flow, one model call.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." The steps are:
split input text into blocks and then sentences within each block, setting the
document title aside unchecked (see `split_units` below), load the whole corpus
for the user, one call to Claude
with the corpus cached and the sentences volatile, parse the structured
response, record exactly one `runs` row. No retrieval, no verifier, no second
call, no loop.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
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

# One result object per input sentence, each with a drift label, cited span IDs and an
# evidence note, so output still scales with document length even without echoed sentence
# text: a whole CV runs to ~150 sentences and blew through 16000, truncating the JSON
# mid-string. That surfaced as a parse error, which named the wrong cause entirely --
# hence the explicit max_tokens check below.
#
# A ceiling this high forces streaming: the SDK refuses a non-streaming request whose
# max_tokens implies a possible >10-minute response. Billing is on tokens actually
# generated, so the headroom costs nothing when the document is short.
MAX_TOKENS = 64000

# Left at `high`, Opus 5's default. Measured low/medium/high on one real CV (85 sentences,
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
#
# Pinned explicitly rather than left to the model's default: `high` was Opus 5's
# default, and Opus 5.5 defaults to `medium`, so if `JFL_GATE_MODEL` ever names
# Opus 5.5 an unpinned call would silently reason less than the measured gate did.
# Kept equal to `jfl_core.context.MODEL_EFFORT` (a test pins that).
EFFORT: Literal["low", "medium", "high", "xhigh", "max"] = "high"

Outcome = Literal["ok", "error", "refused", "skipped"]


class GateError(RuntimeError):
    """Raised when the gate cannot produce a verdict: bad config, an API error, or a
    refusal. A `runs` row recording the failure has already been written by the time
    this is raised -- the caller (the CLI) just needs to report it.
    """


# A markdown h1: one "#", whitespace, then the heading text. "## Section" does not
# match -- its second character is "#", not whitespace.
_H1_LINE = re.compile(r"^#\s+(?P<text>\S.*)$")

# What a title's `evidence_note` says, so the reason it has no verdict travels with
# the result rather than living only in whichever renderer shows it.
TITLE_NOTE = "Document title: not checked against the corpus."


@dataclass(frozen=True, slots=True)
class TextUnit:
    """One unit of a checked document, in document order. `kind="sentence"` is
    sent to the model; `kind="title"` is not, and comes back as a result with no
    verdict (see `jfl_gate.schema.SentenceKind`).
    """

    text: str
    kind: Literal["sentence", "title"]


def _title_line(lines: Sequence[str]) -> int | None:
    """The line number of the document title, or None.

    The title is an h1 that is the only h1 in the text -- the same rule
    jfl_core.ingest.parser applies to corpus markdown, where a lone h1 is the
    document's title and several h1s are structure. It is a rule about markup,
    so it is only as good as the markup: plain text and PDF extractions carry
    none, and a name line at the top of one is not recognised as a title. That
    is deliberate -- nothing in plain text separates "Jane Doe" or "Engineering
    Manager" from a role-and-dates line that is a real, checkable claim.
    """
    h1s = [i for i, line in enumerate(lines) if _H1_LINE.match(line.strip())]
    return h1s[0] if len(h1s) == 1 else None


def _blocks(text: str) -> list[tuple[str, bool]]:
    """(block text, is the document title) in document order. See `split_blocks`."""
    lines = text.splitlines()
    title_at = _title_line(lines)
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append((" ".join(current), False))
            current.clear()

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if i == title_at:
            flush()
            match = _H1_LINE.match(line)
            assert match is not None  # _title_line only returns matching lines
            blocks.append((match.group("text").strip(), True))
            continue
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


def split_blocks(text: str) -> list[str]:
    """Group text into blocks: bullets or paragraphs.

    Mirrors the block model in jfl_core.ingest.parser -- a blank line ends a
    block, and a bullet marker always starts a new one, even directly below
    another bullet -- minus section tracking, which checked text (unlike the
    corpus) has no use for. The one heading this does recognise is the
    document title (a lone markdown h1, see `_title_line`), which is always a
    block of its own, returned without its "#" marker. Text handed in from
    `read_input`'s PDF path already has every real break expressed as a blank
    line, so this only needs blank lines and bullet markers to recover the
    same blocks; plain hand-written or pasted text (never touched by that PDF
    pass) relies on the same two signals, exactly as corpus markdown does.
    """
    return [block for block, _ in _blocks(text)]


def split_units(text: str) -> list[TextUnit]:
    """Every unit of the document in order: one per sentence *within a block*,
    plus the document title as a unit of its own that is never sentence-split.

    Sentences never cross blocks, so a heading, a bullet, and the next bullet
    down can never be fused into one claim the way flat, block-blind
    sentence-splitting fused them before.

    The title is kept rather than dropped so it stays visible in the gate's
    output, marked as not checked. It is not sent to the model because a title
    is not an assertion: a draft titled "<name> -- CV bullets (<target role> --
    <target employer>)" names the job being applied for, and the gate read that
    as a claim to hold the role -- `adjacency_substitution`, unsupported, on
    two of the nine demo drafts.
    """
    units: list[TextUnit] = []
    for block, is_title in _blocks(text):
        if is_title:
            units.append(TextUnit(block, "title"))
            continue
        units.extend(TextUnit(block[s:e], "sentence") for s, e in split_sentences(block))
    return units


def sentences_from_text(text: str) -> list[str]:
    """The sentences `check_text` sends to the model, in order -- `split_units`
    without the document title.

    Public, like jfl_core.ingest.parser.split_sentences, so this can be tested
    directly instead of only through `check_text`, which needs a live (or
    mocked) model call to exercise at all.
    """
    return [unit.text for unit in split_units(text) if unit.kind == "sentence"]


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


def _reject_splitter_only_fields(results: Sequence[SentenceResult]) -> None:
    """`kind="title"` and a missing verdict or drift label belong to the splitter's
    title units only. GATE_OUTPUT_SCHEMA already keeps the model from producing
    them; this makes a response that somehow did a parse failure rather than a
    result that looks like an unchecked title. Raises ValueError so `check_text`
    records it exactly as it records any other unparseable response.
    """
    for r in results:
        if r.kind == "title" or r.verdict is None or r.drift_label is None:
            raise ValueError(
                f"result {r.index} has kind={r.kind!r}, verdict={r.verdict!r}, "
                f"drift_label={r.drift_label!r}; the model must give a claim or "
                "framing kind, a verdict and a drift label"
            )


def _assemble(units: Sequence[TextUnit], checked: Sequence[SentenceResult]) -> GateOutput:
    """Merge the model's aligned results back into document order.

    Each sentence result gets its own text (the model was never asked to echo it;
    see SentenceResult.text) and its position in the whole document as `index`.
    Each title unit becomes a result of its own with kind "title", no verdict, no
    drift label and no citations, so it is shown as not checked rather than dropped.
    Call only after `_check_alignment` has passed: `checked` is consumed in order.
    """
    remaining = iter(checked)
    assembled: list[SentenceResult] = []
    for position, unit in enumerate(units, start=1):
        if unit.kind == "title":
            assembled.append(
                SentenceResult(
                    index=position,
                    kind="title",
                    verdict=None,
                    drift_label=None,
                    cited_span_ids=[],
                    evidence_note=TITLE_NOTE,
                    text=unit.text,
                )
            )
        else:
            sentence = next(remaining)
            assembled.append(sentence.model_copy(update={"text": unit.text, "index": position}))
    return GateOutput(sentences=assembled)


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

    The model is `ctx.gate_model`, never `ctx.model`. The gate is configured
    independently of the product model (`$JFL_GATE_MODEL`, default `claude-opus-5`
    -- see `jfl_core.context.GATE_MODEL`) and stays on Opus 5 while generation
    moves to Opus 5.5, because the published over-claim 0.7% / over-flag 2.9%
    were measured on Opus 5, and Opus 5.5 widens the `reasoning_extraction`
    classifier that once refused every gate call. Switching waits for the paired
    eval (`packages/evals/scripts/compare_eval_runs.py`); until then a product
    model change must not quietly change what the published number describes.

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
    units = split_units(text)
    sentences = [unit.text for unit in units if unit.kind == "sentence"]
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
            model=ctx.gate_model,
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
                model=ctx.gate_model,
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
        ctx.gate_model, tokens_in, tokens_out, cache_read_tokens, cache_write_tokens
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
        _reject_splitter_only_fields(result.sentences)
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

    result = _assemble(units, result.sentences)

    # The rule tier makes no model call and adds no latency worth measuring, so it
    # runs here, after the one parse that can fail, and before the one `runs` row
    # this function writes on success -- never a second row of its own.
    result = apply_rules(result, spans)
    rule_escalations = sum(1 for sentence in result.sentences if sentence.rule_flags)
    # Citations that were not uuids at all, set aside rather than failing the
    # run. Counted on the `runs` row so a model that starts garbling ids is a
    # query, not a surprise.
    unparseable_citations = sum(len(s.unparseable_citations) for s in result.sentences)

    record(
        "ok",
        None,
        tokens_in,
        tokens_out,
        cache_read_tokens,
        cache_write_tokens,
        cost_usd,
        attributes={
            "rule_escalations": rule_escalations,
            "unparseable_citations": unparseable_citations,
        },
    )
    return result
