"""A complete CV behind the claim gate: fixed control flow, two model calls.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not":

    corpus spans -> [no model: jfl_core.cv_skeleton] -> roles, dates, education
    skeleton + job + requirements + coverage + whole corpus
        -> [model call 1: the CV's claims]  -> summary, skills, descriptors, bullets
    every generated line + whole corpus
        -> [model call 2: the claim gate]   -> a verdict per line

**What the model writes and what it does not.** Titles, employers, dates,
locations and education lines are copied from the corpus by
`jfl_core.cv_skeleton`, verbatim. The model is shown the roles numbered and
answers by number; a role number that does not exist is dropped, and anything
else it returns for a role (a title, a date) has nowhere to land -- the parse
model declares no such field. So a date on this CV is always a date the user
confirmed.

**Every generated line is gated, in one pass.** Summary paragraphs, skill
texts and bullets are joined as one block each and sent to
`jfl_gate.gate.check_text` together, and the per-sentence results are mapped
back to the line each came from (`apply_gate_output`). Fact lines are never
sent: they are the corpus, and checking the corpus against itself would only
dilute the number.

**Descriptors are generated and deliberately not gated.** A descriptor ("A UK
wholesale tea distributor") is a claim about the employer, not the person. The
corpus is a record of the person; it is usually silent on what their employers
do, so the claim gate would mostly return corpus silence as `review` or
`unsupported` -- the over-flag the tool is built to avoid, on a line that says
nothing about the candidate. The prompt tells the model to write a descriptor
only from what the corpus says about the employer, or to leave it empty; and
`CvRole.descriptor` is a plain string, not a `CvLine`, so no screen can show
it as checked. A screen should mark it as not checked.

**A flagged line stays in the document.** Nothing here reads a verdict to decide
what to keep -- the claim gate informs, it never blocks.

The model call mirrors `jfl_generate.draft.generate_draft`: same client
construction, the same exception ladder and error-text prefixes (the worker's
classifier matches on them), and exactly one `runs` row per call on every path.
Both calls share `ctx.trace_id`.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import MODEL_EFFORT, RequestContext
from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvRole, CvSkill
from jfl_core.cv_skeleton import CvSkeleton, build_skeleton, name_from_title
from jfl_core.models import RunRecord
from jfl_core.profile import Capability
from jfl_core.repositories import GroundingRepository, JobRepository, RunRepository
from jfl_gate.gate import check_text, split_units
from jfl_gate.pricing import compute_cost_usd
from jfl_gate.schema import GateOutput, SentenceResult

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    CV_DOCUMENT_OUTPUT_SCHEMA,
    CvPromptRole,
    build_cv_document_system_blocks,
    build_cv_document_user_message,
)
from jfl_generate.schema import CvDocumentOutput

# About twenty bullets of 25-40 words, a summary and eight skills is ~1,500
# output tokens; adaptive thinking at `high` effort shares this budget, hence the
# headroom. Not a measured ceiling.
MAX_TOKENS = 16000

Outcome = Literal["ok", "error", "refused", "skipped"]

# Leading markdown a generated line must not carry into the gate: a bullet
# marker would be stripped by the gate's block splitter, and a lone "# " line
# would be read as a document title and not checked at all.
_LEADING_MARKUP = re.compile(r"^(?:#+\s*|(?:[•▪\-*+]|\d+[.)])\s+)+")


@dataclass(frozen=True)
class GeneratedCv:
    document: CvDocument
    gate_result: dict[str, object]
    trace_id: uuid.UUID


def clean_line(text: str) -> str:
    """One generated line as it is shown and as it is gated -- the same string
    both times. Whitespace collapsed to single spaces (so one line is always one
    gate block) and leading bullet or heading markup removed."""
    flattened = " ".join(text.split())
    return _LEADING_MARKUP.sub("", flattened).strip()


def assemble_document(skeleton: CvSkeleton, output: CvDocumentOutput, *, name: str) -> CvDocument:
    """The skeleton with the model's claims attached, by index.

    Every title, employer, date and location comes from `skeleton`; the model
    supplies text only. A role index outside the skeleton is dropped; a role
    the model returns twice keeps its first entry.
    """
    roles = [
        CvRole(
            title=role.title,
            employer=role.employer,
            location=role.location,
            dates=role.dates,
        )
        for role in skeleton.roles
    ]
    written: set[int] = set()
    for entry in output.roles:
        position = entry.index - 1
        if not 0 <= position < len(roles) or position in written:
            continue
        written.add(position)
        roles[position].descriptor = clean_line(entry.descriptor)
        roles[position].bullets = [
            CvLine(text=text) for text in (clean_line(b) for b in entry.bullets) if text
        ]
    return CvDocument(
        header=CvHeader(name=name or name_from_title(skeleton.corpus_title)),
        summary=[CvLine(text=text) for text in (clean_line(p) for p in output.summary) if text],
        skills=[
            CvSkill(label=clean_line(skill.label), text=CvLine(text=clean_line(skill.text)))
            for skill in output.skills
            if clean_line(skill.text)
        ],
        roles=roles,
        education=[CvLine(text=line, origin="fact") for line in skeleton.education],
    )


def gate_text(lines: Sequence[CvLine]) -> str:
    """The text the claim gate checks: one block per generated line, blank-line
    separated, so no sentence ever spans two lines."""
    return "\n\n".join(line.text for line in lines)


_SEVERITY = {"supported": 0, "review": 1, "unsupported": 2}


LineVerdict = Literal["supported", "review", "unsupported", "framing"]


def _line_verdict(results: Sequence[SentenceResult]) -> tuple[LineVerdict, str]:
    """One line's verdict from its sentences': the worst claim verdict, with the
    notes of the sentences that earned it. A line with no claim sentence is
    framing -- shown as not checked, never as supported."""
    claims = [r for r in results if r.kind == "claim" and r.verdict is not None]
    if not claims:
        return "framing", " ".join(r.evidence_note for r in results if r.evidence_note)
    worst = max(claims, key=lambda r: _SEVERITY[r.verdict or "supported"]).verdict
    assert worst is not None
    notes = [r.evidence_note for r in claims if r.verdict == worst and r.evidence_note]
    return worst, " ".join(notes)


def apply_gate_output(lines: Sequence[CvLine], output: GateOutput) -> None:
    """Map the gate's per-sentence results back onto the lines they came from,
    in place. Each line's sentence count is computed by the gate's own splitter
    (`split_units`), so the mapping is the gate's reading, not a second one.
    """
    counts = [len(split_units(line.text)) for line in lines]
    if sum(counts) != len(output.sentences) or any(r.kind == "title" for r in output.sentences):
        raise GenerateError(
            "claim gate output does not line up with the CV's lines: "
            f"expected {sum(counts)} sentences, got {len(output.sentences)}"
        )
    position = 0
    for line, count in zip(lines, counts, strict=True):
        verdict, note = _line_verdict(output.sentences[position : position + count])
        line.verdict = verdict
        line.note = note
        position += count


def generate_cv_document(
    ctx: RequestContext,
    job_repo: JobRepository,
    grounding_repo: GroundingRepository,
    run_repo: RunRepository,
    job_id: uuid.UUID,
    *,
    capabilities: Sequence[Capability] = (),
    name: str = "",
    now: datetime | None = None,
) -> GeneratedCv:
    """Write a complete CV for `job_id` and gate every generated line.

    Same prerequisites as `generate_draft` -- a job, its requirements and
    recorded coverage -- with the same error text, so the worker classifies a
    missing one the same way. `name` is the header name the caller resolved
    (profile, then account); empty falls back to the corpus document's own
    title. `capabilities` are the ceiling on what may be claimed.
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

    # Grounding input is the corpus only -- never the sent-document store
    # (CLAUDE.md, "Generated documents influence form, never truth").
    spans = grounding_repo.all_spans(ctx.user_id)
    skeleton = build_skeleton(spans)
    system_blocks = build_cv_document_system_blocks(spans)
    user_message = build_cv_document_user_message(
        job,
        requirements,
        coverage,
        [
            CvPromptRole(title=r.title, employer=r.employer, dates=r.dates, facts=r.facts)
            for r in skeleton.roles
        ],
        boundaries=skeleton.boundaries,
        capabilities=capabilities,
        now=now or datetime.now(UTC),
    )

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
            system=system_blocks,
            messages=[{"role": "user", "content": user_message}],
            # Effort pinned -- see jfl_core.context.MODEL_EFFORT.
            output_config={
                "format": {"type": "json_schema", "schema": CV_DOCUMENT_OUTPUT_SCHEMA},
                "effort": MODEL_EFFORT,
            },
        )
    # Most-specific-first, as in generate_draft.
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
                stage="cv_document",
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
    billed = (tokens_in, tokens_out, cache_read_tokens, cache_write_tokens, cost_usd)

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        record("refused", f"refusal: {category}", *billed)
        raise GenerateError(f"model refused to respond: {category}")

    if response.stop_reason == "max_tokens":
        record("error", f"truncated: output hit max_tokens ({MAX_TOKENS})", *billed)
        raise GenerateError(
            f"model output was truncated at max_tokens ({MAX_TOKENS}); "
            "the document is too long for one call"
        )

    text_block = next((b for b in response.content if isinstance(b, TextBlock)), None)
    if text_block is None:
        record("error", "no text block in response", *billed)
        raise GenerateError("model response had no text content block")

    try:
        parsed = CvDocumentOutput.model_validate(json.loads(text_block.text))
    except (json.JSONDecodeError, ValueError) as e:
        record("error", f"parse_error: {e}", *billed)
        raise GenerateError(f"could not parse structured output: {e}") from e

    record("ok", None, *billed)

    document = assemble_document(skeleton, parsed, name=name)
    lines = [line for line in document.generated_lines() if line.origin == "generated"]
    if not lines:
        # Nothing was written, so there is nothing to check -- and no second
        # call to pay for. The empty CV is still returned: the skeleton is real.
        return GeneratedCv(document=document, gate_result={"sentences": []}, trace_id=ctx.trace_id)

    # The claim gate runs automatically on generated text (CLAUDE.md's decisions
    # log). Its own `runs` row shares ctx.trace_id. A GateError propagates: the
    # draft call's row has already been written.
    gate_output = check_text(ctx, grounding_repo, run_repo, gate_text(lines))
    apply_gate_output(lines, gate_output)
    return GeneratedCv(
        document=document,
        gate_result=gate_output.model_dump(mode="json"),
        trace_id=ctx.trace_id,
    )
