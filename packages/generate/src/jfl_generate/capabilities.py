"""Grouping confirmed facts into capabilities -- one cheap model call.

Why a model at all. A capability is "FX pricing platforms" or "hiring
engineering managers": it spans several roles and several confirmed facts.
Seeding one row per role gets employers, not capabilities; seeding one row per
fact gets dozens of near-duplicates out of thirty-three overlapping CVs.
Grouping is the judgement in the middle, and it is the only part of this worth
a call.

Modelled closely on `jfl_generate.titles.suggest_titles`: same client
construction, same exception ladder, exactly one `runs` row on every path --
success, API error, refusal alike. **Always `claude-haiku-4-5`, never
`ctx.model`**, for the same reason: this is a classification-shaped task, not a
product-model one, and CLAUDE.md's 2026-09-05 decision keeps the cheaper model
selectable per call site.

Three rules are enforced **in code**, not only asked for in the prompt, because
a prompt cannot promise anything:

  * **an id the model did not receive is dropped.** Facts go over the wire as
    short ids ("f1"), not uuids, and anything that is not one of the ids this
    call sent never reaches storage. A capability left with no valid ids is
    dropped with them -- this path exists to propose *evidenced* rows;
  * **a fact belongs to at most one capability.** The first capability to claim
    an id keeps it; later ones lose it. Two rows citing one span would have the
    claim gate read one fact as two pieces of evidence;
  * **a label that is one of the role labels we sent is rejected.** That rule
    is definitional rather than statistical -- we know exactly which role labels
    went into the prompt, so "this label is a role title" is a lookup, not an
    estimate. CLAUDE.md's 2026-09-01 entry on the deleted heuristics is the
    reason there is no list of seniority words here: unigram overlap cannot tell
    "hiring engineering managers" (a capability) from "Engineering Manager" (a
    role), and guessing at it would flag the good one.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." Fixed control
flow: facts in, labels out, one call.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

import anthropic
from anthropic.types import TextBlock
from jfl_core.context import RequestContext
from jfl_core.models import CandidateFact, ProposedCapability, RunRecord
from jfl_core.profile import capability_key
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import CAPABILITY_CLUSTER_OUTPUT_SCHEMA, build_capability_cluster_prompt
from jfl_generate.schema import CapabilityClusterOutput, ClusteredCapabilityItem

# Not `jfl_gate.pricing.MODEL` and not `ctx.model` -- see the module docstring.
MODEL = "claude-haiku-4-5"

# A hundred-odd short facts in, twenty-odd labels and their ids out. Generous
# for the output this asks for, and a truncation is recorded as an error rather
# than parsed as a short answer.
MAX_TOKENS = 4096

# A capability label longer than this is dropped, never truncated -- a cut label
# is a different, wrong label, and the whole claim of this project is measuring
# distance from what was actually said. The prompt asks for "a few words".
MAX_LABEL_CHARS = 60

# Comfortably above what the prompt asks for, and a ceiling on what a
# misbehaving response could hand back to the screen.
MAX_CAPABILITIES = 25

Outcome = Literal["ok", "error", "refused", "skipped"]


def fact_wire_id(index: int) -> str:
    """The id one fact travels under, 1-based. Short and non-uuid on purpose --
    see `CAPABILITY_CLUSTER_OUTPUT_SCHEMA`'s comment.
    """
    return f"f{index + 1}"


def build_facts_message(facts: Sequence[CandidateFact]) -> str:
    """The volatile half: every fact, with its id and the role it was recorded
    under.

    The role is included because it is what the model needs to group *across*
    roles, and because it is what makes "this label is just the role again"
    visible to a reader of the prompt. The fact text is the user's own
    confirmed words (`CandidateFact.corpus_text`), verbatim and never
    summarised.
    """
    lines = ["Here are the facts this person has confirmed.", ""]
    for index, fact in enumerate(facts):
        role = fact.role_label.strip() or "role not stated"
        lines.append(f"{fact_wire_id(index)} [{role}] {fact.corpus_text}")
    lines.extend(["", "Group them into capabilities now."])
    return "\n".join(lines)


def cluster_capabilities(
    ctx: RequestContext,
    run_repo: RunRepository,
    *,
    facts: Sequence[CandidateFact],
    now: datetime,
) -> list[ProposedCapability]:
    """Capability proposals over this user's confirmed facts.

    Always writes exactly one `runs` row -- on success, on an API error, and on
    a refusal alike -- before returning or raising.

    `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision: every model
    call is told what time it is), not read here, so one handler attempt and
    its test share one timestamp.

    Caller's job to send only confirmed facts and to bound how many
    (`jfl_core.profile.facts_to_cluster`). An empty list never reaches the API:
    there is nothing to group, and a call that would return nothing is not one
    worth charging the user for.
    """
    if not facts:
        raise GenerateError("no confirmed facts to group")

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
            system=build_capability_cluster_prompt(max_capabilities=MAX_CAPABILITIES, now=now),
            messages=[{"role": "user", "content": build_facts_message(facts)}],
            output_config={
                "format": {"type": "json_schema", "schema": CAPABILITY_CLUSTER_OUTPUT_SCHEMA}
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
                stage="cluster_capabilities",
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
        parsed = CapabilityClusterOutput.model_validate(json.loads(text_block.text))
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
    return sanitise(parsed.capabilities, facts)


def _clean_label(raw: str) -> str:
    """Trimmed, whitespace collapsed, commas turned into spaces.

    A comma in a capability label reads as two capabilities wherever a list of
    them is rendered or typed back in, the same hazard
    `jfl_generate.titles._sanitise` handles for filter phrases. Replacing
    rather than dropping the row keeps the grouping the model actually found;
    the user can rename it, and their words then win permanently.
    """
    return " ".join(raw.replace(",", " ").split())


def _role_label_keys(facts: Sequence[CandidateFact]) -> set[str]:
    """Every role label we sent, folded -- plus each of its ` -- `-separated
    parts, because role labels are recorded as "Employer -- Title" and both
    halves are equally not a capability.

    Definitional, not statistical: this is the set of strings we told the model
    about, so rejecting a label that is one of them is a lookup rather than a
    guess about what job titles look like.
    """
    keys: set[str] = set()
    for fact in facts:
        label = fact.role_label.strip()
        if not label:
            continue
        keys.add(capability_key(label))
        for part in label.replace("--", "|").replace(",", "|").split("|"):
            cleaned = part.strip()
            if cleaned:
                keys.add(capability_key(cleaned))
    return keys


def sanitise(
    raw: Sequence[ClusteredCapabilityItem], facts: Sequence[CandidateFact]
) -> list[ProposedCapability]:
    """The model's answer, made safe to store. See the module docstring for the
    three rules; this is where all three happen.

    Exported rather than private because it is the interesting half of this
    module and the tests aim straight at it -- a response is easy to fabricate,
    a call is not.
    """
    by_wire_id = {fact_wire_id(i): fact for i, fact in enumerate(facts)}
    role_keys = _role_label_keys(facts)

    claimed: set[uuid.UUID] = set()
    seen_labels: set[str] = set()
    proposals: list[ProposedCapability] = []

    for item in raw:
        label = _clean_label(item.label)
        if not label or len(label) > MAX_LABEL_CHARS:
            continue
        key = capability_key(label)
        # A role is not a capability, and neither is an employer. Both are
        # strings we sent, so this is exact rather than inferred.
        if key in role_keys or key in seen_labels:
            continue

        fact_ids: list[uuid.UUID] = []
        span_ids: list[uuid.UUID] = []
        for wire_id in item.fact_ids:
            fact = by_wire_id.get(wire_id.strip())
            # An id we never sent -- an invention, or a mangled one. Dropped,
            # never resolved to "the nearest fact".
            if fact is None or fact.span_id is None:
                continue
            if fact.id in claimed or fact.id in fact_ids:
                continue
            fact_ids.append(fact.id)
            span_ids.append(fact.span_id)

        # No valid evidence left: the row would be a claim with nothing behind
        # it, which is not what this path is for. The facts it named (if any)
        # stay unclaimed and are reported as unplaced.
        if not fact_ids:
            continue

        claimed.update(fact_ids)
        seen_labels.add(key)
        proposals.append(ProposedCapability(label=label, fact_ids=fact_ids, span_ids=span_ids))
        if len(proposals) >= MAX_CAPABILITIES:
            break
    return proposals


def unplaced_facts(
    facts: Sequence[CandidateFact], proposals: Sequence[ProposedCapability]
) -> list[uuid.UUID]:
    """The ids of facts no proposal covers, in the order they were sent.

    Recorded on the run and shown on the screen: a confirmed fact the model
    could not place is still the user's fact, and losing track of it silently
    is the failure mode this project is against.
    """
    placed = {fact_id for proposal in proposals for fact_id in proposal.fact_ids}
    return [fact.id for fact in facts if fact.id not in placed]
