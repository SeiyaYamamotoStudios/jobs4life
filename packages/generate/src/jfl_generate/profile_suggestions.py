"""Reading plain profile *settings* off a user's own CVs -- one cheap call.

An uploaded CV states two different kinds of thing, and this project has to
treat them differently.

It makes **claims about the world** -- "rebuilt the FX pricing platform", "grew
the team to fourteen". Those go through `jfl_generate.cv_facts`, arrive as
`proposed` candidate facts, and become corpus only one confirmation at a time.
That path is what keeps the over-claim measurement honest and nothing here
weakens it: this module writes no fact, cites no span and touches no corpus.

It also states plain **settings**: which disciplines the person practises, where
they have worked, what level they have been operating at. Those are not claims
to be measured against a corpus -- they are the sort of thing the user would
otherwise type into `/profile` by hand -- so a model can read them off and the
user can accept each with a click.

**What may be proposed, exhaustively**: `discipline`, `not_discipline`,
`location`, `level`. The exclusions are the complement of that whitelist rather
than a list of rules, so they hold however the model answers.

**Deliberately never proposed: comp, contract type, right to work, notice, a
categorical no -- or a workplace preference.** A CV records what someone has
done; it does not state what they now require. A constraint invented from one
would be read by scoring as *the user's own requirement*, which is the tool
asserting something its subject never said. `level` is proposed only as an
**observation** about the past ("has been operating at engineering-manager
level"), never as a floor: `level_floor` is a choice the user makes, and
accepting the observation still asks them for their own words and a stance.

Modelled closely on `jfl_generate.capabilities`: same client construction, same
exception ladder, exactly one `runs` row on every path -- success, API error,
refusal alike. **Always `claude-haiku-4-5`, never `ctx.model`**, for the same
reason: this is a reading-off task, not a product-model one, and CLAUDE.md's
2026-09-05 decision keeps the cheaper model selectable per call site.

Two rules are enforced **in code**, not merely asked for in the prompt, because
a prompt cannot promise anything:

  * **a kind outside the whitelist is dropped.** That is what makes the
    exclusions above structural;
  * **a source line that appears in none of the CVs we sent is a fabrication,
    and the whole suggestion goes with it.** Definitional rather than
    statistical: we know exactly what text went into the prompt, so "is this
    quote in it" is a lookup. It is also what stops a `not_discipline` being
    proposed from the *absence* of something, since absence has no line to
    quote.

Not agentic -- see CLAUDE.md, "What is agentic, and what is not." Fixed control
flow: CVs in, suggestions out, one call.
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
from jfl_core.ids import fold, setting_key
from jfl_core.models import (
    PROFILE_SUGGESTION_KINDS,
    ProfileSuggestionKind,
    ProposedSetting,
    RunRecord,
)
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd

from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    PROFILE_SUGGESTIONS_OUTPUT_SCHEMA,
    build_profile_suggestions_prompt,
)
from jfl_generate.schema import ProfileSuggestionsOutput, ProposedSettingItem

# Not `jfl_gate.pricing.MODEL` and not `ctx.model` -- see the module docstring.
MODEL = "claude-haiku-4-5"

# Several CVs in, a couple of dozen short objects out. Generous for what this
# asks for, and a truncation is recorded as an error rather than parsed as a
# short answer.
MAX_TOKENS = 4096

# How many stored CVs one call reads, newest first, and how much text they may
# come to between them. The owner has thirty-three CVs and they overlap heavily;
# reading all of them would cost several times as much to say the same thing.
# A cap, never a silent truncation of a document: a CV that does not fit is left
# out whole, and the screen says how many were read.
MAX_CVS = 8
MAX_CV_CHARS = 200_000

# Comfortably above what a real CV set yields, and a ceiling on what a
# misbehaving response could hand back to the screen.
MAX_SUGGESTIONS = 30

# A value longer than this is dropped, never truncated -- a cut discipline or
# place is a different one, and this is about to be offered to its subject as a
# setting on their own profile.
MAX_VALUE_CHARS = 200
# A level observation is a short sentence rather than a label.
MAX_LEVEL_CHARS = 300
MAX_SOURCE_LINE_CHARS = 1_000

Outcome = Literal["ok", "error", "refused", "skipped"]


def build_cvs_message(cv_texts: Sequence[str]) -> str:
    """The volatile half: the CVs themselves, verbatim and numbered.

    Verbatim because the source line a suggestion quotes is checked against
    exactly this text, and because a CV we tidied is a CV the user did not
    write. Nothing from the corpus goes in here -- this call reads what the
    person sent out, never what they have confirmed as true.
    """
    parts = ["Here are this person's CVs.", ""]
    for index, text in enumerate(cv_texts, start=1):
        parts.extend([f"--- CV {index} ---", text, ""])
    parts.append("Read off the settings now.")
    return "\n".join(parts)


def select_cv_texts(texts: Sequence[str]) -> list[str]:
    """Which CVs one call is given, in the caller's order (newest first).

    A cap, not a truncation: a CV that would take the batch past `MAX_CV_CHARS`
    is left out whole rather than cut in half, because half a CV quotes lines
    that are not in it and reads as a document its author never wrote.
    """
    chosen: list[str] = []
    budget = MAX_CV_CHARS
    for text in texts:
        if not text.strip():
            continue
        if len(chosen) >= MAX_CVS:
            break
        if len(text) > budget:
            continue
        chosen.append(text)
        budget -= len(text)
    return chosen


def _clean(value: str, limit: int) -> str | None:
    """Trim and collapse whitespace; None if empty or past `limit`."""
    text = " ".join(value.split())
    if not text or len(text) > limit:
        return None
    return text


def _value_limit(kind: ProfileSuggestionKind) -> int:
    return MAX_LEVEL_CHARS if kind == "level" else MAX_VALUE_CHARS


def to_proposals(
    items: Sequence[ProposedSettingItem],
    *,
    cv_texts: Sequence[str],
    answered_keys: Sequence[str] = (),
) -> list[ProposedSetting]:
    """The model's answer, made safe to store. See the module docstring for the
    two guarantees; this is where both happen.

    Locations are collapsed into **one** proposal carrying the places in the
    order the model returned them -- the profile's location constraint is a
    single ordered list (SEEK's shape), so accepting it once is what the user
    actually wants to do. Every other kind is one proposal each. At most one
    `level` observation survives, because the profile holds one level setting
    and two observations would be a choice nobody asked for.

    `answered_keys` are suggestions this user has already accepted or rejected,
    from any earlier run. They are dropped here so that "no" sticks across runs
    -- the key is content-derived, so the same suggestion from the same CV, or
    from a later one, folds to the same key.

    Exported rather than private because it is the interesting half of this
    module and the tests aim straight at it -- a response is easy to fabricate,
    a call is not.
    """
    haystack = fold("\n".join(cv_texts))
    already = set(answered_keys)

    places: list[str] = []
    place_lines: list[str] = []
    seen_places: set[str] = set()
    singles: list[ProposedSetting] = []
    seen_keys: set[str] = set()
    have_level = False

    for item in items:
        kind = item.kind.strip().casefold()
        # The whitelist, and therefore the exclusions. A kind we did not ask for
        # -- comp_floor, contract, right_to_work, notice, categorical_no,
        # workplace, or anything else -- never becomes a proposal.
        if kind not in PROFILE_SUGGESTION_KINDS:
            continue
        # `in` over the tuple has already narrowed this to the Literal.
        typed: ProfileSuggestionKind = kind

        value = _clean(item.value, _value_limit(typed))
        source_line = _clean(item.source_line, MAX_SOURCE_LINE_CHARS)
        if value is None or source_line is None:
            continue
        # A quote that is in none of the CVs we sent is an invention, and a
        # suggestion whose evidence is invented is dropped with it. This is also
        # what stops a "not this" being read off the absence of something.
        if fold(source_line) not in haystack:
            continue

        if typed == "location":
            if fold(value) in seen_places:
                continue
            seen_places.add(fold(value))
            places.append(value)
            place_lines.append(source_line)
            continue

        if typed == "level":
            if have_level:
                continue
            have_level = True

        key = setting_key(typed, [value])
        if key in already or key in seen_keys:
            continue
        seen_keys.add(key)
        singles.append(
            ProposedSetting(kind=typed, key=key, values=[value], source_lines=[source_line])
        )

    proposals: list[ProposedSetting] = []
    if places:
        key = setting_key("location", places)
        if key not in already:
            proposals.append(
                ProposedSetting(kind="location", key=key, values=places, source_lines=place_lines)
            )
    proposals.extend(singles)
    return proposals[:MAX_SUGGESTIONS]


def suggest_profile_settings(
    ctx: RequestContext,
    run_repo: RunRepository,
    *,
    cv_texts: Sequence[str],
    now: datetime,
) -> list[ProposedSettingItem]:
    """Read this user's CVs and propose profile settings from them. Always
    writes exactly one `runs` row -- on success, on an API error, and on a
    refusal alike -- before returning or raising.

    `now` is the caller's clock (CLAUDE.md's 2026-09-07 decision: every model
    call is told what time it is), not read here, so one handler attempt and its
    test share one timestamp.

    Caller's job to send only this user's own CVs and to bound how many
    (`select_cv_texts`). An empty list never reaches the API: there is nothing
    to read, and a call that would return nothing is not one worth charging the
    user for.
    """
    if not [text for text in cv_texts if text.strip()]:
        raise GenerateError("no CV text to read")

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
            system=build_profile_suggestions_prompt(max_suggestions=MAX_SUGGESTIONS, now=now),
            messages=[{"role": "user", "content": build_cvs_message(cv_texts)}],
            output_config={
                "format": {"type": "json_schema", "schema": PROFILE_SUGGESTIONS_OUTPUT_SCHEMA}
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
                stage="suggest_profile_settings",
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
        parsed = ProfileSuggestionsOutput.model_validate(json.loads(text_block.text))
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
    return parsed.suggestions
