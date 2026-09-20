"""Display-only helpers for B5's drafting screen: draft-kind labels, the
NOT-CHECKED rendering rule for a gate sentence, cost formatting, and plain
English for a failed `generate_cv_draft` / `generate_coverage` task.

Nothing here touches storage or the network -- the same split
`jfl_web.jobads` and `jfl_web.boards` draw for their own screens.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from jfl_core.models import DraftKind

# What a "Generate a ..." button says, and what a draft's kind reads as in its
# own history list.
KIND_LABELS: dict[DraftKind, str] = {
    "cv_bullets": "CV",
    "cover_letter": "cover letter",
}


def kind_label(kind: DraftKind) -> str:
    return KIND_LABELS.get(kind, kind)


# What the "Generate ..." button itself says -- not composed from
# `KIND_LABELS` because "a CV" and "a cover letter" is a fact about English
# indefinite articles, not about this list, and hard-coding two strings reads
# more honestly than a rule that happens to work for both.
GENERATE_LABELS: dict[DraftKind, str] = {
    "cv_bullets": "Generate a CV",
    "cover_letter": "Generate a cover letter",
}


def generate_label(kind: DraftKind) -> str:
    return GENERATE_LABELS.get(kind, f"Generate a {kind_label(kind)}")


def sentence_label(sentence: dict[str, Any]) -> str:
    """SUPPORTED / REVIEW / UNSUPPORTED, or NOT CHECKED.

    The rule from CLAUDE.md ("Framing renders as NOT CHECKED, never as
    supported") and from `jfl_cli.main._print_sentence`'s docstring, ported to
    the template layer verbatim: a document title (`kind="title"`) was never
    sent to the model, and framing is forced to `verdict="supported"` by the
    prompt without ever being compared against the corpus, so displaying
    either as SUPPORTED asserts a verification that did not happen. `sentence`
    is one entry of `Draft.gate_result["sentences"]` -- a plain dict, since
    `jfl_core` holds no dependency on `jfl_gate`'s typed `GateOutput` (see
    CLAUDE.md's architectural constraints) -- so this reads it the same way
    the CLI's `_print_sentence` reads a validated `SentenceResult`, field for
    field.
    """
    if sentence.get("kind") == "claim" and sentence.get("verdict") is not None:
        verdict = str(sentence["verdict"])
        return verdict.upper()
    return "NOT CHECKED"


def sentence_style(sentence: dict[str, Any]) -> str:
    """A CSS class for the verdict badge -- mirrors `sentence_label`'s rule so
    the two can never disagree about what counts as "not checked".
    """
    if sentence.get("kind") == "claim" and sentence.get("verdict") is not None:
        return f"verdict-{sentence['verdict']}"
    return "verdict-not-checked"


def usd(value: Decimal | None) -> str:
    """`$0.4123`, or an em dash when nothing was billed yet (a task still
    running, or every attempt failed before usage was returned).
    """
    if value is None:
        return "—"
    return f"${value:.4f}"


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    """What to tell the user about a failed `generate_cv_draft` or
    `generate_coverage` task, and where to send them to fix it -- the same
    shape `jfl_web.jobads.ExtractionFailure` gives extraction's failures.
    """

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_SHARED_FAILURES: dict[str, GenerationFailure] = {
    "no_api_key": GenerationFailure(
        "This needs your own Anthropic API key -- it is billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": GenerationFailure(
        "Anthropic rejected the API key stored here. Replace it and try again.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "model_refused": GenerationFailure(
        "The model declined to respond. Trying again is worth a go."
    ),
    "model_error": GenerationFailure("That failed. Trying again is worth a go."),
    "credential_unreadable": GenerationFailure(
        "Your stored API key could not be unlocked on the server. Setting it again will fix it.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
}

_COVERAGE_FAILURES: dict[str, GenerationFailure] = {
    **_SHARED_FAILURES,
    "no_job": GenerationFailure("There is no job linked to this application yet."),
    "no_requirements": GenerationFailure(
        "The ad has not been read yet, so there are no requirements to check coverage against."
    ),
}

_DRAFT_FAILURES: dict[str, GenerationFailure] = {
    **_SHARED_FAILURES,
    "no_job": GenerationFailure("There is no job linked to this application yet."),
    "no_requirements": GenerationFailure(
        "The ad has not been read yet, so there is nothing to draft against."
    ),
    "no_coverage": GenerationFailure(
        "Corpus coverage has not been checked for this job yet -- run that first."
    ),
    "ad_too_long": GenerationFailure(
        "That came out too long for one call. This is a bug worth reporting, "
        "not something retrying fixes."
    ),
}

_UNKNOWN = GenerationFailure("That failed. Trying again is worth a go.")

# The exact suffix `jfl_worker.handlers.coverage_generation._permanent` and
# `jfl_worker.handlers.draft_generation._permanent` build `PermanentTaskError`
# messages from -- the coupling this module has to those two, pinned by
# `tests/test_drafts_display.py` the same way `jfl_web.jobads` is coupled to
# `jfl_worker.handlers.extraction`'s wording.
#
# `tasks.last_error` never carries a `PermanentTaskError`'s message alone --
# `jfl_worker.runner.Worker._fail` always wraps it as
# f"{type(exc).__name__}: {exc}", so the stored text is
# "PermanentTaskError: coverage generation failed permanently: <code>". Both
# functions below look for the marker anywhere in the string rather than at
# its start, so that wrapping -- present today and liable to gain more detail
# later -- can never break the parse.
_COVERAGE_MARKER = "coverage generation failed permanently: "
_DRAFT_MARKER = "draft generation failed permanently: "


def coverage_failure(last_error: str | None) -> GenerationFailure:
    """`tasks.last_error` for a failed `generate_coverage` task, turned into a
    sentence. `last_error` is safe to parse here (never to show verbatim):
    every permanent failure in that handler is built only from the closed set
    of codes below, never from a formatted exception or a credential -- see
    its module docstring.
    """
    if not last_error or _COVERAGE_MARKER not in last_error:
        return _UNKNOWN
    code = last_error.rsplit(_COVERAGE_MARKER, 1)[-1].strip()
    return _COVERAGE_FAILURES.get(code, _UNKNOWN)


def draft_failure(last_error: str | None) -> GenerationFailure:
    """The same reading as `coverage_failure`, for `generate_cv_draft`."""
    if not last_error or _DRAFT_MARKER not in last_error:
        return _UNKNOWN
    code = last_error.rsplit(_DRAFT_MARKER, 1)[-1].strip()
    return _DRAFT_FAILURES.get(code, _UNKNOWN)
