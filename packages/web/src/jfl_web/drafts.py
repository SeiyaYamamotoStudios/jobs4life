"""Display-only helpers for B5's drafting screen: draft-kind labels, the
NOT-CHECKED rendering rule for a checked sentence, the headline count and each
flagged sentence's next action, the three steps to a CV and what each costs,
and plain English for a failed step.

The screen's words are the reader's, not the implementation's: "your
confirmed facts", "check this", "not checked" -- never the names the code uses
for the same things. `packages/web/tests/test_wording.py` holds that line.

Nothing here touches storage or the network -- the same split
`jfl_web.jobads` and `jfl_web.boards` draw for their own screens.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from jfl_core.models import DraftKind

from jfl_web.jobads import extraction_failure

# What a "Generate a ..." button says, and what a draft's kind reads as in its
# own history list.
KIND_LABELS: dict[DraftKind, str] = {
    "cv_bullets": "CV",
    "cover_letter": "cover letter",
}


def kind_label(kind: DraftKind) -> str:
    return KIND_LABELS.get(kind, kind)


# What the button that writes one says. Not composed from `KIND_LABELS`
# because "the CV" and "a cover letter" is a fact about English articles, not
# about this list, and two hard-coded strings read more honestly than a rule
# that happens to work for both.
GENERATE_LABELS: dict[DraftKind, str] = {
    "cv_bullets": "Write the CV",
    "cover_letter": "Write a cover letter",
}


def generate_label(kind: DraftKind) -> str:
    return GENERATE_LABELS.get(kind, f"Write a {kind_label(kind)}")


# --------------------------------------------------------------------------
# One sentence's verdict, in plain words
# --------------------------------------------------------------------------

VerdictKey = Literal["supported", "review", "unsupported", "not_checked"]


def verdict_key(sentence: dict[str, Any]) -> VerdictKey:
    """Which of the four marks a sentence gets.

    The rule from CLAUDE.md ("Framing renders as NOT CHECKED, never as
    supported") and from `jfl_cli.main._print_sentence`'s docstring, ported to
    the template layer: a document title (`kind="title"`) was never sent to the
    model, and framing is forced to `verdict="supported"` by the prompt without
    ever being compared against the user's facts, so displaying either as
    supported asserts a verification that did not happen. Only a `claim` with a
    verdict gets one of the three real marks. `sentence` is one entry of
    `Draft.gate_result["sentences"]` -- a plain dict, since `jfl_core` holds no
    dependency on `jfl_gate`'s typed output.

    This is the one place that rule is expressed for this screen:
    `sentence_label`, `sentence_style` and `check_summary` all go through it.
    """
    if sentence.get("kind") == "claim":
        verdict = sentence.get("verdict")
        if verdict == "supported":
            return "supported"
        if verdict == "review":
            return "review"
        if verdict == "unsupported":
            return "unsupported"
    return "not_checked"


# The mark itself, as the reader sees it beside a sentence.
VERDICT_WORDS: dict[VerdictKey, str] = {
    "supported": "Supported",
    "review": "Check this",
    "unsupported": "Not supported",
    "not_checked": "Not checked",
}

# What each mark means, said where the reader meets it.
VERDICT_MEANINGS: dict[VerdictKey, str] = {
    "supported": "Traces to a fact you have confirmed.",
    "review": "Only partly traces to your facts, or says more than they do.",
    "unsupported": "Nothing you have confirmed backs this.",
    "not_checked": (
        "Framing or a heading. It is never compared with your facts, so it is "
        "never marked as supported."
    ),
}


def sentence_label(sentence: dict[str, Any]) -> str:
    """ "Supported" / "Check this" / "Not supported" / "Not checked"."""
    return VERDICT_WORDS[verdict_key(sentence)]


def sentence_style(sentence: dict[str, Any]) -> str:
    """A CSS class for the mark -- the same rule as `sentence_label`, so the
    two can never disagree about what counts as "not checked".
    """
    return "verdict-" + verdict_key(sentence).replace("_", "-")


def sentence_meaning(sentence: dict[str, Any]) -> str:
    return VERDICT_MEANINGS[verdict_key(sentence)]


@dataclass(frozen=True, slots=True)
class NextAction:
    """What to do about one flagged sentence.

    A sentence the facts do not back is one of two things: an over-claim to
    reword, or a true thing missing from the record. The screen offers both
    and does not guess which -- only the author knows. `add_fact` is False
    only where the facts actively say something different (the CLAUDE.md
    taxonomy's `adjacency_substitution`): there the record is not silent, it
    disagrees, and "add the fact" would be the wrong advice.
    """

    why: str
    reword: str
    add_fact: bool = True


# What each drift label means, in words a reader has -- the taxonomy in
# CLAUDE.md, restated for the person holding the CV rather than the eval.
_WHY: dict[str, str] = {
    "invented_quantity": "This number does not appear in anything you have confirmed.",
    "adjacency_substitution": (
        "Your facts describe your part in this differently -- for example, "
        "reviewing rather than building."
    ),
    "scope_inflation": (
        "Your facts do not state the scope this implies -- the squad, team or department."
    ),
    "ownership_inflation": (
        "Your facts do not say what owning this involved -- decisions, budget, on-call, people."
    ),
    "outcome_attribution": "Your facts do not link your work to this outcome.",
    "strategy_scope": "Your facts do not say whose strategy this was -- the team's, the org's.",
    "causality": "Your facts do not show your part in making this happen.",
}

_REWORD = "Reword it to what you can show, or cut it, before you send this."
_REWORD_TO_MATCH = "Reword it to match what your facts say before you send this."


def next_action(sentence: dict[str, Any]) -> NextAction | None:
    """None for a sentence that needs nothing from the reader."""
    if verdict_key(sentence) not in ("review", "unsupported"):
        return None
    label = str(sentence.get("drift_label") or "")
    if label == "adjacency_substitution":
        return NextAction(why=_WHY[label], reword=_REWORD_TO_MATCH, add_fact=False)
    return NextAction(why=_WHY.get(label, ""), reword=_REWORD)


@dataclass(frozen=True, slots=True)
class DraftCheck:
    """A draft's sentence-by-sentence check, counted and grouped for the page.

    `headline` is the one line the page leads with -- see `headline`.
    """

    claims: int
    supported: int
    review: int
    unsupported: int
    not_checked: int
    headline: str
    flagged: tuple[dict[str, Any], ...]
    backed: tuple[dict[str, Any], ...]
    unchecked: tuple[dict[str, Any], ...]


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def headline(supported: int, review: int, unsupported: int) -> str:
    """ "12 of 18 claims trace to your confirmed facts; 3 need checking; 1
    isn't supported." -- the parts that are zero are left out, and a claim
    count of zero says so rather than printing "0 of 0".
    """
    claims = supported + review + unsupported
    if claims == 0:
        return "Nothing in this draft makes a claim that could be checked against your facts."
    if supported == claims:
        if claims == 1:
            return "Its one claim traces to your confirmed facts."
        return f"All {claims} claims trace to your confirmed facts."
    verb = _plural(supported, "traces", "trace")
    noun = _plural(claims, "claim", "claims")
    parts = [f"{supported} of {claims} {noun} {verb} to your confirmed facts"]
    if review:
        parts.append(f"{review} {_plural(review, 'needs', 'need')} checking")
    if unsupported:
        parts.append(f"{unsupported} {_plural(unsupported, "isn't", "aren't")} supported")
    return "; ".join(parts) + "."


def check_summary(gate_result: dict[str, Any] | None) -> DraftCheck:
    sentences = [s for s in (gate_result or {}).get("sentences", []) if isinstance(s, dict)]
    keys = [verdict_key(s) for s in sentences]
    supported = keys.count("supported")
    review = keys.count("review")
    unsupported = keys.count("unsupported")
    return DraftCheck(
        claims=supported + review + unsupported,
        supported=supported,
        review=review,
        unsupported=unsupported,
        not_checked=keys.count("not_checked"),
        headline=headline(supported, review, unsupported),
        # Worst first: "not supported" before "check this", each in the order
        # it appears in the draft.
        flagged=tuple(
            [s for s, k in zip(sentences, keys, strict=True) if k == "unsupported"]
            + [s for s, k in zip(sentences, keys, strict=True) if k == "review"]
        ),
        backed=tuple(s for s, k in zip(sentences, keys, strict=True) if k == "supported"),
        unchecked=tuple(s for s, k in zip(sentences, keys, strict=True) if k == "not_checked"),
    )


def download_name(kind: DraftKind, title: str | None) -> str:
    """`cv-acme-senior-engineer.txt` -- a file name a person would keep."""
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    stem = "-".join(["cv" if kind == "cv_bullets" else "cover-letter", *words])[:80]
    return f"{stem.rstrip('-')}.txt"


# --------------------------------------------------------------------------
# The steps to a CV: read the ad, check it against your facts, write it
# --------------------------------------------------------------------------

StepKey = Literal["ad", "check", "write"]
StepState = Literal["done", "current", "waiting", "running", "failed", "blocked"]

STEP_LABELS: dict[StepKey, str] = {
    "ad": "Read the ad",
    "check": "Check it against your facts",
    "write": "Write the CV",
}

# The task kind that performs each step -- the three handlers in `jfl_worker`.
STEP_KINDS: dict[str, StepKey] = {
    "extract_job_ad": "ad",
    "generate_coverage": "check",
    "generate_cv_draft": "write",
}

# What each step costs on the user's own key, in dollars, low and high.
# Measured, not taken from the rate card: ~$0.012 per ad read and ~$0.157 per
# check are W1's per-call numbers (docs/PLAN-september-demo.md), and a draft
# plus its automatic sentence check came in at $0.35-0.64 (CLAUDE.md,
# 2026-09-07). The check and the draft grow with how many facts you have
# confirmed, which is why the page says "about".
STEP_COSTS: dict[StepKey, tuple[Decimal, Decimal]] = {
    "ad": (Decimal("0.01"), Decimal("0.01")),
    "check": (Decimal("0.16"), Decimal("0.16")),
    "write": (Decimal("0.35"), Decimal("0.65")),
}

_STEP_ORDER: tuple[StepKey, ...] = ("ad", "check", "write")


def cost_range(low: Decimal, high: Decimal) -> str:
    if low == high:
        return f"about ${low:.2f}"
    return f"about ${low:.2f}–{high:.2f}"


@dataclass(frozen=True, slots=True)
class Step:
    key: StepKey
    label: str
    state: StepState
    cost: str
    detail: str = ""
    fix_url: str | None = None
    fix_label: str | None = None


@dataclass(frozen=True, slots=True)
class ChainTask:
    """One task in a button press's chain, as far as the steps need it."""

    kind: str
    status: str
    failure: GenerationFailure | None = None
    # The task's payload still names a step after this one -- so a task that
    # succeeded with nothing queued after it yet is a chain still going.
    has_next: bool = False


@dataclass(frozen=True, slots=True)
class CvPlan:
    """Everything the steps panel says.

    `can_start` -- the button is live. `in_progress` -- something is running,
    so the panel polls. `finished` -- the watched press has stopped, either
    way, so a poll should reload the page to show the result.
    """

    steps: tuple[Step, ...]
    can_start: bool
    in_progress: bool
    finished: bool
    cost_line: str
    blocked_message: str = ""

    @property
    def current(self) -> Step | None:
        return next((s for s in self.steps if s.state != "done"), None)


def plan_steps(
    *,
    has_ad: bool,
    ad_read: bool,
    ad_reading: bool,
    checked: bool,
    chain: Sequence[ChainTask] = (),
    writing: DraftKind = "cv_bullets",
) -> CvPlan:
    """The three steps and where this application is in them.

    `chain` is the watched press, in order -- empty when nothing is running
    and nobody is watching. Pure, so every state is unit-testable.
    """
    by_step = {STEP_KINDS[t.kind]: t for t in chain if t.kind in STEP_KINDS}
    chain_open = bool(chain) and (
        any(t.status in ("pending", "running") for t in chain)
        or (chain[-1].status == "succeeded" and chain[-1].has_next)
    )

    done: dict[StepKey, bool] = {"ad": ad_read, "check": checked, "write": False}
    steps: list[Step] = []
    blocked_message = ""
    for key in _STEP_ORDER:
        low, high = STEP_COSTS[key]
        cost = cost_range(low, high)
        task = by_step.get(key)
        state: StepState
        detail = ""
        fix_url = fix_label = None
        if task is not None and task.status in ("pending", "running"):
            state = "running"
            detail = _RUNNING[key]
        elif task is not None and task.status == "failed":
            state = "failed"
            if task.failure is not None:
                detail = task.failure.message
                fix_url, fix_label = task.failure.fix_url, task.failure.fix_label
        elif done[key] or (task is not None and task.status == "succeeded" and key != "write"):
            # A written CV does not make "write" done: it is the step the
            # button offers again, for another version.
            state = "done"
        elif key == "ad" and not has_ad:
            state = "blocked"
            blocked_message = "There is no job ad stored for this application yet."
        elif key == "ad" and ad_reading:
            state = "running"
            detail = _RUNNING["ad"]
        else:
            state = "waiting"
        label = STEP_LABELS[key]
        if key == "write" and writing != "cv_bullets":
            label = f"Write the {kind_label(writing)}"
        steps.append(Step(key, label, state, cost, detail, fix_url, fix_label))

    # The first step that is not done, and is not already running, failed or
    # blocked, is the one the button starts from.
    in_progress = chain_open or any(s.state == "running" for s in steps)
    if not in_progress:
        for index, step in enumerate(steps):
            if step.state == "done":
                continue
            if step.state == "waiting":
                steps[index] = dataclasses.replace(step, state="current")
            break

    todo = [s.key for s in steps if s.state != "done"]
    can_start = not in_progress and not blocked_message
    low = sum((STEP_COSTS[k][0] for k in todo), Decimal(0))
    high = sum((STEP_COSTS[k][1] for k in todo), Decimal(0))
    if blocked_message or not todo:
        cost_line = ""
    elif todo == ["write"]:
        cost_line = (
            f"Writing it costs {cost_range(low, high)} on your own API key, and every "
            "sentence is checked against your confirmed facts as part of it."
        )
    else:
        verbs = {
            "ad": "read the ad",
            "check": "check it against your confirmed facts",
            "write": f"write the {kind_label(writing)}",
        }
        said = [verbs[k] for k in todo]
        sequence = ", ".join(said[:-1]) + " and then " + said[-1]
        cost_line = (
            f"One press does the lot: it will {sequence} -- {cost_range(low, high)} in "
            "all, on your own API key."
        )
    finished = bool(chain) and not chain_open
    return CvPlan(
        steps=tuple(steps),
        can_start=can_start,
        in_progress=in_progress,
        finished=finished,
        cost_line=cost_line,
        blocked_message=blocked_message,
    )


_RUNNING: dict[StepKey, str] = {
    "ad": "Reading the ad now -- about half a minute.",
    "check": "Checking the job's requirements against your confirmed facts -- about half a minute.",
    "write": "Writing, then checking every sentence against your facts -- about a minute.",
}


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
        "The ad has not been read yet, so there is nothing to check against your facts."
    ),
}

_DRAFT_FAILURES: dict[str, GenerationFailure] = {
    **_SHARED_FAILURES,
    "no_job": GenerationFailure("There is no job linked to this application yet."),
    "no_requirements": GenerationFailure(
        "The ad has not been read yet, so there is nothing to draft against."
    ),
    "no_coverage": GenerationFailure(
        "This job had not been checked against your confirmed facts yet. Press "
        "Write the CV again and it will run that check first."
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


# `jfl_worker.handlers.extraction` raises its permanent failures as
# "extraction failed permanently: <code>" -- the same coupling the two markers
# above have to their handlers.
_EXTRACTION_MARKER = "extraction failed permanently: "


def extraction_step_failure(last_error: str | None) -> GenerationFailure:
    """A failed `extract_job_ad` task in a chain, as the steps panel says it."""
    code: str | None = None
    if last_error and _EXTRACTION_MARKER in last_error:
        code = last_error.rsplit(_EXTRACTION_MARKER, 1)[-1].strip()
    failure = extraction_failure(code)  # type: ignore[arg-type]
    return GenerationFailure(failure.message, failure.fix_url, failure.fix_label)


def step_failure(kind: str, last_error: str | None) -> GenerationFailure:
    if kind == "extract_job_ad":
        return extraction_step_failure(last_error)
    if kind == "generate_coverage":
        return coverage_failure(last_error)
    return draft_failure(last_error)
