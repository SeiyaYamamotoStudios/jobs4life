"""Bulk actions on the applications list beyond archiving: re-scoring a
selection, and retrying everything that has failed. Owner feedback,
2026-09-27, alongside the bulk-archive slice it sits next to.

Pure functions over data the caller already loaded, matching
`jfl_web.archive_rules`'s shape: no SQL, no model call, so what gets selected
and what a bulk action is expected to cost can both be unit-tested without a
database. The routes in `jfl_web.routes.applications` own fetching the data
and writing the enqueue (through the shared helpers there); this module owns
only "which applications does this action touch" and "roughly what will it
cost."

**Re-score skips two kinds of application, not zero.** A re-score already in
flight (`RowScore.state` "scoring" or "retrying") must not be pressed again --
that buys a second charge for the same answer, the same reasoning as the
single Re-score button's own guard against a second press. And an application
whose ad has not been read into at least one requirement cannot be scored at
all: the scoring handler's own `no_requirements` failure is what a call would
come back with, so filtering it out here is honesty, not merely an
optimisation -- it costs the user nothing to learn that in the preview
instead of from a failed run they paid nothing for but still have to read.

**Retry picks the read over the score when both have failed.** An application
whose extraction failed is retried by re-reading the ad; a successful re-read
chains its first score by itself if none is already in flight or done (see
`jfl_worker.handlers.extraction._chain_first_score` and
`jfl_core.storage.scores.PostgresScoreRepository.has_active`), so retrying the
read *and* separately pressing Re-score on the same application would either
double the work or race the chain. `description_unavailable` is the one
extraction failure a retry cannot fix -- there is no ad text to re-read at all
-- so those applications are reported separately, with no action taken on
them, rather than counted toward what will be retried.

**The cost estimate is measured, never the rate card.** It is handed the
user's own recent `runs` costs (`PostgresRunRepository.recent_costs`) and
takes the median of that sample -- median rather than mean, so one unusually
long ad or a big corpus does not swing the number the way it would swing an
average. With no history at all it falls back to a stated rough range,
labelled as a guess rather than a measurement, so it can never be mistaken for
one.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from jfl_core.models import Application

from jfl_web.scores import RowScore

# The one extraction failure a retry cannot fix: there is no ad text stored at
# all (a board fetch whose description never came back), so the fix is the
# paste box, not another attempt at the same call. See `ExtractionErrorCode`'s
# docstring in `jfl_core.models`.
NEEDS_PASTE_CODE = "description_unavailable"

RescoreSkipReason = Literal["already_scoring", "ad_not_read"]

SKIP_REASON_LABELS: dict[RescoreSkipReason, str] = {
    "already_scoring": "already scoring",
    "ad_not_read": "the ad has not been read yet",
}


@dataclass(frozen=True, slots=True)
class RescoreSkip:
    """One application a bulk re-score will not touch, and why -- said in the
    preview so a smaller number than the selection is never a surprise.
    """

    application: Application
    reason: RescoreSkipReason


def partition_rescore(
    applications: Sequence[Application],
    row_scores: Mapping[uuid.UUID, RowScore],
    has_requirements: Mapping[uuid.UUID, bool],
) -> tuple[list[Application], list[RescoreSkip]]:
    """Split a selection into what a bulk re-score will run, and what it will
    skip and why. `has_requirements` is whatever the caller has already
    determined has at least one requirement recorded against it (regardless of
    whether a later re-read of the ad happens to be pending right now) --
    scoring reads `job_requirements` directly and does not care about
    `extraction_status` by itself.
    """
    eligible: list[Application] = []
    skipped: list[RescoreSkip] = []
    for application in applications:
        row = row_scores.get(application.id)
        if row is not None and row.state in ("scoring", "retrying"):
            skipped.append(RescoreSkip(application, "already_scoring"))
            continue
        if not has_requirements.get(application.id, False):
            skipped.append(RescoreSkip(application, "ad_not_read"))
            continue
        eligible.append(application)
    return eligible, skipped


@dataclass(frozen=True, slots=True)
class RetryPlan:
    """What "Retry everything that failed" will do, split by what each
    application needs. The three lists are mutually exclusive -- an
    application appears in exactly one, never more than one action at a time.
    See the module docstring for why a failed read takes priority over a
    failed score on the same application.
    """

    retry_reads: list[Application]
    retry_scores: list[Application]
    needs_paste: list[Application]

    @property
    def count(self) -> int:
        """How many applications will actually be retried -- what the button's
        own label counts. `needs_paste` is never part of it: nothing is
        enqueued for those."""
        return len(self.retry_reads) + len(self.retry_scores)

    @property
    def is_empty(self) -> bool:
        return not (self.retry_reads or self.retry_scores or self.needs_paste)


def plan_retry_failed(
    applications: Sequence[Application], row_scores: Mapping[uuid.UUID, RowScore]
) -> RetryPlan:
    """Classify every application in `applications` (the caller's own live
    list) by what, if anything, retrying it should do.

    A failed extraction is checked first and, when it is not the
    `needs_paste` case, wins outright over a failed score on the same
    application -- re-reading is the fix for both, since a successful re-read
    chains a fresh score by itself when none is active (see
    `jfl_core.storage.scores.PostgresScoreRepository.has_active`). Only when
    the extraction is *not* failed does a failed score get its own retry.
    """
    retry_reads: list[Application] = []
    retry_scores: list[Application] = []
    needs_paste: list[Application] = []
    for application in applications:
        if application.extraction_status == "failed":
            if application.extraction_error_code == NEEDS_PASTE_CODE:
                needs_paste.append(application)
            else:
                retry_reads.append(application)
            continue
        row = row_scores.get(application.id)
        if row is not None and row.state == "failed":
            retry_scores.append(application)
    return RetryPlan(retry_reads=retry_reads, retry_scores=retry_scores, needs_paste=needs_paste)


# --------------------------------------------------------------------------
# Cost estimate -- measured from the user's own `runs` history, never the
# rate card. See the module docstring.
# --------------------------------------------------------------------------

CENTS = Decimal("0.01")

# What is shown when there is no history at all to measure from -- a stated
# rough per-item figure, not a number dressed up as a measurement. The
# template says so in words ("a rough estimate -- no cost history yet"); this
# constant is only the figure itself, kept separate so it is never mistaken
# for a full sentence that happens to be measured.
FALLBACK_RANGE = "$0.10-0.30 each"


def median_cost(costs: Sequence[Decimal]) -> Decimal | None:
    """The middle value of a recent-cost sample, or None with nothing to
    measure from. Median, not mean: one unusually expensive run (a long ad, a
    big corpus) should not swing the estimate the way it would swing an
    average.
    """
    if not costs:
        return None
    ordered = sorted(costs)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """What a bulk action is expected to cost.

    `measured` says whether `total` came from the user's own `runs` history --
    when it is False, `total` is always None and the template shows
    `fallback` instead, so a guess can never be presented as a measurement by
    accident. The wording ("measured from your own recent runs" / "a rough
    estimate") lives in the template, not here, for the same reason every
    other user-facing sentence in this app lives beside the screen rather than
    the logic.
    """

    measured: bool
    total: Decimal | None = None
    fallback: str = FALLBACK_RANGE


def estimate_for_count(costs: Sequence[Decimal], count: int) -> CostEstimate:
    """A single-category estimate: `count` items, each costing about what the
    user's own recent runs in `costs` cost. Used for a bulk re-score, where
    every enqueued item runs the same kind of call.
    """
    if count == 0:
        return CostEstimate(measured=False)
    median = median_cost(costs)
    if median is None:
        return CostEstimate(measured=False)
    return CostEstimate(measured=True, total=(median * count).quantize(CENTS))


def estimate_retry_cost(
    read_costs: Sequence[Decimal],
    score_costs: Sequence[Decimal],
    *,
    reads: int,
    scores: int,
) -> CostEstimate:
    """A two-category estimate for "retry everything that failed", which mixes
    re-reads and re-scores at different measured costs. If either category
    that is actually in play (`reads` or `scores` > 0) has no history to
    measure from, the whole estimate falls back rather than quietly averaging
    a real number with a guess.
    """
    if reads == 0 and scores == 0:
        return CostEstimate(measured=False)
    read_median = median_cost(read_costs)
    score_median = median_cost(score_costs)
    if reads and read_median is None:
        return CostEstimate(measured=False)
    if scores and score_median is None:
        return CostEstimate(measured=False)
    total = Decimal(0)
    if reads:
        assert read_median is not None
        total += read_median * reads
    if scores:
        assert score_median is not None
        total += score_median * scores
    return CostEstimate(measured=True, total=total.quantize(CENTS))
