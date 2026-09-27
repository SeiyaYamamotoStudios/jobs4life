"""Bulk re-score and "retry everything that failed" -- owner feedback,
2026-09-27, alongside the bulk-archive slice. No database, no model: pure
functions over `Application` and `RowScore`, the same data the list screen
already loads.

What is under test, repeated in each test's name where it matters: a bulk
re-score skips an application already in flight and one whose ad has not been
read into any requirement; a retry plan reads the read as the fix for both a
failed read and a failed score on the same application, never both at once;
`description_unavailable` is reported separately, with nothing enqueued for
it; and the cost estimate is measured from a sample and never presented as a
measurement when there is nothing to measure from.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from jfl_core.models import Application, ApplicationStatus, ExtractionErrorCode, ExtractionStatus
from jfl_web.bulk_actions import (
    CostEstimate,
    RescoreSkip,
    RetryPlan,
    estimate_for_count,
    estimate_retry_cost,
    median_cost,
    partition_rescore,
    plan_retry_failed,
)
from jfl_web.scores import RowScore, RowScoreState

NOW = dt.datetime(2026, 9, 27, 12, 0, tzinfo=dt.UTC)
USER = uuid.uuid4()


def _app(
    title: str = "Role",
    *,
    status: ApplicationStatus = "interested",
    extraction_status: ExtractionStatus = "done",
    extraction_error_code: ExtractionErrorCode | None = None,
) -> Application:
    return Application(
        id=uuid.uuid4(),
        user_id=USER,
        title=title,
        status=status,
        extraction_status=extraction_status,
        extraction_error_code=extraction_error_code,
        created_at=NOW,
        updated_at=NOW,
    )


def _row(state: RowScoreState = "done", *, could: int | None = 5, want: int | None = 5) -> RowScore:
    return RowScore(state=state, could_get=could, want_it=want, has_numbers=state == "done")


# --------------------------------------------------------------------------
# partition_rescore
# --------------------------------------------------------------------------


def test_a_never_scored_application_with_requirements_is_eligible() -> None:
    app = _app()
    eligible, skipped = partition_rescore([app], {}, {app.id: True})
    assert eligible == [app]
    assert skipped == []


def test_a_scoring_run_in_flight_is_skipped() -> None:
    app = _app()
    eligible, skipped = partition_rescore([app], {app.id: _row("scoring")}, {app.id: True})
    assert eligible == []
    assert skipped == [RescoreSkip(app, "already_scoring")]


def test_a_retrying_run_is_skipped_the_same_as_scoring() -> None:
    app = _app()
    eligible, skipped = partition_rescore([app], {app.id: _row("retrying")}, {app.id: True})
    assert eligible == []
    assert skipped == [RescoreSkip(app, "already_scoring")]


def test_an_application_with_no_requirements_is_skipped_rather_than_queued_to_fail() -> None:
    app = _app(extraction_status="pending")
    eligible, skipped = partition_rescore([app], {}, {app.id: False})
    assert eligible == []
    assert skipped == [RescoreSkip(app, "ad_not_read")]


def test_a_finished_or_failed_score_is_still_eligible_for_re_score() -> None:
    """Re-scoring a finished or failed run is exactly what the button is for --
    only an in-flight run and an unread ad are skipped."""
    done = _app("Done")
    failed = _app("Failed")
    row_scores = {done.id: _row("done"), failed.id: _row("failed")}
    has_requirements = {done.id: True, failed.id: True}
    eligible, skipped = partition_rescore([done, failed], row_scores, has_requirements)
    assert eligible == [done, failed]
    assert skipped == []


def test_eligibility_is_checked_in_a_fixed_order_so_the_reason_is_never_ambiguous() -> None:
    """An application that is both scoring and unread (impossible in practice,
    since a scoring run implies requirements existed) is reported once, as
    "already scoring" -- the check order is fixed, not incidental."""
    app = _app()
    eligible, skipped = partition_rescore([app], {app.id: _row("scoring")}, {app.id: False})
    assert eligible == []
    assert skipped == [RescoreSkip(app, "already_scoring")]


# --------------------------------------------------------------------------
# plan_retry_failed
# --------------------------------------------------------------------------


def test_a_failed_read_is_retried() -> None:
    app = _app(extraction_status="failed", extraction_error_code="model_error")
    plan = plan_retry_failed([app], {})
    assert plan.retry_reads == [app]
    assert plan.retry_scores == []
    assert plan.needs_paste == []
    assert plan.count == 1


def test_a_failed_score_on_a_read_application_is_retried() -> None:
    app = _app(extraction_status="done")
    plan = plan_retry_failed([app], {app.id: _row("failed")})
    assert plan.retry_scores == [app]
    assert plan.retry_reads == []
    assert plan.count == 1


def test_description_unavailable_needs_a_paste_and_is_never_counted_to_retry() -> None:
    app = _app(extraction_status="failed", extraction_error_code="description_unavailable")
    plan = plan_retry_failed([app], {})
    assert plan.needs_paste == [app]
    assert plan.retry_reads == []
    assert plan.retry_scores == []
    assert plan.count == 0


def test_a_failed_read_wins_over_a_failed_score_on_the_same_application() -> None:
    """Retrying the read is enough -- a successful re-read chains its own first
    score when none is active (`PostgresScoreRepository.has_active`), so
    retrying both would double the work or race the chain.
    """
    app = _app(extraction_status="failed", extraction_error_code="model_error")
    plan = plan_retry_failed([app], {app.id: _row("failed")})
    assert plan.retry_reads == [app]
    assert plan.retry_scores == []
    assert plan.count == 1


def test_a_successful_read_and_a_successful_score_are_never_retried() -> None:
    app = _app(extraction_status="done")
    plan = plan_retry_failed([app], {app.id: _row("done")})
    assert plan.is_empty


def test_a_scoring_run_still_in_flight_is_not_retried() -> None:
    """ "Failed" is the only score state a retry acts on -- pending is not
    stuck, it is running."""
    app = _app(extraction_status="done")
    plan = plan_retry_failed([app], {app.id: _row("scoring")})
    assert plan.is_empty


def test_an_empty_live_list_plans_nothing() -> None:
    plan = plan_retry_failed([], {})
    assert plan.is_empty
    assert plan.count == 0


def test_three_applications_sort_into_their_three_buckets() -> None:
    read_failure = _app(
        "Read failed", extraction_status="failed", extraction_error_code="ad_too_long"
    )
    score_failure_app = _app("Score failed", extraction_status="done")
    paste_needed = _app(
        "Needs paste", extraction_status="failed", extraction_error_code="description_unavailable"
    )
    fine = _app("Fine", extraction_status="done")
    row_scores = {score_failure_app.id: _row("failed"), fine.id: _row("done")}
    plan = plan_retry_failed([read_failure, score_failure_app, paste_needed, fine], row_scores)
    assert plan.retry_reads == [read_failure]
    assert plan.retry_scores == [score_failure_app]
    assert plan.needs_paste == [paste_needed]
    assert plan.count == 2


# --------------------------------------------------------------------------
# Cost estimate: measured from a sample, fallback with nothing to measure.
# --------------------------------------------------------------------------


def test_median_cost_of_an_empty_sample_is_none() -> None:
    assert median_cost([]) is None


def test_median_cost_of_an_odd_sample_is_the_middle_value() -> None:
    assert median_cost([Decimal("0.10"), Decimal("0.50"), Decimal("0.30")]) == Decimal("0.30")


def test_median_cost_of_an_even_sample_averages_the_middle_two() -> None:
    assert median_cost([Decimal("0.10"), Decimal("0.30")]) == Decimal("0.20")


def test_median_is_resistant_to_one_expensive_outlier() -> None:
    """The whole reason for a median over a mean: one long ad or a big corpus
    must not swing the estimate the way it would swing an average."""
    sample = [Decimal("0.10"), Decimal("0.12"), Decimal("0.11"), Decimal("5.00")]
    assert median_cost(sample) == Decimal("0.115")


def test_zero_items_never_needs_an_estimate() -> None:
    assert estimate_for_count([Decimal("0.20")], 0) == CostEstimate(measured=False)


def test_a_measured_estimate_multiplies_the_median_by_the_count() -> None:
    estimate = estimate_for_count([Decimal("0.20"), Decimal("0.30"), Decimal("0.25")], 4)
    assert estimate.measured is True
    assert estimate.total == Decimal("1.00")


def test_no_cost_history_falls_back_rather_than_inventing_a_number() -> None:
    estimate = estimate_for_count([], 5)
    assert estimate.measured is False
    assert estimate.total is None
    assert estimate.fallback == "$0.10-0.30 each"


def test_retry_cost_sums_both_categories_when_both_are_measured() -> None:
    estimate = estimate_retry_cost([Decimal("0.10")], [Decimal("0.40")], reads=2, scores=3)
    assert estimate.measured is True
    assert estimate.total == Decimal("0.10") * 2 + Decimal("0.40") * 3


def test_retry_cost_ignores_the_other_categorys_history_when_that_category_is_unused() -> None:
    """Only reads are happening -- an empty `score_costs` sample must not sink
    the whole estimate to the fallback."""
    estimate = estimate_retry_cost([Decimal("0.10")], [], reads=2, scores=0)
    assert estimate.measured is True
    assert estimate.total == Decimal("0.20")


def test_retry_cost_falls_back_when_a_category_in_play_has_no_history() -> None:
    """Reads are happening and have history; scores are happening and do not --
    the whole estimate falls back rather than quietly averaging a real number
    with a guess."""
    estimate = estimate_retry_cost([Decimal("0.10")], [], reads=2, scores=1)
    assert estimate.measured is False
    assert estimate.total is None


def test_retry_cost_of_nothing_needs_no_estimate() -> None:
    assert estimate_retry_cost([], [], reads=0, scores=0) == CostEstimate(measured=False)


# --------------------------------------------------------------------------
# RetryPlan.count / is_empty
# --------------------------------------------------------------------------


def test_retry_plan_count_never_includes_needs_paste() -> None:
    plan = RetryPlan(retry_reads=[_app()], retry_scores=[], needs_paste=[_app(), _app()])
    assert plan.count == 1
    assert plan.is_empty is False


def test_an_all_empty_plan_is_empty() -> None:
    assert RetryPlan(retry_reads=[], retry_scores=[], needs_paste=[]).is_empty is True
