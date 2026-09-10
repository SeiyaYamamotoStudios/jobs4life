"""The check engine's rules, directly: no database, no network, no clock."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from jfl_core.models import BoardCheckState, CheckPlan, KnownBoardJob, ObservedJob
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import (
    DROP_GUARD_EMPTY_MIN_PREVIOUS,
    DROP_GUARD_MIN_PREVIOUS,
    REPOST_WINDOW,
    plan_check,
)
from jfl_intake.normalise import fingerprint

NOW = dt.datetime(2026, 9, 10, 6, 0, tzinfo=dt.UTC)
BOARD = uuid.UUID("11111111-1111-4111-8111-111111111111")
BASELINE = uuid.UUID("22222222-2222-4222-8222-222222222222")


def observed(ext: str, title: str | None = None, location: str = "London") -> ObservedJob:
    name = title or f"Role {ext}"
    return ObservedJob(
        external_id=ext,
        title=name,
        location=location,
        url=f"https://example.test/{ext}",
        fingerprint=fingerprint(name, location),
    )


def known(
    ext: str,
    *,
    is_open: bool = True,
    closed_days_ago: float | None = None,
    title: str | None = None,
    location: str = "London",
    has_successor: bool = False,
) -> KnownBoardJob:
    return KnownBoardJob(
        job_id=uuid.uuid5(BOARD, ext),
        external_id=ext,
        fingerprint=fingerprint(title or f"Role {ext}", location),
        is_open=is_open,
        last_closed_at=None
        if closed_days_ago is None
        else NOW - dt.timedelta(days=closed_days_ago),
        has_repost_successor=has_successor,
    )


def state(
    *jobs: KnownBoardJob, baseline: bool = True, drop_accepted: bool = False
) -> BoardCheckState:
    return BoardCheckState(
        board_id=BOARD,
        baseline_check_id=BASELINE if baseline else None,
        drop_accepted=drop_accepted,
        known_jobs=list(jobs),
    )


def complete(*jobs: ObservedJob, expected: int | None = None) -> FetchResult:
    return FetchResult(status="complete", jobs=tuple(jobs), expected_total=expected)


def plan(s: BoardCheckState, r: FetchResult) -> CheckPlan:
    return plan_check(s, r, observed_at=NOW)


def job_id(ext: str) -> uuid.UUID:
    return uuid.uuid5(BOARD, ext)


def assert_no_job_changes(p: CheckPlan) -> None:
    assert p.new_jobs == []
    assert p.returned == []
    assert p.still_open == []
    assert p.gone_job_ids == []
    assert not p.changes_job_state


# -- the rule -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("unreachable", "timeout"),
        ("failed", "malformed_response"),
        ("incomplete", "count_mismatch"),
        ("truncated", "listing_ceiling"),
    ],
)
def test_a_check_that_is_not_complete_closes_no_interval_and_opens_none(
    status: str, code: str
) -> None:
    """The rule everything rests on. The board has three open jobs; the fetch
    saw one of them and one it has never seen. None of that may reach the
    history -- not the two absences, and not the new job either.
    """
    s = state(known("a"), known("b"), known("c"))
    partial = FetchResult(
        status=status,  # type: ignore[arg-type]
        jobs=(observed("a"), observed("brand-new")),
        expected_total=40,
        error_code=code,  # type: ignore[arg-type]
    )

    p = plan(s, partial)

    assert p.status == status
    assert p.error_code == code
    assert p.jobs_seen == 2  # still recorded: what it saw is worth knowing
    assert p.expected_total == 40
    assert_no_job_changes(p)


def test_a_fetch_result_must_explain_itself_exactly_when_it_is_not_complete() -> None:
    with pytest.raises(ValueError):
        plan(state(), FetchResult(status="complete", error_code="timeout"))
    with pytest.raises(ValueError):
        plan(state(), FetchResult(status="failed"))


# -- baseline -------------------------------------------------------------------


def test_the_first_complete_check_is_the_baseline_and_marks_nothing_new() -> None:
    p = plan(state(baseline=False), complete(observed("a"), observed("b"), observed("c")))
    assert p.status == "complete"
    assert p.is_baseline
    assert [n.job.external_id for n in p.new_jobs] == ["a", "b", "c"]
    assert all(n.reposted_from_job_id is None for n in p.new_jobs)
    assert p.summary() == {"baseline": 3, "new": 0, "reposted": 0, "returned": 0, "gone": 0}


def test_an_empty_board_can_be_a_baseline() -> None:
    p = plan(state(baseline=False), complete())
    assert p.is_baseline and p.new_jobs == []


# -- a later complete check -------------------------------------------------------


def test_a_later_complete_check_yields_new_gone_and_returned() -> None:
    s = state(
        known("still"),
        known("vanishes"),
        known("comes-back", is_open=False, closed_days_ago=5),
    )
    p = plan(s, complete(observed("still"), observed("comes-back"), observed("fresh")))

    assert p.status == "complete" and not p.is_baseline
    assert [x.job_id for x in p.still_open] == [job_id("still")]
    assert [x.job_id for x in p.returned] == [job_id("comes-back")]
    assert [n.job.external_id for n in p.new_jobs] == ["fresh"]
    assert p.new_jobs[0].reposted_from_job_id is None
    assert p.gone_job_ids == [job_id("vanishes")]
    assert p.summary() == {"baseline": 0, "new": 1, "reposted": 0, "returned": 1, "gone": 1}


def test_duplicate_ids_in_one_result_count_once() -> None:
    p = plan(state(known("a")), complete(observed("a"), observed("a"), observed("b")))
    assert p.jobs_seen == 2
    assert len(p.still_open) == 1 and len(p.new_jobs) == 1


def test_planning_the_same_result_against_the_applied_history_changes_nothing() -> None:
    """Idempotency at the level of the rules: once a check's result is in the
    history, the same result again is all sightings and no events.
    """
    result = complete(observed("a"), observed("b"), observed("c"))
    first = plan(state(known("a"), known("gone-one")), result)
    assert len(first.new_jobs) == 2 and first.gone_job_ids == [job_id("gone-one")]

    applied = state(
        known("a"), known("b"), known("c"), known("gone-one", is_open=False, closed_days_ago=0)
    )
    second = plan(applied, result)
    assert second.new_jobs == [] and second.returned == [] and second.gone_job_ids == []
    assert len(second.still_open) == 3


# -- drop guard ---------------------------------------------------------------------


def test_a_sudden_collapse_is_held_and_closes_nothing() -> None:
    open_jobs = [known(f"j{i}") for i in range(20)]
    p = plan(state(*open_jobs), complete(*[observed(f"j{i}") for i in range(9)]))
    assert p.status == "held"
    assert p.error_code == "drop_guard"
    assert p.jobs_seen == 9
    assert_no_job_changes(p)


def test_exactly_half_is_normal_churn_and_is_applied() -> None:
    open_jobs = [known(f"j{i}") for i in range(20)]
    p = plan(state(*open_jobs), complete(*[observed(f"j{i}") for i in range(10)]))
    assert p.status == "complete"
    assert len(p.gone_job_ids) == 10


def test_small_boards_are_not_held_for_halving() -> None:
    open_jobs = [known(f"j{i}") for i in range(DROP_GUARD_MIN_PREVIOUS - 1)]
    p = plan(state(*open_jobs), complete(observed("j0")))
    assert p.status == "complete"
    assert len(p.gone_job_ids) == DROP_GUARD_MIN_PREVIOUS - 2


def test_a_small_board_returning_zero_is_held() -> None:
    """Zero is the exact shape of the observed silent failure."""
    open_jobs = [known(f"j{i}") for i in range(DROP_GUARD_EMPTY_MIN_PREVIOUS)]
    p = plan(state(*open_jobs), complete())
    assert p.status == "held"
    assert_no_job_changes(p)


def test_a_tiny_board_emptying_is_applied() -> None:
    p = plan(state(known("only"), known("other")), complete())
    assert p.status == "complete"
    assert sorted(p.gone_job_ids) == sorted([job_id("only"), job_id("other")])


def test_an_accepted_drop_is_applied() -> None:
    open_jobs = [known(f"j{i}") for i in range(20)]
    p = plan(state(*open_jobs, drop_accepted=True), complete())
    assert p.status == "complete"
    assert len(p.gone_job_ids) == 20


# -- reposted ------------------------------------------------------------------------


def test_a_new_id_matching_a_recently_closed_role_is_reposted() -> None:
    s = state(known("old", is_open=False, closed_days_ago=30, title="Staff Engineer"))
    p = plan(s, complete(observed("new", title="Staff Engineer")))
    assert p.new_jobs[0].reposted_from_job_id == job_id("old")
    assert p.summary()["reposted"] == 1 and p.summary()["new"] == 0


def test_the_window_edge_is_inclusive_and_outside_it_is_just_new() -> None:
    days = REPOST_WINDOW.days
    inside = state(known("old", is_open=False, closed_days_ago=days, title="Staff Engineer"))
    outside = state(known("old", is_open=False, closed_days_ago=days + 1, title="Staff Engineer"))
    repost = complete(observed("new", title="Staff Engineer"))
    assert plan(inside, repost).new_jobs[0].reposted_from_job_id == job_id("old")
    assert plan(outside, repost).new_jobs[0].reposted_from_job_id is None


def test_taken_down_and_put_back_between_two_checks_is_a_repost() -> None:
    """The commonest repost: the old id closes in the same check the new one
    opens, so the source is a job this very plan is closing.
    """
    s = state(known("old", title="Staff Engineer"), known("other"))
    p = plan(s, complete(observed("other"), observed("new", title="Staff Engineer")))
    assert p.gone_job_ids == [job_id("old")]
    assert p.new_jobs[0].reposted_from_job_id == job_id("old")


def test_a_different_location_is_not_the_same_role() -> None:
    s = state(known("old", is_open=False, closed_days_ago=3, title="Staff Engineer"))
    p = plan(s, complete(observed("new", title="Staff Engineer", location="Zurich")))
    assert p.new_jobs[0].reposted_from_job_id is None


def test_returned_is_not_reposted() -> None:
    """Same external id reopening is `returned`, never a repost -- and a job
    that is returning cannot be the source of a repost either.
    """
    s = state(known("x", is_open=False, closed_days_ago=10, title="Staff Engineer"))
    p = plan(s, complete(observed("x", title="Staff Engineer")))
    assert [r.job_id for r in p.returned] == [job_id("x")]
    assert p.new_jobs == []

    both = plan(
        s, complete(observed("x", title="Staff Engineer"), observed("y", title="Staff Engineer"))
    )
    assert [r.job_id for r in both.returned] == [job_id("x")]
    assert both.new_jobs[0].reposted_from_job_id is None


def test_one_closed_role_is_the_source_of_at_most_one_repost() -> None:
    s = state(known("old", is_open=False, closed_days_ago=2, title="Staff Engineer"))
    p = plan(
        s, complete(observed("n1", title="Staff Engineer"), observed("n2", title="Staff Engineer"))
    )
    assert [n.reposted_from_job_id for n in p.new_jobs] == [job_id("old"), None]


def test_a_role_already_reposted_is_not_matched_again() -> None:
    s = state(
        known("old", is_open=False, closed_days_ago=20, title="Staff Engineer", has_successor=True)
    )
    p = plan(s, complete(observed("later", title="Staff Engineer")))
    assert p.new_jobs[0].reposted_from_job_id is None


def test_the_most_recently_closed_source_is_matched_first() -> None:
    s = state(
        known("older", is_open=False, closed_days_ago=40, title="Staff Engineer"),
        known("recent", is_open=False, closed_days_ago=5, title="Staff Engineer"),
    )
    p = plan(s, complete(observed("new", title="Staff Engineer")))
    assert p.new_jobs[0].reposted_from_job_id == job_id("recent")


def test_nothing_is_reposted_in_a_baseline() -> None:
    p = plan(state(baseline=False), complete(observed("new", title="Staff Engineer")))
    assert p.new_jobs[0].reposted_from_job_id is None
