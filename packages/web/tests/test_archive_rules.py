"""Bulk-archive rule matching -- owner feedback, 2026-09-27. No database, no
model: pure functions over `Application` and `RowScore`, the same data the
list screen already loads.

The rule under test, repeated in each test's name where it matters: a score
rule never touches an application that has not finished a scoring run, "both
axes below N" needs both numbers present, the axis choice is never an
average, and an empty rule set matches nothing.
"""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import Application, ApplicationStatus
from jfl_web.archive_rules import (
    RULE_ERROR_MESSAGES,
    TERMINAL_STATUSES,
    ArchiveRules,
    ScoreRule,
    matches,
    matching_applications,
    parse_rule_form,
)
from jfl_web.routes.applications import _PIPELINE
from jfl_web.scores import RowScore

NOW = dt.datetime(2026, 9, 27, 12, 0, tzinfo=dt.UTC)
USER = uuid.uuid4()


def _app(
    title: str = "Role",
    *,
    status: ApplicationStatus = "interested",
    updated: dt.datetime = NOW,
) -> Application:
    return Application(
        id=uuid.uuid4(),
        user_id=USER,
        title=title,
        status=status,
        created_at=updated,
        updated_at=updated,
    )


def _done(*, could: int | None = 5, want: int | None = 5) -> RowScore:
    return RowScore(state="done", could_get=could, want_it=want, has_numbers=True)


# --------------------------------------------------------------------------
# TERMINAL_STATUSES stays exactly "every status the pipeline does not name"
# --------------------------------------------------------------------------


def test_terminal_statuses_are_exactly_the_statuses_outside_the_pipeline() -> None:
    from typing import get_args

    all_statuses = set(get_args(ApplicationStatus))
    assert set(TERMINAL_STATUSES) == all_statuses - set(_PIPELINE)


# --------------------------------------------------------------------------
# The score rule: never an unscored application, never an averaged axis
# --------------------------------------------------------------------------


class TestScoreRule:
    def test_below_threshold_on_the_chosen_axis_matches(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="want", threshold=6))
        assert matches(_app(), _done(want=5, could=9), rules, now=NOW)

    def test_at_or_above_threshold_does_not_match(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="want", threshold=6))
        assert not matches(_app(), _done(want=6, could=1), rules, now=NOW)

    def test_could_axis_looks_only_at_could_get(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="could", threshold=6))
        assert matches(_app(), _done(want=9, could=5), rules, now=NOW)
        assert not matches(_app(), _done(want=1, could=9), rules, now=NOW)

    def test_either_axis_matches_on_either_one(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="either", threshold=6))
        assert matches(_app(), _done(want=9, could=5), rules, now=NOW)
        assert matches(_app(), _done(want=5, could=9), rules, now=NOW)
        assert not matches(_app(), _done(want=9, could=9), rules, now=NOW)

    def test_both_axes_requires_both_below(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="both", threshold=6))
        assert matches(_app(), _done(want=5, could=5), rules, now=NOW)
        assert not matches(_app(), _done(want=5, could=9), rules, now=NOW)
        assert not matches(_app(), _done(want=9, could=5), rules, now=NOW)

    def test_both_axes_does_not_match_when_want_is_silently_unmeasured(self) -> None:
        """An empty profile can finish a run with `want_it_score` still None.
        "Both axes below N" is not a claim the data supports when one axis
        never got a number -- it must not be read as "the missing one counts
        as zero"."""
        row = RowScore(state="done", could_get=1, want_it=None, has_numbers=True)
        rules = ArchiveRules(score=ScoreRule(axis="both", threshold=6))
        assert not matches(_app(), row, rules, now=NOW)

    def test_want_axis_does_not_match_when_want_is_unmeasured(self) -> None:
        row = RowScore(state="done", could_get=1, want_it=None, has_numbers=True)
        rules = ArchiveRules(score=ScoreRule(axis="want", threshold=10))
        assert not matches(_app(), row, rules, now=NOW)

    def test_an_unscored_application_never_matches_a_score_rule(self) -> None:
        """Not yet scored is not a low score -- the headline case this rule
        exists to get right."""
        rules = ArchiveRules(score=ScoreRule(axis="either", threshold=10))
        assert not matches(_app(), RowScore(state="unscored"), rules, now=NOW)
        assert not matches(_app(), None, rules, now=NOW)

    def test_a_run_in_progress_never_matches_a_score_rule(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="either", threshold=10))
        assert not matches(_app(), RowScore(state="scoring"), rules, now=NOW)
        assert not matches(_app(), RowScore(state="retrying"), rules, now=NOW)

    def test_a_failed_run_never_matches_a_score_rule(self) -> None:
        rules = ArchiveRules(score=ScoreRule(axis="either", threshold=10))
        assert not matches(_app(), RowScore(state="failed"), rules, now=NOW)


# --------------------------------------------------------------------------
# The status rule
# --------------------------------------------------------------------------


class TestStatusRule:
    def test_matches_only_the_checked_statuses(self) -> None:
        rules = ArchiveRules(statuses=frozenset({"rejected"}))
        assert matches(_app(status="rejected"), None, rules, now=NOW)
        assert not matches(_app(status="withdrawn"), None, rules, now=NOW)
        assert not matches(_app(status="interested"), None, rules, now=NOW)

    def test_several_statuses_are_an_or_within_the_rule(self) -> None:
        rules = ArchiveRules(statuses=frozenset({"rejected", "withdrawn"}))
        assert matches(_app(status="rejected"), None, rules, now=NOW)
        assert matches(_app(status="withdrawn"), None, rules, now=NOW)


# --------------------------------------------------------------------------
# The inactivity rule
# --------------------------------------------------------------------------


class TestInactivityRule:
    def test_older_than_the_threshold_matches(self) -> None:
        rules = ArchiveRules(inactive_days=30)
        stale = _app(updated=NOW - dt.timedelta(days=31))
        assert matches(stale, None, rules, now=NOW)

    def test_exactly_the_threshold_matches(self) -> None:
        rules = ArchiveRules(inactive_days=30)
        exactly = _app(updated=NOW - dt.timedelta(days=30))
        assert matches(exactly, None, rules, now=NOW)

    def test_more_recent_than_the_threshold_does_not_match(self) -> None:
        rules = ArchiveRules(inactive_days=30)
        fresh = _app(updated=NOW - dt.timedelta(days=1))
        assert not matches(fresh, None, rules, now=NOW)


# --------------------------------------------------------------------------
# AND combination, and the empty rule set
# --------------------------------------------------------------------------


class TestCombiningRules:
    def test_rules_combine_with_and(self) -> None:
        rules = ArchiveRules(
            score=ScoreRule(axis="either", threshold=6),
            statuses=frozenset({"rejected"}),
        )
        low_scored_and_rejected = _app(status="rejected")
        assert matches(low_scored_and_rejected, _done(want=1, could=1), rules, now=NOW)

        low_scored_but_not_rejected = _app(status="interested")
        assert not matches(low_scored_but_not_rejected, _done(want=1, could=1), rules, now=NOW)

        rejected_but_not_low_scored = _app(status="rejected")
        assert not matches(rejected_but_not_low_scored, _done(want=9, could=9), rules, now=NOW)

    def test_an_empty_rule_set_matches_nothing(self) -> None:
        rules = ArchiveRules()
        assert rules.is_empty
        assert not matches(_app(), _done(), rules, now=NOW)
        assert matching_applications([_app(), _app()], {}, rules, now=NOW) == []

    def test_matching_applications_filters_the_whole_list(self) -> None:
        rules = ArchiveRules(statuses=frozenset({"rejected"}))
        keep = _app(status="rejected")
        drop = _app(status="interested")
        row_scores: dict[uuid.UUID, RowScore] = {}
        assert matching_applications([keep, drop], row_scores, rules, now=NOW) == [keep]


# --------------------------------------------------------------------------
# Parsing the rule form
# --------------------------------------------------------------------------


class TestParseRuleForm:
    def _parse(self, **overrides: object) -> tuple[ArchiveRules | None, str | None]:
        defaults: dict[str, object] = {
            "score_enabled": False,
            "score_axis": "either",
            "score_threshold": "",
            "statuses": None,
            "activity_enabled": False,
            "activity_days": "",
        }
        defaults.update(overrides)
        return parse_rule_form(**defaults)  # type: ignore[arg-type]

    def test_nothing_checked_is_an_error(self) -> None:
        rules, error = self._parse()
        assert rules is None
        assert error == "no_rule"
        assert error in RULE_ERROR_MESSAGES

    def test_score_rule_needs_axis_and_a_number_in_range(self) -> None:
        rules, error = self._parse(score_enabled=True, score_axis="want", score_threshold="6")
        assert error is None
        assert rules is not None
        assert rules.score == ScoreRule(axis="want", threshold=6)

    def test_score_threshold_out_of_range_is_rejected(self) -> None:
        rules, error = self._parse(score_enabled=True, score_axis="want", score_threshold="11")
        assert rules is None
        assert error == "bad_score"

    def test_score_threshold_not_a_number_is_rejected(self) -> None:
        rules, error = self._parse(
            score_enabled=True, score_axis="want", score_threshold="not-a-number"
        )
        assert rules is None
        assert error == "bad_score"

    def test_an_unknown_axis_is_rejected_rather_than_silently_defaulted(self) -> None:
        rules, error = self._parse(score_enabled=True, score_axis="average", score_threshold="6")
        assert rules is None
        assert error == "bad_score"

    def test_statuses_outside_the_closed_set_are_dropped_silently(self) -> None:
        """Same discipline as `change_status`'s `to_status`: a tampered value
        cannot smuggle in a status this form never offered."""
        rules, error = self._parse(statuses=["rejected", "not-a-real-status"])
        assert error is None
        assert rules is not None
        assert rules.statuses == frozenset({"rejected"})

    def test_activity_rule_needs_a_number_in_range(self) -> None:
        rules, error = self._parse(activity_enabled=True, activity_days="30")
        assert error is None
        assert rules is not None
        assert rules.inactive_days == 30

    def test_activity_days_zero_is_rejected(self) -> None:
        rules, error = self._parse(activity_enabled=True, activity_days="0")
        assert rules is None
        assert error == "bad_activity"

    def test_several_rules_together_all_land_in_the_result(self) -> None:
        rules, error = self._parse(
            score_enabled=True,
            score_axis="both",
            score_threshold="4",
            statuses=["withdrawn"],
            activity_enabled=True,
            activity_days="14",
        )
        assert error is None
        assert rules == ArchiveRules(
            score=ScoreRule(axis="both", threshold=4),
            statuses=frozenset({"withdrawn"}),
            inactive_days=14,
        )
