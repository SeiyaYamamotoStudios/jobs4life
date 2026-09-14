"""The pure job filter (`jfl_intake.filtering`). No database, no network."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field

from jfl_core.models import BoardFilterException, JobFilter, Workplace
from jfl_intake.filtering import apply_filter, parse_terms

BOARD = uuid.uuid4()
OTHER_BOARD = uuid.uuid4()
NOW = dt.datetime(2026, 9, 15, 9, 0, tzinfo=dt.UTC)


@dataclass(frozen=True)
class Job:
    title: str
    workplace: Workplace = "unknown"
    locations: tuple[str, ...] = ()
    location: str | None = None
    board_id: uuid.UUID = field(default=BOARD)


def titles(
    jobs: list[Job], saved: JobFilter, *, exceptions: list[BoardFilterException] | None = None
) -> list[str]:
    result = apply_filter(jobs, saved, include_unstated={}, exceptions=exceptions or [])
    return [m.job.title for m in result.matches]


def exception(
    board: uuid.UUID = BOARD,
    *,
    workplaces: list[Workplace] | None = None,
    location: str = "",
    note: str = "",
) -> BoardFilterException:
    return BoardFilterException(
        id=uuid.uuid4(),
        board_id=board,
        workplaces=workplaces or [],
        location=location,
        note=note,
        created_at=NOW,
        updated_at=NOW,
    )


# -- terms --------------------------------------------------------------------------


def test_terms_are_normalised_comma_alternatives_with_blanks_dropped() -> None:
    assert parse_terms("Engineering Manager, head of ENGINEERING , ,-") == (
        frozenset({"engineering", "manager"}),
        frozenset({"head", "of", "engineering"}),
    )
    assert parse_terms("") == ()


# -- title -----------------------------------------------------------------------------


def test_includes_are_word_order_independent() -> None:
    jobs = [
        Job("Manager, Engineering"),
        Job("Engineering Manager, Platform"),
        Job("Engineer"),
        Job("Senior Engineering Program Manager"),
    ]
    assert titles(jobs, JobFilter(title_includes="engineering manager")) == [
        "Manager, Engineering",
        "Engineering Manager, Platform",
        "Senior Engineering Program Manager",
    ]


def test_matching_is_whole_word() -> None:
    jobs = [Job("Engineering Manager"), Job("Software Engineer"), Job("Reengineer")]
    assert titles(jobs, JobFilter(title_includes="engineer")) == ["Software Engineer"]


def test_comma_alternatives_any_matches() -> None:
    jobs = [Job("Head of Engineering"), Job("Engineering Manager"), Job("Designer")]
    saved = JobFilter(title_includes="engineering manager, head of engineering")
    assert titles(jobs, saved) == ["Head of Engineering", "Engineering Manager"]


def test_excludes_win_over_includes() -> None:
    jobs = [Job("Engineering Manager"), Job("Engineering Manager, Sales Engineering")]
    saved = JobFilter(title_includes="engineering manager", title_excludes="sales, intern")
    assert titles(jobs, saved) == ["Engineering Manager"]


def test_matching_ignores_case_punctuation_and_unicode_width() -> None:
    jobs = [Job("ＥＮＧＩＮＥＥＲＩＮＧ-manager (Remote)")]
    assert titles(jobs, JobFilter(title_includes="Engineering Manager")) == [jobs[0].title]


def test_an_empty_filter_matches_everything() -> None:
    jobs = [Job("A"), Job("B", workplace="onsite"), Job("C", workplace="remote")]
    result = apply_filter(jobs, JobFilter(), include_unstated={})
    assert [m.job.title for m in result.matches] == ["A", "B", "C"]
    assert result.open_total == 3 and result.hidden_unstated == 0


# -- location -----------------------------------------------------------------------------


def test_location_matches_any_one_of_the_jobs_locations() -> None:
    jobs = [
        Job("Multi", locations=("New York City, NY", "London, UK")),
        Job("Elsewhere", locations=("Sydney, Australia",)),
        Job("Legacy row", location="London, United Kingdom"),  # no `locations` recorded
    ]
    assert titles(jobs, JobFilter(location="london")) == ["Multi", "Legacy row"]


def test_a_location_alternative_needs_all_its_words_in_one_location() -> None:
    jobs = [Job("Split", locations=("New York, NY", "London, UK"))]
    assert titles(jobs, JobFilter(location="new york london")) == []
    assert titles(jobs, JobFilter(location="new york, london")) == ["Split"]


# -- workplace ------------------------------------------------------------------------------


def test_workplace_must_be_in_the_selected_set() -> None:
    jobs = [
        Job("R", workplace="remote"),
        Job("H", workplace="hybrid"),
        Job("O", workplace="onsite"),
    ]
    assert titles(jobs, JobFilter(workplaces=["remote", "hybrid"])) == ["R", "H"]


def test_unknown_workplace_is_counted_as_hidden_never_silently_dropped() -> None:
    jobs = [
        Job("Remote EM", workplace="remote"),
        Job("Unstated EM", workplace="unknown"),
        Job("Unstated designer", workplace="unknown"),  # fails the title anyway
        Job("Onsite EM", workplace="onsite"),
    ]
    saved = JobFilter(workplaces=["remote"], title_includes="em")
    result = apply_filter(jobs, saved, include_unstated={})
    assert [m.job.title for m in result.matches] == ["Remote EM"]
    assert result.hidden_unstated == 1
    assert result.open_total == 4


def test_show_hidden_unstated_brings_them_back_for_that_view() -> None:
    jobs = [Job("Remote", workplace="remote"), Job("Unstated")]
    saved = JobFilter(workplaces=["remote"])
    result = apply_filter(jobs, saved, include_unstated={}, show_hidden_unstated=True)
    assert [m.job.title for m in result.matches] == ["Remote", "Unstated"]
    assert result.hidden_unstated == 0


def test_a_board_including_unstated_lets_them_through_badged_by_reason() -> None:
    jobs = [
        Job("Unstated here"),
        Job("Unstated elsewhere", board_id=OTHER_BOARD),
        Job("Unstated here, Sydney", locations=("Sydney",)),
    ]
    saved = JobFilter(workplaces=["remote"], location="london, sydney")
    result = apply_filter(jobs, saved, include_unstated={BOARD: True, OTHER_BOARD: False})
    # Location "london, sydney": the first two have no location at all, so only
    # the Sydney job passes location -- and then passes workplace by its board.
    assert [(m.job.title, m.reason) for m in result.matches] == [
        ("Unstated here, Sydney", "unstated")
    ]
    saved = JobFilter(workplaces=["remote"])
    result = apply_filter(jobs, saved, include_unstated={BOARD: True, OTHER_BOARD: False})
    assert [(m.job.title, m.reason) for m in result.matches] == [
        ("Unstated here", "unstated"),
        ("Unstated here, Sydney", "unstated"),
    ]
    assert result.via_unstated == 2
    assert result.hidden_unstated == 1  # the other board's


def test_include_unstated_still_honours_title_excludes() -> None:
    jobs = [Job("Sales Engineering Manager")]
    saved = JobFilter(workplaces=["remote"], title_excludes="sales")
    result = apply_filter(jobs, saved, include_unstated={BOARD: True})
    assert result.matches == [] and result.hidden_unstated == 0


# -- board exceptions ------------------------------------------------------------------------


ANTHROPIC_SHAPED = [
    Job("Engineering Manager, Inference", workplace="onsite", locations=("London, UK",)),
    Job("Engineering Manager, Platform", workplace="onsite", locations=("San Francisco, CA",)),
    Job("Engineering Manager, Remote", workplace="remote", locations=("Remote-Friendly, US",)),
    Job("Sales Engineering Manager", workplace="onsite", locations=("London, UK",)),
    Job("Recruiter", workplace="onsite", locations=("London, UK",)),
]


def test_an_onsite_london_exception_is_ored_with_the_filter() -> None:
    rule = exception(
        workplaces=["onsite", "hybrid"],
        location="london",
        note="Accepts ~25% in office — 1 day a week in London",
    )
    saved = JobFilter(
        workplaces=["remote"], title_includes="engineering manager", title_excludes="sales"
    )
    result = apply_filter(ANTHROPIC_SHAPED, saved, include_unstated={}, exceptions=[rule])
    assert [(m.job.title, m.reason) for m in result.matches] == [
        ("Engineering Manager, Inference", "exception"),
        ("Engineering Manager, Remote", "filter"),
    ]
    assert result.matches[0].exception == rule
    assert result.via_exception == 1


def test_an_exception_never_widens_the_title() -> None:
    rule = exception(workplaces=["onsite"], location="london")
    saved = JobFilter(workplaces=["remote"], title_includes="engineering manager")
    got = titles(ANTHROPIC_SHAPED, saved, exceptions=[rule])
    assert "Recruiter" not in got


def test_an_exception_applies_only_to_its_own_board() -> None:
    elsewhere = Job(
        "Engineering Manager", workplace="onsite", locations=("London, UK",), board_id=OTHER_BOARD
    )
    rule = exception(BOARD, workplaces=["onsite"], location="london")
    assert titles([elsewhere], JobFilter(workplaces=["remote"]), exceptions=[rule]) == []


def test_an_exception_with_no_workplaces_means_any_workplace() -> None:
    rule = exception(location="london")
    jobs = [Job("Any London", workplace="hybrid", locations=("London",))]
    assert titles(jobs, JobFilter(workplaces=["remote"]), exceptions=[rule]) == ["Any London"]


def test_a_job_matching_an_exception_is_not_counted_as_hidden_unstated() -> None:
    rule = exception(workplaces=["unknown"], location="london")
    jobs = [Job("Unstated London", locations=("London",))]
    result = apply_filter(
        jobs, JobFilter(workplaces=["remote"]), include_unstated={}, exceptions=[rule]
    )
    assert [m.reason for m in result.matches] == ["exception"]
    assert result.hidden_unstated == 0


def test_the_filter_itself_takes_precedence_over_an_exception() -> None:
    rule = exception(workplaces=["remote"])
    jobs = [Job("Remote", workplace="remote")]
    result = apply_filter(
        jobs, JobFilter(workplaces=["remote"]), include_unstated={}, exceptions=[rule]
    )
    assert [m.reason for m in result.matches] == ["filter"]
