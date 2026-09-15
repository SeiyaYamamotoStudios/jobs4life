"""The pure job filter (`jfl_intake.filtering`). No database, no network."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from jfl_core.models import BoardFilterException, JobFilter, Workplace, WorkplaceMode
from jfl_intake.adapters import greenhouse
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
    workplace_label: str | None = None


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


# -- workplace presets ------------------------------------------------------------------------

RF_LOCATION = "Remote-Friendly (Travel-Required)"
NOTE = "~1 day a week"


def preset(mode: WorkplaceMode, **fields: object) -> JobFilter:
    return JobFilter.model_validate({"workplace_mode": mode, **fields})


def job_for(workplace: Workplace, evidence: bool, board: uuid.UUID = BOARD) -> Job:
    locations = ("London, UK", RF_LOCATION) if evidence else ("London, UK",)
    return Job(f"{workplace}{'+rf' if evidence else ''}", workplace, locations, board_id=board)


# Every mode x workplace x evidence x hybrid_too_heavy, written out by hand rather
# than derived, so the table is the specification. `custom` is saved as
# workplaces=["remote"]. Outcomes: filter, hidden (hidden_unstated), out (excluded).
# fmt: off
PRESET_TABLE: dict[tuple[WorkplaceMode, Workplace, bool, bool], str] = {
    # remote only: strict, and hybrid_too_heavy is irrelevant.
    ("remote_only", "remote", False, False): "filter",
    ("remote_only", "remote", True, False): "out",
    ("remote_only", "hybrid", False, False): "out",
    ("remote_only", "hybrid", True, False): "out",
    ("remote_only", "onsite", False, False): "out",
    ("remote_only", "onsite", True, False): "out",
    ("remote_only", "unknown", False, False): "hidden",
    ("remote_only", "unknown", True, False): "out",
    ("remote_only", "remote", False, True): "filter",
    ("remote_only", "remote", True, True): "out",
    ("remote_only", "hybrid", False, True): "out",
    ("remote_only", "hybrid", True, True): "out",
    ("remote_only", "onsite", False, True): "out",
    ("remote_only", "onsite", True, True): "out",
    ("remote_only", "unknown", False, True): "hidden",
    ("remote_only", "unknown", True, True): "out",
    # remote friendly.
    ("remote_friendly", "remote", False, False): "filter",
    ("remote_friendly", "remote", True, False): "filter",
    ("remote_friendly", "hybrid", False, False): "filter",
    ("remote_friendly", "hybrid", True, False): "filter",
    ("remote_friendly", "onsite", False, False): "out",
    ("remote_friendly", "onsite", True, False): "filter",
    ("remote_friendly", "unknown", False, False): "hidden",
    ("remote_friendly", "unknown", True, False): "filter",
    # remote friendly on a board whose hybrid is too heavy: plain remote only.
    ("remote_friendly", "remote", False, True): "filter",
    ("remote_friendly", "remote", True, True): "out",
    ("remote_friendly", "hybrid", False, True): "out",
    ("remote_friendly", "hybrid", True, True): "out",
    ("remote_friendly", "onsite", False, True): "out",
    ("remote_friendly", "onsite", True, True): "out",
    ("remote_friendly", "unknown", False, True): "hidden",
    ("remote_friendly", "unknown", True, True): "out",
    # custom: the checkboxes exactly as before -- evidence and the board setting ignored.
    ("custom", "remote", False, False): "filter",
    ("custom", "remote", True, False): "filter",
    ("custom", "hybrid", False, False): "out",
    ("custom", "hybrid", True, False): "out",
    ("custom", "onsite", False, False): "out",
    ("custom", "onsite", True, False): "out",
    ("custom", "unknown", False, False): "hidden",
    ("custom", "unknown", True, False): "hidden",
    ("custom", "remote", False, True): "filter",
    ("custom", "remote", True, True): "filter",
    ("custom", "hybrid", False, True): "out",
    ("custom", "hybrid", True, True): "out",
    ("custom", "onsite", False, True): "out",
    ("custom", "onsite", True, True): "out",
    ("custom", "unknown", False, True): "hidden",
    ("custom", "unknown", True, True): "hidden",
}
# fmt: on


def test_the_preset_table_covers_every_combination() -> None:
    modes: tuple[WorkplaceMode, ...] = ("remote_only", "remote_friendly", "custom")
    workplaces: tuple[Workplace, ...] = ("remote", "hybrid", "onsite", "unknown")
    assert set(PRESET_TABLE) == {
        (m, w, e, h)
        for m in modes
        for w in workplaces
        for e in (False, True)
        for h in (False, True)
    }


@pytest.mark.parametrize(("case", "expected"), sorted(PRESET_TABLE.items()))
def test_each_mode_workplace_evidence_and_board_setting(
    case: tuple[WorkplaceMode, Workplace, bool, bool], expected: str
) -> None:
    mode, workplace, evidence, too_heavy = case
    saved = preset(mode, workplaces=["remote"])
    job = job_for(workplace, evidence)
    result = apply_filter(
        [job],
        saved,
        include_unstated={},
        hybrid_too_heavy={BOARD} if too_heavy else set(),
    )
    got = (
        "hidden"
        if result.hidden_unstated
        else (result.matches[0].reason if result.matches else "out")
    )
    assert got == expected
    assert result.open_total == 1


@pytest.mark.parametrize(
    ("workplace", "evidence", "note"),
    [
        ("remote", False, None),
        ("remote", True, "says_remote_friendly"),
        ("hybrid", False, "hybrid_days_not_stated"),
        ("hybrid", True, "says_remote_friendly"),
        ("onsite", True, "listed_onsite_says_remote_friendly"),
        ("unknown", True, "says_remote_friendly"),
    ],
)
def test_each_match_carries_its_workplace_note(
    workplace: Workplace, evidence: bool, note: str | None
) -> None:
    result = apply_filter(
        [job_for(workplace, evidence)], preset("remote_friendly"), include_unstated={}
    )
    (match,) = result.matches
    assert match.workplace_note == note
    assert match.workplace_exception is None


def test_notes_describe_the_posting_in_every_mode_not_just_remote_friendly() -> None:
    jobs = [job_for("hybrid", False), job_for("onsite", True)]
    result = apply_filter(jobs, JobFilter(), include_unstated={})  # custom, any workplace
    assert [m.workplace_note for m in result.matches] == [
        "hybrid_days_not_stated",
        "listed_onsite_says_remote_friendly",
    ]


ANTHROPIC_CONFLICT = Job(
    "Compute Country Lead, Canada",
    "onsite",
    ("Remote-Friendly (Travel-Required)", "Canada"),
    location="Remote-Friendly (Travel-Required) | Canada",
    workplace_label="On-Site",
)


def test_anthropic_onsite_but_remote_friendly_is_out_of_remote_only_and_in_remote_friendly() -> (
    None
):
    """Owner ruling, 2026-09-15: metadata `On-Site`, location "Remote-Friendly
    (Travel-Required)" -- under remote friendly, with the conflict on the row.
    """
    strict = apply_filter([ANTHROPIC_CONFLICT], preset("remote_only"), include_unstated={})
    assert strict.matches == [] and strict.hidden_unstated == 0

    friendly = apply_filter([ANTHROPIC_CONFLICT], preset("remote_friendly"), include_unstated={})
    (match,) = friendly.matches
    assert match.reason == "filter"
    assert match.workplace_note == "listed_onsite_says_remote_friendly"
    assert match.job.workplace == "onsite"  # shown as listed, never rewritten
    assert friendly.hybrid_days_not_stated == 0


def test_the_captured_anthropic_board_under_each_preset() -> None:
    body = (
        Path(__file__).parent / "fixtures" / "greenhouse_anthropic_workplace_jobs.json"
    ).read_text()
    observed = greenhouse.parse(json.loads(body)).jobs
    jobs = [
        Job(o.title, o.workplace, o.locations, o.location, workplace_label=o.workplace_label)
        for o in observed
    ]

    strict = apply_filter(jobs, preset("remote_only"), include_unstated={BOARD: True})
    # Both remote jobs say Remote-Friendly, so remote only keeps neither; the
    # unstated Sydney job comes in by Greenhouse's include-unstated default.
    assert [(m.job.title, m.reason) for m in strict.matches] == [
        ("Applied AI Architect", "unstated")
    ]

    friendly = apply_filter(jobs, preset("remote_friendly"), include_unstated={BOARD: True})
    assert [(m.job.title, m.reason, m.workplace_note) for m in friendly.matches] == [
        (
            "Anthropic Fellows Program, AI Safety & Security",
            "filter",
            "listed_onsite_says_remote_friendly",
        ),
        ("Business Systems Analyst", "filter", "says_remote_friendly"),
        ("Applied AI Architect, Industries", "filter", "hybrid_days_not_stated"),
        ("Applied AI Architect", "unstated", None),
        ("Staff+ Software Engineer, Data Infrastructure", "filter", "says_remote_friendly"),
        ("Compute Country Lead, Canada", "filter", "listed_onsite_says_remote_friendly"),
    ]
    assert friendly.hybrid_days_not_stated == 1


def test_title_and_location_apply_under_every_preset() -> None:
    jobs = [
        Job("Engineering Manager", "remote", ("London, UK",)),
        Job("Sales Engineering Manager", "remote", ("London, UK",)),
        Job("Engineering Manager, Sydney", "remote", ("Sydney",)),
        Job("Designer", "hybrid", ("London, UK",)),
    ]
    for mode in ("remote_only", "remote_friendly", "custom"):
        saved = preset(
            mode, title_includes="engineering manager", title_excludes="sales", location="london"
        )
        assert titles(jobs, saved) == ["Engineering Manager"], mode


def test_hybrid_too_heavy_applies_only_to_its_own_board() -> None:
    jobs = [job_for("hybrid", False, BOARD), job_for("hybrid", False, OTHER_BOARD)]
    result = apply_filter(
        jobs, preset("remote_friendly"), include_unstated={}, hybrid_too_heavy={BOARD}
    )
    assert [m.job.board_id for m in result.matches] == [OTHER_BOARD]


def test_an_exception_still_ors_in_under_remote_only() -> None:
    rule = exception(workplaces=["onsite"], location="london", note=NOTE)
    jobs = [Job("Onsite London", "onsite", ("London, UK",))]
    result = apply_filter(jobs, preset("remote_only"), include_unstated={}, exceptions=[rule])
    assert [(m.reason, m.exception) for m in result.matches] == [("exception", rule)]


def test_an_exception_still_ors_in_on_a_board_whose_hybrid_is_too_heavy() -> None:
    rule = exception(workplaces=["hybrid"], location="london", note=NOTE)
    result = apply_filter(
        [job_for("hybrid", False)],
        preset("remote_friendly"),
        include_unstated={},
        exceptions=[rule],
        hybrid_too_heavy={BOARD},
    )
    (match,) = result.matches
    assert match.reason == "exception" and match.exception == rule
    # The owner's note replaces "days not stated" on the exception's own match too.
    assert match.workplace_note is None and match.workplace_exception == rule


def test_a_matching_exception_with_a_note_refines_days_not_stated() -> None:
    rule = exception(workplaces=["hybrid"], location="london", note=NOTE)
    result = apply_filter(
        [job_for("hybrid", False)],
        preset("remote_friendly"),
        include_unstated={},
        exceptions=[rule],
    )
    (match,) = result.matches
    assert match.reason == "filter"  # the filter let it through; the exception describes it
    assert match.exception is None
    assert match.workplace_note is None
    assert match.workplace_exception == rule
    assert result.hybrid_days_not_stated == 0


@pytest.mark.parametrize(
    "rule",
    [
        exception(workplaces=["hybrid"], location="london", note="   "),  # no words to offer
        exception(workplaces=["hybrid"], location="dublin", note=NOTE),  # does not match
        exception(workplaces=["onsite"], note=NOTE),  # does not match
        exception(OTHER_BOARD, workplaces=["hybrid"], note=NOTE),  # another board's
    ],
)
def test_days_not_stated_stays_without_a_matching_noted_exception(
    rule: BoardFilterException,
) -> None:
    result = apply_filter(
        [job_for("hybrid", False)],
        preset("remote_friendly"),
        include_unstated={},
        exceptions=[rule],
    )
    (match,) = result.matches
    assert match.workplace_note == "hybrid_days_not_stated"
    assert match.workplace_exception is None
    assert result.hybrid_days_not_stated == 1


def test_evidence_outranks_an_exception_note() -> None:
    rule = exception(workplaces=["hybrid"], note=NOTE)
    result = apply_filter(
        [job_for("hybrid", True)], preset("remote_friendly"), include_unstated={}, exceptions=[rule]
    )
    (match,) = result.matches
    assert match.workplace_note == "says_remote_friendly" and match.workplace_exception is None


def test_show_hidden_unstated_under_a_preset_brings_back_only_jobs_that_state_nothing() -> None:
    jobs = [job_for("unknown", False), job_for("unknown", True), job_for("onsite", False)]
    result = apply_filter(
        jobs, preset("remote_only"), include_unstated={}, show_hidden_unstated=True
    )
    assert [m.job.title for m in result.matches] == ["unknown"]
    assert result.hidden_unstated == 0


def test_a_board_including_unstated_under_a_preset_badges_by_reason() -> None:
    result = apply_filter(
        [job_for("unknown", False), job_for("unknown", True)],
        preset("remote_friendly"),
        include_unstated={BOARD: True},
        hybrid_too_heavy={BOARD},
    )
    # The remote-friendly one has stated something the board setting rules out,
    # so it does not come back in as "unstated".
    assert [(m.job.title, m.reason) for m in result.matches] == [("unknown", "unstated")]


def test_a_filter_saved_before_the_presets_is_custom_and_unchanged() -> None:
    assert JobFilter().workplace_mode == "custom"
    jobs = [job_for("hybrid", False), job_for("onsite", True), job_for("remote", True)]
    assert titles(jobs, JobFilter(workplaces=["remote"])) == ["remote+rf"]
