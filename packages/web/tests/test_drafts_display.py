"""The drafting screen's words and states, without a database: the headline
count, the four marks, each flagged sentence's next action, and where an
application is in the three steps to a CV -- `jfl_web.drafts`.
"""

from __future__ import annotations

from typing import Any

import pytest
from jfl_web.drafts import (
    VERDICT_WORDS,
    ChainTask,
    GenerationFailure,
    check_summary,
    download_name,
    headline,
    next_action,
    plan_steps,
    sentence_label,
    sentence_style,
    step_failure,
)


def _sentence(kind: str, verdict: str | None, drift: str | None = None) -> dict[str, Any]:
    return {"kind": kind, "verdict": verdict, "drift_label": drift, "text": "x"}


# --------------------------------------------------------------------------
# The headline
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        (
            (12, 3, 1),
            "12 of 16 claims trace to your confirmed facts; 3 need checking; 1 isn't supported.",
        ),
        (
            (1, 1, 2),
            "1 of 4 claims traces to your confirmed facts; 1 needs checking; 2 aren't supported.",
        ),
        ((5, 0, 1), "5 of 6 claims trace to your confirmed facts; 1 isn't supported."),
        ((0, 2, 0), "0 of 2 claims trace to your confirmed facts; 2 need checking."),
        ((18, 0, 0), "All 18 claims trace to your confirmed facts."),
        ((1, 0, 0), "Its one claim traces to your confirmed facts."),
        (
            (0, 0, 0),
            "Nothing in this draft makes a claim that could be checked against your facts.",
        ),
    ],
)
def test_the_headline_says_the_count_plainly(counts: tuple[int, int, int], expected: str) -> None:
    assert headline(*counts) == expected


def test_the_summary_counts_claims_only_and_lists_the_worst_first() -> None:
    sentences = [
        _sentence("title", None),
        _sentence("claim", "review", "scope_inflation"),
        _sentence("claim", "supported"),
        _sentence("framing", "supported", "framing"),
        _sentence("claim", "unsupported", "invented_quantity"),
    ]
    check = check_summary({"sentences": sentences})
    assert (check.claims, check.supported, check.review, check.unsupported) == (3, 1, 1, 1)
    assert check.not_checked == 2
    assert [s["verdict"] for s in check.flagged] == ["unsupported", "review"]
    assert check_summary(None).claims == 0


# --------------------------------------------------------------------------
# Framing is never supported
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        _sentence("framing", "supported", "framing"),
        _sentence("title", None),
        _sentence("title", "supported"),
        _sentence("claim", None),
    ],
)
def test_anything_not_a_checked_claim_reads_not_checked(sentence: dict[str, Any]) -> None:
    assert sentence_label(sentence) == "Not checked"
    assert sentence_style(sentence) == "verdict-not-checked"


def test_the_marks_are_plain_words() -> None:
    assert VERDICT_WORDS == {
        "supported": "Supported",
        "review": "Check this",
        "unsupported": "Not supported",
        "not_checked": "Not checked",
    }


# --------------------------------------------------------------------------
# What to do about a flagged sentence
# --------------------------------------------------------------------------


def test_a_supported_or_unchecked_sentence_needs_nothing() -> None:
    assert next_action(_sentence("claim", "supported")) is None
    assert next_action(_sentence("framing", "supported")) is None


def test_a_flagged_sentence_offers_both_rewording_and_adding_the_fact() -> None:
    action = next_action(_sentence("claim", "unsupported", "invented_quantity"))
    assert action is not None
    assert action.add_fact
    assert "number" in action.why
    assert action.reword.startswith("Reword it")


def test_a_contradicted_sentence_is_only_offered_rewording() -> None:
    """The facts say something different, so "add the fact" is the wrong advice."""
    action = next_action(_sentence("claim", "unsupported", "adjacency_substitution"))
    assert action is not None and not action.add_fact


def test_an_unknown_label_still_gets_an_action() -> None:
    action = next_action(_sentence("claim", "review", "something_new"))
    assert action is not None and action.why == "" and action.add_fact


# --------------------------------------------------------------------------
# The three steps
# --------------------------------------------------------------------------


def _states(plan: Any) -> list[str]:
    return [step.state for step in plan.steps]


def test_nothing_done_yet_makes_reading_the_ad_current_and_costs_all_three() -> None:
    plan = plan_steps(has_ad=True, ad_read=False, ad_reading=False, checked=False)
    assert _states(plan) == ["current", "waiting", "waiting"]
    assert plan.can_start and not plan.in_progress
    assert "about $0.52–0.82 in all" in plan.cost_line
    assert [s.cost for s in plan.steps] == ["about $0.01", "about $0.16", "about $0.35–0.65"]


def test_an_ad_being_read_runs_and_the_button_waits() -> None:
    plan = plan_steps(has_ad=True, ad_read=False, ad_reading=True, checked=False)
    assert _states(plan) == ["running", "waiting", "waiting"]
    assert plan.in_progress and not plan.can_start


def test_no_ad_blocks_and_says_so() -> None:
    plan = plan_steps(has_ad=False, ad_read=False, ad_reading=False, checked=False)
    assert plan.steps[0].state == "blocked"
    assert not plan.can_start and plan.blocked_message


def test_only_writing_left() -> None:
    plan = plan_steps(has_ad=True, ad_read=True, ad_reading=False, checked=True)
    assert _states(plan) == ["done", "done", "current"]
    assert plan.cost_line.startswith("Writing it costs about $0.35–0.65")


def test_a_chain_in_flight_shows_each_step_and_keeps_polling_between_steps() -> None:
    running = plan_steps(
        has_ad=True,
        ad_read=True,
        ad_reading=False,
        checked=False,
        chain=[ChainTask("generate_coverage", "running", has_next=True)],
    )
    assert _states(running) == ["done", "running", "waiting"]
    assert running.in_progress and not running.can_start and not running.finished

    # The check has finished and the draft is not queued yet: still going.
    between = plan_steps(
        has_ad=True,
        ad_read=True,
        ad_reading=False,
        checked=True,
        chain=[ChainTask("generate_coverage", "succeeded", has_next=True)],
    )
    assert between.in_progress and not between.finished


def test_a_finished_chain_offers_the_button_again() -> None:
    plan = plan_steps(
        has_ad=True,
        ad_read=True,
        ad_reading=False,
        checked=True,
        chain=[
            ChainTask("generate_coverage", "succeeded", has_next=True),
            ChainTask("generate_cv_draft", "succeeded"),
        ],
    )
    assert _states(plan) == ["done", "done", "current"]
    assert plan.finished and plan.can_start


def test_a_failed_step_is_named_and_can_be_retried() -> None:
    failure = GenerationFailure("No key.", fix_url="/settings", fix_label="Add an API key")
    plan = plan_steps(
        has_ad=True,
        ad_read=True,
        ad_reading=False,
        checked=False,
        chain=[ChainTask("generate_coverage", "failed", failure=failure, has_next=True)],
    )
    assert _states(plan) == ["done", "failed", "waiting"]
    assert plan.steps[1].detail == "No key." and plan.steps[1].fix_url == "/settings"
    assert plan.finished and plan.can_start


def test_a_cover_letter_press_names_the_letter() -> None:
    plan = plan_steps(
        has_ad=True, ad_read=True, ad_reading=False, checked=False, writing="cover_letter"
    )
    assert plan.steps[2].label == "Write the cover letter"
    assert "write the cover letter" in plan.cost_line


def test_a_failed_read_in_a_chain_reads_as_extraction_does() -> None:
    failure = step_failure(
        "extract_job_ad", "PermanentTaskError: extraction failed permanently: no_api_key"
    )
    assert "API key" in failure.message and failure.fix_url == "/settings"
    assert step_failure("extract_job_ad", None).message


def test_the_download_is_named_for_the_job() -> None:
    assert download_name("cv_bullets", "Senior Engineer, Acme!") == "cv-senior-engineer-acme.txt"
    assert download_name("cover_letter", None) == "cover-letter.txt"
