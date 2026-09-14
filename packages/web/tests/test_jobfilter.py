"""Display and form helpers for the job filter. No database, no network."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from jfl_core.models import BoardJob, Workplace
from jfl_web.jobfilter import (
    FormTooLongError,
    checked_text,
    parse_workplaces,
    workplace_display,
)

NOW = dt.datetime(2026, 9, 15, tzinfo=dt.UTC)


def board_job(workplace: Workplace, label: str | None) -> BoardJob:
    return BoardJob(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        board_id=uuid.uuid4(),
        external_id="1",
        title="Engineering Manager",
        fingerprint="engineering manager|london",
        workplace=workplace,
        workplace_label=label,
        first_seen_check_id=uuid.uuid4(),
        first_seen_at=NOW,
        last_seen_at=NOW,
    )


def test_the_employers_label_is_shown_verbatim() -> None:
    assert workplace_display(board_job("onsite", "On-Site")) == "On-Site"
    assert workplace_display(board_job("hybrid", "Hybrid (Travel-Required)")) == (
        "Hybrid (Travel-Required)"
    )


def test_without_a_label_the_canonical_word_is_used() -> None:
    assert workplace_display(board_job("remote", None)) == "Remote"
    assert workplace_display(board_job("onsite", None)) == "On-site"
    assert workplace_display(board_job("unknown", None)) == "Workplace not stated"


def test_parse_workplaces_keeps_only_real_values_in_canonical_order() -> None:
    assert parse_workplaces(["unknown", "office", "remote", "remote"]) == ["remote", "unknown"]


def test_overlong_text_is_rejected_never_truncated() -> None:
    assert checked_text("a" * 10, 10) == "a" * 10
    with pytest.raises(FormTooLongError):
        checked_text("a" * 11, 10)
