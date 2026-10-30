"""Signed email-link tokens and the digest schedule. Pure."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from jfl_core.digest_schedule import DigestSchedule, DigestScheduleError
from jfl_core.email_links import UNSUBSCRIBE, sign_link_token, verify_link_token

SECRET = "s" * 43
USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


def test_a_token_round_trips_to_its_user() -> None:
    token = sign_link_token(SECRET, UNSUBSCRIBE, USER)
    assert verify_link_token(SECRET, UNSUBSCRIBE, token) == USER


def test_tampering_with_either_half_fails() -> None:
    token = sign_link_token(SECRET, UNSUBSCRIBE, USER)
    ident, _, signature = token.partition(".")
    other = uuid.uuid4()
    assert verify_link_token(SECRET, UNSUBSCRIBE, f"{other.hex}.{signature}") is None
    flipped = signature[:-1] + ("A" if signature[-1] != "A" else "B")
    assert verify_link_token(SECRET, UNSUBSCRIBE, f"{ident}.{flipped}") is None
    assert verify_link_token(SECRET, UNSUBSCRIBE, ident) is None
    assert verify_link_token(SECRET, UNSUBSCRIBE, "") is None


def test_a_token_for_one_user_never_verifies_as_another() -> None:
    token = sign_link_token(SECRET, UNSUBSCRIBE, USER)
    assert verify_link_token(SECRET, UNSUBSCRIBE, token) != uuid.uuid4()


def test_wrong_secret_wrong_purpose_and_empty_secret_all_fail() -> None:
    token = sign_link_token(SECRET, UNSUBSCRIBE, USER)
    assert verify_link_token("x" * 43, UNSUBSCRIBE, token) is None
    assert verify_link_token(SECRET, "something-else", token) is None
    assert verify_link_token("", UNSUBSCRIBE, token) is None
    with pytest.raises(ValueError):
        sign_link_token("", UNSUBSCRIBE, USER)


@pytest.mark.parametrize("garbage", ["....", "zz.zz", "a" * 500, "\x00.\x00", f"{USER}.abc"])
def test_garbage_is_just_none(garbage: str) -> None:
    assert verify_link_token(SECRET, UNSUBSCRIBE, garbage) is None


def utc(year: int, month: int, day: int, hour: int, minute: int) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)


def test_latest_due_is_today_once_the_hour_has_passed_else_yesterday() -> None:
    schedule = DigestSchedule(hour=8, timezone="Europe/London")
    # October: BST (UTC+1), so 08:00 local is 07:00 UTC.
    assert schedule.latest_due(utc(2026, 10, 20, 7, 30)) == utc(2026, 10, 20, 7, 0)
    assert schedule.latest_due(utc(2026, 10, 20, 6, 59)) == utc(2026, 10, 19, 7, 0)
    # After the clocks go back (25 Oct 2026) 08:00 local is 08:00 UTC.
    assert schedule.latest_due(utc(2026, 10, 26, 8, 1)) == utc(2026, 10, 26, 8, 0)


def test_schedule_from_env_defaults_and_validation() -> None:
    default = DigestSchedule.from_env({})
    assert (default.hour, default.timezone) == (8, "Europe/London")
    assert DigestSchedule.from_env({"JFL_DIGEST_HOUR": "6", "JFL_DIGEST_TIMEZONE": "UTC"}).hour == 6
    for bad in ({"JFL_DIGEST_HOUR": "24"}, {"JFL_DIGEST_HOUR": "x"}):
        with pytest.raises(DigestScheduleError):
            DigestSchedule.from_env(bad)
    with pytest.raises(DigestScheduleError):
        DigestSchedule.from_env({"JFL_DIGEST_TIMEZONE": "Mars/Olympus"})
