"""Unit tests for A6's absolute-plus-relative timestamp pairing. Pure
functions, no database, no network -- see jfl_web.timeformat for the
thresholds this pins down.
"""

from __future__ import annotations

import datetime as dt

from jfl_web.timeformat import absolute, compact_relative, humanize, relative, time_compact

NOW = dt.datetime(2026, 9, 8, 12, 0, 0, tzinfo=dt.UTC)  # a Tuesday


def test_relative_just_now_covers_both_directions() -> None:
    assert relative(NOW - dt.timedelta(seconds=30), now=NOW) == "just now"
    assert relative(NOW + dt.timedelta(seconds=30), now=NOW) == "just now"


def test_relative_minutes() -> None:
    assert relative(NOW - dt.timedelta(minutes=1), now=NOW) == "1 minute ago"
    assert relative(NOW - dt.timedelta(minutes=5), now=NOW) == "5 minutes ago"
    assert relative(NOW + dt.timedelta(minutes=5), now=NOW) == "in 5 minutes"


def test_relative_hours() -> None:
    assert relative(NOW - dt.timedelta(hours=1), now=NOW) == "1 hour ago"
    assert relative(NOW - dt.timedelta(hours=3), now=NOW) == "3 hours ago"


def test_relative_yesterday_and_tomorrow() -> None:
    assert relative(NOW - dt.timedelta(hours=30), now=NOW) == "yesterday"
    assert relative(NOW + dt.timedelta(hours=30), now=NOW) == "tomorrow"


def test_relative_days() -> None:
    assert relative(NOW - dt.timedelta(days=3), now=NOW) == "3 days ago"
    assert relative(NOW + dt.timedelta(days=10), now=NOW) == "in 10 days"


def test_relative_months() -> None:
    assert relative(NOW - dt.timedelta(days=60), now=NOW) == "2 months ago"


def test_relative_years() -> None:
    assert relative(NOW - dt.timedelta(days=400), now=NOW) == "1 year ago"
    assert relative(NOW - dt.timedelta(days=800), now=NOW) == "2 years ago"


def test_absolute_has_no_leading_zero_on_the_day() -> None:
    assert absolute(dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.UTC)) == "Tue 8 Sep 2026"


def test_absolute_converts_utc_to_london_across_the_bst_boundary() -> None:
    # 23:30 UTC on 7 Sep is after midnight in London (BST, UTC+1) -- the 8th.
    late = dt.datetime(2026, 9, 7, 23, 30, tzinfo=dt.UTC)
    assert absolute(late) == "Tue 8 Sep 2026"


def test_humanize_pairs_both_halves() -> None:
    value = NOW - dt.timedelta(days=3)
    assert humanize(value, now=NOW) == "Sat 5 Sep 2026, 3 days ago"


def test_relative_defaults_to_the_real_now_when_not_given() -> None:
    """Not pinned to an exact string -- just proves the `now` parameter is
    genuinely optional and produces something sane for a very old timestamp.
    """
    ancient = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
    assert "ago" in relative(ancient)


def test_compact_relative_shortens_only_the_long_units() -> None:
    assert compact_relative(NOW - dt.timedelta(minutes=13), now=NOW) == "13 min ago"
    assert compact_relative(NOW - dt.timedelta(minutes=1), now=NOW) == "1 min ago"
    assert compact_relative(NOW - dt.timedelta(hours=3), now=NOW) == "3 hr ago"
    assert compact_relative(NOW - dt.timedelta(hours=30), now=NOW) == "yesterday"
    assert compact_relative(NOW - dt.timedelta(days=8), now=NOW) == "8 days ago"
    assert compact_relative(NOW - dt.timedelta(days=60), now=NOW) == "2 mo ago"
    assert compact_relative(NOW - dt.timedelta(days=800), now=NOW) == "2 yr ago"
    assert compact_relative(NOW + dt.timedelta(minutes=5), now=NOW) == "in 5 min"
    assert compact_relative(NOW - dt.timedelta(seconds=10), now=NOW) == "just now"


def test_time_compact_keeps_both_halves_one_in_the_title() -> None:
    """The cell shows the short relative form; the absolute date (with the
    minute) and the long relative form move into `title`, and the instant into
    `datetime` -- A6's pairing moved out of the column's way, not dropped."""
    value = dt.datetime(2026, 9, 8, 11, 47, tzinfo=dt.UTC)
    html = str(time_compact(value, now=NOW))
    assert html == (
        '<time datetime="2026-09-08T11:47:00+00:00" '
        'title="Tue 8 Sep 2026, 12:47 · 13 minutes ago">13 min ago</time>'
    )
