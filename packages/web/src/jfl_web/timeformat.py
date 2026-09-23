"""Absolute-plus-relative timestamps -- slice A6.

Per CLAUDE.md's 2026-09-07 decision, every event shows both an absolute date
and a relative one, because relative time on its own is exactly what a chat
conversation loses: "tomorrow" stops meaning anything once the conversation is
a week old. Storage is UTC throughout; the absolute half renders in
Europe/London, a fixed choice for now -- a per-user timezone preference is a
later-slice concern, not this one.

The exact minute is not thrown away: every place this is rendered also carries
the full ISO-8601 instant in a `title` attribute, so it is one hover away.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from markupsafe import Markup

LONDON = ZoneInfo("Europe/London")

_MINUTE = 60
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR
_MONTH = 30 * _DAY
_YEAR = 365 * _DAY


def relative(value: dt.datetime, *, now: dt.datetime | None = None) -> str:
    """ "3 days ago" / "in 2 hours" / "just now".

    Coarse on purpose -- this is for orientation, not a stopwatch. Thresholds:

      * under 60 seconds either side       -> "just now"
      * under 60 minutes                   -> minutes
      * under 24 hours                     -> hours
      * under 2 days                       -> "yesterday" / "tomorrow"
      * under 30 days                      -> days
      * under 365 days                     -> months (delta // 30 days)
      * beyond that                        -> years (delta // 365 days)
    """
    now = now if now is not None else dt.datetime.now(dt.UTC)
    delta_seconds = (value - now).total_seconds()
    future = delta_seconds > 0
    magnitude = abs(delta_seconds)

    if magnitude < _MINUTE:
        return "just now"
    if magnitude < _HOUR:
        n, unit = int(magnitude // _MINUTE), "minute"
    elif magnitude < _DAY:
        n, unit = int(magnitude // _HOUR), "hour"
    elif magnitude < 2 * _DAY:
        return "tomorrow" if future else "yesterday"
    elif magnitude < _MONTH:
        n, unit = int(magnitude // _DAY), "day"
    elif magnitude < _YEAR:
        n, unit = int(magnitude // _MONTH), "month"
    else:
        n, unit = int(magnitude // _YEAR), "year"

    unit = unit if n == 1 else f"{unit}s"
    return f"in {n} {unit}" if future else f"{n} {unit} ago"


def absolute(value: dt.datetime) -> str:
    """ "Tue 8 Sep 2026" in Europe/London -- no leading zero on the day."""
    local = value.astimezone(LONDON)
    return f"{local:%a} {local.day} {local:%b %Y}"


def humanize(value: dt.datetime, *, now: dt.datetime | None = None) -> str:
    """ "Tue 8 Sep 2026, 3 days ago" -- the pairing A6 requires everywhere an
    event or record timestamp is shown.
    """
    return f"{absolute(value)}, {relative(value, now=now)}"


# The compact form's short units. Only minutes, hours, months and years get
# abbreviated -- "days" is already short, and "yesterday" reads better than
# any abbreviation of it.
_COMPACT_UNITS = {
    "minute": "min",
    "minutes": "min",
    "hour": "hr",
    "hours": "hr",
    "month": "mo",
    "months": "mo",
    "year": "yr",
    "years": "yr",
}


def compact_relative(value: dt.datetime, *, now: dt.datetime | None = None) -> str:
    """ "13 min ago" / "3 hr ago" / "yesterday" / "8 days ago" / "2 mo ago".

    The same thresholds as `relative`, with the long units shortened. For a
    table cell, where the full "Wed 23 Sep 2026, 13 minutes ago" repeated in
    two columns was most of /boards' width. It is never shown alone: the
    `time_compact` filter puts the full `humanize` form in the element's
    `title`, so the absolute date is one hover away, and the instant itself in
    `datetime`.
    """
    words = relative(value, now=now).split(" ")
    return " ".join(_COMPACT_UNITS.get(word, word) for word in words)


def absolute_with_time(value: dt.datetime) -> str:
    """ "Wed 23 Sep 2026, 13:04" in Europe/London."""
    local = value.astimezone(LONDON)
    return f"{absolute(value)}, {local:%H:%M}"


def time_compact(value: dt.datetime, *, now: dt.datetime | None = None) -> Markup:
    """`<time datetime="…" title="Wed 23 Sep 2026, 13:04 · 13 minutes ago">13 min ago</time>`.

    The compact cell form of A6's pairing: the relative half is what the cell
    shows, the absolute half is in the `title` (with the minute, which the
    visible form never had room for), and the machine-readable instant is in
    `datetime`. Neither half is dropped -- one moved out of the column's way.
    """
    title = f"{absolute_with_time(value)} · {relative(value, now=now)}"
    return Markup('<time datetime="{}" title="{}">{}</time>').format(
        value.isoformat(), title, compact_relative(value, now=now)
    )
