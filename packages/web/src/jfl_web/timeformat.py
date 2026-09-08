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
