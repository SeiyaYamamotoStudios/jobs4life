"""When the daily digest is due: a local hour in a named timezone. Pure.

Shared by the web app (which tells the user when to expect the email) and the
worker (which decides who is due), so the two cannot disagree about "08:00".
Each reads the two variables once at its own environment boundary and builds a
`DigestSchedule`; nothing below that touches the environment.

**The rule.** The digest for a day is due from `hour:00` local time. At any
moment the *most recent* due instant is `latest_due(now)`; a user is due when
their last completed run is older than that. Expressed that way a worker that was
down over 08:00 sends the missed digest on restart (one, not a backlog), a
daylight-saving change moves the instant with the wall clock rather than drifting
by an hour for half the year, and "has today's digest run?" is a single
comparison with no calendar arithmetic in the scheduler.

The timezone is the deployment's, not the user's: users have no stored timezone
and inventing a per-user setting for a one-owner-sized feature would be scope.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_DIGEST_HOUR = 8
DEFAULT_DIGEST_TIMEZONE = "Europe/London"


class DigestScheduleError(ValueError):
    """Configuration names an impossible hour or an unknown timezone."""


@dataclass(frozen=True, slots=True)
class DigestSchedule:
    hour: int = DEFAULT_DIGEST_HOUR
    timezone: str = DEFAULT_DIGEST_TIMEZONE

    def __post_init__(self) -> None:
        if not 0 <= self.hour <= 23:
            raise DigestScheduleError("JFL_DIGEST_HOUR must be a whole hour from 0 to 23")
        self.tz  # noqa: B018 -- validates the name now, at boot, not at the first run

    @property
    def tz(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            raise DigestScheduleError("JFL_DIGEST_TIMEZONE is not a known timezone") from None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DigestSchedule:
        source = os.environ if env is None else env
        raw_hour = (source.get("JFL_DIGEST_HOUR") or "").strip()
        try:
            hour = int(raw_hour) if raw_hour else DEFAULT_DIGEST_HOUR
        except ValueError:
            raise DigestScheduleError("JFL_DIGEST_HOUR must be a whole hour from 0 to 23") from None
        timezone = (source.get("JFL_DIGEST_TIMEZONE") or "").strip() or DEFAULT_DIGEST_TIMEZONE
        return cls(hour=hour, timezone=timezone)

    def latest_due(self, now: dt.datetime) -> dt.datetime:
        """The most recent `hour:00` local time at or before `now`, in UTC."""
        tz = self.tz
        local = now.astimezone(tz)
        candidate = dt.datetime.combine(local.date(), dt.time(self.hour), tzinfo=tz)
        if candidate > now:
            candidate = dt.datetime.combine(
                local.date() - dt.timedelta(days=1), dt.time(self.hour), tzinfo=tz
            )
        return candidate.astimezone(dt.UTC)

    def describe(self) -> str:
        """ "08:00 Europe/London", for the settings page."""
        return f"{self.hour:02d}:00 {self.timezone}"
