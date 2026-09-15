"""Which "what changed" events a user sees. Pure: no database, no network, no clock.

The feed (PLAN.md C7) shows events on watched boards -- new, gone, returned,
reposted -- **since the user last looked**, not since the service last checked.
A daily check that ran while nobody was reading is exactly what the feed is for;
measuring from the check would make every event news for one check's worth of
time and then silently gone.

**The rule, per event.** An event is named by `(job_id, kind, check_id)`; a
`JobFeedMark` is this user's record of it.

  * **No mark** -- the user has never been shown it. Visible iff it happened
    after they last looked (`event.at > last_looked_at`). Displaying it is what
    creates its mark, with `first_seen_at = now`.
  * **A mark** -- the mark decides, and `last_looked_at` does not. Visible iff it
    is not dismissed and was first seen less than `VISIBLE_FOR` ago. The mark
    wins even for an event dated after the last look, so a dismissal is final
    whatever the clocks did: "until they dismiss it" is the owner's rule, and an
    event reappearing after dismissal is the feed contradicting the user.

The 24 hours run from when the user first saw an event, not from when it
happened, so an event from a check at 06:00 read at 21:00 is still there the next
morning. It is visible while `now - first_seen_at < VISIBLE_FOR`; at exactly 24
hours it is gone.

**First visit.** A user who has never looked has no `last_looked_at`. Treating
that as "the beginning of time" would dump every event in the history on day one;
treating it as `now` would show an empty page to someone who just watched their
first boards and wants to know the feed works. **The choice made here:** a first
visit looks back `FIRST_VISIT_LOOKBACK` (7 days). Baselines are never events
(C4), so this cannot flood the page with a new board's existing jobs.

Filtering by the saved job filter is not this module's business; the caller runs
the same lens `/jobs` uses and creates marks only for what it actually displays.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from jfl_core.models import BoardJobEvent, BoardJobEventKind, JobFeedMark

VISIBLE_FOR = dt.timedelta(hours=24)
FIRST_VISIT_LOOKBACK = dt.timedelta(days=7)

EventKey = tuple[uuid.UUID, BoardJobEventKind, uuid.UUID]


def event_key(event: BoardJobEvent) -> EventKey:
    return (event.job.id, event.kind, event.check_id)


def mark_key(mark: JobFeedMark) -> EventKey:
    return (mark.job_id, mark.kind, mark.check_id)


def effective_last_looked(last_looked_at: dt.datetime | None, *, now: dt.datetime) -> dt.datetime:
    """`last_looked_at`, or `now - FIRST_VISIT_LOOKBACK` for a first visit."""
    return now - FIRST_VISIT_LOOKBACK if last_looked_at is None else last_looked_at


def mark_is_live(mark: JobFeedMark, *, now: dt.datetime) -> bool:
    """A mark that still keeps its event on the page."""
    return mark.dismissed_at is None and now - mark.first_seen_at < VISIBLE_FOR


def derive_since(
    last_looked_at: dt.datetime | None, live_marks: Iterable[JobFeedMark], *, now: dt.datetime
) -> dt.datetime:
    """How far back events must be derived to evaluate the rule: the last look,
    or earlier if a still-visible event happened before it. Exclusive, like
    `PostgresBoardRepository.events_since`, so it sits one microsecond before the
    oldest such event -- Postgres's own resolution -- to keep that event in.
    """
    since = effective_last_looked(last_looked_at, now=now)
    oldest = min((m.event_at for m in live_marks if mark_is_live(m, now=now)), default=None)
    if oldest is not None and oldest <= since:
        since = oldest - dt.timedelta(microseconds=1)
    return since


@dataclass(frozen=True, slots=True)
class FeedItem:
    event: BoardJobEvent
    # None: never shown to this user before -- the caller marks it as it displays it.
    mark: JobFeedMark | None


def visible_events(
    events: Sequence[BoardJobEvent],
    marks: Iterable[JobFeedMark],
    *,
    last_looked_at: dt.datetime | None,
    now: dt.datetime,
) -> list[FeedItem]:
    """The events this user sees now, newest first. See the module docstring."""
    since = effective_last_looked(last_looked_at, now=now)
    by_key = {mark_key(m): m for m in marks}
    items: list[FeedItem] = []
    for event in events:
        mark = by_key.get(event_key(event))
        if mark is None:
            if event.at > since:
                items.append(FeedItem(event=event, mark=None))
        elif mark_is_live(mark, now=now):
            items.append(FeedItem(event=event, mark=mark))
    return sorted(items, key=_newest_first)


# Within one check, the same kind order `PostgresBoardRepository` uses: "gone"
# before "returned" before "reposted" before "new". Only the time is reversed.
_KIND_ORDER: dict[BoardJobEventKind, int] = {"gone": 0, "returned": 1, "reposted": 2, "new": 3}


def _newest_first(item: FeedItem) -> tuple[float, int, str]:
    event = item.event
    return (-event.at.timestamp(), _KIND_ORDER[event.kind], event.job.title)
