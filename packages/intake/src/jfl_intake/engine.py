"""The check engine: what one adapter result means for a board's history.

Pure -- no database, no network, no clock. It takes the board's state as read
under a row lock, the adapter's result, and the moment the result was observed,
and returns a `CheckPlan` that `jfl_core.storage.boards` applies verbatim. That
split is what lets every rule below be unit-tested directly.

**The rule everything rests on: only a complete, successful check may close a
presence interval.** A job absent from a check is "gone" only if that check
provably saw the whole board. A check that was unreachable, errored, partial or
truncated is recorded with that status and changes no job's state. A failed
fetch is not evidence that jobs vanished, and writing false disappearances into
the history would corrupt exactly what the owner relies on -- the history is the
product, and a history that says forty roles closed on the day an API timed out
is worse than no history. This function is where that rule is enforced: every
non-complete result returns a plan with no job changes before any diffing
happens, so there is no code path from a partial result to a closed interval.

**The baseline.** The first complete check of a board opens an interval for
every job it saw and marks nothing new. Those jobs were open when watching
started; announcing all 595 of Anthropic's as "new" would make the feed noise
from its first day.

**The drop guard** holds a complete check that returns far fewer jobs than the
board currently has open. It is recorded as `held`, closes nothing, and flags
the board. This comes from an observed failure, not an invented one: Workday
answered a bad request with HTTP 200 and zero jobs, which is exactly what a
broken adapter or a quietly changed API looks like, and it is far likelier than
an employer closing most of its roles overnight. The numbers:

  * `DROP_GUARD_RATIO = 0.5` -- hold when fewer than half the open jobs remain.
    Ordinary churn is a few percent a day (postings live weeks to months); even
    a hiring freeze tends to take roles down over days. Losing half in one check
    is an order of magnitude past normal, and the cost is asymmetric: a false
    hold costs one click to accept, a false apply writes false "gone" events
    and, when the adapter recovers, false "returned" ones;
  * `DROP_GUARD_MIN_PREVIOUS = 10` -- the ratio only applies to boards with at
    least ten open jobs. Below that, halving is two to five roles, which one
    team finishing a hiring round explains, and holding it would flag small
    boards for normal behaviour;
  * `DROP_GUARD_EMPTY_MIN_PREVIOUS = 3` -- separately, a board with three or
    more open jobs that suddenly returns *zero* is held. Zero is the exact shape
    of the observed failure, so it gets a lower floor than a partial drop. A
    board going from one or two jobs to none is normal and is applied.

A held board stays flagged; while it is, each later collapse is held too. It
clears when a complete check comes back above the threshold (the adapter
recovered) or when a person accepts the drop (`drop_accepted`), after which the
next complete check is applied whatever its count.

**Reposted** is a *new* job (an external id never seen on this board) whose
fingerprint matches a job on the same board whose interval closed within
`REPOST_WINDOW` (60 days) -- including a job closing in this same check, which
is the commonest repost of all: taken down and put back up between two checks.
**Returned** is the same external id reopening. They are different events with
different code paths and must stay distinct: a returned job cannot be a repost
(its id is not new), and a job being returned is never a repost *source*.

Each closed job can be the source of at most one repost, matched most recently
closed first. Without that, one closed role would be reported as reposted by
every same-titled posting for the next 60 days.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections import defaultdict
from dataclasses import dataclass

from jfl_core.models import (
    BoardCheckState,
    CheckPlan,
    ObservedJob,
    PlannedNewJob,
    PlannedSighting,
)

from jfl_intake.adapters.base import FetchResult

REPOST_WINDOW = dt.timedelta(days=60)
DROP_GUARD_RATIO = 0.5
DROP_GUARD_MIN_PREVIOUS = 10
DROP_GUARD_EMPTY_MIN_PREVIOUS = 3


@dataclass(frozen=True, slots=True)
class DropGuard:
    ratio: float = DROP_GUARD_RATIO
    min_previous: int = DROP_GUARD_MIN_PREVIOUS
    empty_min_previous: int = DROP_GUARD_EMPTY_MIN_PREVIOUS

    def trips(self, *, previous: int, seen: int) -> bool:
        """`previous` is the number of jobs currently open on the board, which is
        exactly what the last applied complete check saw.
        """
        if seen == 0 and previous >= self.empty_min_previous:
            return True
        return previous >= self.min_previous and seen < previous * self.ratio


DEFAULT_DROP_GUARD = DropGuard()


def _dedupe(jobs: tuple[ObservedJob, ...]) -> list[ObservedJob]:
    seen: dict[str, ObservedJob] = {}
    for job in jobs:
        seen.setdefault(job.external_id, job)
    return list(seen.values())


def plan_check(
    state: BoardCheckState,
    result: FetchResult,
    *,
    observed_at: dt.datetime,
    repost_window: dt.timedelta = REPOST_WINDOW,
    drop_guard: DropGuard = DEFAULT_DROP_GUARD,
) -> CheckPlan:
    if (result.status == "complete") != (result.error_code is None):
        raise ValueError("a fetch result carries an error code if and only if it is not complete")

    jobs = _dedupe(result.jobs)
    board_id = state.board_id
    jobs_seen = len(jobs)
    expected_total = result.expected_total

    # THE RULE. Anything short of a complete fetch changes no job's state. See
    # the module docstring; nothing below this line runs for a partial result.
    if result.status != "complete":
        return CheckPlan(
            board_id=board_id,
            status=result.status,
            error_code=result.error_code,
            jobs_seen=jobs_seen,
            expected_total=expected_total,
        )

    if state.baseline_check_id is None:
        return CheckPlan(
            board_id=board_id,
            status="complete",
            jobs_seen=jobs_seen,
            expected_total=expected_total,
            is_baseline=True,
            new_jobs=[PlannedNewJob(job=job) for job in jobs],
        )

    known_open = [k for k in state.known_jobs if k.is_open]
    if not state.drop_accepted and drop_guard.trips(previous=len(known_open), seen=jobs_seen):
        return CheckPlan(
            board_id=board_id,
            status="held",
            error_code="drop_guard",
            jobs_seen=jobs_seen,
            expected_total=expected_total,
        )

    known = {k.external_id: k for k in state.known_jobs}
    observed_ids = {job.external_id for job in jobs}

    new: list[ObservedJob] = []
    returned: list[PlannedSighting] = []
    still_open: list[PlannedSighting] = []
    for job in jobs:
        record = known.get(job.external_id)
        if record is None:
            new.append(job)
        elif record.is_open:
            still_open.append(PlannedSighting(job_id=record.job_id, job=job))
        else:
            returned.append(PlannedSighting(job_id=record.job_id, job=job))
    gone = [k.job_id for k in known_open if k.external_id not in observed_ids]

    return CheckPlan(
        board_id=board_id,
        status="complete",
        jobs_seen=jobs_seen,
        expected_total=expected_total,
        new_jobs=_match_reposts(state, new, observed_ids, observed_at, repost_window),
        returned=returned,
        still_open=still_open,
        gone_job_ids=gone,
    )


def _match_reposts(
    state: BoardCheckState,
    new: list[ObservedJob],
    observed_ids: set[str],
    observed_at: dt.datetime,
    window: dt.timedelta,
) -> list[PlannedNewJob]:
    cutoff = observed_at - window
    candidates: dict[str, list[tuple[dt.datetime, uuid.UUID]]] = defaultdict(list)
    for record in state.known_jobs:
        if record.has_repost_successor or record.external_id in observed_ids:
            # Already reposted once; or present in this check (still open, or
            # returning) -- a job that is here was not replaced by anything.
            continue
        if record.is_open:
            # Open and absent from a complete check: it closes now, at
            # `observed_at`, which is inside any window.
            closed_at = observed_at
        elif record.last_closed_at is not None and record.last_closed_at >= cutoff:
            closed_at = record.last_closed_at
        else:
            continue
        candidates[record.fingerprint].append((closed_at, record.job_id))

    for sources in candidates.values():
        # Most recently closed first; the id only breaks exact ties, so the
        # match is deterministic.
        sources.sort(key=lambda c: (c[0], str(c[1])), reverse=True)

    planned: list[PlannedNewJob] = []
    for job in new:
        remaining = candidates.get(job.fingerprint)
        source = remaining.pop(0)[1] if remaining else None
        planned.append(PlannedNewJob(job=job, reposted_from_job_id=source))
    return planned
