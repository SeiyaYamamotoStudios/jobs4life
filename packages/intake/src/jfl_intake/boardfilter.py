"""The saved job filter applied to jobs on watched boards: the one computation
`/jobs`, `/boards`, `/changes` and the emailed digest all share.

It lived in `jfl_web.jobfilter` until the digest (PLAN.md C9). It depends only on
`jfl_core` models and `jfl_intake.filtering`, and the worker -- which builds the
email -- must not import the web package, so the pure computation moved down here
and the web module re-imports it. There is one filter, not a copy that could
drift: an email that included a job the page hides, or hid one the page shows,
would be the tool contradicting itself in the owner's inbox.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from jfl_core.models import BoardFilterException, BoardJob, JobFilter, WatchedBoard

from jfl_intake.filtering import FilterResult, apply_filter
from jfl_intake.workplace import effective_include_unstated


def include_unstated_map(boards: Iterable[WatchedBoard]) -> dict[uuid.UUID, bool]:
    return {
        b.id: effective_include_unstated(b.platform, b.include_unstated_workplace) for b in boards
    }


def filter_open_jobs(
    jobs: Sequence[BoardJob],
    saved: JobFilter,
    boards: Sequence[WatchedBoard],
    exceptions: Sequence[BoardFilterException],
    *,
    show_hidden_unstated: bool = False,
) -> FilterResult[BoardJob]:
    return apply_filter(
        jobs,
        saved,
        include_unstated=include_unstated_map(boards),
        exceptions=exceptions,
        show_hidden_unstated=show_hidden_unstated,
        hybrid_too_heavy=frozenset(b.id for b in boards if b.hybrid_too_heavy),
    )


__all__ = ["filter_open_jobs", "include_unstated_map"]
