"""Display and wiring for the job filter: workplace words, form parsing, and the
one computation both `/jobs` and `/boards` show.

No SQL and no matching logic here -- matching is `jfl_intake.filtering` (pure),
storage is the repositories. This module only turns their outputs into what a
page says, the same split `jfl_web.boards` draws for check error codes.

**The employer's words are never rewritten.** A job whose workplace came with
the employer's own label (a Greenhouse custom field, e.g. "On-Site") is shown
with that label, verbatim; our canonical word is used only where the platform
gave an enum or a boolean, which are not anyone's prose.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import get_args

from jfl_core.models import (
    BoardFilterException,
    BoardJob,
    JobFilter,
    WatchedBoard,
    Workplace,
    WorkplaceMode,
)
from jfl_intake.filtering import FilterResult, apply_filter
from jfl_intake.workplace import effective_include_unstated, include_unstated_by_default

WORKPLACE_NAMES: dict[Workplace, str] = {
    "remote": "Remote",
    "hybrid": "Hybrid",
    "onsite": "On-site",
    "unknown": "Workplace not stated",
}
WORKPLACE_VALUES: tuple[Workplace, ...] = get_args(Workplace)

# The presets first: they are the main control, and Custom is the old checkboxes.
WORKPLACE_MODE_NAMES: dict[WorkplaceMode, str] = {
    "remote_only": "Remote only",
    "remote_friendly": "Remote friendly",
    "custom": "Custom",
}
WORKPLACE_MODE_VALUES: tuple[WorkplaceMode, ...] = get_args(WorkplaceMode)

# Generous for comma-separated alternatives and for a note in the owner's own
# words; the limit exists to bound a row, not to shape what anyone writes.
MAX_FILTER_TEXT = 500
MAX_NOTE_TEXT = 1000

# Rows rendered on /jobs before the list says "showing first X of N".
JOBS_PAGE_CAP = 300


class FormTooLongError(ValueError):
    """A submitted text field exceeds its limit. Rejected, never truncated --
    a silently shortened filter or note is not what the owner wrote.
    """


def workplace_display(job: BoardJob) -> str:
    if job.workplace != "unknown" and job.workplace_label:
        return job.workplace_label
    return WORKPLACE_NAMES[job.workplace]


def parse_workplaces(values: Iterable[str]) -> list[Workplace]:
    """The submitted checkbox values that are real workplaces, in canonical
    order. Anything else is ignored rather than stored.
    """
    chosen = set(values)
    return [w for w in WORKPLACE_VALUES if w in chosen]


def parse_workplace_mode(value: str) -> WorkplaceMode | None:
    """The submitted mode if it is one of the choices, else None -- which the
    route refuses, rather than quietly saving some other mode.
    """
    for mode in WORKPLACE_MODE_VALUES:
        if value == mode:
            return mode
    return None


def checked_text(value: str, limit: int) -> str:
    if len(value) > limit:
        raise FormTooLongError(f"That text is longer than {limit} characters.")
    return value


def include_unstated_map(boards: Iterable[WatchedBoard]) -> dict[uuid.UUID, bool]:
    return {
        b.id: effective_include_unstated(b.platform, b.include_unstated_workplace) for b in boards
    }


def unstated_setting_view(board: WatchedBoard) -> dict[str, object]:
    """What the board page says about its include-unstated setting."""
    return {
        "value": board.include_unstated_workplace,
        "effective": effective_include_unstated(board.platform, board.include_unstated_workplace),
        "default": include_unstated_by_default(board.platform),
    }


@dataclass(frozen=True, slots=True)
class BoardMatchCount:
    matching: int
    open_total: int


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


def match_counts_by_board(
    jobs: Sequence[BoardJob],
    saved: JobFilter,
    boards: Sequence[WatchedBoard],
    exceptions: Sequence[BoardFilterException],
) -> dict[uuid.UUID, BoardMatchCount]:
    """Per board, "N of M match" under exactly the filter /jobs applies."""
    result = filter_open_jobs(jobs, saved, boards, exceptions)
    totals: dict[uuid.UUID, int] = {}
    for job in jobs:
        totals[job.board_id] = totals.get(job.board_id, 0) + 1
    matching: dict[uuid.UUID, int] = {}
    for match in result.matches:
        matching[match.job.board_id] = matching.get(match.job.board_id, 0) + 1
    return {
        board.id: BoardMatchCount(matching.get(board.id, 0), totals.get(board.id, 0))
        for board in boards
    }
