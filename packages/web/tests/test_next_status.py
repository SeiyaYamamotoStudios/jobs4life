"""The pipeline order behind the status quick-actions.

The quick action is the most-used control in the app, and getting `next_status`
wrong would send an application somewhere the user did not ask for with a single
click -- so the ordering is asserted rather than trusted.
"""

from __future__ import annotations

import pytest
from jfl_web.routes.applications import STATUSES, next_status


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ("interested", "applied"),
        ("applied", "screening"),
        ("screening", "interviewing"),
        ("interviewing", "offer"),
    ],
)
def test_each_stage_advances_one_step(current: str, expected: str) -> None:
    assert next_status(current) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("terminal", ["offer", "rejected", "withdrawn"])
def test_outcomes_have_nothing_to_advance_to(terminal: str) -> None:
    """`offer` ends the line; `rejected` and `withdrawn` are outcomes, not
    stages, so none of them offers a next step."""
    assert next_status(terminal) is None  # type: ignore[arg-type]


def test_every_status_is_handled() -> None:
    """A status added later must not fall through to an exception -- the quick
    action degrades to "no next step" rather than breaking the detail page."""
    for status in STATUSES:
        next_status(status)
