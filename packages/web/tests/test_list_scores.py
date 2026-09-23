"""The applications list's two scores, and the orderings it offers -- no
database, no model.

The rule under test is CLAUDE.md's domain 4: two axes, never one composite.
On the list that means two cells, and sorting by one axis alone or by none;
never a ranking by a blend.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from jfl_core.models import Application, ApplicationScore
from jfl_web.scores import (
    COULD_GET_LABEL,
    SORTS,
    WANT_IT_LABEL,
    RowScore,
    row_score,
    sort_applications,
)

NOW = dt.datetime(2026, 9, 23, 9, 0, tzinfo=dt.UTC)
USER = uuid.uuid4()


def _app(title: str) -> Application:
    return Application(
        id=uuid.uuid4(),
        user_id=USER,
        title=title,
        status="interested",
        created_at=NOW,
        updated_at=NOW,
    )


def _score(
    status: str = "done",
    *,
    could: int | None = 5,
    want: int | None = 5,
    error_code: str | None = None,
) -> ApplicationScore:
    return ApplicationScore(
        id=uuid.uuid4(),
        application_id=uuid.uuid4(),
        status=status,  # type: ignore[arg-type]
        error_code=error_code,  # type: ignore[arg-type]
        could_get_score=could if status == "done" else None,
        could_get_assessment="",
        want_it_score=want if status == "done" else None,
        want_it_assessment="",
        model=None,
        cost_usd=Decimal("0"),
        trace_id=None,
        created_at=NOW,
        updated_at=NOW,
    )


class TestRowScore:
    def test_no_run_is_unscored(self) -> None:
        assert row_score(None, None) == RowScore(state="unscored")

    def test_a_pending_run_is_scoring_and_a_noted_one_is_retrying(self) -> None:
        assert row_score(_score("pending"), None).state == "scoring"
        assert row_score(_score("pending", error_code="model_error"), None).state == "retrying"

    def test_a_re_score_in_flight_keeps_the_last_numbers(self) -> None:
        done = _score(could=7, want=2)
        row = row_score(_score("pending"), done)
        assert row.state == "scoring"
        assert (row.could_get, row.want_it, row.has_numbers) == (7, 2, True)

    def test_a_failed_run_with_no_earlier_one_has_no_numbers(self) -> None:
        row = row_score(_score("failed", error_code="model_error"), None)
        assert row.state == "failed"
        assert not row.has_numbers

    def test_the_row_carries_two_numbers_and_no_third(self) -> None:
        """No field on the row combines the axes -- there is nothing a template
        could render as an overall score."""
        row = row_score(_score(could=7, want=2), _score(could=7, want=2))
        numeric = {
            name: getattr(row, name)
            for name in RowScore.__slots__
            if isinstance(getattr(row, name), int) and not isinstance(getattr(row, name), bool)
        }
        assert numeric == {"could_get": 7, "want_it": 2}


class TestSort:
    def _rows(self) -> tuple[list[Application], dict[uuid.UUID, RowScore]]:
        # a: would love it, will not get it. b: would walk into it, dislikes it.
        # c: never scored. d: scored, but nothing to measure "want" against.
        a, b, c, d = _app("a"), _app("b"), _app("c"), _app("d")
        rows = {
            a.id: row_score(_score(could=3, want=9), _score(could=3, want=9)),
            b.id: row_score(_score(could=8, want=1), _score(could=8, want=1)),
            c.id: row_score(None, None),
            d.id: row_score(_score(could=5, want=None), _score(could=5, want=None)),
        }
        return [a, b, c, d], rows

    def test_the_offered_orders_name_one_axis_or_none(self) -> None:
        assert SORTS == {
            "updated": "Recently updated",
            "could_get": COULD_GET_LABEL,
            "want_it": WANT_IT_LABEL,
        }

    def test_by_could_get_alone(self) -> None:
        apps, rows = self._rows()
        assert [a.title for a in sort_applications(apps, rows, "could_get")] == [
            "b",
            "d",
            "a",
            "c",
        ]

    def test_by_want_it_alone_with_no_number_last(self) -> None:
        apps, rows = self._rows()
        assert [a.title for a in sort_applications(apps, rows, "want_it")] == [
            "a",
            "b",
            "c",
            "d",
        ]

    def test_the_other_axis_is_never_a_tie_break(self) -> None:
        """Equal on the chosen axis keeps the incoming (most recently updated)
        order, whatever the other axis says."""
        x, y = _app("x"), _app("y")
        rows = {
            x.id: row_score(_score(could=6, want=1), _score(could=6, want=1)),
            y.id: row_score(_score(could=6, want=9), _score(could=6, want=9)),
        }
        assert [a.title for a in sort_applications([x, y], rows, "could_get")] == ["x", "y"]

    def test_an_unknown_sort_leaves_the_order_alone(self) -> None:
        apps, rows = self._rows()
        assert sort_applications(apps, rows, "blend") == apps
