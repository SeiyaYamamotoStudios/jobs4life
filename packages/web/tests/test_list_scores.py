"""The applications list's two scores, and the orderings it offers -- no
database, no model.

The rule under test is CLAUDE.md's domain 4: two axes, never one composite.
On the list that means two cells, and sorting by one column at a time --
each score column by its own axis alone; never a ranking by a blend.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from jfl_core.models import Application, ApplicationScore, ApplicationStatus
from jfl_web.scores import (
    COULD_GET_LABEL,
    SORT_COLUMNS,
    WANT_IT_LABEL,
    RowScore,
    parse_sort,
    row_score,
    sort_applications,
    sort_headers,
    sort_query,
)

NOW = dt.datetime(2026, 9, 23, 9, 0, tzinfo=dt.UTC)
USER = uuid.uuid4()


def _app(
    title: str,
    *,
    employer: str | None = None,
    status: ApplicationStatus = "interested",
    updated: dt.datetime = NOW,
) -> Application:
    return Application(
        id=uuid.uuid4(),
        user_id=USER,
        title=title,
        employer=employer,
        status=status,
        created_at=NOW,
        updated_at=updated,
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

    def test_every_column_but_the_actions_is_a_sort_and_none_is_a_blend(self) -> None:
        assert SORT_COLUMNS == {
            "title": "Title",
            "employer": "Employer",
            "status": "Status",
            "could_get": COULD_GET_LABEL,
            "want_it": WANT_IT_LABEL,
            "updated": "Updated",
        }

    def test_by_could_get_alone(self) -> None:
        apps, rows = self._rows()
        assert _titles(sort_applications(apps, rows, "could_get")) == ["b", "d", "a", "c"]

    def test_by_want_it_alone_with_no_number_last(self) -> None:
        apps, rows = self._rows()
        assert _titles(sort_applications(apps, rows, "want_it")) == ["a", "b", "c", "d"]

    def test_the_axes_disagreeing_give_opposite_orders_and_no_blend(self) -> None:
        """a is 3/9, b is 8/1. A blend (sum 12 vs 9, or any weighting) would put
        them in one fixed order; each axis alone puts them in opposite orders,
        and reversing one axis gives exactly the other axis's reverse."""
        apps, rows = self._rows()
        by_could = _titles(sort_applications(apps, rows, "could_get", "desc"))
        by_want = _titles(sort_applications(apps, rows, "want_it", "desc"))
        assert by_could.index("b") < by_could.index("a")
        assert by_want.index("a") < by_want.index("b")
        assert _titles(sort_applications(apps, rows, "could_get", "asc")) == ["a", "d", "b", "c"]

    def test_unscored_rows_go_last_in_both_directions(self) -> None:
        apps, rows = self._rows()
        for key in ("could_get", "want_it"):
            for direction in ("asc", "desc"):
                ordered = _titles(sort_applications(apps, rows, key, direction))
                assert ordered[-1] in {"c", "d"} and "c" in ordered[-2:], (key, direction)
        assert _titles(sort_applications(apps, rows, "want_it", "asc")) == ["b", "a", "c", "d"]

    def test_the_other_axis_is_never_a_tie_break(self) -> None:
        """Equal on the chosen axis: most recently updated first, whatever the
        other axis says -- and in both directions."""
        x = _app("x", updated=NOW)
        y = _app("y", updated=NOW - dt.timedelta(hours=1))
        rows = {
            x.id: row_score(_score(could=6, want=1), _score(could=6, want=1)),
            y.id: row_score(_score(could=6, want=9), _score(could=6, want=9)),
        }
        for direction in ("asc", "desc"):
            assert _titles(sort_applications([y, x], rows, "could_get", direction)) == ["x", "y"]

    def test_words_and_status_and_time(self) -> None:
        first = _app("beta", employer="Zeta Fictional", status="offer", updated=NOW)
        second = _app(
            "Alpha", employer=None, status="interested", updated=NOW - dt.timedelta(days=1)
        )
        third = _app(
            "gamma",
            employer="alpha fictional",
            status="applied",
            updated=NOW - dt.timedelta(days=2),
        )
        apps = [first, second, third]
        assert _titles(sort_applications(apps, {}, "title", "asc")) == ["Alpha", "beta", "gamma"]
        assert _titles(sort_applications(apps, {}, "title", "desc")) == ["gamma", "beta", "Alpha"]
        # No employer recorded goes last either way, like an unscored row.
        assert _titles(sort_applications(apps, {}, "employer", "asc")) == ["gamma", "beta", "Alpha"]
        assert _titles(sort_applications(apps, {}, "employer", "desc")) == [
            "beta",
            "gamma",
            "Alpha",
        ]
        # The pipeline's own order, not the alphabet.
        assert _titles(sort_applications(apps, {}, "status", "asc")) == ["Alpha", "gamma", "beta"]
        assert _titles(sort_applications(apps, {}, "updated", "desc")) == ["beta", "Alpha", "gamma"]
        assert _titles(sort_applications(apps, {}, "updated", "asc")) == ["gamma", "Alpha", "beta"]

    def test_an_unknown_sort_is_the_default_order(self) -> None:
        older = _app("older", updated=NOW - dt.timedelta(days=1))
        newer = _app("newer", updated=NOW)
        assert _titles(sort_applications([older, newer], {}, "blend")) == ["newer", "older"]


class TestSortControls:
    def test_the_query_is_read_from_a_closed_set(self) -> None:
        assert parse_sort(None, None) == ("updated", "desc")
        assert parse_sort("could_get", None) == ("could_get", "desc")
        assert parse_sort("title", None) == ("title", "asc")
        assert parse_sort("want_it", "asc") == ("want_it", "asc")
        assert parse_sort("want_it", "sideways") == ("want_it", "desc")
        # An unknown key drops its direction too: it meant another column.
        assert parse_sort("overall", "asc") == ("updated", "desc")
        assert parse_sort("could_get+want_it", None) == ("updated", "desc")

    def test_clicking_the_active_header_reverses_it(self) -> None:
        headers = sort_headers("could_get", "desc", status=None)
        assert headers["could_get"].href == "/applications?sort=could_get&dir=asc"
        assert headers["could_get"].aria_sort == "descending"
        reversed_ = sort_headers("could_get", "asc", status=None)
        assert reversed_["could_get"].href == "/applications?sort=could_get"
        assert reversed_["could_get"].aria_sort == "ascending"

    def test_other_headers_start_at_their_default_and_carry_no_aria_sort(self) -> None:
        headers = sort_headers("could_get", "desc", status=None)
        assert headers["want_it"].href == "/applications?sort=want_it"
        assert headers["title"].href == "/applications?sort=title"
        assert headers["updated"].href == "/applications"
        assert [k for k, h in headers.items() if h.aria_sort] == ["could_get"]

    def test_the_default_sort_reverses_to_oldest_first(self) -> None:
        headers = sort_headers("updated", "desc", status=None)
        assert headers["updated"].href == "/applications?sort=updated&dir=asc"

    def test_the_status_filter_rides_along(self) -> None:
        headers = sort_headers("title", "asc", status="applied")
        assert headers["title"].href == "/applications?status=applied&sort=title&dir=desc"
        assert headers["updated"].href == "/applications?status=applied"
        assert sort_query("title", "desc") == {"sort": "title", "dir": "desc"}
        assert sort_query("updated", "desc") == {}


def _titles(apps: list[Application]) -> list[str]:
    return [a.title for a in apps]
