"""`jfl_web.sorting`, the generic engine every table but `jfl_web.scores`'s
applications list sorts through -- see that module's docstring for why the
applications list keeps its own (behaviourally identical) copy.

No database, no model, no live app: a plain `SortSpec` over small records
stands in for a real table's rows, and `fastapi.Request` is built from a bare
ASGI scope rather than a running server.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from fastapi import Request
from jfl_core.storage.ui_table_sorts import TableSortState
from jfl_web.sorting import (
    SortColumn,
    SortSpec,
    parse_sort,
    resolve_sort,
    sort_headers,
    sort_rows,
)

NOW = dt.datetime(2026, 9, 23, 9, 0, tzinfo=dt.UTC)


@dataclass(frozen=True, slots=True)
class Item:
    name: str
    count: int | None
    updated: dt.datetime


def _spec() -> SortSpec[Item]:
    return SortSpec(
        columns={
            "name": SortColumn("Name", value=lambda i: i.name.casefold()),
            "count": SortColumn("Count", value=lambda i: i.count, default_direction="desc"),
            "updated": SortColumn("Updated", value=lambda i: i.updated, default_direction="desc"),
        },
        default_key="updated",
        tiebreak=lambda i: i.updated,
    )


def _request(query: str = "") -> Request:
    return Request({"type": "http", "query_string": query.encode(), "headers": []})


class TestParseSort:
    def test_unknown_key_falls_back_to_the_default_and_its_own_direction(self) -> None:
        spec = _spec()
        assert parse_sort(spec, None, None) == ("updated", "desc")
        assert parse_sort(spec, "nonsense", "asc") == ("updated", "desc")

    def test_a_known_key_with_no_direction_uses_its_own_default(self) -> None:
        spec = _spec()
        assert parse_sort(spec, "name", None) == ("name", "asc")
        assert parse_sort(spec, "count", None) == ("count", "desc")

    def test_an_explicit_direction_is_kept_an_invalid_one_falls_back(self) -> None:
        spec = _spec()
        assert parse_sort(spec, "name", "desc") == ("name", "desc")
        assert parse_sort(spec, "name", "sideways") == ("name", "asc")


class TestSortRows:
    def _items(self) -> list[Item]:
        return [
            Item("beta", 3, NOW),
            Item("Alpha", None, NOW - dt.timedelta(days=1)),
            Item("gamma", 5, NOW - dt.timedelta(days=2)),
        ]

    def test_text_ascending_and_descending(self) -> None:
        spec = _spec()
        assert [i.name for i in sort_rows(spec, self._items(), "name", "asc")] == [
            "Alpha",
            "beta",
            "gamma",
        ]
        assert [i.name for i in sort_rows(spec, self._items(), "name", "desc")] == [
            "gamma",
            "beta",
            "Alpha",
        ]

    def test_missing_values_sort_last_in_both_directions(self) -> None:
        spec = _spec()
        for direction in ("asc", "desc"):
            ordered = [i.name for i in sort_rows(spec, self._items(), "count", direction)]
            assert ordered[-1] == "Alpha", (direction, ordered)

    def test_ties_break_on_the_spec_tiebreak_never_the_other_column(self) -> None:
        """Two rows with the same count: most recently updated first, in either
        direction -- the standing two-axis rule generalised to this engine."""
        spec = _spec()
        x = Item("x", 4, NOW)
        y = Item("y", 4, NOW - dt.timedelta(hours=1))
        for direction in ("asc", "desc"):
            assert [i.name for i in sort_rows(spec, [y, x], "count", direction)] == ["x", "y"]

    def test_an_unknown_key_sorts_as_the_default(self) -> None:
        spec = _spec()
        older = Item("older", 1, NOW - dt.timedelta(days=1))
        newer = Item("newer", 1, NOW)
        assert [i.name for i in sort_rows(spec, [older, newer], "blend")] == ["newer", "older"]

    def test_no_direction_uses_the_columns_own_default(self) -> None:
        spec = _spec()
        assert [i.name for i in sort_rows(spec, self._items(), "count")][0] == "gamma"


class TestSortHeaders:
    def test_every_link_names_its_sort_explicitly(self) -> None:
        """Unlike `jfl_web.scores.sort_headers`, this module never omits the
        sort at the table's default -- see the module docstring for why that
        omission becomes ambiguous once sort is persisted."""
        spec = _spec()
        headers = sort_headers(spec, "updated", "desc", base_path="/items")
        assert headers["updated"].href == "/items?sort=updated&dir=asc"
        assert headers["name"].href == "/items?sort=name&dir=asc"
        assert headers["count"].href == "/items?sort=count&dir=desc"

    def test_clicking_the_active_header_reverses_it(self) -> None:
        spec = _spec()
        headers = sort_headers(spec, "count", "desc", base_path="/items")
        assert headers["count"].active
        assert headers["count"].aria_sort == "descending"
        assert headers["count"].href == "/items?sort=count&dir=asc"

    def test_other_headers_carry_no_aria_sort(self) -> None:
        spec = _spec()
        headers = sort_headers(spec, "count", "desc", base_path="/items")
        assert [k for k, h in headers.items() if h.aria_sort] == ["count"]

    def test_extra_params_ride_along(self) -> None:
        spec = _spec()
        headers = sort_headers(
            spec, "updated", "desc", base_path="/items", extra_params={"status": "open"}
        )
        assert headers["name"].href == "/items?status=open&sort=name&dir=asc"


class _FakeRepo:
    """Stands in for `PostgresUiTableSortRepository` -- an in-memory dict."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], tuple[str, str]] = {}

    def get_sort(self, table_key: str) -> TableSortState | None:
        row = self._rows.get(("u", table_key))
        if row is None:
            return None
        return TableSortState(
            table_key=table_key, sort_key=row[0], direction=row[1], updated_at=NOW
        )

    def save_sort(self, table_key: str, sort_key: str, direction: str) -> None:
        self._rows[("u", table_key)] = (sort_key, direction)


class TestResolveSort:
    def test_a_header_click_saves_and_applies(self) -> None:
        spec = _spec()
        repo = _FakeRepo()
        assert resolve_sort(_request("sort=name&dir=desc"), repo, "applications", spec) == (
            "name",
            "desc",
        )
        saved = repo.get_sort("applications")
        assert saved is not None and saved.sort_key == "name"

    def test_a_plain_visit_applies_what_was_saved(self) -> None:
        spec = _spec()
        repo = _FakeRepo()
        resolve_sort(_request("sort=name&dir=desc"), repo, "applications", spec)
        assert resolve_sort(_request(""), repo, "applications", spec) == ("name", "desc")

    def test_a_plain_visit_with_nothing_saved_is_the_default(self) -> None:
        spec = _spec()
        repo = _FakeRepo()
        assert resolve_sort(_request(""), repo, "applications", spec) == ("updated", "desc")

    def test_a_stale_saved_key_falls_back_silently(self) -> None:
        """A column a screen has since dropped must not raise -- it reads back
        as the table's own default."""
        spec = _spec()
        repo = _FakeRepo()
        repo.save_sort("applications", "no_longer_a_column", "asc")
        assert resolve_sort(_request(""), repo, "applications", spec) == ("updated", "desc")

    def test_an_unrecognised_direction_on_a_click_falls_back_to_the_columns_own(self) -> None:
        spec = _spec()
        repo = _FakeRepo()
        assert resolve_sort(_request("sort=name&dir=sideways"), repo, "applications", spec) == (
            "name",
            "asc",
        )
