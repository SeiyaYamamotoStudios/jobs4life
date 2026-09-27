"""A sortable table sorts from its column headers -- one engine, every table.

`jfl_web.scores` built this once, for the applications list, before there was
a second table to share it with: a closed set of column keys, a link per
header that reverses the active column and starts any other at its own
default direction, and rows with nothing to sort on going last **in both**
directions rather than reading as a low score. `docs/ui-sections.md`'s
"a sortable table sorts from its column headers" is the contract this
generalises; `jfl_web.scores` keeps its own copy for the applications list
(matched here, and left alone, because a unit test file already pins its
exact public shape) while every table added since uses this module directly.

**Persistence is the reason this exists as a module at all** (owner feedback:
"the sorting of the applications isn't persistent ... it needs to be
persistent on any tables"). `resolve_sort` is the one function every list
route calls: a header click (`?sort=` present) is read, applied and saved; a
plain visit reads back what was saved; a stale or unknown saved key silently
falls back to the table's own default, via `parse_sort`, which already treats
an unrecognised key that way for the same reason CLAUDE.md gives for the
board-check guard -- "the tool measures what is true", and a value nothing
recognises is not evidence of an intentional choice.

Every header link this module builds carries its sort **explicitly** (`sort`
and `dir` both, always) rather than the shorter, cleaner URL a stateless list
could get away with by omitting them at the table's global default. That
extra verbosity is the fix for a real bug, not a style choice: once a table's
order can be *read back*, a link that means "go to the default order" and a
plain visit that means "show me whatever I last chose" become indistinguishable
the moment both would otherwise produce a bare path with no query string at
all. Only `jfl_web.scores`'s own header links still take that shortcut, since
its test file pins the shorter URLs; the applications route compensates by
rewriting those hrefs before persistence was added on top -- see
`jfl_web.routes.applications._persistent_sort_headers`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urlencode

from fastapi import Request
from jfl_core.models import TableKey
from jfl_core.storage.ui_table_sorts import TableSortState

SortDirection = Literal["asc", "desc"]


class _Comparable(Protocol):
    """What a column's `value` and a spec's `tiebreak` return -- a string, a
    number or a date all satisfy this; `object` does not, which is the point:
    it is what lets `sort_rows` actually call `sorted` on the result."""

    def __lt__(self, other: Any) -> bool: ...


class TableSortStore(Protocol):
    """What `resolve_sort` needs from a table-sort repository -- structural,
    not `PostgresUiTableSortRepository` by name, so a test's in-memory stand-in
    satisfies it without a subclass or a `type: ignore`."""

    def get_sort(self, table_key: str) -> TableSortState | None: ...
    def save_sort(self, table_key: str, sort_key: str, direction: str) -> None: ...


@dataclass(frozen=True, slots=True)
class SortColumn[T]:
    """One sortable column.

    `value` returns `None` for a row with nothing to sort on -- an unscored
    application, a job with no location recorded -- which `sort_rows` always
    places last, in both directions: "nothing recorded" is not a low value,
    and ranking it as one would be the table asserting something nothing
    measured.
    """

    label: str
    value: Callable[[T], _Comparable | None]
    default_direction: SortDirection = "asc"


@dataclass(frozen=True, slots=True)
class SortSpec[T]:
    """Every sortable column of one table, plus the tie-break used when two
    rows compare equal on the chosen column, or neither has anything to sort
    on. Almost always "most recently touched" -- **never** the other column of
    a two-axis pair, which must stay uninfluenced by the one not chosen.
    """

    columns: Mapping[str, SortColumn[T]]
    default_key: str
    tiebreak: Callable[[T], _Comparable]
    tiebreak_reverse: bool = True


@dataclass(frozen=True, slots=True)
class SortHeader:
    """One column header: where its link goes, and what it says about the
    current order. `aria_sort` is set only on the active header -- ARIA puts
    it on one header at a time."""

    key: str
    label: str
    href: str
    active: bool
    direction: SortDirection | None
    aria_sort: Literal["ascending", "descending"] | None
    next_direction: SortDirection


def parse_sort[T](
    spec: SortSpec[T], raw_key: str | None, raw_direction: str | None
) -> tuple[str, SortDirection]:
    """The sort a query string (or a saved row) asks for, from a closed set.

    An unknown key falls back to the table's default **and its own default
    direction** -- a direction means nothing without the column it was chosen
    for. An unknown direction falls back to the chosen column's own default.
    """
    if raw_key is None or raw_key not in spec.columns:
        key = spec.default_key
        return key, spec.columns[key].default_direction
    default_direction = spec.columns[raw_key].default_direction
    direction: SortDirection = default_direction
    if raw_direction == "asc":
        direction = "asc"
    elif raw_direction == "desc":
        direction = "desc"
    return raw_key, direction


def sort_headers[T](
    spec: SortSpec[T],
    key: str,
    direction: SortDirection,
    *,
    base_path: str,
    extra_params: Mapping[str, str] | None = None,
) -> dict[str, SortHeader]:
    """A link per column, in the order `spec.columns` declares them. Clicking
    the active column reverses it; clicking any other starts that column at
    its own default direction. Every other query parameter the page needs
    (a status filter, a view toggle) rides along on every link.

    Every link names its sort explicitly -- see the module docstring for why
    this table never takes the shorter, ambiguous "omit it at the default"
    route `jfl_web.scores` does.
    """
    extra = dict(extra_params or {})
    headers: dict[str, SortHeader] = {}
    for column_key, column in spec.columns.items():
        active = column_key == key
        next_direction: SortDirection = (
            ("asc" if direction == "desc" else "desc") if active else column.default_direction
        )
        params = {**extra, "sort": column_key, "dir": next_direction}
        headers[column_key] = SortHeader(
            key=column_key,
            label=column.label,
            href=f"{base_path}?{urlencode(params)}",
            active=active,
            direction=direction if active else None,
            aria_sort=("ascending" if direction == "asc" else "descending") if active else None,
            next_direction=next_direction,
        )
    return headers


def sort_rows[T](
    spec: SortSpec[T], rows: Sequence[T], key: str, direction: SortDirection | None = None
) -> list[T]:
    """Order `rows` by **one** column.

    Ties -- and every row with nothing in the chosen column -- keep the
    spec's tie-break order. Python's sort is stable, including with
    `reverse=True`, so every later sort leaves equal rows in that order. A key
    outside the closed set sorts as the table's default.
    """
    resolved_key = key if key in spec.columns else spec.default_key
    column = spec.columns[resolved_key]
    resolved_direction: SortDirection = (
        direction if direction is not None else column.default_direction
    )

    base = sorted(rows, key=spec.tiebreak, reverse=spec.tiebreak_reverse)
    valued = [(column.value(row), row) for row in base]
    present = [(value, row) for value, row in valued if value is not None]
    missing = [row for value, row in valued if value is None]
    present.sort(key=lambda pair: pair[0], reverse=(resolved_direction == "desc"))
    return [row for _, row in present] + missing


def resolve_sort[T](
    request: Request,
    table_sorts: TableSortStore,
    table_key: TableKey,
    spec: SortSpec[T],
) -> tuple[str, SortDirection]:
    """The sort in effect for this render, and the one place a header click
    is told apart from a plain visit.

    A header click puts `?sort=` on the URL -- so its presence, not its value,
    is what triggers a save; `parse_sort` still normalises whatever value
    arrived, so a hand-edited or stale `?sort=` cannot write a value outside
    the table's own closed set. A plain visit (`?sort=` absent) reads back
    what was last saved; nothing having been saved yet, or a saved key that no
    longer names a column, falls back to the table's default silently -- the
    owner should never see an error banner about a layout preference.
    """
    raw_sort = request.query_params.get("sort")
    if raw_sort is not None:
        key, direction = parse_sort(spec, raw_sort, request.query_params.get("dir"))
        table_sorts.save_sort(table_key, key, direction)
        return key, direction
    saved = table_sorts.get_sort(table_key)
    if saved is None:
        return parse_sort(spec, None, None)
    return parse_sort(spec, saved.sort_key, saved.direction)


__all__ = [
    "SortColumn",
    "SortDirection",
    "SortHeader",
    "SortSpec",
    "TableSortStore",
    "parse_sort",
    "resolve_sort",
    "sort_headers",
    "sort_rows",
]
