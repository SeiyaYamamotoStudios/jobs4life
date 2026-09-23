"""Wide tables on a desktop: text columns keep their words, actions never clip.

The failure this guards, seen on a ~1750px window: /changes set its job titles
one syllable per line ("En / gin / eer / ing") in a ~40px column, /boards broke
employer names ("Anthr / opic"), and both clipped the actions column at the
container edge. The cause was `overflow-wrap: anywhere` applied to `td`, `th`,
`a`, `li` and `.note` wholesale: `anywhere` counts every character as a soft
wrap opportunity when the browser measures min-content width, so the automatic
table layout was free to squeeze every text cell to one character while the
nowrap timestamp and action cells kept their full width.

There is no browser here, so nothing below claims a page *looks* right. It
pins the structural facts the fix depends on:

  1. `anywhere` survives only on selectors for genuinely unbreakable runs;
  2. each wide table's primary column carries the class its min-width floor
     hangs on, and the floor exists in the stylesheet (and is lifted on a
     phone, where the rows are cards);
  3. table timestamps render compactly as `<time datetime>` with the full
     date in `title`.

No database, no model call, no network.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import re
import uuid
from types import SimpleNamespace

import pytest
from jfl_web.templating import _templates

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "packages/web/src/jfl_web"
TEMPLATES = WEB / "templates"
STYLESHEET = WEB / "static/style.css"

# The only selectors `overflow-wrap: anywhere` may sit on: code, a URL shown as
# its own text, an id, a draft's citation lines (span ids), and a drift
# dimension's monospace name inside a flex row. Everything else is text, and
# text gets `break-word` from body.
UNBREAKABLE_RUNS = {
    "code",
    ".url-text",
    ".id-text",
    ".draft-citations li",
    ".drift-dimension",
}

# Selectors that must never carry it, whatever else is added later. A bare
# element here would reach every table cell, link or note in the app.
GENERAL_TEXT = {"td", "th", "a", "li", "dd", "dt", "p", "blockquote", "body", "span", "div"}
GENERAL_TEXT_CLASSES = {".note", ".answer-text", ".pushback-words"}

# (template, the cell's own class) for each wide table's primary column.
PRIMARY_COLUMNS = {
    "/jobs": ("jobs_list.html", "col-job"),
    "/changes": ("changes.html", "col-job"),
    "/boards": ("_board_row.html", "col-board"),
    "/applications": ("_application_row.html", "col-title"),
    "/applications?archived=1": ("applications_list.html", "col-title"),
}

# Every table cell that shows a timestamp, and the template it lives in.
TIMESTAMP_CELLS = (
    "_board_row.html",
    "changes.html",
    "jobs_list.html",
    "_application_row.html",
    "applications_list.html",
)


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLESHEET.read_text(), flags=re.DOTALL)


def _rules(css: str) -> list[tuple[str, str]]:
    """(selector list, declarations) for every innermost rule, media queries
    flattened -- a rule inside `@media` counts exactly as one outside."""
    css = re.sub(r"@media[^{]+\{", "", css)
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]


def _selectors_with(css: str, declaration: str) -> list[str]:
    found = []
    for selector_list, body in _rules(css):
        if re.search(declaration, body):
            found.extend(s.strip() for s in selector_list.split(","))
    return found


# ----------------------------------------------------------------------
# 1. `anywhere` only where a run genuinely cannot break
# ----------------------------------------------------------------------


def test_anywhere_is_never_on_general_text() -> None:
    offenders = []
    for selector in _selectors_with(_css(), r"overflow-wrap:\s*anywhere"):
        if selector in UNBREAKABLE_RUNS:
            continue  # e.g. `.draft-citations li`: scoped by its class
        # The part that actually receives the property is the last compound.
        subject = selector.split()[-1]
        is_bare_element = subject in GENERAL_TEXT
        if is_bare_element or any(c in selector for c in GENERAL_TEXT_CLASSES):
            offenders.append(selector)
    assert not offenders, (
        f"`overflow-wrap: anywhere` on general text {offenders}: it lowers the "
        "min-content width to about one character, and a table then crushes "
        "that column. Use `break-word` (inherited from body) instead."
    )


def test_anywhere_survives_only_on_the_unbreakable_run_selectors() -> None:
    selectors = set(_selectors_with(_css(), r"overflow-wrap:\s*anywhere"))
    unexpected = selectors - UNBREAKABLE_RUNS
    assert not unexpected, (
        f"`overflow-wrap: anywhere` on {sorted(unexpected)}. It belongs only on "
        f"unbreakable runs {sorted(UNBREAKABLE_RUNS)} -- anything else is text."
    )
    # The shared rule still exists, so a URL or id does not start pushing
    # tables wide again.
    assert {"code", ".url-text"} <= selectors


def test_body_breaks_words_only_when_they_cannot_fit() -> None:
    assert "body" in _selectors_with(_css(), r"overflow-wrap:\s*break-word")


def test_a_board_shown_by_its_url_carries_the_url_class() -> None:
    """A board with no label is shown as its URL, which is one unbreakable run
    -- the case `.url-text` exists for."""
    for name in ("_board_row.html", "changes.html", "jobs_list.html"):
        source = (TEMPLATES / name).read_text()
        assert "{% if not " in source and 'class="url-text"' in source, name


# ----------------------------------------------------------------------
# 2. the primary column has a floor; secondary columns give way
# ----------------------------------------------------------------------


@pytest.mark.parametrize("page,spec", PRIMARY_COLUMNS.items(), ids=list(PRIMARY_COLUMNS))
def test_primary_column_carries_its_min_width_class(page: str, spec: tuple[str, str]) -> None:
    name, own_class = spec
    cells = re.findall(r"<td\b[^>]*>", (TEMPLATES / name).read_text())
    primary = [c for c in cells if "col-primary" in c]
    assert len(primary) == 1, f"{page}: expected one primary cell in {name}, found {primary}"
    assert own_class in primary[0], f"{page}: the primary column is not {own_class}"


def test_the_primary_column_floor_exists_and_is_lifted_on_a_phone() -> None:
    css = _css()
    floors = {
        sel: body
        for sel_list, body in _rules(css[: css.index("@media (max-width: 40rem)")])
        for sel in (s.strip() for s in sel_list.split(","))
        if "col-primary" in sel and "min-width" in body
    }
    assert floors, "no min-width on td.col-primary"
    for body in floors.values():
        width = re.search(r"min-width:\s*([\d.]+)rem", body)
        assert width and 8 <= float(width.group(1)) <= 20, body

    phone = css[css.index("@media (max-width: 40rem)") :]
    reset = [
        s.strip()
        for sel_list, body in _rules(phone)
        if re.search(r"min-width:\s*0\b", body)
        for s in sel_list.split(",")
    ]
    for selector in floors:
        assert selector in reset, f"{selector}'s floor is never lifted for the stacked cards"


def test_actions_keep_their_labels_whole_but_may_stack() -> None:
    """An action's label never wraps (so the column's minimum is its widest
    control, and it cannot clip); the cell itself wraps, so Track and Dismiss
    stack instead of demanding the sum of their widths."""
    nowrap = _selectors_with(_css(), r"white-space:\s*nowrap")
    assert "table.boards .col-actions button" in nowrap
    assert "table.boards .col-track button" in nowrap
    assert "table.boards .col-actions" not in nowrap


def test_table_pages_get_desktop_room() -> None:
    """/applications is included: at the 56rem reading measure its seven
    columns overflowed into the sideways scroll and the Archive control was
    cut off at the table's right edge, leaving only a disclosure triangle."""
    wide = _selectors_with(_css(), r"max-width:\s*80rem")
    assert "main:has(table.boards)" in wide
    assert "main:has(table.applications)" in wide


def test_the_archive_control_is_a_labelled_button_that_never_wraps() -> None:
    nowrap = _selectors_with(_css(), r"white-space:\s*nowrap")
    assert "table.applications .col-actions summary" in nowrap
    assert "table.applications .col-actions button" in nowrap
    assert "table.applications td.col-actions" not in nowrap
    source = (TEMPLATES / "_application_row.html").read_text()
    assert '<summary class="row-archive-toggle">Archive…</summary>' in source


# ----------------------------------------------------------------------
# 3. compact timestamps, with the full date one hover away
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", TIMESTAMP_CELLS)
def test_table_timestamps_use_the_compact_form(name: str) -> None:
    source = (TEMPLATES / name).read_text()
    table = source[source.index("<tr") :]
    assert "time_compact" in table, f"{name}: no compact timestamp in the table"
    for cell in re.findall(r'<td class="col-updated.*?</td>', table, re.S):
        assert "humanize_dt" not in cell, f"{name}: a table cell still shows the long form"


def _board_row_html() -> str:
    # The filter reads the real clock, so the instants are placed relative to
    # it; a few seconds of test runtime cannot move either across a unit.
    now = dt.datetime.now(dt.UTC)
    finished = now - dt.timedelta(minutes=13, seconds=20)
    created = now - dt.timedelta(days=8, hours=3)
    row = SimpleNamespace(
        board=SimpleNamespace(
            id=uuid.uuid4(),
            label=None,
            board_url="https://boards.greenhouse.io/averyveryverylongemployername",
            created_at=created,
        ),
        platform_label="Greenhouse",
        last_check=SimpleNamespace(status="complete", error_code=None, finished_at=finished),
        open_job_count=12,
        held=False,
        checking=False,
        match_count=None,
    )
    template = _templates.env.get_template("_board_row.html")
    return template.render(row=row, session=SimpleNamespace(csrf_token="t"))


def test_a_rendered_board_row_has_time_elements_with_the_full_date() -> None:
    html = _board_row_html()
    times = re.findall(r'<time datetime="([^"]+)" title="([^"]+)">([^<]+)</time>', html)
    assert [shown for _, _, shown in times] == ["13 min ago", "8 days ago"], html
    for instant, title, _ in times:
        assert dt.datetime.fromisoformat(instant)  # machine-readable
        # The full date, with the minute, and the long relative form.
        assert re.search(r"^\w{3} \d{1,2} \w{3} \d{4}, \d{2}:\d{2} · .+ ago$", title), title


def test_a_rendered_board_row_marks_its_url_and_primary_cell() -> None:
    html = _board_row_html()
    assert 'class="col-board col-primary"' in html
    assert 'class="url-text"' in html
