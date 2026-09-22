"""The phone layout, asserted against markup and CSS rather than a screenshot.

There is no browser in this test run, so nothing here claims a page *looks*
right. What it does claim is the set of structural facts the small-screen
rules depend on, each of which is the kind of thing a later edit removes
without noticing:

  1. every wide table sits in a `.table-wrap`, so the one that scrolls can
     scroll and the ones that stack have somewhere to stack inside;
  2. every cell of a table that stacks carries the `data-label` its column
     header would otherwise have given it -- a bare value with no label is
     worse than a sideways scroll, which is the whole reason the header row
     is allowed to disappear;
  3. the nav is still links and buttons, so wrapping it cost nothing in the
     tab order and nothing to a screen reader;
  4. nothing the phone rules hide is only visible in the place they hide it;
  5. no template hard-codes a pixel width that would overflow 320px;
  6. the stylesheet is not truncated and every block and media query is
     closed. A merge in this repo has silently cut CSS rules before, and a
     stylesheet that stops mid-rule fails silently in a browser and not at
     all in a test run.

No database, no model call, no network: this reads source.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "packages/web/src/jfl_web"
TEMPLATES = WEB / "templates"
STYLESHEET = WEB / "static/style.css"

# The two breakpoints, named. Anything else in the stylesheet is either an
# undocumented third one or a typo, and both should fail here.
BREAKPOINTS = {
    "(max-width: 64rem)": "--lap: the header stops fitting on one line",
    "(max-width: 40rem)": "--phone: one hand, one column",
}

# Tables that become cards under --phone, as {label: (files, column headers)}.
# The header row and the row markup live in different files for two of them,
# so a table is a set of files rather than one.
STACKED_TABLES = {
    "applications": ("applications_list.html", "_application_row.html"),
    "boards": ("boards_list.html", "_board_row.html"),
    "jobs": ("jobs_list.html",),
    "changes": ("changes.html",),
}

# The one table that keeps its horizontal scroll: four read-only status
# columns with no control in any cell, so scrolling puts nothing out of reach.
SCROLLING_TABLE = "background.html"

# Cells exempt from `data-label`, because what is in them names itself:
# "Track as application", "Stop watching", "Restore", "Dismiss".
SELF_NAMING_CELLS = ("col-actions", "col-track")


def _read(name: str) -> str:
    return (TEMPLATES / name).read_text()


def _template_files() -> list[pathlib.Path]:
    return sorted(TEMPLATES.glob("*.html"))


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


# ----------------------------------------------------------------------
# 1. every table is inside a scroll wrapper
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.name)
def test_every_table_sits_in_a_scroll_wrapper(path: pathlib.Path) -> None:
    """A `<table>` with no `.table-wrap` around it is a page that scrolls
    sideways as a whole on a phone instead of scrolling the table."""
    source = path.read_text()
    for match in re.finditer(r"<table\b", source):
        before = source[: match.start()]
        # The nearest preceding tag must be the wrapper, and it must not have
        # been closed again in between.
        opened = before.rfind('<div class="table-wrap">')
        assert opened != -1, f"{path.name}: <table> with no .table-wrap around it"
        assert "</div>" not in before[opened:], (
            f"{path.name}: .table-wrap is closed before the <table> it should wrap"
        )


# ----------------------------------------------------------------------
# 2. a stacked cell keeps its column's meaning
# ----------------------------------------------------------------------


@pytest.mark.parametrize("table,files", STACKED_TABLES.items(), ids=list(STACKED_TABLES))
def test_stacked_cells_carry_their_column_label(table: str, files: tuple[str, ...]) -> None:
    headers = set()
    for name in files:
        headers |= {
            re.sub(r"<[^>]+>", "", text).strip()
            for text in re.findall(r'<th scope="col">(.*?)</th>', _read(name))
        }
    headers.discard("")
    assert headers, f"{table}: no column headers found"

    labelled = set()
    for name in files:
        for cell in re.findall(r"<td\b[^>]*>", _read(name)):
            if any(c in cell for c in SELF_NAMING_CELLS):
                assert "data-label" not in cell, (
                    f"{table}: {cell} is a self-naming action cell and should not "
                    "also carry a label"
                )
                continue
            label = re.search(r'data-label="([^"]*)"', cell)
            assert label, (
                f"{table}: {cell} has no data-label, so it would render on a phone "
                "as a bare value with nothing saying which column it came from"
            )
            labelled.add(label.group(1))

    unknown = labelled - headers
    assert not unknown, (
        f"{table}: data-label(s) {sorted(unknown)} match no column header "
        f"{sorted(headers)} -- the label and the header have drifted apart"
    )


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.name)
def test_every_column_header_declares_its_scope(path: pathlib.Path) -> None:
    """`scope="col"` is what keeps the scrolling table readable once it is
    wider than the viewport and the header is off to the left."""
    for header in re.findall(r"<th\b[^>]*>", path.read_text()):
        assert 'scope="col"' in header, f"{path.name}: {header} has no scope"


def test_the_scrolling_table_has_no_controls_in_its_cells() -> None:
    """The justification for leaving /background's CV table scrolling rather
    than stacking it: there is nothing in a cell to press, so nothing is put
    out of reach by being off-screen. If a control is ever added, this fails
    and the decision gets made again."""
    source = _read(SCROLLING_TABLE)
    body = source[source.index('<table class="cvs"') : source.index("</table>")]
    assert "<button" not in body, (
        "a control was added to the CV table -- it now needs stacking, not scrolling"
    )
    assert "<select" not in body
    assert "<input" not in body


# ----------------------------------------------------------------------
# 3. the nav survives wrapping
# ----------------------------------------------------------------------


def test_nav_is_links_and_buttons_with_no_script() -> None:
    """The header wraps instead of collapsing into a menu, so every
    destination must still be a plain link or a submit button: nothing is
    behind a disclosure, nothing needs JavaScript, and the tab order is the
    document order."""
    base = _read("base.html")
    nav = base[base.index("<nav") : base.index("</nav>")]

    assert 'aria-label="Main"' in nav, "the header nav needs an accessible name"
    assert "onclick" not in nav and "<script" not in nav, "the nav must need no script"
    assert "hx-" not in nav, "the nav must not need htmx to navigate"

    links = re.findall(r'<a href="(/[^"]*)"', nav)
    assert len(links) >= 9, f"expected the full nav, found {links}"
    assert "/settings" in links and "/applications" in links

    # The sign-out form is the one non-link, and it is a real submit button.
    assert '<button type="submit" class="link">Sign out</button>' in nav


def test_hidden_on_a_phone_means_available_somewhere_else() -> None:
    """The signed-in address comes off the header under --phone. That is only
    acceptable while Settings still names the account."""
    css = STYLESHEET.read_text()
    phone = css[css.index("@media (max-width: 40rem)") :]
    assert ".signed-in-as { display: none; }" in phone

    settings = _read("settings.html")
    assert "Signed in as" in settings and "{{ user.email }}" in settings


# ----------------------------------------------------------------------
# 4. nothing hard-codes a width that overflows a 320px screen
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.name)
def test_no_template_hard_codes_a_pixel_width(path: pathlib.Path) -> None:
    source = path.read_text()
    # The viewport meta is the one legitimate `width=`.
    source = source.replace('content="width=device-width, initial-scale=1"', "")
    for bad in re.findall(r'(?:min-)?width\s*[:=]\s*"?\s*(\d+)(?:px)?', source):
        pytest.fail(f"{path.name}: hard-coded width {bad} -- use CSS, and a relative unit")


def test_stylesheet_hard_codes_no_width_wider_than_a_phone() -> None:
    """A `min-width` above 320px anywhere outside the deliberate scrolling
    table is a horizontal page scroll waiting to happen."""
    css = _strip_css_comments(STYLESHEET.read_text())
    deliberate = {"30rem"}  # table.cvs, which is meant to scroll sideways
    for value in re.findall(r"min-width:\s*([\d.]+)(rem|px)", css):
        size, unit = value
        px = float(size) * (16 if unit == "rem" else 1)
        if px <= 320 or f"{size}{unit}" in deliberate:
            continue
        pytest.fail(f"min-width: {size}{unit} is wider than a 320px viewport")


def test_text_inputs_do_not_trigger_an_ios_zoom() -> None:
    """Under 16px, iOS Safari zooms the page on focus and does not zoom back."""
    css = STYLESHEET.read_text()
    phone = css[css.index("@media (max-width: 40rem)") :]
    block = phone[phone.index('input[type="text"]') :]
    rule = block[: block.index("}")]
    assert "font-size: 16px" in rule
    for control in ('input[type="url"]', 'input[type="password"]', "select", "textarea"):
        assert control in rule, f"{control} is missing from the 16px rule"


# ----------------------------------------------------------------------
# 5. the stylesheet is whole
# ----------------------------------------------------------------------


def test_stylesheet_blocks_are_balanced_and_nothing_is_truncated() -> None:
    raw = STYLESHEET.read_text()
    assert raw.rstrip().endswith("}"), "stylesheet ends mid-rule"
    assert raw.count("/*") == raw.count("*/"), "unterminated CSS comment"

    css = _strip_css_comments(raw)
    depth = 0
    for line_no, line in enumerate(css.splitlines(), start=1):
        depth += line.count("{") - line.count("}")
        assert depth >= 0, f"line {line_no}: a `}}` closes a block that was never opened"
    assert depth == 0, f"stylesheet leaves {depth} block(s) open"


def test_every_media_query_is_a_declared_breakpoint_and_is_closed() -> None:
    css = _strip_css_comments(STYLESHEET.read_text())
    queries = re.findall(r"@media\s*([^{]+)\{", css)
    assert queries, "the stylesheet has no media queries at all"

    for query in queries:
        condition = query.strip()
        assert condition in BREAKPOINTS, (
            f"undeclared breakpoint {condition!r}; the declared ones are {sorted(BREAKPOINTS)}"
        )

    # Every declared breakpoint is actually used, and each `@media` block
    # closes: walk from each one and check the depth returns to zero.
    for condition in BREAKPOINTS:
        assert f"@media {condition}" in css, f"{condition} is declared but never used"

    for match in re.finditer(r"@media[^{]+\{", css):
        depth = 0
        closed = False
        for char in css[match.end() - 1 :]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
        assert closed, f"unclosed @media block at offset {match.start()}"
