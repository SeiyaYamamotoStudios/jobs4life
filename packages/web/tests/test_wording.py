"""The screens use the words a new user has, and old links still work.

The word **corpus** is right in the code -- `jfl_core.corpus_source`, the
`spans` table, the claim gate's docstrings -- and wrong on a screen: nobody
uploads a corpus, they upload old CVs. The user-facing half of the vocabulary
is pinned here so the internal half can keep its own words.

Three rules, none of which needs a database:

  1. the pages say "your CVs" and "facts you confirmed", never "corpus";
  2. the nav has one link per destination, each with its own label (it briefly
     had two links both labelled "Corpus" -- a merge artefact);
  3. the paths the pages used to live at still answer, permanently.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from jfl_web.templating import TEMPLATE_DIR

# Words that belong to the implementation and mean nothing to someone who has
# not read CLAUDE.md. There is no allow-list: no screen has needed one.
JARGON = ("corpus", "provenance", "adjudicated")

_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


class _StubGoogle:
    """Enough to satisfy `create_app` without building a real provider."""

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        raise NotImplementedError

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        raise NotImplementedError


@pytest.fixture
def client(settings: WebSettings) -> TestClient:
    from jfl_web.app import create_app

    return TestClient(create_app(settings, identity_provider=_StubGoogle()))


def _rendered(path: pathlib.Path) -> str:
    """A template's text with its comments removed.

    A `{# ... #}` block is a note to the next developer and never reaches a
    browser, so it may say "corpus" for the same reason the Python around it
    does. Everything else here is read by a user.
    """
    return _HTML_COMMENT.sub("", _JINJA_COMMENT.sub("", path.read_text()))


def test_no_screen_says_corpus() -> None:
    offenders = []
    for template in sorted(TEMPLATE_DIR.rglob("*.html")):
        text = _rendered(template)
        for number, line in enumerate(text.splitlines(), 1):
            if any(word in line.lower() for word in JARGON):
                offenders.append(f"{template.name}:{number}: {line.strip()}")
    assert not offenders, (
        "these lines put an implementation word on a screen -- say what the user "
        "has instead (their CVs, the facts they confirmed):\n" + "\n".join(offenders)
    )


def _nav_links() -> list[tuple[str, str]]:
    """(href, label) for every link in the signed-in nav, in order."""
    base = (TEMPLATE_DIR / "base.html").read_text()
    nav = re.search(r"<nav\b[^>]*>(.*?)</nav>", base, re.S)
    assert nav is not None, "base.html has no <nav>"
    return re.findall(r'<a href="([^"]+)">([^<]+)</a>', nav.group(1))


def test_the_nav_links_each_destination_once() -> None:
    hrefs = [href for href, _ in _nav_links()]
    assert len(hrefs) == len(set(hrefs)), f"a destination is linked twice: {hrefs}"


def test_every_nav_label_is_distinct() -> None:
    """Two links reading "Corpus" tell a user nothing about either."""
    labels = [label.strip() for _, label in _nav_links()]
    assert len(labels) == len(set(labels)), f"two nav items share a label: {labels}"


def test_the_nav_reaches_the_cv_pages() -> None:
    assert "/background" in [href for href, _ in _nav_links()]


@pytest.mark.parametrize(
    ("old", "new"),
    [("/corpus", "/background"), ("/corpus/facts", "/background/facts")],
)
def test_the_old_paths_redirect_permanently(client: TestClient, old: str, new: str) -> None:
    """Bookmarks exist, and a 404 is a bad way to learn a word changed."""
    response = client.get(old, follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == new


# --------------------------------------------------------------------------
# Writing a CV: a stricter list, for one feature
# --------------------------------------------------------------------------

# The drafting screens had the most internal vocabulary on them -- "check
# coverage", a legend about the gate -- and the owner's complaint was that
# generating a CV was unclear. These words name parts of the machine, not
# anything the reader has. Checked on these screens only: "coverage" is fine in
# a stylesheet class, and other screens have their own history.
DRAFTING_JARGON = re.compile(
    r"\b(gate|span|spans|grounded|grounding|coverage|corpus|draft kind)\b", re.I
)
DRAFTING_TEMPLATES = (
    "application_drafts.html",
    "_draft.html",
    "_cv_steps.html",
)

_JINJA_TAG = re.compile(r"\{%.*?%\}|\{\{.*?\}\}", re.S)
_HTML_TAG = re.compile(r"<[^>]+>")


def _visible(text: str) -> str:
    """What a template would show, minus anything computed: comments, Jinja
    tags and expressions, and HTML tags (so a class name is not read as
    words). What the Python puts on screen is swept separately below."""
    return _HTML_TAG.sub(
        " ", _JINJA_TAG.sub(" ", _HTML_COMMENT.sub("", _JINJA_COMMENT.sub("", text)))
    )


@pytest.mark.parametrize("name", DRAFTING_TEMPLATES)
def test_the_drafting_templates_use_the_readers_words(name: str) -> None:
    found = DRAFTING_JARGON.findall(_visible((TEMPLATE_DIR / name).read_text()))
    assert not found, f"{name} shows {found}"


def test_the_cv_panel_on_the_application_page_uses_the_readers_words() -> None:
    text = (TEMPLATE_DIR / "application_detail.html").read_text()
    start = text.index("{% call sections.section(cv_section) %}")
    panel = text[start : text.index("{% endcall %}", start)]
    assert not DRAFTING_JARGON.findall(_visible(panel))


def test_what_the_drafting_helpers_put_on_screen_uses_the_readers_words() -> None:
    """Every string `jfl_web.drafts` renders: the marks and what they mean, each
    step, the running lines, the next actions and every failure sentence."""
    from jfl_web import drafts

    said: list[str] = [
        *drafts.VERDICT_WORDS.values(),
        *drafts.VERDICT_MEANINGS.values(),
        *drafts.STEP_LABELS.values(),
        *drafts._RUNNING.values(),
        *drafts._WHY.values(),
        drafts._REWORD,
        drafts._REWORD_TO_MATCH,
        *(f.message for f in drafts._COVERAGE_FAILURES.values()),
        *(f.message for f in drafts._DRAFT_FAILURES.values()),
        *(drafts.generate_label(kind) for kind in drafts.GENERATE_LABELS),
    ]
    for has_ad, ad_read, checked in [(True, False, False), (True, True, False), (True, True, True)]:
        plan = drafts.plan_steps(has_ad=has_ad, ad_read=ad_read, ad_reading=False, checked=checked)
        said.append(plan.cost_line)
    said.append(
        drafts.plan_steps(
            has_ad=False, ad_read=False, ad_reading=False, checked=False
        ).blocked_message
    )
    offenders = [line for line in said if DRAFTING_JARGON.search(line)]
    assert not offenders, offenders
