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
