"""Render a `CvDocument` to the PDF that gets sent, and to an HTML preview of it.

One template (`templates/cv/document.html`) and two stylesheets (`classic`,
`modern`) drawn from the owner's own CVs. The same HTML is what WeasyPrint lays
out as the PDF and what the browser shows as the preview, so the preview cannot
drift from the file.

The PDF is what an employer receives, so three rules hold:

- **Only text goes in.** The template is handed the document's text fields, and
  nothing from the claim gate: `CvLine.verdict` and `CvLine.note` never reach it.
  The gate informs the author; it does not annotate what they send.
- **Real, selectable text** with the fonts embedded (WeasyPrint subsets and
  embeds every font it uses) -- applicant tracking systems parse the text layer.
- **Nothing is fetched.** The page has no external resources, and the URL
  fetcher refuses anything anyway, so a CV render can never reach the network.

Placement: this lives in `jfl_web`, not `jfl_core`, because WeasyPrint pulls in
Pango through cffi and a presentation stylesheet -- neither belongs in the
storage/schema package every other package imports, and the web app is the only
caller (the preview and the download). If the worker ever renders, it already
depends on the same image, and the module has no FastAPI import to drag along.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlsplit

from jfl_core.cv_document import CvDocument, CvLink
from jinja2 import Environment, FileSystemLoader, select_autoescape

_CV_DIR = Path(__file__).parent / "templates" / "cv"

# A link in the header becomes a live PDF annotation, so only schemes a reader
# expects to click are kept; anything else renders as plain text.
_SAFE_SCHEMES = frozenset({"http", "https", "mailto"})


@cache
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_CV_DIR)),
        autoescape=select_autoescape(default=True, default_for_string=True),
        keep_trailing_newline=True,
    )


@cache
def _css(template: str) -> str:
    base = (_CV_DIR / "base.css").read_text(encoding="utf-8")
    own = (_CV_DIR / f"{template}.css").read_text(encoding="utf-8")
    return base + "\n" + own


def _tagline_items(tagline: str) -> list[str]:
    return [item.strip() for item in tagline.split("|") if item.strip()]


def _safe_link(link: CvLink) -> CvLink | None:
    return link if urlsplit(link.url).scheme.lower() in _SAFE_SCHEMES else None


def _sendable(doc: CvDocument) -> CvDocument:
    """The document stripped to what may be sent: gate verdicts and evidence
    notes removed, unclickable link schemes dropped to plain text.

    Belt and braces -- the template never reads `verdict` or `note` -- but a
    future template edit should not be one `{{ line.note }}` away from mailing
    the gate's opinion of a sentence to an employer.
    """
    clean = doc.model_copy(deep=True)
    for line in clean.generated_lines() + clean.education:
        line.verdict = None
        line.note = ""
    links = []
    for link in clean.header.links:
        if _safe_link(link) is None:
            clean.header.contact.append(link.label)
        else:
            links.append(link)
    clean.header.links = links
    return clean


def render_cv_html(doc: CvDocument) -> str:
    """The CV as a standalone HTML page (styles inline, no external assets).

    On screen it shows as an A4 sheet; printed, or passed to `render_cv_pdf`, it
    is the page itself.
    """
    clean = _sendable(doc)
    context: dict[str, Any] = {
        "doc": clean,
        "tagline": _tagline_items(clean.header.tagline),
        "css": _css(clean.template),
    }
    return _env().get_template("document.html").render(**context)


def _refuse_fetch(url: str) -> NoReturn:
    from weasyprint.urls import URLFetchingError

    raise URLFetchingError(f"CV rendering fetches nothing: {url}")


def render_cv_pdf(doc: CvDocument) -> bytes:
    """The CV as an A4 PDF with a real text layer and embedded fonts.

    Metadata title is "<Name> — CV", taken from the page's <title>.
    """
    # Imported here so importing jfl_web (and its tests) does not load Pango.
    from weasyprint import HTML

    pdf = HTML(string=render_cv_html(doc), url_fetcher=_refuse_fetch).write_pdf()
    assert isinstance(pdf, bytes)
    return pdf
