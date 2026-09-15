"""Slice C7 "Track as application": fetch one posting's description, lazily.

The board-watching engine (`jfl_intake.engine`) never stores a description --
a board check is about presence and reposting, and fetching every job's full
text on every check would be a lot of bytes for data nobody has asked to read
yet. Only when a user turns a watched job into an application does the worker
call `fetch_description` for that **one** posting, and feed the text to the
existing job-ad extraction (domain 2). This module is that one call.

**Same honesty rule as the adapters, applied to a single posting instead of a
whole board**: this never raises for anything the network or the ATS did. A
404, a timeout, a shape the platform's docs do not promise -- each comes back
as a `DescriptionResult` with a code, not an exception. `is_transient` is
true only for `unreachable` (a timeout, a 429, a 5xx): the kind of failure
that plausibly clears on a retry. Everything else -- `not_found`,
`bad_response`, `unsupported_platform`, `empty` -- is the same answer next
time, because nothing about *this* posting changed by waiting.

**Where a platform exposes a public per-posting JSON/XML endpoint, this uses
it** (one request). **Where it does not, and the description was already
present in the board's own listing response** (Ashby, Pinpoint, Recruitee's
list shape, Teamtailor's RSS, Personio's XML), the listing is re-fetched and
the matching posting is picked out by id -- still one request, and still the
platform's own structured API, never its human-facing page. Every endpoint
below was hit live on 2026-09-15 with the project's honest user agent
(`jfl_intake.http.USER_AGENT`), a handful of requests per platform, before
being relied on; where that did not hold, the platform maps to
`unsupported_platform` and the docstring says what was tried. See each
`_fetch_*` function for the endpoint and the shape it was verified against.

**`InvalidBoardKeyError` -- the stored `board_key` does not have what this
platform's endpoint needs -- always becomes `bad_response`, never
`not_found`.** `not_found` means the ATS said the posting is gone (a 404, or
absence from a re-fetched listing); a malformed key is not a fact about the
posting, it is this call being unable to even ask the question, which is the
same kind of "we cannot construct a valid request" failure as Workday's
missing `url` below. A `TransportError` or `RequestBudgetExceeded` raised out
of the one HTTP call a handler makes is caught centrally in
`fetch_description` and becomes `unreachable`, `requests=1` -- one attempt
was made and it did not land, exactly `adapters/base.py`'s
`from_http_failure` / `failure(..., requests=1)` convention for the board
fetchers.
"""

from __future__ import annotations

import html as html_module
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import quote

from jfl_core.models import BoardPlatform

from jfl_intake.adapters.base import InvalidBoardKeyError, require_key
from jfl_intake.adapters.personio import PersonioAdapter
from jfl_intake.adapters.single import MalformedResponseError
from jfl_intake.adapters.workday import WorkdayAdapter
from jfl_intake.adapters.xml_single import parse_xml
from jfl_intake.http import RequestBudgetExceeded, Transport, TransportError

DescriptionErrorCode = Literal[
    "unsupported_platform", "not_found", "unreachable", "bad_response", "empty"
]

# A CV bullet or a cover letter needs a paragraph or two of context, not the
# whole posting twice over with a benefits deck bolted on. 60,000 characters
# is a large multiple of every real posting seen while building this (the
# longest fixture captured live, Rippling's, is under 12,000) -- generous
# enough that truncation should be rare, bounded enough that a malformed feed
# or a platform bug cannot hand the drafting pipeline an unbounded prompt.
MAX_CHARS = 60_000


@dataclass(frozen=True, slots=True)
class DescriptionResult:
    text: str | None
    error_code: DescriptionErrorCode | None
    requests: int = 0

    @property
    def is_transient(self) -> bool:
        return self.error_code == "unreachable"


def _ok(text: str) -> DescriptionResult:
    return DescriptionResult(text=text, error_code=None, requests=1)


def _fail(code: DescriptionErrorCode, *, requests: int = 1) -> DescriptionResult:
    return DescriptionResult(text=None, error_code=code, requests=requests)


def _status_code(status: int) -> DescriptionErrorCode:
    """What a non-200 means for a single-posting fetch. Same split as
    `adapters.base.status_failure`, translated onto this module's smaller
    error set: 404 is the ATS saying the posting is gone; 429 and 5xx are
    plausibly transient; every other 4xx is a request this module built
    wrong or a posting the ATS refuses for a reason that will not change.
    """
    if status == 404:
        return "not_found"
    if status == 429 or status >= 500:
        return "unreachable"
    return "bad_response"


def _finish(text: str) -> DescriptionResult:
    """Bound and land the successful text, or report `empty`."""
    cleaned = text.strip()
    if not cleaned:
        return _fail("empty")
    if len(cleaned) > MAX_CHARS:
        cleaned = cleaned[:MAX_CHARS].rstrip() + f"\n\n[truncated at {MAX_CHARS} characters]"
    return _ok(cleaned)


# --------------------------------------------------------------------------
# HTML -> text. Headings, paragraphs and list items each land on their own
# line; inline markup (strong, em, span, a, ...) does not break the line it
# sits inside. Entities are decoded once by `HTMLParser` itself
# (`convert_charrefs=True`, the default), which is enough for every platform
# here except Greenhouse -- see `_fetch_greenhouse`.
# --------------------------------------------------------------------------

_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "aside",
        "main",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "table",
        "thead",
        "tbody",
        "tr",
        "td",
        "th",
        "blockquote",
        "figure",
        "figcaption",
        "hr",
    }
)
# Content of these never reaches `handle_data`: a job description is never
# supposed to carry script/style, but a platform's WYSIWYG export sometimes
# leaves an empty one behind, and this keeps it from becoming stray text.
_SKIP_TAGS = frozenset({"script", "style", "head", "noscript", "template"})


# An explicit line break within one block (`<br>`), kept apart from a plain
# space during accumulation so `_flush` can tell "the source HTML happened to
# wrap this paragraph onto two lines" (collapses to one space -- source
# formatting is not structure) from "the employer asked for a line break here"
# (kept as one). Not a character real markup or a `handle_data` chunk
# produces, so it cannot collide with content.
_BR = "\x00"


class _HTMLTextExtractor(HTMLParser):
    """Collects `_blocks`, one already-clean string per heading/paragraph/list
    item -- whitespace collapsed and `<br>` line breaks resolved right here in
    `_flush`, so `html_to_text` below does no further processing. A `<li>`
    (however deeply its own text is wrapped, e.g. `<li><p>...</p></li>` --
    SmartRecruiters does this) gets a leading `"- "` on whichever block is the
    first text to actually land inside it, tracked with `_li_pending` rather
    than prepended blindly, so a `<li>` that wraps its text in a `<p>` does
    not produce a bare `"-"` line followed by an unmarked one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._blocks: list[str] = []
        self._current: list[str] = []
        self._skip_depth = 0
        self._li_pending: list[bool] = []

    def _flush(self) -> None:
        raw = "".join(self._current)
        self._current = []
        # `.split()` with no argument treats any run of whitespace -- spaces,
        # tabs, source newlines, a decoded `&nbsp;` -- as one separator, so
        # this both collapses formatting whitespace and normalises it to a
        # plain space, without touching the `_BR` sentinel.
        segments = [" ".join(segment.split()) for segment in raw.split(_BR)]
        text = "\n".join(segment for segment in segments if segment)
        if not text:
            return
        if self._li_pending and self._li_pending[-1]:
            text = "- " + text
            self._li_pending[-1] = False
        self._blocks.append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            if not self._skip_depth:
                # A plain space, not `_BR`: dropped script/style content
                # should not fuse the words on either side of it, but it is
                # not the employer asking for a line break either.
                self._current.append(" ")
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._current.append(_BR)
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            if tag == "li":
                self._li_pending.append(True)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br" and not self._skip_depth:
            self._current.append(_BR)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in _BLOCK_TAGS:
            self._flush()
            if tag == "li" and self._li_pending:
                self._li_pending.pop()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._current.append(data)

    def text(self) -> str:
        self._flush()
        return "\n".join(self._blocks)


def html_to_text(markup: str) -> str:
    """Readable plain text from one HTML fragment or document: headings,
    paragraphs and list items each on their own line, comments and
    script/style content dropped, intra-line whitespace (including a
    decoded `&nbsp;`) collapsed to single spaces, blank lines dropped.
    """
    extractor = _HTMLTextExtractor()
    extractor.feed(markup)
    extractor.close()
    return extractor.text()


def _plain_text(text: str) -> str:
    """Already-plain text (Ashby's `descriptionPlain`, used only when
    `descriptionHtml` is absent): there are no tags to mark a paragraph
    boundary, so the platform's own blank lines are trusted for that instead
    -- unlike `html_to_text`, which has real structure to read and does not
    need this. Other whitespace runs within a paragraph still collapse.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    lines = (" ".join(paragraph.split()) for paragraph in paragraphs)
    return "\n".join(line for line in lines if line)


# --------------------------------------------------------------------------
# Per platform.
# --------------------------------------------------------------------------

_Handler = Callable[[Mapping[str, str], str, str | None, Transport], DescriptionResult]


def _fetch_greenhouse(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """`GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs/{id}`.

    Verified live 2026-09-15 against Anthropic (`token=anthropic`, a real job
    id from that board): 200, `content` present. A missing id is a clean 404
    (also checked live).

    **`content` is HTML-encoded HTML** -- the JSON string is literally
    `"&lt;div class=&quot;...&quot;&gt;..."`, not `"<div class=\\"...\\">..."`.
    One `html.unescape()` turns it into the real markup `html_to_text` expects;
    skipping it would feed the parser a document that is one big text node of
    escaped angle brackets. This is specific to Greenhouse -- every other
    platform below hands back markup that is already real HTML (or, for the
    two XML feeds, was already decoded once by the XML parser itself).
    """
    (token,) = require_key(board_key, "token")
    response = transport.get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{quote(token, safe='')}"
        f"/jobs/{quote(external_id, safe='')}"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    content = body.get("content") if isinstance(body, dict) else None
    if not isinstance(content, str):
        return _fail("bad_response")
    return _finish(html_to_text(html_module.unescape(content)))


def _fetch_ashby(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """No per-posting endpoint: `GET .../posting-api/job-board/{name}/{id}`
    answers 401 even for a real id (checked live 2026-09-15) -- the posting
    detail lives only behind the same board listing this project already
    fetches for a check. So this re-fetches
    `GET https://api.ashbyhq.com/posting-api/job-board/{name}` (verified live
    against OpenAI) and finds the job whose `id` matches, by `descriptionHtml`
    (falling back to the already-plain `descriptionPlain` on the rare posting
    that has one but not the other). Not present in the listing at all
    (unlisted, or gone since the board was last checked) is `not_found`.
    """
    (name,) = require_key(board_key, "name")
    response = transport.get_json(
        f"https://api.ashbyhq.com/posting-api/job-board/{quote(name, safe='')}"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    jobs = body.get("jobs") if isinstance(body, dict) else None
    if not isinstance(jobs, list):
        return _fail("bad_response")
    job = next((j for j in jobs if isinstance(j, dict) and str(j.get("id")) == external_id), None)
    if job is None:
        return _fail("not_found")
    described = job.get("descriptionHtml")
    if isinstance(described, str):
        return _finish(html_to_text(described))
    plain = job.get("descriptionPlain")
    if isinstance(plain, str):
        return _finish(_plain_text(plain))
    return _fail("bad_response")


def _fetch_lever(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """`GET https://api.lever.co/v0/postings/{company}/{id}?mode=json`.

    Verified live 2026-09-15 against Palantir: 200 with `description`
    (opening + role body, already combined) and `lists` -- named groups
    (`text` as the heading, `content` as raw `<li>...</li>` HTML, no
    enclosing `<ul>`; feeding bare `<li>` tags to `html_to_text` is fine, it
    does not require valid nesting) -- and `additional` (a closing section,
    "Life at Palantir" on this posting). A missing id is a clean 404 with
    `{"ok": false, ...}` (also checked live). All three parts are
    concatenated in that order; `lists` entries that are not the expected
    shape are skipped rather than failing the whole fetch.
    """
    (company,) = require_key(board_key, "company")
    response = transport.get_json(
        f"https://api.lever.co/v0/postings/{quote(company, safe='')}"
        f"/{quote(external_id, safe='')}?mode=json"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    if not isinstance(body, dict):
        return _fail("bad_response")
    description = body.get("description")
    if not isinstance(description, str):
        return _fail("bad_response")
    parts = [description]
    lists = body.get("lists")
    for entry in lists if isinstance(lists, list) else []:
        if not isinstance(entry, dict):
            continue
        heading = entry.get("text")
        content = entry.get("content")
        if isinstance(heading, str) and heading:
            parts.append(f"<h3>{html_module.escape(heading)}</h3>")
        if isinstance(content, str):
            parts.append(content)
    additional = body.get("additional")
    if isinstance(additional, str):
        parts.append(additional)
    return _finish(html_to_text("".join(parts)))


def _fetch_workday(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """`GET https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{path}`,
    where `{path}` is the posting's `/job/...` slug -- the same one the
    listing's `externalPath` carries and `POSTING_URL` in
    `adapters.workday` builds the stored `url` from. Verified live
    2026-09-15 against NVIDIA: `jobPostingInfo.jobDescription` is HTML. A
    POST to this path (the listing's own verb) answers 400; GET is correct.

    `external_id` alone -- the `_`-suffix `adapters.workday` keys on -- is
    not enough to rebuild `{path}`; only the stored `url` has the full slug.
    So this **requires `url`** and derives `{path}` by stripping the
    `https://{tenant}.{wd}.myworkdayjobs.com/{site}` prefix from it; a
    missing `url`, or one that does not start with that exact prefix followed
    by `/job/`, is `bad_response` -- Workday is labelled fragile by decision
    (see `adapters.workday`'s docstring), so an unexpected shape here fails
    loudly rather than guessing a path.
    """
    # `validate_key` (not just `require_key`) so the `wd` shape check
    # (`adapters.workday.WorkdayAdapter`) is the single source of truth
    # rather than duplicated here.
    WorkdayAdapter().validate_key(board_key)
    tenant, wd, site = require_key(board_key, "tenant", "wd", "site")
    prefix = f"https://{tenant}.{wd}.myworkdayjobs.com/{site}"
    if url is None or not url.startswith(prefix) or not url[len(prefix) :].startswith("/job/"):
        return _fail("bad_response", requests=0)
    path = url[len(prefix) :]
    response = transport.get_json(
        f"https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{path}"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    info = body.get("jobPostingInfo") if isinstance(body, dict) else None
    description = info.get("jobDescription") if isinstance(info, dict) else None
    if not isinstance(description, str):
        return _fail("bad_response")
    return _finish(html_to_text(description))


def _fetch_smartrecruiters(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """`GET https://api.smartrecruiters.com/v1/companies/{companyId}/postings/{id}`.

    Verified live 2026-09-15 against Bosch: 200,
    `jobAd.sections` with (observed) `companyDescription`, `jobDescription`,
    `qualifications`, `additionalInformation`, each `{title, text}` with
    `text` as HTML. A missing id is a clean 404 (also checked live). Sections
    are emitted in the fixed order above -- a real order the API happened to
    return them in, kept deterministic here rather than trusted to repeat --
    with their `title` as a heading; any other section key observed later
    would need adding to the list (nothing here would silently drop it as
    `bad_response`, but nor would it appear -- there is no evidence yet of a
    fifth section to build a case around).
    """
    (company_id,) = require_key(board_key, "company_id")
    response = transport.get_json(
        f"https://api.smartrecruiters.com/v1/companies/{quote(company_id, safe='')}"
        f"/postings/{quote(external_id, safe='')}"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    job_ad = body.get("jobAd") if isinstance(body, dict) else None
    sections = job_ad.get("sections") if isinstance(job_ad, dict) else None
    if not isinstance(sections, dict):
        return _fail("bad_response")
    parts: list[str] = []
    for key in ("companyDescription", "jobDescription", "qualifications", "additionalInformation"):
        section = sections.get(key)
        if not isinstance(section, dict):
            continue
        title = section.get("title")
        text = section.get("text")
        if isinstance(title, str) and title:
            parts.append(f"<h3>{html_module.escape(title)}</h3>")
        if isinstance(text, str):
            parts.append(text)
    return _finish(html_to_text("".join(parts)))


def _fetch_workable(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """No working per-shortcode endpoint was found (`/api/v1/jobs/{shortcode}`
    and `/api/v3/accounts/{subdomain}/jobs/{shortcode}` both answer 404 for a
    real, live shortcode -- checked 2026-09-15). What does carry the
    description is `GET https://apply.workable.com/api/v1/widget/accounts/
    {subdomain}?details=true`, the same public unauthenticated endpoint the
    careers-page widget itself calls: verified live against Hugging Face and
    against Devsinc (35 jobs, one request, no paging) -- every job entry
    includes `description` (HTML).

    That listing keys jobs by `shortcode`, not the numeric `id` this project
    stores as `external_id` (`adapters.workable._external_id`) -- the two are
    unrelated strings. So this extracts the shortcode from the stored `url`
    (`https://apply.workable.com/{subdomain}/j/{shortcode}`, built in
    `adapters.workable.WorkableAdapter.fetch`) rather than from `external_id`.
    A missing `url`, or one not matching that shape, is `bad_response`: there
    is no way to ask the question without it.
    """
    (subdomain,) = require_key(board_key, "subdomain")
    prefix = f"https://apply.workable.com/{subdomain}/j/"
    if url is None or not url.startswith(prefix) or len(url) <= len(prefix):
        return _fail("bad_response", requests=0)
    shortcode = url[len(prefix) :]
    account = quote(subdomain, safe="")
    response = transport.get_json(
        f"https://apply.workable.com/api/v1/widget/accounts/{account}?details=true"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    jobs = body.get("jobs") if isinstance(body, dict) else None
    if not isinstance(jobs, list):
        return _fail("bad_response")
    job = next((j for j in jobs if isinstance(j, dict) and j.get("shortcode") == shortcode), None)
    if job is None:
        return _fail("not_found")
    description = job.get("description")
    if not isinstance(description, str):
        return _fail("bad_response")
    return _finish(html_to_text(description))


def _fetch_pinpoint(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """No per-posting endpoint is public (`/api/v1/*` is the authenticated
    API, 401 unauthenticated -- see `adapters.pinpoint`'s docstring). The
    listing itself already carries the full text, though: `GET
    https://{company}.pinpointhq.com/postings.json`, verified live 2026-09-15
    against Sun King, has `description`, `key_responsibilities`,
    `skills_knowledge_expertise` and `benefits` (each HTML) plus their
    `*_header` labels directly on every posting in `data`. Re-fetched and
    matched by `id`; a posting no longer listed is `not_found`.
    """
    (company,) = require_key(board_key, "company")
    response = transport.get_json(f"https://{quote(company, safe='')}.pinpointhq.com/postings.json")
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list):
        return _fail("bad_response")
    posting = next(
        (p for p in data if isinstance(p, dict) and str(p.get("id")) == external_id), None
    )
    if posting is None:
        return _fail("not_found")
    parts: list[str] = []
    for header_key, body_key in (
        (None, "description"),
        ("key_responsibilities_header", "key_responsibilities"),
        ("skills_knowledge_expertise_header", "skills_knowledge_expertise"),
        ("benefits_header", "benefits"),
    ):
        text = posting.get(body_key)
        if not isinstance(text, str):
            continue
        if header_key is not None:
            header = posting.get(header_key)
            if isinstance(header, str) and header:
                parts.append(f"<h3>{html_module.escape(header)}</h3>")
        parts.append(text)
    return _finish(html_to_text("".join(parts)))


def _fetch_recruitee(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """`GET https://{company}.recruitee.com/api/offers/{id}` -> `{"offer": {...}}`.

    Verified live 2026-09-15 against Make: 200, `description` and
    `requirements` both present (HTML) on a real offer. A missing id is a
    clean 404 (also checked live). The two fields are concatenated;
    `requirements` on the postings checked already carries its own heading
    (an `<h4>`), so none is added here.
    """
    (company,) = require_key(board_key, "company")
    response = transport.get_json(
        f"https://{quote(company, safe='')}.recruitee.com/api/offers/{quote(external_id, safe='')}"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    offer = body.get("offer") if isinstance(body, dict) else None
    # `id` is checked as a sanity marker that this really is an offer object
    # (present on every real offer, listed or single) -- distinguishing a
    # genuinely wrong shape (`bad_response`) from an offer whose description
    # and requirements both happen to be blank (`empty`, via `_finish` below).
    if not isinstance(offer, dict) or "id" not in offer:
        return _fail("bad_response")
    parts = [v for v in (offer.get("description"), offer.get("requirements")) if isinstance(v, str)]
    return _finish(html_to_text("".join(parts)))


def _fetch_teamtailor(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """The RSS feed (`adapters.teamtailor`) is the only public surface, so
    this re-fetches `GET https://{site}/jobs.rss` (`get_text` -- an RSS body
    is never JSON) and matches the item whose `<guid>` is `external_id`.

    Verified live 2026-09-15 against `career.teamtailor.com`: an item's
    `<description>` is already real HTML -- the XML parser decodes entities
    once while reading the element's text, so (unlike Greenhouse) no extra
    `html.unescape()` is needed. A malformed or channel-less feed is
    `bad_response`, matching `xml_single.fetch_single_xml`'s rule for the
    board check; a well-formed feed with no matching `<guid>` is `not_found`;
    an item with no `<description>` at all is `empty`, not `bad_response` --
    RSS does not require every item to carry one, so a blank one here is
    information (the employer wrote nothing), not a broken response.
    """
    (site,) = require_key(board_key, "site")
    response = transport.get_text(f"https://{site}/jobs.rss")
    if response.status != 200:
        return _fail(_status_code(response.status))
    if not isinstance(response.body, str):
        return _fail("bad_response")
    try:
        root = parse_xml(response.body)
    except MalformedResponseError:
        return _fail("bad_response")
    channel = root.find("channel")
    if channel is None:
        return _fail("bad_response")
    item = next((i for i in channel.findall("item") if i.findtext("guid") == external_id), None)
    if item is None:
        return _fail("not_found")
    description = item.findtext("description")
    if not description:
        return _fail("empty")
    return _finish(html_to_text(description))


def _fetch_personio(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """The XML feed (`adapters.personio`) is the only public surface -- the
    adapter's own docstring records that a guessed per-posting URL could not
    be verified live (two attempts both 429'd). So this re-fetches `GET
    https://{company}.jobs.personio.{tld}/xml` and matches the `<position>`
    whose `<id>` is `external_id`.

    A position's text lives under `<jobDescriptions><jobDescription>
    <name>...</name><value>...</value></jobDescription>...</jobDescriptions>`
    -- Personio's documented recruiting XML export shape. **This shape is not
    verified populated live**: the one tenant checked (`personio.jobs.
    personio.de`, its own careers board, 2026-09-15) has exactly one position,
    and its `<jobDescriptions>` element is present but empty
    (`<jobDescriptions></jobDescriptions>`), which does confirm the element
    exists and exercises the `empty` path below, but not the `name`/`value`
    pairing for a populated one. A `<jobDescriptions>` that is missing
    entirely, rather than present-and-empty, is `bad_response` -- a
    genuinely different shape from what was observed, not the same "nothing
    to say" case.
    """
    # `validate_key` (not just `require_key`) so the `tld` check
    # (`adapters.personio.PersonioAdapter`) is the single source of truth.
    PersonioAdapter().validate_key(board_key)
    company, tld = require_key(board_key, "company", "tld")
    response = transport.get_text(f"https://{company}.jobs.personio.{tld}/xml")
    if response.status != 200:
        return _fail(_status_code(response.status))
    if not isinstance(response.body, str):
        return _fail("bad_response")
    try:
        root = parse_xml(response.body)
    except MalformedResponseError:
        return _fail("bad_response")
    if root.tag != "workzag-jobs":
        return _fail("bad_response")
    position = next((p for p in root.findall("position") if p.findtext("id") == external_id), None)
    if position is None:
        return _fail("not_found")
    descriptions = position.find("jobDescriptions")
    if descriptions is None:
        return _fail("bad_response")
    parts: list[str] = []
    for entry in descriptions.findall("jobDescription"):
        name = entry.findtext("name")
        value = entry.findtext("value")
        if isinstance(name, str) and name:
            parts.append(f"<h3>{html_module.escape(name)}</h3>")
        if isinstance(value, str):
            parts.append(value)
    return _finish(html_to_text("".join(parts)))


def _fetch_rippling(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """`GET https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs/{uuid}`
    -- the same host and prefix as the listing (`adapters.rippling`'s `API`),
    with the posting's `uuid` (this platform's `external_id`) appended.

    Verified live 2026-09-15 against Rippling's own board: 200,
    `description.role` (the job-specific content) and `description.company`
    (a closing "About Rippling" blurb), both HTML. A missing uuid is a clean
    404 (also checked live). Concatenated `role` then `company`, matching a
    posting page's own order -- role-specific content before the boilerplate.
    """
    (slug,) = require_key(board_key, "slug")
    response = transport.get_json(
        f"https://api.rippling.com/platform/api/ats/v1/board/{quote(slug, safe='')}"
        f"/jobs/{quote(external_id, safe='')}"
    )
    if response.status != 200:
        return _fail(_status_code(response.status))
    body = response.body
    description = body.get("description") if isinstance(body, dict) else None
    if not isinstance(description, dict):
        return _fail("bad_response")
    parts = [v for v in (description.get("role"), description.get("company")) if isinstance(v, str)]
    if not parts:
        return _fail("bad_response")
    return _finish(html_to_text("".join(parts)))


def _fetch_breezy(
    board_key: Mapping[str, str], external_id: str, url: str | None, transport: Transport
) -> DescriptionResult:
    """**Unsupported.** The only public endpoint, `GET
    https://{company}.breezy.hr/json` (`adapters.breezy`), carries no
    description field at all on any job. Three candidate per-posting JSON
    paths were tried live 2026-09-15 against Breezy's own trial board
    (`json/{id}`, `position/{id}.json`, `p/{id}.json`): each answers `302` to
    `/`, i.e. does not exist. What does hold the text is the public HTML
    posting page (`{job.url}`, e.g. `.../p/{id}-{slug}`) -- a page meant for a
    browser, not a documented API, and this project's intake policy is ATS
    APIs and RSS only (`CLAUDE.md`'s no-scraping rule; see also Workday's own
    fragile-but-in-scope carve-out, which this platform does not have an
    equivalent structured-data argument for). So: no reachable description
    source, and `unsupported_platform` always, at zero cost.
    """
    return _fail("unsupported_platform", requests=0)


_HANDLERS: dict[BoardPlatform, _Handler] = {
    "greenhouse": _fetch_greenhouse,
    "ashby": _fetch_ashby,
    "lever": _fetch_lever,
    "workday": _fetch_workday,
    "smartrecruiters": _fetch_smartrecruiters,
    "rippling": _fetch_rippling,
    "breezy": _fetch_breezy,
    "teamtailor": _fetch_teamtailor,
    "personio": _fetch_personio,
    "recruitee": _fetch_recruitee,
    "pinpoint": _fetch_pinpoint,
    "workable": _fetch_workable,
}
# Exported so a test can assert this agrees with `BoardPlatform` (`get_args`)
# -- the same drift guard `test_board_platform_sources_agree` runs for the
# adapter registry -- and fail the build if a platform is ever added here
# without a handler, rather than silently falling through to
# `unsupported_platform` for a platform this module never actually decided
# about.
PLATFORM_HANDLERS: Mapping[BoardPlatform, _Handler] = _HANDLERS


def fetch_description(
    platform: BoardPlatform,
    board_key: Mapping[str, str],
    external_id: str,
    url: str | None,
    transport: Transport,
) -> DescriptionResult:
    handler = _HANDLERS.get(platform)
    if handler is None:  # pragma: no cover -- unreachable while BoardPlatform agrees
        return _fail("unsupported_platform", requests=0)
    try:
        return handler(board_key, external_id, url, transport)
    except InvalidBoardKeyError:
        return _fail("bad_response", requests=0)
    except TransportError:
        return _fail("unreachable")
    except RequestBudgetExceeded:
        return _fail("unreachable")
