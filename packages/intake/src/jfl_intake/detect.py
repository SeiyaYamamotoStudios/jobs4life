"""Pasted URL -> which ATS, and the identifier its API needs. By pattern only.

The input to the whole feature is one pasted careers-board URL, and this is the
first thing that happens to it. Deterministic on purpose: a URL either matches a
known board shape or it does not, and "does not" is answered with a message the
owner can act on rather than a guess.

Supported, each verified live on 2026-09-10:

  * Greenhouse -- `boards.greenhouse.io/{token}`, `job-boards.greenhouse.io/{token}`
    (any path below the token, e.g. `/anthropic/jobs/4461450008`), and the
    embed form `boards.greenhouse.io/embed/job_board?for={token}`;
  * Ashby -- `jobs.ashbyhq.com/{name}`;
  * Lever -- `jobs.lever.co/{company}`;
  * Workday -- `{tenant}.{wdN}.myworkdayjobs.com/{site}`, optionally with a
    locale segment first (`/en-US/{site}`), and any path below the site.

EU-hosted variants (`job-boards.eu.greenhouse.io`, `jobs.eu.lever.co`) exist but
were not verified, so they are rejected as unsupported rather than mapped to an
API host nobody has seen answer.

**LinkedIn and Indeed are rejected, in any form** -- including their short-link
hosts. Not only for legal reasons: a scraper in this architecture is a judgement
call a hiring manager reads, and CLAUDE.md rules it out. The message says what to
paste instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlsplit

from jfl_core.models import BoardPlatform


class BoardUrlError(ValueError):
    """A URL that can never become a watched board. The message is user-facing."""


class ForbiddenSourceError(BoardUrlError):
    pass


class UnsupportedBoardError(BoardUrlError):
    pass


class MalformedBoardUrlError(BoardUrlError):
    pass


FORBIDDEN_SOURCE_MESSAGE = (
    "LinkedIn and Indeed are not supported, and will not be: jobs4life does not "
    "scrape them. Paste the employer's own careers board instead -- a Greenhouse, "
    "Ashby, Lever or Workday URL."
)

UNSUPPORTED_MESSAGE = (
    "That does not look like a careers board jobs4life can watch. Supported boards "
    "are Greenhouse (boards.greenhouse.io/...), Ashby (jobs.ashbyhq.com/...), "
    "Lever (jobs.lever.co/...) and Workday (....myworkdayjobs.com/...)."
)


@dataclass(frozen=True, slots=True)
class BoardRef:
    platform: BoardPlatform
    board_key: dict[str, str] = field(hash=False)
    board_url: str


# Matched against the host's registrable suffix, so `uk.linkedin.com`,
# `www.indeed.co.uk` and `lnkd.in` are all caught.
_FORBIDDEN_HOST = re.compile(
    r"(^|\.)(linkedin\.com|lnkd\.in|indeed\.[a-z.]+|indeedjobs\.com)$", re.IGNORECASE
)
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_WORKDAY_HOST = re.compile(
    r"^(?P<tenant>[a-z0-9][a-z0-9-]*)\.(?P<wd>wd\d{1,3})\.myworkdayjobs\.com$", re.IGNORECASE
)
_LOCALE = re.compile(r"^[a-z]{2}(-[A-Za-z]{2})?$")
# Path segments that are never a Greenhouse token or a Workday site.
_GREENHOUSE_RESERVED = frozenset({"embed", "v1", "boards"})
_WORKDAY_RESERVED = frozenset({"wday", "job", "details"})


def is_forbidden_host(host: str) -> bool:
    return bool(_FORBIDDEN_HOST.search(host.rstrip(".")))


def detect_board(url: str) -> BoardRef:
    """Parse a pasted URL. Raises a `BoardUrlError` subclass with a user-facing
    message; never returns a guess.
    """
    raw = (url or "").strip()
    if not raw:
        raise MalformedBoardUrlError("Paste the URL of a careers board.")
    if "://" not in raw:
        raw = "https://" + raw

    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
    except ValueError:
        raise MalformedBoardUrlError("That is not a valid URL.") from None
    if parts.scheme.lower() not in ("http", "https") or not host or "." not in host:
        raise MalformedBoardUrlError("That is not a valid URL.")

    # Before anything else, so no LinkedIn or Indeed URL can ever fall through to
    # a pattern that happens to match it.
    if is_forbidden_host(host):
        raise ForbiddenSourceError(FORBIDDEN_SOURCE_MESSAGE)

    segments = [unquote(s) for s in parts.path.split("/") if s]

    if host in ("boards.greenhouse.io", "job-boards.greenhouse.io"):
        return _greenhouse(url.strip(), segments, parts.query)
    if host == "jobs.ashbyhq.com":
        return _single_segment(url.strip(), "ashby", "name", segments)
    if host == "jobs.lever.co":
        return _single_segment(url.strip(), "lever", "company", segments)
    workday = _WORKDAY_HOST.match(host)
    if workday:
        return _workday(url.strip(), workday.group("tenant"), workday.group("wd"), segments)

    raise UnsupportedBoardError(UNSUPPORTED_MESSAGE)


def _greenhouse(url: str, segments: list[str], query: str) -> BoardRef:
    token: str | None = None
    if segments[:2] == ["embed", "job_board"]:
        values = parse_qs(query).get("for")
        token = values[0] if values else None
    elif segments and segments[0] not in _GREENHOUSE_RESERVED:
        token = segments[0]
    if not token or not _TOKEN.match(token):
        raise UnsupportedBoardError(
            "That Greenhouse URL does not name a board. It should look like "
            "boards.greenhouse.io/<company>."
        )
    return BoardRef(platform="greenhouse", board_key={"token": token.lower()}, board_url=url)


def _single_segment(
    url: str, platform: BoardPlatform, key_name: str, segments: list[str]
) -> BoardRef:
    if not segments or not _TOKEN.match(segments[0]):
        host = "jobs.ashbyhq.com" if platform == "ashby" else "jobs.lever.co"
        raise UnsupportedBoardError(
            f"That URL does not name a board. It should look like {host}/<company>."
        )
    # Case is kept: Ashby board names are case-sensitive in the API path.
    return BoardRef(platform=platform, board_key={key_name: segments[0]}, board_url=url)


def _workday(url: str, tenant: str, wd: str, segments: list[str]) -> BoardRef:
    rest = segments[1:] if segments and _LOCALE.match(segments[0]) else segments
    if not rest or rest[0].lower() in _WORKDAY_RESERVED or not _TOKEN.match(rest[0]):
        raise UnsupportedBoardError(
            "That Workday URL does not name a careers site. It should look like "
            "<company>.wd5.myworkdayjobs.com/<site>."
        )
    return BoardRef(
        platform="workday",
        board_key={"tenant": tenant.lower(), "wd": wd.lower(), "site": rest[0]},
        board_url=url,
    )
