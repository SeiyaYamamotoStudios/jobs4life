"""Shared shape for the feed-based adapters (Teamtailor, Personio): one
RSS/XML response is the whole board, parsed with `defusedxml` -- never the
standard library's `xml.etree`, because both feeds come from a server we do
not control and a malicious one could use the parse to exhaust memory
(entity expansion) or read local files (external entities). `defusedxml`
refuses both before anything is expanded.

**Why this is a separate module from `single.py`, not a reuse of it.**
`Transport.get_json` parses a response body as JSON and hands back `None`
when it is not; an RSS or XML feed is never JSON. This module instead uses
`Transport.get_text` (added 2026-09-11), which hands back the body as decoded
text verbatim, so a live feed reaches `parse` intact. A response that is not
well-formed XML, or well-formed but the wrong shape, still becomes
`failed` / `malformed_response` -- never an empty board -- exactly like every
other adapter's rule for an unexpected 200. `response.body` from `get_text`
is always a string on any status; the `isinstance` check below is a belt and
braces against a fake transport in a test giving the wrong shape, not
something a real `HttpxTransport` response can produce.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring

from jfl_intake.adapters.base import FetchResult, dedupe, failure, from_http_failure
from jfl_intake.adapters.single import MalformedResponseError, Parsed
from jfl_intake.http import RequestBudgetExceeded, Transport, TransportError


def parse_xml(text: str) -> Any:
    """The root element of a well-formed, safely-parsed document.

    Raises `MalformedResponseError` for anything not well-formed, and for an
    entity-expansion or external-entity payload -- `defusedxml` refuses those
    before they are ever expanded, and that refusal is folded into the same
    "malformed" outcome an adapter already has a rule for.
    """
    try:
        return fromstring(text)
    except (DefusedXmlException, ParseError) as exc:
        raise MalformedResponseError from exc


def fetch_single_xml(transport: Transport, url: str, parse: Callable[[str], Parsed]) -> FetchResult:
    try:
        response = transport.get_text(url)
    except TransportError as exc:
        return failure("unreachable", exc.code, requests=1)
    except RequestBudgetExceeded as exc:
        return failure("incomplete", exc.code)

    if response.status != 200:
        return from_http_failure(response, requests=1)
    if not isinstance(response.body, str):
        # `get_text` always hands back a string on any real transport; this
        # guards only against a test double built with the wrong shape.
        # Never treated as an empty board.
        return failure("failed", "malformed_response", requests=1)
    try:
        parsed = parse(response.body)
    except MalformedResponseError:
        return failure("failed", "malformed_response", requests=1)

    jobs = dedupe(parsed.jobs)
    if parsed.unidentified:
        return FetchResult(
            status="incomplete",
            jobs=jobs,
            expected_total=parsed.expected_total,
            error_code="unidentifiable_job",
            requests=1,
        )
    return FetchResult(
        status="complete", jobs=jobs, expected_total=parsed.expected_total, requests=1
    )
