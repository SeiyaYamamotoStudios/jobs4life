"""Teamtailor and Personio: RSS/XML feeds, parsed with `defusedxml`. No test
opens a socket; the live fixtures are trimmed and were captured 2026-09-10.

Both adapters fetch via `Transport.get_text` (added 2026-09-11 specifically
for these two -- `get_json` discards a non-JSON body as `None`, which is what
an RSS/XML feed always is). See `jfl_intake.adapters.xml_single` and
`test_httpx_transport_get_text_returns_an_rss_body_intact` in
`test_intake_adapters.py` for the real-transport proof; here the fake
transport's `get_text` just hands back the text it was given.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jfl_intake.adapters.base import InvalidBoardKeyError
from jfl_intake.adapters.personio import PersonioAdapter
from jfl_intake.adapters.teamtailor import TeamtailorAdapter
from jfl_intake.http import HttpResponse, TransportError

FIXTURES = Path(__file__).parent / "fixtures"
TEAMTAILOR_URL = "https://career.teamtailor.com/jobs.rss"
PERSONIO_URL = "https://personio.jobs.personio.de/xml"


def text_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class FakeTransport:
    def __init__(self, responses: Mapping[str, HttpResponse | Exception]) -> None:
        self._responses = dict(responses)

    def get_json(self, url: str) -> HttpResponse:
        raise AssertionError("neither feed adapter fetches JSON")

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        raise AssertionError("neither feed adapter POSTs")

    def get_text(self, url: str) -> HttpResponse:
        response = self._responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def text(body: str) -> HttpResponse:
    return HttpResponse(status=200, body=body)


# -- Teamtailor ---------------------------------------------------------


def test_teamtailor_fixture_is_a_complete_board() -> None:
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: text(text_fixture("teamtailor_jobs.rss"))}),
    )
    assert result.status == "complete"
    assert result.expected_total is None  # no total declared in the feed
    assert result.jobs_seen == 3
    first = result.jobs[0]
    assert first.external_id == "3ce2c88b-cbc6-4ae9-8ecb-000466c69037"  # <guid>, not a URL
    assert first.title == "Group Financial Controller"
    assert first.location == "Stockholm"
    assert first.url == "https://career.teamtailor.com/jobs/8124573-group-financial-controller"


def test_teamtailor_falls_back_to_tt_city_when_tt_name_is_blank() -> None:
    """Observed live: one real item had an empty `<tt:name/>` but a populated
    `<tt:city>` -- a blank name is not the same as no location."""
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: text(text_fixture("teamtailor_jobs.rss"))}),
    )
    toronto = next(j for j in result.jobs if "Toronto" in j.title)
    assert toronto.location == "Toronto"


def test_teamtailor_malformed_xml_is_failed() -> None:
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: text("<rss><channel><item>unclosed</channel></rss>")}),
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_teamtailor_a_feed_with_no_channel_is_failed() -> None:
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: text("<rss></rss>")}),
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_teamtailor_a_channel_with_no_items_is_genuinely_empty() -> None:
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: text("<rss><channel><title>x</title></channel></rss>")}),
    )
    assert result.status == "complete"
    assert result.jobs == ()


def test_teamtailor_entity_expansion_payload_is_rejected_not_expanded() -> None:
    """`defusedxml` refuses a billion-laughs payload before it is ever
    expanded; the stdlib parser would not."""
    bomb = """<?xml version="1.0"?>
<!DOCTYPE channel [
 <!ENTITY a "a">
 <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
]>
<rss><channel><item><title>&b;</title></channel></rss>"""
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"}, FakeTransport({TEAMTAILOR_URL: text(bomb)})
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_teamtailor_a_200_with_no_string_body_is_malformed() -> None:
    """`get_text` always hands back a string on a real transport; this is a
    belt-and-braces check against a transport built with the wrong shape.
    Never treated as an empty board."""
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: HttpResponse(status=200, body=None)}),
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_teamtailor_no_response_at_all_is_unreachable() -> None:
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: TransportError("timeout")}),
    )
    assert result.status == "unreachable"
    assert result.error_code == "timeout"


def test_teamtailor_a_404_is_failed() -> None:
    result = TeamtailorAdapter().fetch(
        {"site": "career.teamtailor.com"},
        FakeTransport({TEAMTAILOR_URL: HttpResponse(status=404, body=None)}),
    )
    assert result.status == "failed"
    assert result.error_code == "not_found"


# -- Personio -------------------------------------------------------------


def test_personio_fixture_is_a_complete_board() -> None:
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"},
        FakeTransport({PERSONIO_URL: text(text_fixture("personio_position.xml"))}),
    )
    assert result.status == "complete"
    assert result.expected_total is None
    assert result.jobs_seen == 1
    first = result.jobs[0]
    assert first.external_id == "1834171"
    assert first.title == "Staff Software Engineer, Data Platform"
    assert first.location == "Munich"
    assert first.url is None  # no verified public-URL pattern -- see the docstring


def test_personio_a_root_element_that_is_not_workzag_jobs_is_failed() -> None:
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"},
        FakeTransport(
            {PERSONIO_URL: text("<positions><position><id>1</id></position></positions>")}
        ),
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_personio_malformed_xml_is_failed() -> None:
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"},
        FakeTransport({PERSONIO_URL: text("<workzag-jobs><position>unclosed</workzag-jobs>")}),
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_personio_zero_positions_is_genuinely_empty() -> None:
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"},
        FakeTransport({PERSONIO_URL: text("<workzag-jobs></workzag-jobs>")}),
    )
    assert result.status == "complete"
    assert result.jobs == ()


def test_personio_entity_expansion_payload_is_rejected() -> None:
    bomb = """<?xml version="1.0"?>
<!DOCTYPE workzag-jobs [
 <!ENTITY a "a">
 <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
]>
<workzag-jobs><position><name>&b;</name></position></workzag-jobs>"""
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"}, FakeTransport({PERSONIO_URL: text(bomb)})
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_personio_a_200_with_no_string_body_is_malformed() -> None:
    """Belt-and-braces, same reasoning as the Teamtailor case above."""
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"},
        FakeTransport({PERSONIO_URL: HttpResponse(status=200, body=None)}),
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_personio_rejects_an_invalid_tld() -> None:
    with pytest.raises(InvalidBoardKeyError):
        PersonioAdapter().validate_key({"company": "personio", "tld": "net"})


def test_personio_no_response_at_all_is_unreachable() -> None:
    result = PersonioAdapter().fetch(
        {"company": "personio", "tld": "de"},
        FakeTransport({PERSONIO_URL: TransportError("connection_error")}),
    )
    assert result.status == "unreachable"
    assert result.error_code == "connection_error"
