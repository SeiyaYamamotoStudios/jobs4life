"""SmartRecruiters: the echoed-limit page-size cap, verified live against
Bosch on 2026-09-10 (see `docs/ats-platforms.md`). No test opens a socket --
the live fixture is trimmed and real paging traps are reproduced with a fake
transport using small numbers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jfl_intake.adapters.smartrecruiters import MAX_PAGES, SmartRecruitersAdapter
from jfl_intake.http import HttpResponse, TransportError

FIXTURES = Path(__file__).parent / "fixtures"
ENDPOINT = "https://api.smartrecruiters.com/v1/companies/BoschGroup/postings"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def page_url(offset: int) -> str:
    # The adapter always *requests* limit=100, regardless of what a server
    # echoes back -- see the module docstring.
    return f"{ENDPOINT}?offset={offset}&limit=100"


class FakeTransport:
    def __init__(self, responses: Mapping[str, HttpResponse | Exception]) -> None:
        self._responses = dict(responses)
        self.urls: list[str] = []

    def get_json(self, url: str) -> HttpResponse:
        self.urls.append(url)
        response = self._responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        raise AssertionError("SmartRecruiters is GET only")

    def get_text(self, url: str) -> HttpResponse:
        raise AssertionError("SmartRecruiters adapter never fetches text")


def ok(body: Any) -> HttpResponse:
    return HttpResponse(status=200, body=body)


def posting(n: int) -> dict[str, Any]:
    return {
        "id": f"P{n}",
        "name": f"Job {n}",
        "refNumber": f"REF{n}",
        "location": {"city": "Stuttgart", "region": "BW", "country": "de"},
    }


def page_body(offset: int, echoed_limit: int, total: int, content: list[Any]) -> Any:
    return {"offset": offset, "limit": echoed_limit, "totalFound": total, "content": content}


# -- field parsing, from the live-captured fixture ---------------------------


def test_bosch_fixture_is_a_complete_board() -> None:
    """The trimmed fixture: 4 postings, a short first page, `totalFound`
    adjusted to 4 to match -- one request is the whole board."""
    result = SmartRecruitersAdapter().fetch(
        {"company_id": "BoschGroup"},
        FakeTransport({page_url(0): ok(fixture("smartrecruiters_bosch_postings.json"))}),
    )
    assert result.status == "complete"
    assert result.expected_total == 4 == result.jobs_seen
    first = result.jobs[0]
    assert first.external_id == "744000148872789"
    assert first.title == "ANALISTA DE LOGÍSTICA"
    assert first.location == "Itatiba, SP, br"
    assert first.url == "https://jobs.smartrecruiters.com/BoschGroup/744000148872789"
    assert first.requisition_id == "REF295991U"


# -- the echoed-limit trap ---------------------------------------------------


def test_a_page_size_over_the_echoed_limit_is_silently_capped_and_the_adapter_advances_by_it() -> (
    None
):
    """Requesting `limit=100` and receiving an echoed `limit=2` must advance
    the offset by 2, not by 100 -- the trap verified live against Bosch."""
    transport = FakeTransport(
        {
            page_url(0): ok(page_body(0, 2, 5, [posting(1), posting(2)])),
            page_url(2): ok(page_body(2, 2, 5, [posting(3), posting(4)])),
            page_url(4): ok(page_body(4, 2, 5, [posting(5)])),  # short: 1 < 2
        }
    )
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert transport.urls == [page_url(0), page_url(2), page_url(4)]
    assert result.status == "complete"
    assert result.expected_total == 5 == result.jobs_seen
    assert [j.external_id for j in result.jobs] == ["P1", "P2", "P3", "P4", "P5"]


def test_a_union_short_of_total_found_is_incomplete() -> None:
    transport = FakeTransport(
        {
            page_url(0): ok(page_body(0, 2, 10, [posting(1), posting(2)])),
            page_url(2): ok(page_body(2, 2, 10, [])),  # short (empty) -- terminates
        }
    )
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "count_mismatch"
    assert result.expected_total == 10
    assert result.jobs_seen == 2


def test_terminates_cleanly_with_no_wrap() -> None:
    """A short final page stops the check; nothing past it is ever requested."""
    transport = FakeTransport(
        {
            page_url(0): ok(page_body(0, 2, 3, [posting(1), posting(2)])),
            page_url(2): ok(page_body(2, 2, 3, [posting(3)])),  # short: 1 < 2
        }
    )
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "complete"
    assert transport.urls == [page_url(0), page_url(2)]  # never asked for offset=4


def test_totals_are_trusted_only_from_the_first_page() -> None:
    """Even though SmartRecruiters was observed consistent (unlike Workday),
    the adapter still only reads `totalFound` once, from the first page."""
    transport = FakeTransport(
        {
            page_url(0): ok(page_body(0, 2, 3, [posting(1), posting(2)])),
            # A later page lying about the total must not override the first.
            page_url(2): ok(page_body(2, 2, 999, [posting(3)])),
        }
    )
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.expected_total == 3
    assert result.status == "complete"


# -- the hard page cap -------------------------------------------------------


class _NeverEndingPages:
    """Every page is full (never short), so only the hard cap stops it."""

    def __init__(self) -> None:
        self.requests = 0

    def get_json(self, url: str) -> HttpResponse:
        n = self.requests
        self.requests += 1
        return ok(page_body(n, 1, 10_000, [posting(n)]))

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        raise AssertionError("SmartRecruiters is GET only")

    def get_text(self, url: str) -> HttpResponse:
        raise AssertionError("SmartRecruiters adapter never fetches text")


def test_a_hard_page_cap_makes_an_endless_listing_incomplete_not_complete() -> None:
    transport = _NeverEndingPages()
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "page_cap_reached"
    assert transport.requests == MAX_PAGES


# -- malformed shapes and http failures --------------------------------------


def test_missing_content_key_is_malformed() -> None:
    transport = FakeTransport({page_url(0): ok({"offset": 0, "limit": 100, "totalFound": 0})})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_missing_limit_is_malformed_never_an_empty_board() -> None:
    transport = FakeTransport({page_url(0): ok({"offset": 0, "totalFound": 0, "content": []})})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_a_valid_empty_board_is_genuinely_empty() -> None:
    transport = FakeTransport({page_url(0): ok(page_body(0, 100, 0, []))})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "complete"
    assert result.jobs == ()
    assert result.expected_total == 0


def test_a_non_dict_posting_is_malformed() -> None:
    transport = FakeTransport({page_url(0): ok(page_body(0, 100, 1, ["not-a-job"]))})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_a_posting_without_an_id_makes_the_board_incomplete() -> None:
    bad = posting(1)
    del bad["id"]
    transport = FakeTransport({page_url(0): ok(page_body(0, 100, 2, [bad, posting(2)]))})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "unidentifiable_job"
    assert result.jobs_seen == 1


def test_a_404_is_failed() -> None:
    transport = FakeTransport({page_url(0): HttpResponse(status=404, body=None)})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "failed"
    assert result.error_code == "not_found"


def test_a_503_is_unreachable() -> None:
    transport = FakeTransport({page_url(0): HttpResponse(status=503, body=None)})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "unreachable"
    assert result.error_code == "server_error"


def test_no_response_at_all_is_unreachable() -> None:
    transport = FakeTransport({page_url(0): TransportError("connection_error")})
    result = SmartRecruitersAdapter().fetch({"company_id": "BoschGroup"}, transport)
    assert result.status == "unreachable"
    assert result.error_code == "connection_error"
