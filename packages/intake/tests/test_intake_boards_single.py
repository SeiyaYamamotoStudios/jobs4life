"""The four plain single-response JSON adapters added for slice C:
Rippling, Breezy, Recruitee, Pinpoint. Every fixture here is a live response
captured 2026-09-10 and trimmed to a handful of jobs; no test opens a socket.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jfl_intake.adapters import BreezyAdapter, PinpointAdapter, RecruiteeAdapter, RipplingAdapter
from jfl_intake.adapters.base import BoardAdapter
from jfl_intake.http import HttpResponse, TransportError

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


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
        raise AssertionError("none of these adapters POST")

    def get_text(self, url: str) -> HttpResponse:
        raise AssertionError("none of these adapters fetch text")


def ok(body: Any) -> HttpResponse:
    return HttpResponse(status=200, body=body)


RIPPLING_URL = "https://api.rippling.com/platform/api/ats/v1/board/rippling/jobs"
BREEZY_URL = "https://breezy.breezy.hr/json"
RECRUITEE_URL = "https://make.recruitee.com/api/offers/"
PINPOINT_URL = "https://sunking.pinpointhq.com/postings.json"

CASES: list[tuple[BoardAdapter, dict[str, str], str]] = [
    (RipplingAdapter(), {"slug": "rippling"}, RIPPLING_URL),
    (BreezyAdapter(), {"company": "breezy"}, BREEZY_URL),
    (RecruiteeAdapter(), {"company": "make"}, RECRUITEE_URL),
    (PinpointAdapter(), {"company": "sunking"}, PINPOINT_URL),
]
CASE_IDS = ["rippling", "breezy", "recruitee", "pinpoint"]


# -- Rippling ---------------------------------------------------------------


def test_rippling_fixture_is_a_complete_board() -> None:
    result = RipplingAdapter().fetch(
        {"slug": "rippling"}, FakeTransport({RIPPLING_URL: ok(fixture("rippling_jobs.json"))})
    )
    assert result.status == "complete"
    assert result.expected_total is None  # Rippling declares no total
    assert result.jobs_seen == 4
    first = result.jobs[0]
    assert first.external_id == "75ad50c6-778f-42ee-9c63-70d1cd687202"
    assert first.title == "Account Executive, Broker Channel (Austin & San Antonio)"
    assert first.location == "Austin, TX"
    assert (
        first.url == "https://ats.rippling.com/rippling/jobs/75ad50c6-778f-42ee-9c63-70d1cd687202"
    )
    assert first.requisition_id is None


def _dupe_entry(location: str) -> dict[str, Any]:
    return {
        "uuid": "dupe-1",
        "name": "Account Executive (Pittsburgh or Cleveland)",
        "url": "https://ats.rippling.com/rippling/jobs/dupe-1",
        "workLocation": {"label": location},
    }


def test_rippling_a_repeated_uuid_for_two_locations_merges_them_deterministically() -> None:
    """Observed live on Rippling's own board, not in `docs/ats-platforms.md`:
    one combined listing ("Pittsburgh or Cleveland") is repeated under the
    same uuid with a different `workLocation` each time. Locations merge
    distinct, sorted, joined -- never "whichever came first", since location
    feeds the repost fingerprint and a pick-the-first rule would make that
    fingerprint depend on API ordering.
    """
    body = [_dupe_entry("Cleveland, OH"), _dupe_entry("Pittsburgh, PA")]
    result = RipplingAdapter().fetch({"slug": "rippling"}, FakeTransport({RIPPLING_URL: ok(body)}))
    assert result.status == "complete"
    assert result.jobs_seen == 1
    assert result.jobs[0].location == "Cleveland, OH; Pittsburgh, PA"


def test_rippling_a_repeated_uuid_in_reversed_order_produces_an_identical_record() -> None:
    forward = [_dupe_entry("Cleveland, OH"), _dupe_entry("Pittsburgh, PA")]
    reversed_ = [_dupe_entry("Pittsburgh, PA"), _dupe_entry("Cleveland, OH")]

    forward_result = RipplingAdapter().fetch(
        {"slug": "rippling"}, FakeTransport({RIPPLING_URL: ok(forward)})
    )
    reversed_result = RipplingAdapter().fetch(
        {"slug": "rippling"}, FakeTransport({RIPPLING_URL: ok(reversed_)})
    )
    assert forward_result.jobs == reversed_result.jobs


def test_rippling_a_single_occurrence_uuid_is_unaffected_by_merging() -> None:
    body = [_dupe_entry("Cleveland, OH")]
    result = RipplingAdapter().fetch({"slug": "rippling"}, FakeTransport({RIPPLING_URL: ok(body)}))
    assert result.jobs[0].location == "Cleveland, OH"


# -- Breezy -------------------------------------------------------------


def test_breezy_fixture_is_a_complete_board() -> None:
    result = BreezyAdapter().fetch(
        {"company": "breezy"}, FakeTransport({BREEZY_URL: ok(fixture("breezy_jobs.json"))})
    )
    assert result.status == "complete"
    assert result.expected_total is None
    assert result.jobs_seen == 3
    first = result.jobs[0]
    assert first.external_id == "98323abf2296"
    assert first.title == "Employee #12"
    assert first.location == "Chaos, FL"
    assert first.url == "https://breezy.breezy.hr/p/98323abf2296-employee-12"
    assert first.requisition_id is None


# -- Recruitee ------------------------------------------------------------


def test_recruitee_fixture_is_a_complete_board() -> None:
    result = RecruiteeAdapter().fetch(
        {"company": "make"}, FakeTransport({RECRUITEE_URL: ok(fixture("recruitee_offers.json"))})
    )
    assert result.status == "complete"
    assert result.expected_total is None
    assert result.jobs_seen == 3
    first = result.jobs[0]
    assert first.external_id == "2695896"
    assert first.title == "Delivery Lead"
    assert first.location == "Remote job"  # already a plain string, not an object
    assert first.url == "https://make.recruitee.com/o/delivery-lead-3"
    assert first.requisition_id is None


def test_recruitee_object_body_is_malformed() -> None:
    result = RecruiteeAdapter().fetch(
        {"company": "make"}, FakeTransport({RECRUITEE_URL: ok({"jobs": []})})
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


# -- Pinpoint ---------------------------------------------------------------


def test_pinpoint_fixture_is_a_complete_board() -> None:
    result = PinpointAdapter().fetch(
        {"company": "sunking"},
        FakeTransport({PINPOINT_URL: ok(fixture("pinpoint_sunking_postings.json"))}),
    )
    assert result.status == "complete"
    assert result.expected_total is None
    assert result.jobs_seen == 4
    first = result.jobs[0]
    assert first.external_id == "332887"  # the posting id, numeric
    assert first.title == "Sun King Store Executive, Soweto"
    assert first.location == "South Africa"  # location.name
    assert (
        first.url
        == "https://sunking.pinpointhq.com/en/postings/d73a7468-0dc5-4adc-8717-57bcbcfdfb5b"
    )
    assert first.requisition_id == "347536"  # job.id -- the parent, never job.requisition_id


def test_pinpoint_job_requisition_id_field_is_never_used() -> None:
    """`job.requisition_id` is the employer's own reference and was empty on
    every posting checked live. Even when populated, `job.id` is what gets
    stored -- confirmed by feeding a fixture where they disagree.
    """
    body = fixture("pinpoint_sunking_postings.json")
    body["data"][0]["job"]["requisition_id"] = "EMPLOYER-REF-999"
    result = PinpointAdapter().fetch(
        {"company": "sunking"}, FakeTransport({PINPOINT_URL: ok(body)})
    )
    assert result.jobs[0].requisition_id == "347536"


def test_pinpoint_postings_and_jobs_are_different_resources() -> None:
    """`/jobs.json` is never fetched -- only `/postings.json` is in the URL
    this adapter calls."""
    transport = FakeTransport({PINPOINT_URL: ok(fixture("pinpoint_sunking_postings.json"))})
    PinpointAdapter().fetch({"company": "sunking"}, transport)
    assert transport.urls == [PINPOINT_URL]


# -- shared failure classification ------------------------------------------


@pytest.mark.parametrize(("adapter", "key", "url"), CASES, ids=CASE_IDS)
@pytest.mark.parametrize(
    ("status", "expected_status", "code"),
    [
        (404, "failed", "not_found"),
        (403, "failed", "http_client_error"),
        (429, "unreachable", "rate_limited"),
        (503, "unreachable", "server_error"),
    ],
)
def test_http_failures_are_classified_not_raised(
    adapter: BoardAdapter,
    key: dict[str, str],
    url: str,
    status: int,
    expected_status: str,
    code: str,
) -> None:
    result = adapter.fetch(key, FakeTransport({url: HttpResponse(status=status, body=None)}))
    assert result.status == expected_status
    assert result.error_code == code
    assert result.jobs == ()


@pytest.mark.parametrize(("adapter", "key", "url"), CASES, ids=CASE_IDS)
def test_no_response_at_all_is_unreachable(
    adapter: BoardAdapter, key: dict[str, str], url: str
) -> None:
    result = adapter.fetch(key, FakeTransport({url: TransportError("timeout")}))
    assert result.status == "unreachable"
    assert result.error_code == "timeout"


@pytest.mark.parametrize(("adapter", "key", "url"), CASES, ids=CASE_IDS)
def test_a_200_that_is_not_json_is_malformed_never_empty(
    adapter: BoardAdapter, key: dict[str, str], url: str
) -> None:
    """A 200 whose body httpx could not parse as JSON becomes `body=None`;
    that is a shape failure, never a genuinely empty board."""
    result = adapter.fetch(key, FakeTransport({url: ok(None)}))
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


@pytest.mark.parametrize(
    ("adapter", "key", "url", "empty_body"),
    [
        (RipplingAdapter(), {"slug": "rippling"}, RIPPLING_URL, []),
        (BreezyAdapter(), {"company": "breezy"}, BREEZY_URL, []),
        (RecruiteeAdapter(), {"company": "make"}, RECRUITEE_URL, {"offers": []}),
        (PinpointAdapter(), {"company": "sunking"}, PINPOINT_URL, {"data": []}),
    ],
    ids=CASE_IDS,
)
def test_a_valid_empty_list_is_a_genuinely_empty_board(
    adapter: BoardAdapter, key: dict[str, str], url: str, empty_body: Any
) -> None:
    result = adapter.fetch(key, FakeTransport({url: ok(empty_body)}))
    assert result.status == "complete"
    assert result.jobs == ()


@pytest.mark.parametrize(
    ("adapter", "key", "url", "fixture_name", "jobs_path", "strip"),
    [
        (RipplingAdapter(), {"slug": "rippling"}, RIPPLING_URL, "rippling_jobs.json", None, "uuid"),
        (BreezyAdapter(), {"company": "breezy"}, BREEZY_URL, "breezy_jobs.json", None, "id"),
        (
            RecruiteeAdapter(),
            {"company": "make"},
            RECRUITEE_URL,
            "recruitee_offers.json",
            "offers",
            "id",
        ),
        (
            PinpointAdapter(),
            {"company": "sunking"},
            PINPOINT_URL,
            "pinpoint_sunking_postings.json",
            "data",
            "id",
        ),
    ],
    ids=CASE_IDS,
)
def test_a_job_without_an_id_makes_the_board_incomplete(
    adapter: BoardAdapter,
    key: dict[str, str],
    url: str,
    fixture_name: str,
    jobs_path: str | None,
    strip: str,
) -> None:
    body = copy.deepcopy(fixture(fixture_name))
    jobs = body if jobs_path is None else body[jobs_path]
    listed = len(jobs)
    del jobs[0][strip]
    result = adapter.fetch(key, FakeTransport({url: ok(body)}))
    assert result.status == "incomplete"
    assert result.error_code == "unidentifiable_job"
    assert result.jobs_seen == listed - 1
