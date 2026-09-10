"""Workable: token paging, verified live against Devsinc and Hugging Face on
2026-09-10 (see `docs/ats-platforms.md`). No test opens a socket.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jfl_intake.adapters.workable import MAX_PAGES, WorkableAdapter
from jfl_intake.http import HttpResponse, TransportError

FIXTURES = Path(__file__).parent / "fixtures"
HUGGINGFACE_URL = "https://apply.workable.com/api/v3/accounts/huggingface/jobs"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FakeTransport:
    """Keyed by request body's `token` (or its absence), since one subdomain
    is one POST endpoint throughout a check -- the URL never changes between
    pages of the same board."""

    def __init__(self, by_token: Mapping[str | None, HttpResponse | Exception]) -> None:
        self._by_token = dict(by_token)
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def get_json(self, url: str) -> HttpResponse:
        raise AssertionError("Workable is POST only")

    def get_text(self, url: str) -> HttpResponse:
        raise AssertionError("Workable adapter never fetches text")

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        self.urls.append(url)
        assert url.startswith("https://apply.workable.com/api/v3/accounts/")
        self.bodies.append(dict(body))
        token = body.get("token")
        response = self._by_token[token]
        if isinstance(response, Exception):
            raise response
        return response


def ok(body: Any) -> HttpResponse:
    return HttpResponse(status=200, body=body)


def page(results: list[dict[str, Any]], total: int, next_token: str | None) -> Any:
    body: dict[str, Any] = {"results": results, "total": total}
    if next_token is not None:
        body["nextPage"] = next_token
    return body


def job(n: int) -> dict[str, Any]:
    return {
        "id": n,
        "title": f"Engineer {n}",
        "shortcode": f"CODE{n}",
        "location": {"city": "Lahore", "country": "Pakistan"},
    }


# -- field parsing, from the live-captured fixture ---------------------------


def test_huggingface_fixture_is_a_complete_single_page_board() -> None:
    transport = FakeTransport({None: ok(fixture("workable_huggingface_jobs.json"))})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "complete"
    assert result.expected_total == 3 == result.jobs_seen
    first = result.jobs[0]
    assert first.external_id == "6074222"
    assert first.title == "Senior Machine Learning Engineer, Voice Agents - EMEA Remote"
    assert first.location == "Paris, France"
    assert first.url == "https://apply.workable.com/huggingface/j/9E2A4C02C7"
    assert first.requisition_id is None
    assert transport.bodies == [
        {"query": "", "location": [], "department": [], "worktype": [], "remote": []}
    ]


# -- token paging -------------------------------------------------------


def test_token_paging_to_a_clean_termination() -> None:
    transport = FakeTransport(
        {
            None: ok(page([job(1), job(2)], 5, "tok-1")),
            "tok-1": ok(page([job(3), job(4)], 5, "tok-2")),
            "tok-2": ok(page([job(5)], 5, None)),  # last page: no nextPage
        }
    )
    result = WorkableAdapter().fetch({"subdomain": "devsinc-17"}, transport)
    assert result.status == "complete"
    assert result.expected_total == 5 == result.jobs_seen
    assert [j.external_id for j in result.jobs] == ["1", "2", "3", "4", "5"]
    assert [b.get("token") for b in transport.bodies] == [None, "tok-1", "tok-2"]


def test_completeness_needs_both_no_token_and_the_count_to_match() -> None:
    """A last page with no token but a short count is still incomplete."""
    transport = FakeTransport({None: ok(page([job(1)], 5, None))})
    result = WorkableAdapter().fetch({"subdomain": "devsinc-17"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "count_mismatch"
    assert result.expected_total == 5
    assert result.jobs_seen == 1


def test_a_repeated_id_is_never_treated_as_complete_even_if_the_count_matches() -> None:
    """A page handing back an id already collected is a failure to make
    progress. Two jobs, the second page repeating the first id -- the count
    would coincidentally match 2 if repeats were tolerated, so the test
    would wrongly pass `complete` unless the repeat itself is caught."""
    transport = FakeTransport(
        {
            None: ok(page([job(1)], 2, "tok-1")),
            "tok-1": ok(page([job(1), job(2)], 2, None)),  # job(1) again, plus a new one
        }
    )
    result = WorkableAdapter().fetch({"subdomain": "devsinc-17"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "duplicate_posting"


def test_total_is_read_from_the_first_page_and_trusted_throughout() -> None:
    transport = FakeTransport(
        {
            None: ok(page([job(1)], 2, "tok-1")),
            "tok-1": ok(page([job(2)], 999, None)),  # a later page lying about total
        }
    )
    result = WorkableAdapter().fetch({"subdomain": "devsinc-17"}, transport)
    assert result.expected_total == 2
    assert result.status == "complete"


def test_a_hard_page_cap_makes_endless_paging_incomplete() -> None:
    class _NeverEndingTokens:
        def __init__(self) -> None:
            self.requests = 0

        def get_json(self, url: str) -> HttpResponse:
            raise AssertionError("Workable is POST only")

        def get_text(self, url: str) -> HttpResponse:
            raise AssertionError("Workable adapter never fetches text")

        def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
            n = self.requests
            self.requests += 1
            return ok(page([job(n)], 100_000, f"tok-{n + 1}"))

    transport = _NeverEndingTokens()
    result = WorkableAdapter().fetch({"subdomain": "devsinc-17"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "page_cap_reached"
    assert transport.requests == MAX_PAGES


# -- the 403 trap -------------------------------------------------------


def test_a_403_is_failed_never_an_empty_board() -> None:
    """Workable refused `Python-urllib` with a bare 403 while accepting an
    honest client, confirmed live. `status_failure` maps it to `failed`."""
    transport = FakeTransport({None: HttpResponse(status=403, body=None)})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "failed"
    assert result.error_code == "http_client_error"
    assert result.jobs == ()


# -- malformed shapes and other http failures --------------------------------


def test_missing_results_key_is_malformed() -> None:
    transport = FakeTransport({None: ok({"total": 0})})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_missing_total_is_malformed_never_an_empty_board() -> None:
    transport = FakeTransport({None: ok({"results": []})})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_a_valid_empty_board_is_genuinely_empty() -> None:
    transport = FakeTransport({None: ok(page([], 0, None))})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "complete"
    assert result.jobs == ()
    assert result.expected_total == 0


def test_a_job_without_an_id_makes_the_board_incomplete() -> None:
    bad = job(1)
    del bad["id"]
    transport = FakeTransport({None: ok(page([bad, job(2)], 2, None))})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "incomplete"
    assert result.error_code == "unidentifiable_job"
    assert result.jobs_seen == 1


def test_a_job_with_no_shortcode_gets_no_url_but_is_still_identified() -> None:
    j = job(1)
    del j["shortcode"]
    transport = FakeTransport({None: ok(page([j], 1, None))})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "complete"
    assert result.jobs[0].url is None


def test_a_404_is_failed() -> None:
    transport = FakeTransport({None: HttpResponse(status=404, body=None)})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "failed"
    assert result.error_code == "not_found"


def test_a_503_is_unreachable() -> None:
    transport = FakeTransport({None: HttpResponse(status=503, body=None)})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "unreachable"
    assert result.error_code == "server_error"


def test_no_response_at_all_is_unreachable() -> None:
    transport = FakeTransport({None: TransportError("timeout")})
    result = WorkableAdapter().fetch({"subdomain": "huggingface"}, transport)
    assert result.status == "unreachable"
    assert result.error_code == "timeout"
