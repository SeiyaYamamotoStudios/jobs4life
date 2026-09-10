"""The one-request adapters (Greenhouse, Ashby, Lever), and the transports.

Every response here is either a recorded fixture -- captured live on 2026-09-10
and trimmed to a handful of jobs -- or a status code. No test in this file opens
a socket: adapters get a fake transport, and `HttpxTransport` is driven through
`httpx.MockTransport`.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, get_args

import httpx
import pytest
from jfl_core.db.tables import _BOARD_PLATFORMS
from jfl_core.models import BoardPlatform
from jfl_intake.adapters import (
    AshbyAdapter,
    GreenhouseAdapter,
    InvalidBoardKeyError,
    LeverAdapter,
    WorkdayAdapter,
    default_registry,
)
from jfl_intake.adapters.base import BoardAdapter
from jfl_intake.http import (
    HttpResponse,
    HttpxTransport,
    PoliteTransport,
    RequestBudget,
    RequestBudgetExceeded,
    TransportError,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FakeTransport:
    """Replays one response (or raises one error) per URL, and records calls."""

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
        raise AssertionError("a one-request adapter never POSTs")

    def get_text(self, url: str) -> HttpResponse:
        raise AssertionError("none of these adapters fetch text")


GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs"
ASHBY_URL = "https://api.ashbyhq.com/posting-api/job-board/openai"
LEVER_URL = "https://api.lever.co/v0/postings/palantir?mode=json"

CASES: list[tuple[BoardAdapter, dict[str, str], str]] = [
    (GreenhouseAdapter(), {"token": "anthropic"}, GREENHOUSE_URL),
    (AshbyAdapter(), {"name": "openai"}, ASHBY_URL),
    (LeverAdapter(), {"company": "palantir"}, LEVER_URL),
]
CASE_IDS = ["greenhouse", "ashby", "lever"]


def ok(body: Any) -> HttpResponse:
    return HttpResponse(status=200, body=body)


# -- recorded fixtures ------------------------------------------------------


def test_greenhouse_fixture_is_a_complete_board() -> None:
    transport = FakeTransport({GREENHOUSE_URL: ok(fixture("greenhouse_anthropic_jobs.json"))})
    result = GreenhouseAdapter().fetch({"token": "anthropic"}, transport)

    assert transport.urls == [GREENHOUSE_URL]
    assert result.status == "complete"
    assert result.error_code is None
    assert result.expected_total == 4 == result.jobs_seen
    first = result.jobs[0]
    assert first.external_id == "4461450008"  # the public id, never internal_job_id
    assert first.title == "Account Executive, AI Native"
    assert first.location == "New York City, NY; San Francisco, CA | New York City, NY"
    assert first.url == "https://job-boards.greenhouse.io/anthropic/jobs/4461450008"
    assert first.fingerprint.startswith("account executive ai native|")
    assert first.requisition_id == "3356"  # stored beside the id, never as it


@pytest.mark.parametrize(("raw", "stored"), [(3356, "3356"), (None, None), ("  ", None)])
def test_greenhouse_requisition_id_is_optional(raw: object, stored: str | None) -> None:
    body = fixture("greenhouse_anthropic_jobs.json")
    body["jobs"][0]["requisition_id"] = raw
    result = GreenhouseAdapter().fetch(
        {"token": "anthropic"}, FakeTransport({GREENHOUSE_URL: ok(body)})
    )
    assert result.status == "complete"  # a missing requisition never makes a job unidentifiable
    assert result.jobs[0].requisition_id == stored


def test_platforms_without_a_requisition_leave_it_empty() -> None:
    ashby = AshbyAdapter().fetch(
        {"name": "openai"}, FakeTransport({ASHBY_URL: ok(fixture("ashby_openai_job_board.json"))})
    )
    lever = LeverAdapter().fetch(
        {"company": "palantir"},
        FakeTransport({LEVER_URL: ok(fixture("lever_palantir_postings.json"))}),
    )
    assert ashby.jobs and all(j.requisition_id is None for j in ashby.jobs)
    assert lever.jobs and all(j.requisition_id is None for j in lever.jobs)


def test_greenhouse_count_disagreeing_with_meta_total_is_incomplete() -> None:
    body = fixture("greenhouse_anthropic_jobs.json")
    body["meta"]["total"] = 5
    result = GreenhouseAdapter().fetch(
        {"token": "anthropic"}, FakeTransport({GREENHOUSE_URL: ok(body)})
    )
    assert result.status == "incomplete"
    assert result.error_code == "count_mismatch"
    assert result.expected_total == 5
    assert result.jobs_seen == 4


def test_ashby_fixture_is_a_complete_board() -> None:
    result = AshbyAdapter().fetch(
        {"name": "openai"}, FakeTransport({ASHBY_URL: ok(fixture("ashby_openai_job_board.json"))})
    )
    assert result.status == "complete"
    assert result.expected_total is None  # Ashby declares no total
    assert result.jobs_seen == 3
    first = result.jobs[0]
    assert first.external_id == "8fb1615c-34bf-47c4-a1d1-b7b2f836bbd3"
    assert first.title == "Technical Program Manager, Compute Infrastructure"
    assert first.location == "San Francisco"
    assert first.url == "https://jobs.ashbyhq.com/openai/8fb1615c-34bf-47c4-a1d1-b7b2f836bbd3"


def test_ashby_unlisted_jobs_are_not_part_of_the_board() -> None:
    body = fixture("ashby_openai_job_board.json")
    body["jobs"][1]["isListed"] = False
    result = AshbyAdapter().fetch({"name": "openai"}, FakeTransport({ASHBY_URL: ok(body)}))
    assert result.status == "complete"
    assert result.jobs_seen == 2
    assert body["jobs"][1]["id"] not in {j.external_id for j in result.jobs}


def test_lever_fixture_is_a_complete_board() -> None:
    result = LeverAdapter().fetch(
        {"company": "palantir"},
        FakeTransport({LEVER_URL: ok(fixture("lever_palantir_postings.json"))}),
    )
    assert result.status == "complete"
    assert result.jobs_seen == 3
    first = result.jobs[0]
    assert first.external_id == "ac978161-6f46-4f6b-ad9e-a258e642751c"
    assert first.title == "Administrative Business Partner"  # from `text`
    assert first.location == "London, United Kingdom"  # from `categories.location`
    assert first.url == "https://jobs.lever.co/palantir/ac978161-6f46-4f6b-ad9e-a258e642751c"


def test_lever_object_body_is_malformed_never_an_empty_board() -> None:
    result = LeverAdapter().fetch(
        {"company": "palantir"}, FakeTransport({LEVER_URL: ok({"ok": False})})
    )
    assert result.status == "failed"
    assert result.error_code == "malformed_response"
    assert result.jobs == ()


# -- failure classification, shared by all three -------------------------


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
@pytest.mark.parametrize("code", ["timeout", "connection_error"])
def test_no_response_at_all_is_unreachable(
    adapter: BoardAdapter, key: dict[str, str], url: str, code: str
) -> None:
    result = adapter.fetch(key, FakeTransport({url: TransportError(code)}))  # type: ignore[arg-type]
    assert result.status == "unreachable"
    assert result.error_code == code


@pytest.mark.parametrize(("adapter", "key", "url"), CASES, ids=CASE_IDS)
def test_a_200_that_is_not_json_is_malformed(
    adapter: BoardAdapter, key: dict[str, str], url: str
) -> None:
    result = adapter.fetch(key, FakeTransport({url: ok(None)}))
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


@pytest.mark.parametrize(
    ("adapter", "key", "url", "fixture_name", "strip"),
    [
        (
            GreenhouseAdapter(),
            {"token": "anthropic"},
            GREENHOUSE_URL,
            "greenhouse_anthropic_jobs.json",
            "id",
        ),
        (AshbyAdapter(), {"name": "openai"}, ASHBY_URL, "ashby_openai_job_board.json", "id"),
        (LeverAdapter(), {"company": "palantir"}, LEVER_URL, "lever_palantir_postings.json", "id"),
    ],
    ids=CASE_IDS,
)
def test_a_job_without_an_id_makes_the_board_incomplete(
    adapter: BoardAdapter, key: dict[str, str], url: str, fixture_name: str, strip: str
) -> None:
    """It cannot be tracked, so the result cannot claim to account for the board."""
    body = copy.deepcopy(fixture(fixture_name))
    jobs = body if isinstance(body, list) else body["jobs"]
    listed = len(jobs)
    del jobs[0][strip]
    result = adapter.fetch(key, FakeTransport({url: ok(body)}))
    assert result.status == "incomplete"
    assert result.error_code == "unidentifiable_job"
    assert result.jobs_seen == listed - 1  # the others are still reported as seen


@pytest.mark.parametrize(
    "key",
    [{}, {"token": ""}, {"token": "../boards"}, {"token": "a/b"}, {"token": "x?y"}],
)
def test_a_board_key_that_could_escape_its_path_is_refused(key: dict[str, str]) -> None:
    with pytest.raises(InvalidBoardKeyError):
        GreenhouseAdapter().validate_key(key)


def test_the_default_registry_has_exactly_the_twelve_verified_platforms() -> None:
    registry = default_registry()
    assert registry.platforms() == (
        "ashby",
        "breezy",
        "greenhouse",
        "lever",
        "personio",
        "pinpoint",
        "recruitee",
        "rippling",
        "smartrecruiters",
        "teamtailor",
        "workable",
        "workday",
    )
    assert isinstance(registry.get("workday"), WorkdayAdapter)


def test_board_platform_sources_agree() -> None:
    """Three copies of the platform list have already drifted apart once,
    silently: `BoardPlatform` (typing), `_BOARD_PLATFORMS` (the DB CHECK
    constraint's source), and `default_registry()` (what actually runs) each
    listed a different set between 2026-09-10 and 2026-09-11, and only the
    third would have failed loudly -- at INSERT, in production, long after
    the adapter code shipped and its tests passed. This test is what makes
    that impossible to repeat unnoticed.
    """
    literal_platforms = frozenset(get_args(BoardPlatform))
    table_platforms = frozenset(_BOARD_PLATFORMS)
    registry_platforms = frozenset(default_registry().platforms())

    assert literal_platforms == table_platforms == registry_platforms


# -- transports -----------------------------------------------------------


def test_polite_transport_waits_between_requests_but_not_before_the_first() -> None:
    sleeps: list[float] = []
    inner = FakeTransport({"u": ok([])})
    polite = PoliteTransport(
        inner, RequestBudget(delay_seconds=0.5, max_requests=10), sleep=sleeps.append
    )
    for _ in range(3):
        polite.get_json("u")
    assert sleeps == [0.5, 0.5]
    assert polite.requests == 3


def test_polite_transport_enforces_its_request_budget() -> None:
    polite = PoliteTransport(
        FakeTransport({"u": ok([])}), RequestBudget(max_requests=2), sleep=lambda _: None
    )
    polite.get_json("u")
    polite.get_json("u")
    with pytest.raises(RequestBudgetExceeded) as caught:
        polite.get_json("u")
    assert caught.value.code == "request_budget_exhausted"


def test_polite_transport_enforces_its_deadline() -> None:
    now = {"t": 0.0}
    polite = PoliteTransport(
        FakeTransport({"u": ok([])}),
        RequestBudget(deadline_seconds=10.0),
        sleep=lambda _: None,
        clock=lambda: now["t"],
    )
    polite.get_json("u")
    now["t"] = 11.0
    with pytest.raises(RequestBudgetExceeded) as caught:
        polite.get_json("u")
    assert caught.value.code == "deadline_exceeded"


def test_polite_transport_enforces_its_budget_on_get_text_too() -> None:
    class _TextOnly:
        def get_json(self, url: str) -> HttpResponse:
            raise AssertionError("not used in this test")

        def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
            raise AssertionError("not used in this test")

        def get_text(self, url: str) -> HttpResponse:
            return HttpResponse(status=200, body="<rss></rss>")

    polite = PoliteTransport(_TextOnly(), RequestBudget(max_requests=1), sleep=lambda _: None)
    assert polite.get_text("u").body == "<rss></rss>"
    with pytest.raises(RequestBudgetExceeded) as caught:
        polite.get_text("u")
    assert caught.value.code == "request_budget_exhausted"


def _httpx(handler: Any) -> HttpxTransport:
    return HttpxTransport(httpx.Client(transport=httpx.MockTransport(handler)))


def test_httpx_transport_returns_every_status_with_its_parsed_body() -> None:
    transport = _httpx(lambda request: httpx.Response(404, json={"error": "nope"}))
    response = transport.get_json("https://boards-api.greenhouse.io/v1/boards/x/jobs")
    assert response == HttpResponse(status=404, body={"error": "nope"})


def test_httpx_transport_posts_json_and_tolerates_a_non_json_body() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, text="<html>maintenance</html>")

    response = _httpx(handler).post_json("https://x.wd5.myworkdayjobs.com/j", {"limit": 20})
    assert seen == [{"limit": 20}]
    assert response == HttpResponse(status=200, body=None)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "connection_error"),
    ],
)
def test_httpx_transport_maps_no_response_to_a_code(error: Exception, code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(TransportError) as caught:
        _httpx(handler).get_json("https://api.lever.co/v0/postings/x?mode=json")
    assert caught.value.code == code
    assert str(caught.value) == code  # a code, never the exception's text


def test_httpx_transport_get_text_returns_an_rss_body_intact() -> None:
    """The gap Teamtailor and Personio hit: `get_json` discards a non-JSON
    body as `None`. `get_text` must hand back the real body, unparsed --
    proved here against `httpx.MockTransport`, not just a fake transport a
    test built by hand, since the missing coverage was exactly this: nothing
    tested the real transport against a non-JSON response.
    """
    rss = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss><channel><item><title>Engineer</title></item></channel></rss>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=rss, headers={"content-type": "application/rss+xml"})

    response = _httpx(handler).get_text("https://career.teamtailor.com/jobs.rss")
    assert response.status == 200
    assert response.body == rss  # not None, not re-encoded, not stripped


def test_httpx_transport_get_text_on_a_non_200_still_returns_the_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html>not found</html>")

    response = _httpx(handler).get_text("https://career.teamtailor.com/jobs.rss")
    assert response == HttpResponse(status=404, body="<html>not found</html>")


def test_httpx_transport_get_text_maps_no_response_to_a_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(TransportError) as caught:
        _httpx(handler).get_text("https://career.teamtailor.com/jobs.rss")
    assert caught.value.code == "timeout"
