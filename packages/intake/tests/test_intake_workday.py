"""Workday: every behaviour verified live on 2026-09-10, reproduced by a fake.

`FakeWorkday` is the specification of what the real thing does, written as code:
`limit` above 20 answers 200 with no `total` and no postings; a listing exposes
at most 2,000 postings and caps page 0's `total` there; offsets past the end
wrap round to page 0; `total` is 0 on every page but offset 0. The adapter is
then held to terminating, splitting and verifying against that.

No network anywhere. Facet shapes come from NVIDIA's and Adobe's real page 0,
recorded on 2026-09-10.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jfl_core.models import BoardCheckState, KnownBoardJob
from jfl_intake.adapters.workday import (
    LISTING_CEILING,
    MAX_PAGES_PER_LISTING,
    WorkdayAdapter,
    choose_partition,
    facet_groups,
)
from jfl_intake.engine import plan_check
from jfl_intake.http import (
    HttpResponse,
    PoliteTransport,
    RequestBudget,
    TransportError,
)

FIXTURES = Path(__file__).parent / "fixtures"
NVIDIA = {"tenant": "nvidia", "wd": "wd5", "site": "NVIDIAExternalCareerSite"}
ADOBE = {"tenant": "adobe", "wd": "wd5", "site": "external_experienced"}


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def posting(n: int, *, title: str | None = None, suffix: str = "") -> dict[str, Any]:
    """A posting in the real shape. Requisition ids are unique per `n`."""
    requisition = f"JR{2_000_000 + n}"
    name = title or f"Engineer {n}"
    return {
        "title": name,
        "externalPath": f"/job/US-CA-Santa-Clara/{'-'.join(name.split())}_{requisition}{suffix}",
        "locationsText": "US, CA, Santa Clara",
        "postedOn": "Posted Today",
        "bulletFields": [requisition],
    }


class FakeWorkday:
    """A Workday careers site, faithful to the verified quirks."""

    def __init__(
        self,
        postings: list[dict[str, Any]],
        *,
        facets: list[Any] | None = None,
        slices: Mapping[tuple[str, str], list[dict[str, Any]]] | None = None,
        reported_total: int | None = None,
    ) -> None:
        self.postings: list[dict[str, Any]] = postings
        self.facets: list[Any] = facets or []
        self.slices: dict[tuple[str, str], list[dict[str, Any]]] = dict(slices or {})
        self.reported_total = reported_total
        self.bodies: list[dict[str, Any]] = []
        # Request index -> a response (or error) to give instead.
        self.overrides: dict[int, HttpResponse | Exception] = {}

    def get_json(self, url: str) -> HttpResponse:
        raise AssertionError("Workday is POST only")

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        assert url.endswith("/jobs") and "/wday/cxs/" in url
        self.bodies.append(copy.deepcopy(dict(body)))
        override = self.overrides.get(len(self.bodies) - 1)
        if isinstance(override, Exception):
            raise override
        if override is not None:
            return override

        if body["limit"] > 20:
            # NVIDIA's answer to limit=50: a 200 that is silently empty.
            return HttpResponse(200, {"jobPostings": [], "facets": [], "userAuthenticated": False})

        applied = body["appliedFacets"]
        listing = self._listing(applied)
        visible = listing[:LISTING_CEILING]
        offset = body["offset"]
        period = max(20, -(-len(visible) // 20) * 20)
        start = offset % period  # past the end, it wraps to page 0
        page = visible[start : start + 20]
        if offset == 0:
            total = len(visible)
            if self.reported_total is not None and not applied:
                total = self.reported_total
        else:
            total = 0  # intermediate pages always say 0
        return HttpResponse(
            200,
            {
                "total": total,
                "jobPostings": page,
                "facets": self.facets if offset == 0 and not applied else [],
                "userAuthenticated": False,
            },
        )

    def _listing(self, applied: Mapping[str, list[str]]) -> list[dict[str, Any]]:
        if not applied:
            return self.postings
        ((parameter, values),) = applied.items()
        return self.slices[(parameter, values[0])]

    def offsets(self, applied: Mapping[str, list[str]] | None = None) -> list[int]:
        wanted = dict(applied or {})
        return [b["offset"] for b in self.bodies if b["appliedFacets"] == wanted]


def nvidia_board(*, adjust: Mapping[str, int] | None = None) -> FakeWorkday:
    """NVIDIA's shape: 2,630 postings, page 0 capped at 2,000, and the real
    `jobFamilyGroup` facet (15 values summing to 2,630) to split by. `adjust`
    changes how many postings a slice really serves, by value descriptor.
    """
    facets = fixture("workday_nvidia_page0.json")["facets"]
    family = next(g for g in facets if g["facetParameter"] == "jobFamilyGroup")
    slices: dict[tuple[str, str], list[dict[str, Any]]] = {}
    everything: list[dict[str, Any]] = []
    n = 0
    for value in family["values"]:
        served = value["count"] + (adjust or {}).get(value["descriptor"], 0)
        chunk = [posting(n + i) for i in range(served)]
        n += served
        slices[("jobFamilyGroup", value["id"])] = chunk
        everything.extend(chunk)
    return FakeWorkday(everything, facets=facets, slices=slices)


def fetch(board: Any, key: Mapping[str, str] = NVIDIA) -> Any:
    return WorkdayAdapter().fetch(key, board)


# -- the recorded facets ------------------------------------------------------


def test_the_nvidia_facets_partition_by_job_family_to_2630() -> None:
    chosen = choose_partition(fixture("workday_nvidia_page0.json")["facets"])
    assert chosen is not None
    group, total = chosen
    assert group.parameter == "jobFamilyGroup"
    assert total == 2630
    assert group.largest < LISTING_CEILING


def test_nested_facet_groups_are_lifted_and_multi_valued_ones_do_not_agree() -> None:
    groups = {g.parameter: g for g in facet_groups(fixture("workday_nvidia_page0.json")["facets"])}
    # `locationMainGroup` holds only nested groups, so it is not a group itself.
    assert "locationMainGroup" not in groups
    assert {"locationHierarchy2", "locationHierarchy1", "locations"} <= set(groups)
    assert groups["locations"].total != 2630  # a job has several locations
    assert groups["workerSubType"].total == groups["timeType"].total == 2630
    assert groups["timeType"].largest >= LISTING_CEILING  # agrees, but is not listable


def test_ambiguous_consensus_is_no_partition_at_all() -> None:
    def group(name: str, *counts: int) -> dict[str, Any]:
        return {
            "facetParameter": name,
            "values": [{"id": f"{name}{i}", "count": c} for i, c in enumerate(counts)],
        }

    facets = [
        group("a", 1100, 1000),
        group("b", 1050, 1050),
        group("c", 1200, 1000),
        group("d", 1100, 1100),
    ]
    assert choose_partition(facets) is None


# -- limits and shapes (behaviour 1) ------------------------------------------


def test_a_200_with_no_total_and_no_postings_is_a_failed_check_never_an_empty_board() -> None:
    board = FakeWorkday([posting(1)])
    board.overrides[0] = HttpResponse(
        200, {"jobPostings": [], "facets": [], "userAuthenticated": False}
    )
    result = fetch(board)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"
    assert result.jobs == ()


def test_the_fake_reproduces_the_silent_failure_and_the_adapter_never_provokes_it() -> None:
    board = FakeWorkday([posting(i) for i in range(45)])
    silent = board.post_json(
        "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/x/jobs",
        {"appliedFacets": {}, "limit": 50, "offset": 0, "searchText": ""},
    )
    assert "total" not in silent.body and silent.body["jobPostings"] == []

    board.bodies.clear()
    assert fetch(board).status == "complete"
    assert board.bodies and all(b["limit"] == 20 for b in board.bodies)


def test_http_400_is_a_failed_check() -> None:
    board = FakeWorkday([posting(1)])
    board.overrides[0] = HttpResponse(400, {"errorCode": "HTTP_400"})
    result = fetch(board, ADOBE)
    assert result.status == "failed"
    assert result.error_code == "http_client_error"


def test_total_zero_with_an_empty_list_on_offset_zero_is_an_empty_board() -> None:
    result = fetch(FakeWorkday([]))
    assert result.status == "complete"
    assert result.error_code is None
    assert result.expected_total == 0
    assert result.jobs == ()


@pytest.mark.parametrize(
    "body",
    [
        {"total": 12},  # no jobPostings
        {"total": 12, "jobPostings": []},  # a total with nothing behind it
        {"total": 0, "jobPostings": [posting(1)]},  # postings behind a zero
        {"total": "12", "jobPostings": [posting(1)]},
        ["not", "an", "object"],
    ],
)
def test_an_inconsistent_page_zero_is_malformed(body: Any) -> None:
    board = FakeWorkday([posting(1)])
    board.overrides[0] = HttpResponse(200, body)
    result = fetch(board)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"


def test_a_later_page_without_postings_fails_and_reports_what_it_saw() -> None:
    board = FakeWorkday([posting(i) for i in range(100)])
    board.overrides[2] = HttpResponse(200, {"total": 0})
    result = fetch(board)
    assert result.status == "failed"
    assert result.error_code == "malformed_response"
    assert result.jobs_seen == 40


def test_a_5xx_part_way_is_unreachable() -> None:
    board = FakeWorkday([posting(i) for i in range(100)])
    board.overrides[3] = HttpResponse(503, None)
    result = fetch(board)
    assert result.status == "unreachable"
    assert result.error_code == "server_error"


def test_a_timeout_part_way_is_unreachable() -> None:
    board = FakeWorkday([posting(i) for i in range(100)])
    board.overrides[1] = TransportError("timeout")
    result = fetch(board)
    assert result.status == "unreachable"
    assert result.error_code == "timeout"
    assert result.jobs_seen == 20


# -- termination (behaviours 2, 3, 4) -----------------------------------------


def test_adobe_short_page_ends_the_listing_and_offset_740_is_never_requested() -> None:
    """Adobe: 700 -> 20, 720 -> 10 (the real end), 740 -> page 0 again."""
    board = FakeWorkday([posting(i) for i in range(730)])
    # The fake really does wrap there -- the adapter just never asks.
    assert board.post_json("x/wday/cxs/x/jobs", {"appliedFacets": {}, "limit": 20, "offset": 740})
    assert board.bodies[-1]["offset"] == 740
    board.bodies.clear()

    result = fetch(board, ADOBE)
    assert result.status == "complete"
    assert result.expected_total == 730 == result.jobs_seen
    assert board.offsets() == list(range(0, 740, 20))
    assert 740 not in board.offsets()


def test_a_short_page_terminates_even_when_page_zero_claimed_more() -> None:
    board = FakeWorkday([posting(i) for i in range(730)], reported_total=750)
    result = fetch(board, ADOBE)
    assert board.offsets()[-1] == 720  # the short page stopped it, not the total
    assert result.status == "incomplete"
    assert result.error_code == "count_mismatch"
    assert result.expected_total == 750
    assert result.jobs_seen == 730


def test_pagination_terminates_on_wrap_below_the_ceiling() -> None:
    """740 postings, a whole number of pages, and a page 0 that overstates the
    total: nothing short ends the listing, so only the wrap at 740 can.
    """
    board = FakeWorkday([posting(i) for i in range(740)], reported_total=800)
    result = fetch(board, ADOBE)
    assert board.offsets()[-1] == 740  # the request that came back as page 0
    assert 760 not in board.offsets()
    assert result.status == "incomplete"
    assert result.jobs_seen == 740


def test_pagination_terminates_on_wrap_at_offset_2000_which_equals_page_zero() -> None:
    """NVIDIA's behaviour 2, inside a facet slice whose count undersold it: the
    Engineering slice really holds 2,300 postings, so its own page 0 says 2,000,
    and offset 2000 comes back as page 0. Paging stops there -- 101 requests,
    never offset 2020 -- and the union no longer matches the partition total.
    """
    board = nvidia_board(adjust={"Engineering": 2300 - 1724})
    engineering = next(
        value_id for (parameter, value_id), items in board.slices.items() if len(items) == 2300
    )
    applied = {"jobFamilyGroup": [engineering]}

    result = fetch(board)

    offsets = board.offsets(applied)
    assert offsets == list(range(0, 2020, 20))
    assert len(offsets) == MAX_PAGES_PER_LISTING
    assert (
        board.post_json(
            "x/wday/cxs/x/jobs", {**board.bodies[0], "appliedFacets": applied, "offset": 2000}
        ).body["jobPostings"]
        == (board.slices[("jobFamilyGroup", engineering)][:20])
    )
    assert result.status == "incomplete"
    assert result.error_code == "count_mismatch"
    assert result.expected_total == 2630


def test_the_page_cap_makes_a_never_ending_listing_incomplete() -> None:
    class Endless(FakeWorkday):
        def _listing(self, applied: Mapping[str, list[str]]) -> list[dict[str, Any]]:
            raise AssertionError("unused")

        def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
            applied = body["appliedFacets"]
            if applied == {"A": ["a1"]}:
                self.bodies.append(copy.deepcopy(dict(body)))
                offset = body["offset"]
                page = [posting(100_000 + offset + i) for i in range(20)]  # always new
                total = LISTING_CEILING if offset == 0 else 0
                return HttpResponse(200, {"total": total, "jobPostings": page})
            if applied == {"A": ["a2"]}:
                self.bodies.append(copy.deepcopy(dict(body)))
                return HttpResponse(200, {"total": 1, "jobPostings": [posting(1)]})
            self.bodies.append(copy.deepcopy(dict(body)))
            facets = [
                {
                    "facetParameter": "A",
                    "values": [{"id": "a1", "count": 1999}, {"id": "a2", "count": 1}],
                },
                {"facetParameter": "B", "values": [{"id": "b1", "count": 2000}]},
            ]
            page = [posting(i) for i in range(20)]
            return HttpResponse(
                200, {"total": LISTING_CEILING, "jobPostings": page, "facets": facets}
            )

    board = Endless([])
    result = fetch(board)
    assert result.status == "incomplete"
    assert result.error_code == "page_cap_reached"
    assert len(board.offsets({"A": ["a1"]})) == MAX_PAGES_PER_LISTING


def test_the_request_budget_makes_the_check_incomplete_with_what_it_saw() -> None:
    board = FakeWorkday([posting(i) for i in range(730)])
    polite = PoliteTransport(board, RequestBudget(max_requests=3), sleep=lambda _: None)
    result = fetch(polite, ADOBE)
    assert result.status == "incomplete"
    assert result.error_code == "request_budget_exhausted"
    assert result.jobs_seen == 60


# -- the ceiling ---------------------------------------------------------------


def test_a_ceiling_board_is_facet_split_and_verified_against_the_partition_sum() -> None:
    board = nvidia_board()
    assert len(board.postings) == 2630
    polite = PoliteTransport(board, RequestBudget(), sleep=lambda _: None)

    result = fetch(polite)

    assert result.status == "complete"
    assert result.error_code is None
    assert result.expected_total == 2630  # the facet sum, not page 0's 2000
    assert result.jobs_seen == 2630
    assert result.detail["facet"] == "jobFamilyGroup"
    # Every slice request is filtered by the chosen facet and nothing else, and
    # the whole check fits the default politeness budget.
    sliced = [b for b in board.bodies if b["appliedFacets"]]
    assert sliced and all(set(b["appliedFacets"]) == {"jobFamilyGroup"} for b in sliced)
    assert all(b["limit"] == 20 for b in board.bodies)
    assert polite.requests <= RequestBudget().max_requests


def test_a_facet_split_that_comes_up_one_short_is_incomplete() -> None:
    board = nvidia_board(adjust={"Engineering": -1})
    result = fetch(board)
    assert result.status == "incomplete"
    assert result.error_code == "count_mismatch"
    assert result.expected_total == 2630
    assert result.jobs_seen == 2629


def test_a_ceiling_board_with_no_agreeing_facet_is_truncated_without_paging() -> None:
    facets = fixture("workday_nvidia_page0.json")["facets"]
    only_family = [g for g in facets if g["facetParameter"] == "jobFamilyGroup"]
    board = nvidia_board()
    board.facets = only_family  # one group cannot corroborate its own sum

    result = fetch(board)
    assert result.status == "truncated"
    assert result.error_code == "listing_ceiling"
    assert result.expected_total is None
    assert len(board.bodies) == 1


def test_a_ceiling_board_whose_agreeing_facets_cannot_be_listed_is_truncated() -> None:
    facets = fixture("workday_nvidia_page0.json")["facets"]
    board = nvidia_board()
    board.facets = [g for g in facets if g["facetParameter"] in ("workerSubType", "timeType")]
    result = fetch(board)
    assert result.status == "truncated"
    assert result.error_code == "listing_ceiling"


# -- identity --------------------------------------------------------------------


def _posting_id(p: Mapping[str, Any]) -> str:
    return str(p["externalPath"]).rsplit("_", 1)[1]


def test_the_recorded_adobe_page_is_keyed_by_posting_with_the_requisition_beside_it() -> None:
    page = fixture("workday_adobe_page0.json")["jobPostings"]
    result = fetch(FakeWorkday(page), ADOBE)
    assert result.status == "complete"
    assert result.jobs_seen == len(page) == 20
    assert {j.external_id for j in result.jobs} == {_posting_id(p) for p in page}

    # As recorded on 2026-09-10: five postings whose id is not their requisition.
    suffixed = [p for p in page if _posting_id(p) != p["bulletFields"][0]]
    assert len(suffixed) == 5
    reposting = suffixed[0]
    job = next(j for j in result.jobs if j.external_id == _posting_id(reposting))
    assert job.external_id == reposting["bulletFields"][0] + "-1"  # R171808-1, not R171808
    assert job.requisition_id == reposting["bulletFields"][0]
    assert (
        job.url
        == "https://adobe.wd5.myworkdayjobs.com/external_experienced" + (reposting["externalPath"])
    )
    assert job.title == reposting["title"]
    assert "Posted" not in job.model_dump_json()  # `postedOn` is prose, never kept


def test_two_postings_of_one_requisition_are_two_jobs_sharing_it() -> None:
    postings = [posting(i) for i in range(5)]
    postings.append(posting(4, suffix="-2"))  # same requisition, second posting
    result = fetch(FakeWorkday(postings))
    assert result.status == "complete"
    assert result.expected_total == 6 == result.jobs_seen
    shared = {
        j.external_id: j.requisition_id for j in result.jobs if j.requisition_id == "JR2000004"
    }
    assert shared == {"JR2000004": "JR2000004", "JR2000004-2": "JR2000004"}


def test_a_posting_with_no_posting_id_makes_the_check_incomplete() -> None:
    postings = [posting(i) for i in range(5)]
    postings[2] = {**postings[2], "externalPath": "/job/nowhere"}  # requisition, but no id
    result = fetch(FakeWorkday(postings))
    assert result.status == "incomplete"
    assert result.error_code == "unidentifiable_job"
    assert result.jobs_seen == 4


def test_a_posting_without_bullet_fields_is_tracked_with_no_requisition() -> None:
    postings = [posting(i) for i in range(3)]
    postings[1] = {**postings[1], "bulletFields": []}
    result = fetch(FakeWorkday(postings))
    assert result.status == "complete"
    by_id = {j.external_id: j.requisition_id for j in result.jobs}
    assert by_id == {"JR2000000": "JR2000000", "JR2000001": None, "JR2000002": "JR2000002"}


def test_a_role_relisted_under_a_new_posting_id_is_reposted_not_continuous() -> None:
    """The reason identity is the posting. `R171808-1` vanishes in the same check
    that `R171808-2` -- same requisition, same title, same location -- appears.
    Keyed by requisition that would be one job that never left; keyed by posting
    it is a repost, and neither `returned` nor still open.
    """
    title = "Senior Manager, Digital Monetization Growth"

    def adobe(posting_id: str, requisition: str, name: str) -> dict[str, Any]:
        slug = "-".join(name.replace(",", "").split())
        return {
            "title": name,
            "externalPath": f"/job/San-Jose/{slug}_{posting_id}",
            "locationsText": "San Jose",
            "postedOn": "Posted Today",
            "bulletFields": [requisition],
        }

    day0 = fetch(
        FakeWorkday(
            [adobe("R171808-1", "R171808", title), adobe("R170001", "R170001", "Designer")]
        ),
        ADOBE,
    )
    day1 = fetch(
        FakeWorkday(
            [adobe("R171808-2", "R171808", title), adobe("R170001", "R170001", "Designer")]
        ),
        ADOBE,
    )
    assert day0.status == day1.status == "complete"

    board = uuid.uuid4()
    now = dt.datetime(2026, 9, 10, 6, 0, tzinfo=dt.UTC)
    baseline = plan_check(BoardCheckState(board_id=board), day0, observed_at=now)
    ids = {n.job.external_id: uuid.uuid5(board, n.job.external_id) for n in baseline.new_jobs}
    # The history as that baseline left it: every job open.
    history = BoardCheckState(
        board_id=board,
        baseline_check_id=uuid.uuid4(),
        known_jobs=[
            KnownBoardJob(
                job_id=ids[n.job.external_id],
                external_id=n.job.external_id,
                fingerprint=n.job.fingerprint,
                is_open=True,
            )
            for n in baseline.new_jobs
        ],
    )

    later = plan_check(history, day1, observed_at=now + dt.timedelta(days=1))

    assert later.status == "complete"
    assert later.returned == []
    assert [s.job.external_id for s in later.still_open] == ["R170001"]
    assert later.gone_job_ids == [ids["R171808-1"]]
    (relisted,) = later.new_jobs
    assert relisted.job.external_id == "R171808-2"
    assert relisted.job.requisition_id == "R171808"
    assert relisted.reposted_from_job_id == ids["R171808-1"]
