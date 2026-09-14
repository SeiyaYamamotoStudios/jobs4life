"""Workday: `POST https://{tenant}.{wdN}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs`.

The hard one. Every rule below comes from behaviour verified live on 2026-09-10
against NVIDIA (`nvidia.wd5` / `NVIDIAExternalCareerSite`) and Adobe (`adobe.wd5`
/ `external_experienced`), and each has a test reproducing it.

**1. `limit` above 20 fails, and not consistently.** NVIDIA answered HTTP 200
with zero `jobPostings` and no `total` key at all; Adobe answered HTTP 400. So
every request asks for exactly `PAGE_SIZE = 20`, and both a 4xx and a 200 that
is missing `total` (on offset 0) or `jobPostings` are a **failed** check -- never
an empty board. A genuinely empty board is distinguishable: `total: 0` present
with an empty list at offset 0.

**2. An unfiltered listing exposes at most 2,000 jobs, then wraps.** NVIDIA's
offset 2000 returned page 0 exactly, and offsets well beyond kept returning full
pages. Page 0's `total` said 2000; NVIDIA really has 2,630 -- its facet counts
sum to that -- so 630 jobs are invisible to plain paging.

**3. It wraps past the end even below that ceiling.** Adobe (`total` 730): offset
700 gave 20, offset 720 gave 10 (the real end), offset 740 gave page 0 again.

**4. `total` reads 0 on intermediate pages.** It is only trusted from offset 0.

So a listing stops paging on the FIRST of: a short page (< 20); a posting already
collected in this listing (a wrap); the offset reaching offset-0's `total` when
that total is under the ceiling; or the page cap, a safety net that makes the
check `incomplete`. It never loops unbounded, and the politeness budget in
`jfl_intake.http.RequestBudget` bounds it a second time.

**A board at the ceiling is split by a facet.** Page 0 carries `facets`: groups
with a `facetParameter` and `values` (id + count), where a value may itself be a
nested group one level down. A group is usable when its values *partition* the
board -- every job in exactly one value -- and no value reaches the ceiling. See
`choose_partition` for how "partition" is decided without knowing the true
total. Each value is fetched as its own listing with
`appliedFacets: {parameter: [value id]}`, and the listings are unioned.

**Completeness is verified, never assumed**: the number of distinct postings
collected must equal the expected total -- the partition's summed count, or
offset 0's total. Equal is `complete`; anything else is `incomplete`; a ceiling
board with no usable facet is `truncated`.

**Identity is the posting, not the requisition.** The external id is the suffix
of `externalPath` after its last `_`: `R171808-1` in Adobe's
`.../Senior-Manager--Digital-Monetization-Growth_R171808-1`. The requisition,
`bulletFields[0]` (`R171808`), is recorded beside it as `requisition_id`. On
NVIDIA the two are equal; on Adobe's page 0 (2026-09-10) 5 of 20 postings carry
a `-1` suffix, with no requisition listed twice.

Why the posting: the point of watching is repost patterns. Keyed by requisition,
a role taken down and re-listed as `R171808-2` would read as a job that never
left, and could never read as "reposted". History accumulated at the coarser
grain can never be re-keyed, whereas the requisition is recoverable from the
posting. **`requisition_id` is stored only** -- no rule reads it, because a
`-2` re-listing has not been observed yet and rules come from observed patterns.

A posting whose `externalPath` has no `_`-suffix cannot be identified and makes
the check `incomplete`. Completeness counts distinct postings (by `externalPath`)
against the total, which also counts postings.

**`postedOn` is prose** ("Posted Today", "Posted 30+ Days Ago") and is never
parsed into a timestamp. `first_seen_at` is when *we* saw a job, which is the
only date this history can vouch for.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from jfl_core.models import BoardCheckErrorCode, BoardPlatform, FetchStatus, ObservedJob

from jfl_intake.adapters.base import (
    FetchResult,
    InvalidBoardKeyError,
    dedupe,
    require_key,
    status_failure,
)
from jfl_intake.http import RequestBudgetExceeded, Transport, TransportError
from jfl_intake.normalise import clean_text, fingerprint
from jfl_intake.workplace import dedupe_locations, from_location_text

PAGE_SIZE = 20
LISTING_CEILING = 2000
# 100 pages reach the ceiling and the 101st is the wrap that proves it.
MAX_PAGES_PER_LISTING = LISTING_CEILING // PAGE_SIZE + 1

API = "https://{tenant}.{wd}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
POSTING_URL = "https://{tenant}.{wd}.myworkdayjobs.com/{site}{path}"


@dataclass(frozen=True, slots=True)
class FacetValue:
    id: str
    count: int


@dataclass(frozen=True, slots=True)
class FacetGroup:
    parameter: str
    values: tuple[FacetValue, ...]

    @property
    def total(self) -> int:
        return sum(v.count for v in self.values)

    @property
    def largest(self) -> int:
        return max((v.count for v in self.values), default=0)

    @property
    def nonzero(self) -> int:
        return sum(1 for v in self.values if v.count > 0)


def facet_groups(facets: object) -> list[FacetGroup]:
    """Every facet group on page 0, with nested groups (one level) lifted to
    the top. A group with no countable values -- NVIDIA's `locationMainGroup`,
    which holds only nested groups -- contributes its nested groups and nothing
    of its own.
    """
    groups: list[FacetGroup] = []
    if not isinstance(facets, list):
        return groups

    def values_of(raw: object) -> Iterator[FacetValue]:
        if not isinstance(raw, list):
            return
        for value in raw:
            if (
                isinstance(value, dict)
                and isinstance(value.get("id"), str)
                and isinstance(value.get("count"), int)
                and not isinstance(value.get("count"), bool)
                and value["count"] >= 0
            ):
                yield FacetValue(id=value["id"], count=value["count"])

    for group in facets:
        if not isinstance(group, dict):
            continue
        raw_values = group.get("values")
        if isinstance(raw_values, list):
            for nested in raw_values:
                if isinstance(nested, dict) and isinstance(nested.get("facetParameter"), str):
                    nested_values = tuple(values_of(nested.get("values")))
                    if nested_values:
                        groups.append(FacetGroup(nested["facetParameter"], nested_values))
        own = tuple(values_of(raw_values))
        if own and isinstance(group.get("facetParameter"), str):
            groups.append(FacetGroup(group["facetParameter"], own))
    return groups


def choose_partition(
    facets: object, *, ceiling: int = LISTING_CEILING
) -> tuple[FacetGroup, int] | None:
    """The facet group to split a ceiling board by, and the total it implies.

    The difficulty is that the true total is exactly what the listing will not
    say. Facet counts give it away, but not every group is a partition:

      * a **multi-valued** facet counts a job once per value, so its sum is too
        HIGH (NVIDIA's `locationHierarchy2` sums to 3,065, `locations` to 4,497);
      * a facet some jobs **lack** a value for sums too LOW.

    A group that genuinely partitions the board sums to the true total exactly,
    and so does every other one. So the rule is **consensus**: the total is a
    sum shared by at least two independent groups, and that sum must reach the
    ceiling (anything smaller cannot be the total of a board the listing
    capped). On NVIDIA, `jobFamilyGroup`, `workerSubType` and `timeType` all sum
    to 2,630 and nothing else agrees.

    Two coincidentally equal wrong sums would pass this, which is why the result
    is still verified after fetching: the distinct postings collected must equal
    the total, or the check is `incomplete`. Consensus protects against the
    dangerous direction -- a too-low sum that fetching could not contradict.

    Among groups summing to that total, only those whose largest value is under
    the ceiling can be listed completely (`workerSubType`'s 2,290 and
    `timeType`'s 2,628 cannot). Of those, fewest non-empty values wins, since
    every value costs at least one request; then the smaller largest value; then
    the parameter name, so the choice is deterministic.

    None if there is no consensus, the consensus is ambiguous (two different
    sums each agreed by two groups), or no agreeing group is listable.
    """
    groups = facet_groups(facets)
    agreement = Counter(group.total for group in groups)
    agreed = [total for total, n in agreement.items() if n >= 2 and total >= ceiling]
    if len(agreed) != 1:
        return None
    total = agreed[0]
    listable = [g for g in groups if g.total == total and g.largest < ceiling]
    if not listable:
        return None
    best = min(listable, key=lambda g: (g.nonzero, g.largest, g.parameter))
    return best, total


@dataclass(frozen=True, slots=True)
class _Page:
    total: int | None
    postings: list[dict[str, Any]]
    facets: object


class _Abort(Exception):
    def __init__(self, status: FetchStatus, code: BoardCheckErrorCode) -> None:
        super().__init__(code)
        self.status: FetchStatus = status
        self.code: BoardCheckErrorCode = code


class _WorkdayCheck:
    """One check's worth of state: the transport, and every posting collected so
    far (kept so that an abort part-way still reports what it saw).
    """

    def __init__(self, transport: Transport, tenant: str, wd: str, site: str) -> None:
        self._transport = transport
        self._tenant = tenant
        self._wd = wd
        self._site = site
        self._endpoint = API.format(tenant=tenant, wd=wd, site=site)
        self._collected: dict[str, ObservedJob | None] = {}
        self.requests = 0
        self.pages = 0

    def run(self) -> FetchResult:
        expected: int | None = None
        detail: dict[str, object] = {}
        try:
            first = self._page({}, 0, first=True)
            assert first.total is not None
            if first.total < LISTING_CEILING:
                expected = first.total
                detail["strategy"] = "listing"
                self._listing({}, first)
                return self._verdict(expected, detail)

            chosen = choose_partition(first.facets)
            if chosen is None:
                detail["strategy"] = "truncated"
                return self._result("truncated", "listing_ceiling", None, detail)
            group, expected = chosen
            detail.update(strategy="facet_split", facet=group.parameter, slices=group.nonzero)
            for value in group.values:
                if value.count == 0:
                    continue
                applied = {group.parameter: [value.id]}
                self._listing(applied, self._page(applied, 0, first=True))
            return self._verdict(expected, detail)
        except _Abort as abort:
            return self._result(abort.status, abort.code, expected, detail)
        except TransportError as exc:
            return self._result("unreachable", exc.code, expected, detail)

    # -- paging ------------------------------------------------------------

    def _page(self, applied: Mapping[str, list[str]], offset: int, *, first: bool) -> _Page:
        body = {
            "appliedFacets": dict(applied),
            "limit": PAGE_SIZE,
            "offset": offset,
            "searchText": "",
        }
        try:
            response = self._transport.post_json(self._endpoint, body)
        except RequestBudgetExceeded as exc:
            raise _Abort("incomplete", exc.code) from None
        self.requests += 1
        self.pages += 1

        if response.status != 200:
            status, code = status_failure(response.status)
            raise _Abort(status, code)
        payload = response.body
        if not isinstance(payload, dict) or not isinstance(payload.get("jobPostings"), list):
            raise _Abort("failed", "malformed_response")
        postings = payload["jobPostings"]
        if not all(isinstance(p, dict) for p in postings):
            raise _Abort("failed", "malformed_response")

        total = payload.get("total")
        if first:
            # Offset 0 is the only place `total` means anything, so it must be
            # there and must agree with the page. NVIDIA's silent failure was a
            # 200 with no `total` and no postings; a positive total with an empty
            # page, or a zero total with postings, is the same kind of lie.
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                raise _Abort("failed", "malformed_response")
            if (total == 0) != (len(postings) == 0):
                raise _Abort("failed", "malformed_response")
        return _Page(
            total=total if first else None, postings=postings, facets=payload.get("facets")
        )

    def _listing(self, applied: Mapping[str, list[str]], first: _Page) -> None:
        """Page one listing (the whole board, or one facet slice) to its end."""
        assert first.total is not None
        total = first.total
        seen: set[str] = set()
        self._absorb(first.postings, seen)
        if len(first.postings) < PAGE_SIZE:
            return

        pages = 1
        offset = PAGE_SIZE
        while True:
            if total < LISTING_CEILING and offset >= total:
                return
            if pages >= MAX_PAGES_PER_LISTING:
                raise _Abort("incomplete", "page_cap_reached")
            page = self._page(applied, offset, first=False)
            pages += 1
            wrapped = self._absorb(page.postings, seen)
            if wrapped or len(page.postings) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

    def _absorb(self, postings: list[dict[str, Any]], seen: set[str]) -> bool:
        """Collect a page's postings. True if any was already collected in this
        listing -- a wrap. New postings on that page are still kept.
        """
        wrapped = False
        for posting in postings:
            key, record = self._record(posting)
            if key in seen:
                wrapped = True
                continue
            seen.add(key)
            self._collected.setdefault(key, record)
        return wrapped

    def _record(self, posting: Mapping[str, Any]) -> tuple[str, ObservedJob | None]:
        path = clean_text(posting.get("externalPath"))
        title = clean_text(posting.get("title"))
        location = clean_text(posting.get("locationsText"))

        # Identity: the posting id, what follows the last `_` of `externalPath`
        # (`R171808-1`). No fallback to the requisition -- mixing grains within one
        # board's history is exactly what keying by posting exists to prevent.
        posting_id = clean_text(path.rsplit("_", 1)[1]) if path and "_" in path else None

        # Stored beside it, never used as identity and read by no rule: see the
        # module docstring. Absent `bulletFields` leaves it None, not unidentified.
        requisition: str | None = None
        bullets = posting.get("bulletFields")
        if isinstance(bullets, list) and bullets:
            requisition = clean_text(bullets[0])

        key = path or f"untracked:{title}|{location}"
        if posting_id is None or title is None:
            return key, None
        url = (
            POSTING_URL.format(tenant=self._tenant, wd=self._wd, site=self._site, path=path)
            if path and path.startswith("/")
            else None
        )
        return key, ObservedJob(
            external_id=posting_id,
            requisition_id=requisition,
            title=title,
            location=location,
            url=url,
            fingerprint=fingerprint(title, location),
            # No structured workplace field in the listing: the text rule only.
            # `locationsText` is stored as given -- a multi-location posting reads
            # "2 Locations", and the individual places are never invented.
            workplace=from_location_text([location]),
            locations=dedupe_locations([location]),
        )

    # -- verdict -----------------------------------------------------------

    def _verdict(self, expected: int, detail: dict[str, object]) -> FetchResult:
        detail["postings"] = len(self._collected)
        if any(record is None for record in self._collected.values()):
            return self._result("incomplete", "unidentifiable_job", expected, detail)
        if len(self._collected) != expected:
            return self._result("incomplete", "count_mismatch", expected, detail)
        return self._result("complete", None, expected, detail)

    def _result(
        self,
        status: FetchStatus,
        code: BoardCheckErrorCode | None,
        expected: int | None,
        detail: dict[str, object],
    ) -> FetchResult:
        jobs = dedupe([r for r in self._collected.values() if r is not None])
        detail["pages"] = self.pages
        return FetchResult(
            status=status,
            jobs=jobs,
            expected_total=expected,
            error_code=code,
            requests=self.requests,
            detail=detail,
        )


class WorkdayAdapter:
    platform: BoardPlatform = "workday"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        _, wd, _ = require_key(board_key, "tenant", "wd", "site")
        if not (wd.startswith("wd") and wd[2:].isdigit()):
            raise InvalidBoardKeyError("board key part 'wd' is missing or invalid")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        self.validate_key(board_key)
        tenant, wd, site = require_key(board_key, "tenant", "wd", "site")
        return _WorkdayCheck(transport, tenant, wd, site).run()
