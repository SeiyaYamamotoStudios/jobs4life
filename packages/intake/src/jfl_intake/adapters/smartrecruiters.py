"""SmartRecruiters: `GET https://api.smartrecruiters.com/v1/companies/{companyId}/postings?offset=N&limit=100`.

Verified 2026-09-10 against Bosch (`BoschGroup`, 4,835 postings) -- five
targeted requests, not an exhaustive page-through of a board this size.

**A page size over 100 is silently capped, not rejected.** `limit=200`
answered HTTP 200 with 100 postings and echoed `"limit": 100` -- no error, and
the page was not short or empty, so an adapter that advances `offset` by the
size it *requested* skips 100 postings every page and never notices. Every
page here is requested at `limit=100`, but the offset is always advanced by
the **echoed** `limit` from the response, never the requested one.

`totalFound` read 4,835 consistently at offsets 0, 2400 and 4800 -- unlike
Workday, which lies about the total on intermediate pages, so it is trusted
from the first page only, the same choice Workday makes for the same reason.
The last page (offset 4800) returned exactly the 35-posting remainder, and
offset 4835 returned 0 with no overlap with page 0: **SmartRecruiters
terminates cleanly, no wrap**. Paging stops at the first page shorter than the
limit it echoed (including an empty one), and **completeness is verified,
never assumed**: the count of distinct posting ids collected must equal
`totalFound` from the first page. A hard page cap is kept as a safety net that
has not been observed to trigger; hitting it makes the check `incomplete`.

Identity is the posting `id`. The employer's own requisition reference,
`refNumber`, is stored as `requisition_id` and read by no rule. The response
carries no public URL for a posting; `https://jobs.smartrecruiters.com/{companyId}/{id}`
is the public page (verified live against Bosch) and is built rather than
returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jfl_core.models import BoardCheckErrorCode, BoardPlatform, FetchStatus, ObservedJob

from jfl_intake.adapters.base import FetchResult, dedupe, require_key, status_failure
from jfl_intake.adapters.single import as_int
from jfl_intake.http import RequestBudgetExceeded, Transport, TransportError
from jfl_intake.normalise import clean_text, fingerprint

API = "https://api.smartrecruiters.com/v1/companies/{company_id}/postings"
REQUEST_LIMIT = 100
# 150 pages at the observed 100-per-page cap covers 15,000 postings, roughly
# 3x Bosch's board. A safety net, not a design ceiling: hitting it makes the
# check `incomplete`, never `complete`.
MAX_PAGES = 150


class _Abort(Exception):
    def __init__(self, status: FetchStatus, code: BoardCheckErrorCode) -> None:
        super().__init__(code)
        self.status: FetchStatus = status
        self.code: BoardCheckErrorCode = code


def _external_id(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return clean_text(value)


def _location(location: Mapping[str, Any]) -> str | None:
    parts = [clean_text(location.get(k)) for k in ("city", "region", "country")]
    joined = ", ".join(p for p in parts if p)
    return joined or None


class _SmartRecruitersCheck:
    def __init__(self, transport: Transport, company_id: str) -> None:
        self._transport = transport
        self._company_id = company_id
        self._endpoint = API.format(company_id=company_id)
        self._collected: dict[str, ObservedJob | None] = {}
        self.requests = 0
        self.pages = 0

    def run(self) -> FetchResult:
        try:
            offset = 0
            total: int | None = None
            while True:
                postings, echoed_limit, page_total = self._page(offset)
                if total is None:
                    total = page_total
                for posting in postings:
                    key, record = self._record(posting)
                    self._collected.setdefault(key, record)
                if len(postings) < echoed_limit:
                    break
                self.pages += 1
                if self.pages >= MAX_PAGES:
                    raise _Abort("incomplete", "page_cap_reached")
                offset += echoed_limit
            assert total is not None
            return self._verdict(total)
        except _Abort as abort:
            return self._result(abort.status, abort.code, None)
        except TransportError as exc:
            return self._result("unreachable", exc.code, None)

    def _page(self, offset: int) -> tuple[list[dict[str, Any]], int, int]:
        url = f"{self._endpoint}?offset={offset}&limit={REQUEST_LIMIT}"
        try:
            response = self._transport.get_json(url)
        except RequestBudgetExceeded as exc:
            raise _Abort("incomplete", exc.code) from None
        self.requests += 1

        if response.status != 200:
            status, code = status_failure(response.status)
            raise _Abort(status, code)
        body = response.body
        if not isinstance(body, dict) or not isinstance(body.get("content"), list):
            raise _Abort("failed", "malformed_response")
        echoed_limit = as_int(body.get("limit"))
        total = as_int(body.get("totalFound"))
        # `limit=0` would loop forever advancing nowhere; either missing
        # number is the same silent-empty shape the drop guard exists for.
        if echoed_limit is None or echoed_limit <= 0 or total is None:
            raise _Abort("failed", "malformed_response")
        postings = body["content"]
        if not all(isinstance(p, dict) for p in postings):
            raise _Abort("failed", "malformed_response")
        return postings, echoed_limit, total

    def _record(self, posting: Mapping[str, Any]) -> tuple[str, ObservedJob | None]:
        title = clean_text(posting.get("name"))
        location = posting.get("location")
        location_text = _location(location) if isinstance(location, dict) else None
        ext_id = _external_id(posting.get("id"))
        key = ext_id or f"untracked:{title}|{location_text}"
        if ext_id is None or title is None:
            return key, None
        url = f"https://jobs.smartrecruiters.com/{self._company_id}/{ext_id}"
        return key, ObservedJob(
            external_id=ext_id,
            requisition_id=clean_text(posting.get("refNumber")),
            title=title,
            location=location_text,
            url=url,
            fingerprint=fingerprint(title, location_text),
        )

    def _verdict(self, expected: int) -> FetchResult:
        if any(record is None for record in self._collected.values()):
            return self._result("incomplete", "unidentifiable_job", expected)
        if len(self._collected) != expected:
            return self._result("incomplete", "count_mismatch", expected)
        return self._result("complete", None, expected)

    def _result(
        self, status: FetchStatus, code: BoardCheckErrorCode | None, expected: int | None
    ) -> FetchResult:
        jobs = dedupe([r for r in self._collected.values() if r is not None])
        return FetchResult(
            status=status,
            jobs=jobs,
            expected_total=expected,
            error_code=code,
            requests=self.requests,
            detail={"pages": self.pages + 1},
        )


class SmartRecruitersAdapter:
    platform: BoardPlatform = "smartrecruiters"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "company_id")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (company_id,) = require_key(board_key, "company_id")
        return _SmartRecruitersCheck(transport, company_id).run()
