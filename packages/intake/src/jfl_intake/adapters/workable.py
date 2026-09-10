"""Workable: `POST https://apply.workable.com/api/v3/accounts/{subdomain}/jobs`.

Verified 2026-09-10 against Devsinc (`devsinc-17`, 33 postings, paged) and
Hugging Face (`huggingface`, 6, single page). Body:
`{"query": "", "location": [], "department": [], "worktype": [], "remote": []}`.

**Token paging, clean.** A page carries `nextPage`; send it back as `"token"`
in the next request's body. Devsinc paged 10, 10, 10, 3; `total` read 33 on
every page -- unlike Workday, which lies about the total on intermediate
pages; no id repeated across the four pages; the last page omitted
`nextPage`. **Complete** = no token remaining **and** the distinct ids
collected equal page 1's `total`. A page handing back an id already collected
in this check is a failure to make progress, not a legitimate way to finish:
paging stops there and the check is `incomplete`, never `complete`, even if
the count happens to match. A hard page cap is kept as a safety net that has
not been observed to trigger.

**A 403 is `failed`, never an empty board.** Workable refused `Python-urllib`
with HTTP 403 while accepting an honest client (confirmed live, both against
the same board). `status_failure` already maps a bare 403 to
`failed`/`http_client_error`, which is exactly the behaviour wanted here --
nothing platform-specific is needed for it.

Identity is `id` (numeric on the wire, stored as a string). There is no
public URL on a job object -- only `shortcode`, which the public posting page
uses: `https://apply.workable.com/{subdomain}/j/{shortcode}` (verified live
against a Hugging Face posting). Location is an object (`city`, `country`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jfl_core.models import BoardCheckErrorCode, BoardPlatform, FetchStatus, ObservedJob

from jfl_intake.adapters.base import FetchResult, dedupe, require_key, status_failure
from jfl_intake.adapters.single import as_int
from jfl_intake.http import RequestBudgetExceeded, Transport, TransportError
from jfl_intake.normalise import clean_text, fingerprint

API = "https://apply.workable.com/api/v3/accounts/{subdomain}/jobs"
BASE_BODY: dict[str, Any] = {
    "query": "",
    "location": [],
    "department": [],
    "worktype": [],
    "remote": [],
}
# 60 pages at the observed 10-per-page rate covers 600 postings, well past
# any tenant seen. A safety net, not a design ceiling.
MAX_PAGES = 60


class _Abort(Exception):
    def __init__(self, status: FetchStatus, code: BoardCheckErrorCode) -> None:
        super().__init__(code)
        self.status: FetchStatus = status
        self.code: BoardCheckErrorCode = code


def _location(location: object) -> str | None:
    if not isinstance(location, Mapping):
        return None
    parts = [clean_text(location.get(k)) for k in ("city", "country")]
    joined = ", ".join(p for p in parts if p)
    return joined or None


def _external_id(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return clean_text(value)


class _WorkableCheck:
    def __init__(self, transport: Transport, subdomain: str) -> None:
        self._transport = transport
        self._subdomain = subdomain
        self._endpoint = API.format(subdomain=subdomain)
        self._collected: dict[str, ObservedJob | None] = {}
        self.requests = 0
        self.pages = 0

    def run(self) -> FetchResult:
        try:
            token: str | None = None
            total: int | None = None
            while True:
                results, page_total, next_token = self._page(token)
                if total is None:
                    total = page_total
                for job in results:
                    key, record = self._record(job)
                    if record is not None and key in self._collected:
                        raise _Abort("incomplete", "duplicate_posting")
                    self._collected.setdefault(key, record)
                self.pages += 1
                if next_token is None:
                    break
                if self.pages >= MAX_PAGES:
                    raise _Abort("incomplete", "page_cap_reached")
                token = next_token
            assert total is not None
            return self._verdict(total)
        except _Abort as abort:
            return self._result(abort.status, abort.code, None)
        except TransportError as exc:
            return self._result("unreachable", exc.code, None)

    def _page(self, token: str | None) -> tuple[list[dict[str, Any]], int, str | None]:
        body = dict(BASE_BODY)
        if token is not None:
            body["token"] = token
        try:
            response = self._transport.post_json(self._endpoint, body)
        except RequestBudgetExceeded as exc:
            raise _Abort("incomplete", exc.code) from None
        self.requests += 1

        if response.status != 200:
            status, code = status_failure(response.status)
            raise _Abort(status, code)
        payload = response.body
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise _Abort("failed", "malformed_response")
        total = as_int(payload.get("total"))
        if total is None:
            raise _Abort("failed", "malformed_response")
        results = payload["results"]
        if not all(isinstance(r, dict) for r in results):
            raise _Abort("failed", "malformed_response")
        next_page = payload.get("nextPage")
        next_token = next_page if isinstance(next_page, str) and next_page else None
        return results, total, next_token

    def _record(self, job: Mapping[str, Any]) -> tuple[str, ObservedJob | None]:
        title = clean_text(job.get("title"))
        location_text = _location(job.get("location"))
        ext_id = _external_id(job.get("id"))
        shortcode = clean_text(job.get("shortcode"))
        key = ext_id or f"untracked:{title}|{location_text}"
        if ext_id is None or title is None:
            return key, None
        url = f"https://apply.workable.com/{self._subdomain}/j/{shortcode}" if shortcode else None
        return key, ObservedJob(
            external_id=ext_id,
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
            detail={"pages": self.pages},
        )


class WorkableAdapter:
    platform: BoardPlatform = "workable"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "subdomain")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (subdomain,) = require_key(board_key, "subdomain")
        return _WorkableCheck(transport, subdomain).run()
