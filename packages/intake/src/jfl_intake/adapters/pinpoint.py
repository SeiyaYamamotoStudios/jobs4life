"""Pinpoint: `GET https://{company}.pinpointhq.com/postings.json`.

Verified 2026-09-10 against Sun King (`sunking`, 206 postings). Returns
`{"data": [...]}` with **flat** keys -- not the JSON:API `attributes`
envelope some Pinpoint responses use elsewhere -- and no paging keys, so
completeness is the single response itself. A cap on a much larger board has
not been observed.

**Postings, never jobs.** `/postings.json` and `/jobs.json` are different
resources with disjoint id sets (confirmed live: 206 postings, 206 jobs, ids
essentially disjoint). Postings are the grain this adapter watches -- `/jobs`
is not fetched.

Each posting nests a `job` object that is its parent requisition. **Store
`job.id` as `requisition_id`** -- always present -- **never `job.requisition_id`**,
which is the employer's own reference and was empty on every posting checked.
The public posting URL (`url`) uses a UUID path; the API's own `id` is
numeric and is what identifies the posting. `/api/v1/*` is the authenticated
API (401 unauthenticated) and is not used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jfl_core.models import BoardPlatform

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, fetch_single
from jfl_intake.http import Transport

API = "https://{company}.pinpointhq.com/postings.json"


def parse(body: Any) -> Parsed:
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise MalformedResponseError
    parsed = Parsed()
    for posting in body["data"]:
        if not isinstance(posting, dict):
            parsed.unidentified += 1
            continue
        location = posting.get("location")
        job = posting.get("job")
        parsed.add(
            posting.get("id"),
            posting.get("title"),
            location.get("name") if isinstance(location, dict) else None,
            posting.get("url"),
            requisition_id=job.get("id") if isinstance(job, dict) else None,
        )
    return parsed


class PinpointAdapter:
    platform: BoardPlatform = "pinpoint"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "company")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (company,) = require_key(board_key, "company")
        return fetch_single(transport, API.format(company=company), parse)
