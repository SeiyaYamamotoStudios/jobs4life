"""Greenhouse: `GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs`.

Verified 2026-09-10 against Anthropic's board: 595 jobs, `meta.total` 595.

The external id is the public job `id`. **Not `internal_job_id`** -- that was
checked and is not a stable identity for a posting: one requisition is listed
under two public ids with different titles, so keying on it would merge two
live jobs into one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from jfl_core.models import BoardPlatform

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, as_int, fetch_single
from jfl_intake.http import Transport

API = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"


def parse(body: Any) -> Parsed:
    if not isinstance(body, dict) or not isinstance(body.get("jobs"), list):
        raise MalformedResponseError
    parsed = Parsed()
    for job in body["jobs"]:
        if not isinstance(job, dict):
            parsed.unidentified += 1
            continue
        location = job.get("location")
        parsed.add(
            job.get("id"),
            job.get("title"),
            location.get("name") if isinstance(location, dict) else None,
            job.get("absolute_url"),
            # Stored only -- see `ObservedJob.requisition_id`. Greenhouse lists
            # one requisition under several public ids, so it is never identity.
            requisition_id=job.get("requisition_id"),
        )
    meta = body.get("meta")
    parsed.expected_total = as_int(meta.get("total")) if isinstance(meta, dict) else None
    return parsed


class GreenhouseAdapter:
    platform: BoardPlatform = "greenhouse"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "token")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (token,) = require_key(board_key, "token")
        return fetch_single(transport, API.format(token=quote(token, safe="")), parse)
