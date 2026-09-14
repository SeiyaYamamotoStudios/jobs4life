"""Ashby: `GET https://api.ashbyhq.com/posting-api/job-board/{name}`.

Verified 2026-09-10. Returns `{"jobs": [...], "apiVersion": ...}` with no total,
so completeness is the single response itself.

**`isListed` is honoured.** A job with `isListed: false` is reachable by direct
link but not on the public board; it is not part of what the owner is watching,
so it is skipped entirely -- neither recorded nor counted as unidentified. A
missing `isListed` is treated as listed, since the public board endpoint is
what returned it.

Workplace is `workplaceType`, never `isRemote`; locations are `location` plus
each `secondaryLocations[].location`. See `jfl_intake.workplace`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from jfl_core.models import BoardPlatform

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, fetch_single
from jfl_intake.http import Transport
from jfl_intake.workplace import dedupe_locations, from_enum, resolve

API = "https://api.ashbyhq.com/posting-api/job-board/{name}"


def parse(body: Any) -> Parsed:
    if not isinstance(body, dict) or not isinstance(body.get("jobs"), list):
        raise MalformedResponseError
    parsed = Parsed()
    for job in body["jobs"]:
        if not isinstance(job, dict):
            parsed.unidentified += 1
            continue
        if job.get("isListed") is False:
            continue
        secondary = job.get("secondaryLocations")
        locations = dedupe_locations(
            [
                job.get("location"),
                *(
                    s.get("location")
                    for s in (secondary if isinstance(secondary, list) else [])
                    if isinstance(s, dict)
                ),
            ]
        )
        parsed.add(
            job.get("id"),
            job.get("title"),
            job.get("location"),
            job.get("jobUrl"),
            # `workplaceType` only -- never `isRemote`, which Cohere's board sets
            # on hybrid jobs too. See `jfl_intake.workplace`.
            workplace=resolve(from_enum(job.get("workplaceType")), locations),
            locations=locations,
        )
    return parsed


class AshbyAdapter:
    platform: BoardPlatform = "ashby"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "name")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (name,) = require_key(board_key, "name")
        return fetch_single(transport, API.format(name=quote(name, safe="")), parse)
