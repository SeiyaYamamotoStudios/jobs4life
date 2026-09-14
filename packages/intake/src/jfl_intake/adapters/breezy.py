"""Breezy: `GET https://{company}.breezy.hr/json`.

Verified 2026-09-10 against Breezy's own trial board (`breezy`, 3 postings).
No total is declared, so completeness is the single response itself -- a cap
on a very large board has not been observed, since the only tenant checked
was this small.

The external id is `id`; the title is `name`; the public URL is given
directly (`url`). Location is an object (`location.name`, plus `city`,
`state`, `country` parts); `location.name` is already the composed display
string ("Chaos, FL") and is used as-is, the same choice as Ashby's plain
`location` string.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jfl_core.models import BoardPlatform, Workplace

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, fetch_single
from jfl_intake.http import Transport
from jfl_intake.workplace import as_bool, dedupe_locations, from_location_text

API = "https://{company}.breezy.hr/json"


def parse(body: Any) -> Parsed:
    if not isinstance(body, list):
        raise MalformedResponseError
    parsed = Parsed()
    for job in body:
        if not isinstance(job, dict):
            parsed.unidentified += 1
            continue
        location = job.get("location")
        primary = location.get("name") if isinstance(location, dict) else None
        listed = job.get("locations")
        names = [
            entry.get("name")
            for entry in (listed if isinstance(listed, list) else [])
            if isinstance(entry, dict)
        ]
        locations = dedupe_locations(names or [primary])
        is_remote = as_bool(location, "is_remote") if isinstance(location, dict) else None
        # True is remote; false is NOT on-site -- Breezy does not tell hybrid
        # from on-site -- so it is unknown, and the text is not consulted.
        if is_remote is True:
            workplace: Workplace = "remote"
        elif is_remote is False:
            workplace = "unknown"
        else:
            workplace = from_location_text(locations)
        parsed.add(
            job.get("id"),
            job.get("name"),
            primary,
            job.get("url"),
            workplace=workplace,
            locations=locations,
        )
    return parsed


class BreezyAdapter:
    platform: BoardPlatform = "breezy"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "company")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (company,) = require_key(board_key, "company")
        return fetch_single(transport, API.format(company=company), parse)
