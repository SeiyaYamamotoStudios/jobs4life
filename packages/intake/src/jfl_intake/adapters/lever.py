"""Lever: `GET https://api.lever.co/v0/postings/{company}?mode=json`.

Verified 2026-09-10. Returns a bare JSON list -- 310 postings for Palantir in
one response, with no `skip`/`limit` given -- and no total, so completeness is
the single response itself. The title is `text`; the location is
`categories.location`.

A 200 whose body is not a list is malformed, never an empty board: Lever's
not-found body is an object, and treating an object as "no jobs" is exactly the
silent-empty failure the drop guard exists for.
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

API = "https://api.lever.co/v0/postings/{company}?mode=json"


def parse(body: Any) -> Parsed:
    if not isinstance(body, list):
        raise MalformedResponseError
    parsed = Parsed()
    for posting in body:
        if not isinstance(posting, dict):
            parsed.unidentified += 1
            continue
        categories = posting.get("categories")
        primary = categories.get("location") if isinstance(categories, dict) else None
        all_locations = categories.get("allLocations") if isinstance(categories, dict) else None
        locations = dedupe_locations(
            all_locations if isinstance(all_locations, list) and all_locations else [primary]
        )
        parsed.add(
            posting.get("id"),
            posting.get("text"),
            primary,
            posting.get("hostedUrl"),
            workplace=resolve(from_enum(posting.get("workplaceType")), locations),
            locations=locations,
        )
    return parsed


class LeverAdapter:
    platform: BoardPlatform = "lever"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "company")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (company,) = require_key(board_key, "company")
        return fetch_single(transport, API.format(company=quote(company, safe="")), parse)
