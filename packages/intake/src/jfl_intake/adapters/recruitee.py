"""Recruitee: `GET https://{company}.recruitee.com/api/offers/`.

Verified 2026-09-10 against Make (`make`, 3 offers). Returns
`{"offers": [...]}` with no total, so completeness is the single response
itself.

The external id is `id`; the title is `title`; **`location` is already a
plain display string** ("Remote job"), not an object -- unlike most of the
other platforms here. The public URL is `careers_url`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jfl_core.models import BoardPlatform

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, fetch_single
from jfl_intake.http import Transport

API = "https://{company}.recruitee.com/api/offers/"


def parse(body: Any) -> Parsed:
    if not isinstance(body, dict) or not isinstance(body.get("offers"), list):
        raise MalformedResponseError
    parsed = Parsed()
    for offer in body["offers"]:
        if not isinstance(offer, dict):
            parsed.unidentified += 1
            continue
        parsed.add(
            offer.get("id"),
            offer.get("title"),
            offer.get("location"),
            offer.get("careers_url"),
        )
    return parsed


class RecruiteeAdapter:
    platform: BoardPlatform = "recruitee"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "company")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (company,) = require_key(board_key, "company")
        return fetch_single(transport, API.format(company=company), parse)
