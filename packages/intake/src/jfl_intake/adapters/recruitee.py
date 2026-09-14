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

from jfl_core.models import BoardPlatform, Workplace

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, fetch_single
from jfl_intake.http import Transport
from jfl_intake.workplace import as_bool, dedupe_locations, from_location_text

API = "https://{company}.recruitee.com/api/offers/"


def _workplace(offer: dict[str, Any], locations: tuple[str, ...]) -> Workplace:
    """Exactly one of `remote` / `hybrid` / `on_site` true is that workplace.
    None or several true, with the booleans present, is unknown. Only when all
    three are absent may the location text decide.
    """
    flags: dict[Workplace, bool | None] = {
        "remote": as_bool(offer, "remote"),
        "hybrid": as_bool(offer, "hybrid"),
        "onsite": as_bool(offer, "on_site"),
    }
    if all(v is None for v in flags.values()):
        return from_location_text(locations)
    true = [kind for kind, v in flags.items() if v is True]
    return true[0] if len(true) == 1 else "unknown"


def parse(body: Any) -> Parsed:
    if not isinstance(body, dict) or not isinstance(body.get("offers"), list):
        raise MalformedResponseError
    parsed = Parsed()
    for offer in body["offers"]:
        if not isinstance(offer, dict):
            parsed.unidentified += 1
            continue
        listed = offer.get("locations")
        names = [
            entry.get("name")
            for entry in (listed if isinstance(listed, list) else [])
            if isinstance(entry, dict)
        ]
        locations = dedupe_locations(names or [offer.get("location")])
        parsed.add(
            offer.get("id"),
            offer.get("title"),
            offer.get("location"),
            offer.get("careers_url"),
            workplace=_workplace(offer, locations),
            locations=locations,
        )
    return parsed


class RecruiteeAdapter:
    platform: BoardPlatform = "recruitee"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "company")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (company,) = require_key(board_key, "company")
        return fetch_single(transport, API.format(company=company), parse)
