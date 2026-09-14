"""Teamtailor: `GET https://{site}/jobs.rss`.

Verified 2026-09-10 against `career.teamtailor.com` (12 items). One `<item>`
per job in a standard RSS 2.0 channel, plus a `tt:` namespace
(`https://teamtailor.com/locations`) carrying structured location data
alongside the plain RSS fields. No total is declared anywhere in the feed, so
completeness is the single feed itself, parsed without error.

The external id is `<guid>` (a UUID, not a URL -- Teamtailor does not set
`isPermaLink`). The public URL is `<link>`. The title is the plain RSS
`<title>`.

Location is namespaced: `tt:locations/tt:location/tt:name`. **Observed
quirk**: on one real item `tt:name` was present but empty while `tt:city`
carried the actual value ("Toronto") -- a blank name is not the same as no
location, so `tt:name` is preferred and `tt:city` is the fallback, never the
other way round. Only the first `tt:location` is used for `location` (and so
the fingerprint); `locations` lists every one -- live on 2026-09-15 one item
listed Sydney and Melbourne. Workplace is `<remoteStatus>`; see
`jfl_intake.workplace`.

A feed that is not well-formed XML, or is well-formed but has no `<channel>`
to find items under, is `failed`/`malformed_response` -- never an empty
board. A `<channel>` with zero `<item>` elements is a genuinely empty board.
Fetched via `Transport.get_text` -- see `xml_single.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jfl_core.models import BoardPlatform, Workplace

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed
from jfl_intake.adapters.xml_single import fetch_single_xml, parse_xml
from jfl_intake.http import Transport
from jfl_intake.normalise import clean_text, normalise
from jfl_intake.workplace import dedupe_locations, from_location_text

API = "https://{site}/jobs.rss"
_NS = {"tt": "https://teamtailor.com/locations"}


def _location_name(loc: Any) -> str | None:
    return clean_text(loc.findtext("tt:name", namespaces=_NS)) or clean_text(
        loc.findtext("tt:city", namespaces=_NS)
    )


def _location(item: Any) -> str | None:
    loc = item.find("tt:locations/tt:location", _NS)
    return None if loc is None else _location_name(loc)


# `<remoteStatus>`. `none` (observed live) is Teamtailor's "not remote" and also
# what an unset job carries, so it is not a statement of on-site; `temporary` is
# remote for now, which is not a workplace. Both are unknown, and final.
_REMOTE_STATUS: dict[str, Workplace] = {
    "fully": "remote",
    "remote": "remote",
    "hybrid": "hybrid",
    "none": "unknown",
    "temporary": "unknown",
}


def _workplace(item: Any, locations: tuple[str, ...]) -> Workplace:
    status = normalise(item.findtext("remoteStatus"))
    if status in _REMOTE_STATUS:
        return _REMOTE_STATUS[status]
    return from_location_text(locations)


def parse(text: str) -> Parsed:
    root = parse_xml(text)
    channel = root.find("channel")
    if channel is None:
        raise MalformedResponseError
    parsed = Parsed()
    for item in channel.findall("item"):
        locations = dedupe_locations(
            _location_name(loc) for loc in item.findall("tt:locations/tt:location", _NS)
        )
        parsed.add(
            item.findtext("guid"),
            item.findtext("title"),
            _location(item),
            item.findtext("link"),
            workplace=_workplace(item, locations),
            locations=locations,
        )
    return parsed


class TeamtailorAdapter:
    platform: BoardPlatform = "teamtailor"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "site")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (site,) = require_key(board_key, "site")
        return fetch_single_xml(transport, API.format(site=site), parse)
