"""Personio: `GET https://{company}.jobs.personio.{de,com}/xml`.

Verified 2026-09-10 against Personio's own board (`personio`, 1 position --
the smallest tenant checked across all twelve platforms). The root element is
`<workzag-jobs>` with one `<position>` per job; no total is declared, so
completeness is the single feed itself, parsed without error.

The external id is `<id>`; the title is `<name>`; the location is the
primary `<office>` -- the one the platform itself treats as the listing's
place, and the one the fingerprint uses. `locations` adds every
`<additionalOffices>/<office>`. No structured workplace field: the text rule
in `jfl_intake.workplace` only.

**No URL is recorded.** Unlike the RSS platforms here, Personio's XML feed
carries no link field, and a guessed `https://{company}.jobs.personio.de/job/{id}`
pattern could not be verified live: two independent attempts (2026-09-10 and
2026-09-11) both came back HTTP 429, the second redirecting to Personio's own
marketing homepage rather than a posting. Two rate-limited attempts against a
board this small is enough evidence that this is not the right pattern, or
that this endpoint is simply not meant to be hit directly -- either way,
guessing further and shipping an unverified URL as fact is exactly the kind
of unearned claim this project exists to catch elsewhere, so `url` stays
`None` rather than invented. Do not request it again.

A feed that is not well-formed XML, or is well-formed but rooted in something
other than `<workzag-jobs>`, is `failed`/`malformed_response` -- never an
empty board. A `<workzag-jobs>` with zero `<position>` elements is a
genuinely empty board. Fetched via `Transport.get_text` -- see `xml_single.py`.
"""

from __future__ import annotations

from collections.abc import Mapping

from jfl_core.models import BoardPlatform

from jfl_intake.adapters.base import FetchResult, InvalidBoardKeyError, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed
from jfl_intake.adapters.xml_single import fetch_single_xml, parse_xml
from jfl_intake.http import Transport
from jfl_intake.workplace import dedupe_locations, from_location_text

API = "https://{company}.jobs.personio.{tld}/xml"


def parse(text: str) -> Parsed:
    root = parse_xml(text)
    if root.tag != "workzag-jobs":
        raise MalformedResponseError
    parsed = Parsed()
    for position in root.findall("position"):
        locations = dedupe_locations(
            [
                position.findtext("office"),
                *(o.text for o in position.findall("additionalOffices/office")),
            ]
        )
        parsed.add(
            position.findtext("id"),
            position.findtext("name"),
            position.findtext("office"),
            None,
            workplace=from_location_text(locations),
            locations=locations,
        )
    return parsed


class PersonioAdapter:
    platform: BoardPlatform = "personio"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        _, tld = require_key(board_key, "company", "tld")
        if tld not in ("de", "com"):
            raise InvalidBoardKeyError("board key part 'tld' is missing or invalid")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        self.validate_key(board_key)
        company, tld = require_key(board_key, "company", "tld")
        return fetch_single_xml(transport, API.format(company=company, tld=tld), parse)
