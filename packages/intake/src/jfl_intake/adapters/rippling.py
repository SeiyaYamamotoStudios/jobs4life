"""Rippling: `GET https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs`.

Verified 2026-09-10 against Rippling's own board (`rippling`, 648 postings in
one response). **Use v1** -- the v2 endpoint pages at 20, and this platform
declares no total, so a page cannot be proven incomplete or complete against
anything but the shape of the one response it returns.

No `id` field on a job -- the posting's identity is `uuid`. The title is
`name`. Both the job and its public URL are already given directly (`url`),
unlike SmartRecruiters or Workable, which expose no public link and need one
built.

**Observed quirk, not in `docs/ats-platforms.md`:** the same `uuid` can appear
twice for one combined listing spanning two locations -- Rippling's own board
lists "Account Executive, Broker Channel (Pittsburgh or Cleveland)" twice,
identical in every field but `workLocation`. This is not a repost and not a
data error; it is one posting id doing double duty for two places, which the
platform's own id scheme cannot tell apart.

**Locations for a repeated `uuid` are merged deterministically: distinct,
sorted, joined with `"; "`.** The location feeds the repost fingerprint
(`jfl_intake.normalise.fingerprint`), so picking "whichever occurrence came
first" would make the stored fingerprint depend on an ordering the API makes
no promise about -- a harmless-looking dependency that could manufacture a
false repost the day the platform happens to answer in a different order.
Sorting first removes the ordering dependency entirely: the same set of
locations always produces the same string, however many occurrences arrive
and in whatever order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from jfl_core.models import BoardPlatform, ObservedJob

from jfl_intake.adapters.base import FetchResult, require_key
from jfl_intake.adapters.single import MalformedResponseError, Parsed, fetch_single
from jfl_intake.http import Transport
from jfl_intake.normalise import clean_text, fingerprint
from jfl_intake.workplace import dedupe_locations, from_location_text

API = "https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs"


def _external_id(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return clean_text(value)


def _location(job: Mapping[str, Any]) -> str | None:
    location = job.get("workLocation")
    return clean_text(location.get("label")) if isinstance(location, dict) else None


def parse(body: Any) -> Parsed:
    if not isinstance(body, list):
        raise MalformedResponseError
    parsed = Parsed()

    # Grouped by external id before anything is built, so a repeated uuid's
    # locations are gathered before the merge -- never appended to an
    # already-built job, which would bake in whichever occurrence was first.
    groups: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for job in body:
        if not isinstance(job, dict):
            parsed.unidentified += 1
            continue
        ext_id = _external_id(job.get("uuid"))
        title = clean_text(job.get("name"))
        if ext_id is None or title is None:
            parsed.unidentified += 1
            continue
        if ext_id not in groups:
            groups[ext_id] = []
            order.append(ext_id)
        groups[ext_id].append(job)

    for ext_id in order:
        entries = groups[ext_id]
        first = entries[0]
        title = clean_text(first.get("name"))
        assert title is not None  # guaranteed by the filter above
        locations = sorted({loc for e in entries if (loc := _location(e)) is not None})
        location_text = "; ".join(locations) if locations else None
        merged = dedupe_locations(locations)
        parsed.jobs.append(
            ObservedJob(
                external_id=ext_id,
                title=title,
                location=location_text,
                url=clean_text(first.get("url")),
                fingerprint=fingerprint(title, location_text),
                # No structured workplace field: the text rule only.
                workplace=from_location_text(merged),
                locations=merged,
            )
        )
    return parsed


class RipplingAdapter:
    platform: BoardPlatform = "rippling"

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        require_key(board_key, "slug")

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult:
        (slug,) = require_key(board_key, "slug")
        return fetch_single(transport, API.format(slug=quote(slug, safe="")), parse)
