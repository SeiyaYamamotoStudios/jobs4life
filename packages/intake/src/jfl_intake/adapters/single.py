"""The shared shape of a board that answers in ONE response: Greenhouse, Ashby, Lever.

For these, one 200 response with the documented shape *is* the whole board --
there is no pagination to fall short in -- so completeness is provable from the
response alone. Two things can still make it `incomplete`: a listed job with no
usable id or title (it cannot be tracked, so the result cannot claim to account
for everything), and, where the platform declares a total (Greenhouse's
`meta.total`), a count that disagrees with it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jfl_core.models import ObservedJob

from jfl_intake.adapters.base import FetchResult, dedupe, failure, from_http_failure
from jfl_intake.http import RequestBudgetExceeded, Transport, TransportError
from jfl_intake.normalise import clean_text, fingerprint


class MalformedResponseError(Exception):
    """A 200 whose body is not the platform's documented shape."""


@dataclass
class Parsed:
    jobs: list[ObservedJob] = field(default_factory=list)
    unidentified: int = 0
    expected_total: int | None = None

    def add(
        self,
        external_id: object,
        title: object,
        location: object,
        url: object,
        *,
        requisition_id: object = None,
    ) -> None:
        """Record one listed job. `requisition_id` is for platforms that expose
        one; it is stored and never used to identify the job.
        """
        ext = _external_id(external_id)
        clean_title = clean_text(title)
        if ext is None or clean_title is None:
            self.unidentified += 1
            return
        clean_location = clean_text(location)
        self.jobs.append(
            ObservedJob(
                external_id=ext,
                title=clean_title,
                location=clean_location,
                url=clean_text(url),
                fingerprint=fingerprint(clean_title, clean_location),
                requisition_id=_external_id(requisition_id),
            )
        )


def _external_id(value: object) -> str | None:
    # `bool` is an `int`; a boolean id is a shape error, not job number 1.
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return clean_text(value)


def as_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def fetch_single(transport: Transport, url: str, parse: Callable[[Any], Parsed]) -> FetchResult:
    try:
        response = transport.get_json(url)
    except TransportError as exc:
        return failure("unreachable", exc.code, requests=1)
    except RequestBudgetExceeded as exc:
        return failure("incomplete", exc.code)

    if response.status != 200:
        return from_http_failure(response, requests=1)
    try:
        parsed = parse(response.body)
    except MalformedResponseError:
        return failure("failed", "malformed_response", requests=1)

    jobs = dedupe(parsed.jobs)
    if parsed.unidentified:
        return FetchResult(
            status="incomplete",
            jobs=jobs,
            expected_total=parsed.expected_total,
            error_code="unidentifiable_job",
            requests=1,
        )
    if parsed.expected_total is not None and len(jobs) != parsed.expected_total:
        return FetchResult(
            status="incomplete",
            jobs=jobs,
            expected_total=parsed.expected_total,
            error_code="count_mismatch",
            requests=1,
        )
    return FetchResult(
        status="complete", jobs=jobs, expected_total=parsed.expected_total, requests=1
    )
