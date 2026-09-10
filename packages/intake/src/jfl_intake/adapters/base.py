"""What every adapter returns, and the rules shared between them.

An adapter's job is narrow: fetch every listed job on one board, normalise each
into an `ObservedJob`, and say honestly **whether it saw the whole board**. It
never decides what the result means for the board's history -- that is
`jfl_intake.engine` -- and it never raises for anything the network or the ATS
did. A timeout, a 404, a page of the wrong shape: each is a `FetchResult` with a
status and a code, because each is a fact about this check worth recording.

The completeness verdict is the adapter's most important output. A result is
`complete` only when the adapter can show it saw everything: the single response
that is by definition the whole board, or a paged listing whose unique count
matched the total the board declared. Anything short of proof is `incomplete`
or `truncated`, and the engine then changes no job's state.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from jfl_core.models import BoardCheckErrorCode, BoardPlatform, FetchStatus, ObservedJob

from jfl_intake.http import HttpResponse, Transport


class InvalidBoardKeyError(ValueError):
    """The stored `board_key` is not one this adapter can use. Permanent: a key
    does not repair itself between attempts.
    """


@dataclass(frozen=True, slots=True)
class FetchResult:
    status: FetchStatus
    jobs: tuple[ObservedJob, ...] = ()
    expected_total: int | None = None
    error_code: BoardCheckErrorCode | None = None
    requests: int = 0
    # Free-form counters for the log line (pages fetched, facet used). Never
    # persisted; never carries anything from a response body.
    detail: Mapping[str, object] = field(default_factory=dict)

    @property
    def jobs_seen(self) -> int:
        return len(self.jobs)


class BoardAdapter(Protocol):
    platform: BoardPlatform

    def validate_key(self, board_key: Mapping[str, str]) -> None:
        """Raise `InvalidBoardKeyError` if this key cannot be fetched."""
        ...

    def fetch(self, board_key: Mapping[str, str], transport: Transport) -> FetchResult: ...


def status_failure(status: int) -> tuple[FetchStatus, BoardCheckErrorCode]:
    """What a non-200 HTTP status means for a check.

    `unreachable` is for trouble that plausibly clears by itself (429, 5xx) and
    rides the queue's retry; `failed` is for an answer that will be the same
    next time (404, other 4xx) and does not.
    """
    if status == 404:
        return "failed", "not_found"
    if status == 429:
        return "unreachable", "rate_limited"
    if status >= 500:
        return "unreachable", "server_error"
    return "failed", "http_client_error"


def failure(status: FetchStatus, code: BoardCheckErrorCode, *, requests: int = 0) -> FetchResult:
    return FetchResult(status=status, error_code=code, requests=requests)


def from_http_failure(response: HttpResponse, *, requests: int = 0) -> FetchResult:
    status, code = status_failure(response.status)
    return failure(status, code, requests=requests)


def dedupe(jobs: list[ObservedJob]) -> tuple[ObservedJob, ...]:
    """First occurrence of each external id wins, order kept."""
    seen: dict[str, ObservedJob] = {}
    for job in jobs:
        seen.setdefault(job.external_id, job)
    return tuple(seen.values())


# Every board-key part ends up as a URL path segment or hostname label, so it is
# held to a pattern that cannot smuggle a `/`, `?`, `@` or `..` into a request.
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def require_key(board_key: Mapping[str, str], *names: str) -> tuple[str, ...]:
    """The named parts of a stored board key, validated. Raises
    `InvalidBoardKeyError` -- with the part's name, never its value.
    """
    values: list[str] = []
    for name in names:
        value = board_key.get(name)
        if not isinstance(value, str) or not SAFE_SEGMENT.match(value) or ".." in value:
            raise InvalidBoardKeyError(f"board key part {name!r} is missing or invalid")
        values.append(value)
    return tuple(values)
