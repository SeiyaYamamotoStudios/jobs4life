"""Fetch one job posting's full description, lazily.

Slice C7's "Track as application" button, and nothing else. Every other
watched-board path (the daily check, the listing pages) works from the thin
record an adapter's `fetch()` returns -- title, location, url. A description is
fetched only for the one job a user turns into an application, on their own
request, because the whole point of watching boards whole is that nothing
past the listing is fetched for a job the user merely browses.

**Placeholder.** This file is being built as its own slice, in parallel with
the worker handler and web route that call it. Every platform currently
answers `unsupported_platform` so callers see a permanent, honest failure
rather than a silent success -- see `jfl_worker.handlers.description`, which
treats that answer as "nothing to retry" and fails the application with
`description_unavailable`. The real per-platform fetch lands here later,
behind this exact interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jfl_core.models import BoardPlatform

from jfl_intake.http import Transport

# `unreachable` is the only transient one -- a timeout, a 5xx, a dropped
# connection -- and is what `DescriptionResult.is_transient` keys on. The rest
# are permanent: retrying an unsupported platform or a 404 spends nothing but
# time and, once a real fetch lands here, a request.
DescriptionErrorCode = Literal[
    "unsupported_platform", "not_found", "unreachable", "bad_response", "empty"
]


@dataclass(frozen=True, slots=True)
class DescriptionResult:
    """`text` is the posting's description, or None on any failure -- in which
    case `error_code` says why. Never both set, never both absent.
    """

    text: str | None
    error_code: DescriptionErrorCode | None
    requests: int = 0

    @property
    def is_transient(self) -> bool:
        """True only for `unreachable`. Everything else is a fact that will be
        the same on the next attempt, so a caller must not retry it.
        """
        return self.error_code == "unreachable"


def fetch_description(
    platform: BoardPlatform,
    board_key: Mapping[str, str],
    external_id: str,
    url: str | None,
    transport: Transport,
) -> DescriptionResult:
    """Fetch one posting's description. Placeholder: every platform is
    unsupported until the real per-platform fetch replaces this.
    """
    return DescriptionResult(text=None, error_code="unsupported_platform")
