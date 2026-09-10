"""One adapter per ATS, looked up by platform.

Same shape as the worker's handler registry, for the same reason: an explicit
list someone can read, no discovery, and a duplicate is an error rather than a
silent override.
"""

from __future__ import annotations

from collections.abc import Iterable

from jfl_intake.adapters.ashby import AshbyAdapter
from jfl_intake.adapters.base import BoardAdapter, FetchResult, InvalidBoardKeyError
from jfl_intake.adapters.greenhouse import GreenhouseAdapter
from jfl_intake.adapters.lever import LeverAdapter
from jfl_intake.adapters.workday import WorkdayAdapter

__all__ = [
    "AdapterRegistry",
    "AshbyAdapter",
    "BoardAdapter",
    "FetchResult",
    "GreenhouseAdapter",
    "InvalidBoardKeyError",
    "LeverAdapter",
    "UnsupportedPlatformError",
    "WorkdayAdapter",
    "default_registry",
]


class UnsupportedPlatformError(LookupError):
    """No adapter for this platform. Permanent for the task that hit it."""


class AdapterRegistry:
    def __init__(self, adapters: Iterable[BoardAdapter]) -> None:
        self._adapters: dict[str, BoardAdapter] = {}
        for adapter in adapters:
            if adapter.platform in self._adapters:
                raise ValueError(f"two adapters registered for {adapter.platform!r}")
            self._adapters[adapter.platform] = adapter

    def get(self, platform: str) -> BoardAdapter:
        adapter = self._adapters.get(platform)
        if adapter is None:
            raise UnsupportedPlatformError(f"no adapter for platform {platform!r}")
        return adapter

    def platforms(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


def default_registry() -> AdapterRegistry:
    return AdapterRegistry([GreenhouseAdapter(), AshbyAdapter(), LeverAdapter(), WorkdayAdapter()])
