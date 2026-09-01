"""Repo-wide test guards.

**No test spends API credits by accident.** The marker exclusion in pyproject.toml
(`addopts = "-m 'not integration and not e2e'"`) is a default, and defaults lose: an
explicit `-m e2e`, an `--override-ini`, or a CI job that sets its own `addopts` all
walk straight past it. Worse, it protects nothing against the likeliest mistake --
a newly written test that constructs a real client and that nobody remembered to
mark.

So the guard is at the client instead of at the selector. Every test runs with
`anthropic.Anthropic` and `anthropic.AsyncAnthropic` replaced by something that
raises, and a real client requires **both** conditions to be true:

  * the test is marked `e2e`, and
  * `JFL_ALLOW_REAL_API=1` is set in the environment

Either alone is not enough. That way an unmarked test cannot bill even on a machine
where the opt-in is exported, and a marked test cannot bill on a machine where it
is not.

Tests that install their own fake client keep working untouched -- their
`monkeypatch.setattr` runs after this fixture and simply replaces the block with
their double.
"""

from __future__ import annotations

import os

import anthropic
import pytest

ALLOW_REAL_API_ENV = "JFL_ALLOW_REAL_API"


class RealApiCallBlocked(RuntimeError):
    """Raised when a test tries to construct a real Anthropic client."""


@pytest.fixture(autouse=True)
def _block_real_anthropic_clients(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    is_e2e = request.node.get_closest_marker("e2e") is not None
    opted_in = os.environ.get(ALLOW_REAL_API_ENV) == "1"
    if is_e2e and opted_in:
        return

    if is_e2e:
        detail = (
            f"this test is marked `e2e` but {ALLOW_REAL_API_ENV}=1 is not set. "
            f"Run it deliberately: {ALLOW_REAL_API_ENV}=1 uv run pytest -m e2e"
        )
    else:
        detail = (
            "this test is not marked `e2e`, so it must not reach the API at all. "
            "Install a fake client with monkeypatch, or mark the test `e2e` if it "
            "genuinely needs a real call."
        )

    def _blocked(*args: object, **kwargs: object) -> object:
        raise RealApiCallBlocked(f"blocked a real Anthropic client in a test: {detail}")

    monkeypatch.setattr(anthropic, "Anthropic", _blocked)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _blocked)
