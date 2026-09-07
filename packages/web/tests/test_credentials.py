"""The write-only credential path, and the guard that stops it billing in tests."""

from __future__ import annotations

import pytest
from jfl_web.credentials import InvalidApiKeyError, normalise_submitted_key, validate_api_key

GOOD = "sk-ant-api03-0000000000000000000000000000000000004f2a"


def test_accepts_a_plausible_key() -> None:
    assert normalise_submitted_key(f"  {GOOD}\n") == GOOD


@pytest.mark.parametrize("bad", ["", "   ", "sk-ant short", "sk-ant"])
def test_rejects_the_common_bad_pastes(bad: str) -> None:
    with pytest.raises(InvalidApiKeyError):
        normalise_submitted_key(bad)


def test_rejection_messages_never_quote_the_key() -> None:
    with pytest.raises(InvalidApiKeyError) as exc:
        normalise_submitted_key("sk-ant with spaces in it")
    assert "sk-ant" not in str(exc.value)


def test_validation_cannot_reach_the_api_from_an_unmarked_test() -> None:
    """The root conftest guard, seen from this slice.

    `validate_api_key` builds a real client, which is exactly what the autouse
    fixture in the repository's root `conftest.py` replaces with something that
    raises. Both the `e2e` marker and JFL_ALLOW_REAL_API=1 are required, and this
    test has neither -- so the call must fail rather than bill.

    The guard's own class is not imported here: this directory has its own
    `conftest.py`, which shadows the root one for a plain `import conftest`.
    `RealApiCallBlocked` is a RuntimeError, and the message is asserted instead.
    """
    with pytest.raises(RuntimeError) as exc:
        validate_api_key(GOOD)
    assert "blocked a real Anthropic client" in str(exc.value)
