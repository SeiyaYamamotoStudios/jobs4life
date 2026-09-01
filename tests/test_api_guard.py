"""The guard in conftest.py is the thing standing between a stray test and a bill,
so it gets tested like any other load-bearing code.

These run in the default suite: they must never need the network, a database, or an
API key -- they assert that the *block* is in place, never that a call succeeds.
"""

from __future__ import annotations

import anthropic
import pytest

from conftest import ALLOW_REAL_API_ENV, RealApiCallBlocked


def test_constructing_a_real_client_is_blocked_in_an_unmarked_test() -> None:
    with pytest.raises(RealApiCallBlocked) as excinfo:
        anthropic.Anthropic(api_key="sk-ant-not-a-real-key")
    assert "not marked `e2e`" in str(excinfo.value)


def test_the_async_client_is_blocked_too() -> None:
    with pytest.raises(RealApiCallBlocked):
        anthropic.AsyncAnthropic(api_key="sk-ant-not-a-real-key")


def test_a_bare_client_is_blocked_so_an_ambient_profile_cannot_be_used() -> None:
    """The gate constructs `anthropic.Anthropic()` with no key when the context
    carries none, so it resolves an `ant auth login` profile. That path bills just
    as a key does, and the guard must cover it.
    """
    with pytest.raises(RealApiCallBlocked):
        anthropic.Anthropic()


def test_a_test_installing_its_own_fake_client_is_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not get in the way of the normal pattern: the autouse fixture
    installs the block first, then a test's own monkeypatch replaces it with a double.
    """

    class _Fake:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(anthropic, "Anthropic", _Fake)
    client = anthropic.Anthropic(api_key="test-key")
    assert isinstance(client, _Fake)


def test_the_opt_in_env_var_alone_does_not_unblock_an_unmarked_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both conditions are required. Setting the opt-in here has no effect, because
    the fixture already ran for this (unmarked) test and installed the block --
    which is exactly the property being asserted: an exported opt-in on a developer's
    machine cannot turn an unmarked test into a billable one.
    """
    monkeypatch.setenv(ALLOW_REAL_API_ENV, "1")
    with pytest.raises(RealApiCallBlocked):
        anthropic.Anthropic()
