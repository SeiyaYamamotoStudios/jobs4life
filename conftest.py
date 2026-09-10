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

**No test reaches the network by accident either.** Watched job boards call public
ATS APIs that are flaky, rate-limited and not ours to hammer, and a test that
dispatches the real `check_board` handler against a database left dirty by a
crashed run could otherwise make a live request. The second guard is at the socket,
for the same reason the first is at the client rather than the selector: it does
not depend on anyone having wired a fake in correctly, and it covers every HTTP
library -- httpx, urllib, anything a future adapter or SDK brings.

  * `socket.socket.connect` / `connect_ex` refuse any IPv4/IPv6 address that is not
    loopback, and `socket.getaddrinfo` refuses to resolve any name that is not
    `localhost`, so a request fails before DNS rather than after;
  * loopback and Unix sockets are allowed, so Postgres on `localhost:5433` and any
    local test server keep working (libpq connects from C and never passes through
    here anyway);
  * `httpx.MockTransport`, FastAPI's `TestClient` and every fake transport never
    open a socket, so they are untouched.

It is lifted under exactly the Anthropic guard's two conditions -- an `e2e` test
with `JFL_ALLOW_REAL_API=1` -- because a real API call is a real network call.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any, NoReturn

import anthropic
import pytest

ALLOW_REAL_API_ENV = "JFL_ALLOW_REAL_API"


class RealApiCallBlocked(RuntimeError):
    """Raised when a test tries to construct a real Anthropic client."""


class RealNetworkBlocked(RuntimeError):
    """Raised when a test tries to open a network connection off this machine."""


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


def is_local_host(host: object) -> bool:
    """True for names and addresses that cannot leave this machine."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if not isinstance(host, str):
        return False
    name = host.split("%", 1)[0].strip("[]").lower()
    if name in ("", "localhost") or name.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _is_local_address(family: int, address: Any) -> bool:
    if family not in (socket.AF_INET, socket.AF_INET6):
        return True  # Unix sockets and the like never leave the machine
    return isinstance(address, tuple) and bool(address) and is_local_host(address[0])


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    is_e2e = request.node.get_closest_marker("e2e") is not None
    if is_e2e and os.environ.get(ALLOW_REAL_API_ENV) == "1":
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def refuse(target: object) -> NoReturn:
        raise RealNetworkBlocked(
            f"blocked a real network connection in a test (to {target!r}). Tests must use "
            "a fake transport or httpx.MockTransport; only loopback is reachable. A test "
            f"that genuinely needs the network is `e2e` and runs with {ALLOW_REAL_API_ENV}=1."
        )

    def connect(self: socket.socket, address: Any) -> None:
        if not _is_local_address(self.family, address):
            refuse(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_local_address(self.family, address):
            refuse(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_local_host(host):
            refuse(host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
