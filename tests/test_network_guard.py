"""The network guard in conftest.py, tested like the API guard beside it.

These run in the default suite and must never need the network: every assertion
is that a connection is *refused*, or that something local still works. If the
guard were broken, the refusal tests would fail on DNS or a timeout rather than
pass -- the targets are `.invalid` names and a documentation-range address.
"""

from __future__ import annotations

import socket
import threading
import urllib.request

import httpx
import pytest
from jfl_intake.adapters import GreenhouseAdapter, WorkdayAdapter
from jfl_worker.handlers.boards import polite_httpx_transport

from conftest import RealNetworkBlocked, is_local_host


def test_a_real_httpx_request_is_blocked() -> None:
    with pytest.raises(RealNetworkBlocked):
        httpx.get("https://boards-api.greenhouse.invalid/v1/boards/acme/jobs", timeout=2)


def test_the_production_board_transport_cannot_reach_a_board() -> None:
    """The case that motivated the guard: the real `check_board` transport,
    handed to a real adapter. It fails loudly -- not as a recorded `unreachable`
    check, which is what a caught `TransportError` would have produced.
    """
    with polite_httpx_transport()() as transport:
        with pytest.raises(RealNetworkBlocked):
            GreenhouseAdapter().fetch({"token": "anthropic"}, transport)
        with pytest.raises(RealNetworkBlocked):
            WorkdayAdapter().fetch(
                {"tenant": "nvidia", "wd": "wd5", "site": "NVIDIAExternalCareerSite"}, transport
            )


def test_urllib_is_blocked_too() -> None:
    with pytest.raises(RealNetworkBlocked):
        urllib.request.urlopen("https://jobs.lever.invalid/acme", timeout=2)


def test_a_raw_socket_to_a_public_address_is_blocked() -> None:
    with pytest.raises(RealNetworkBlocked):
        socket.create_connection(("192.0.2.1", 443), timeout=2)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(RealNetworkBlocked):
            sock.connect(("192.0.2.1", 443))
        with pytest.raises(RealNetworkBlocked):
            sock.connect_ex(("192.0.2.1", 443))


def test_loopback_is_still_reachable() -> None:
    """Postgres lives on localhost, and so does any local test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        accepted: list[bool] = []
        thread = threading.Thread(target=lambda: accepted.append(bool(server.accept())))
        thread.start()
        with socket.create_connection(("localhost", port), timeout=2):
            pass
        thread.join(timeout=2)
    assert accepted == [True]


def test_mock_transports_are_unaffected() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"jobs": []}))
    with httpx.Client(transport=transport) as client:
        assert client.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").json() == {
            "jobs": []
        }


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("::1", True),
        ("[::1]", True),
        (None, True),
        (b"localhost", True),
        ("boards-api.greenhouse.io", False),
        ("192.0.2.1", False),
        ("localhost.evil.example", False),
    ],
)
def test_what_counts_as_local(host: object, local: bool) -> None:
    assert is_local_host(host) is local
