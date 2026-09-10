"""The one seam between the adapters and the network.

Adapters speak to a `Transport`, never to httpx. That is what keeps every
default test off the network: an ATS API is flaky, rate-limited and not ours to
hammer, so tests hand an adapter a fake transport replaying recorded responses,
and only `HttpxTransport` ever opens a socket.

A transport returns an `HttpResponse` for every HTTP status, including 4xx and
5xx -- deciding what a status *means* for a check is the adapter's job, because
it differs by platform. It raises `TransportError` only when there is no HTTP
response at all (timeout, refused connection, DNS), and `RequestBudgetExceeded`
when the politeness limits below say stop.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

TransportErrorCode = Literal["timeout", "connection_error"]
BudgetErrorCode = Literal["request_budget_exhausted", "deadline_exceeded"]

USER_AGENT = "jobs4life-board-check/0.1 (+https://jobs4life.hiltonlabs.org)"


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    # Parsed JSON, or None when the body was not JSON. An adapter treats None on
    # a 200 as a malformed response, never as an empty board.
    body: Any


class TransportError(Exception):
    """No HTTP response was received. Carries a code, never the exception text:
    the code is what reaches `board_checks.error_code`.
    """

    def __init__(self, code: TransportErrorCode) -> None:
        super().__init__(code)
        self.code: TransportErrorCode = code


class RequestBudgetExceeded(Exception):
    def __init__(self, code: BudgetErrorCode) -> None:
        super().__init__(code)
        self.code: BudgetErrorCode = code


class Transport(Protocol):
    def get_json(self, url: str) -> HttpResponse: ...

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse: ...


def _parse(response: httpx.Response) -> HttpResponse:
    try:
        body = response.json()
    except ValueError:
        body = None
    return HttpResponse(status=response.status_code, body=body)


class HttpxTransport:
    """The real one. Redirects are followed (boards move between hostnames);
    everything else about a response is handed back untouched.
    """

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def get_json(self, url: str) -> HttpResponse:
        return self._send(lambda: self._client.get(url))

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        return self._send(lambda: self._client.post(url, json=dict(body)))

    @staticmethod
    def _send(call: Callable[[], httpx.Response]) -> HttpResponse:
        try:
            return _parse(call())
        except httpx.TimeoutException:
            raise TransportError("timeout") from None
        except httpx.TransportError:
            # Connection refused, DNS failure, TLS failure, a dropped connection.
            raise TransportError("connection_error") from None


@contextmanager
def httpx_transport(timeout_seconds: float = 30.0) -> Iterator[HttpxTransport]:
    """One client per check: connections are reused across a Workday check's
    hundred-odd requests and closed when it ends.
    """
    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    ) as client:
        yield HttpxTransport(client)


@dataclass(frozen=True, slots=True)
class RequestBudget:
    """Politeness limits for one check.

    Defaults, and why:

      * `delay_seconds=0.5` between requests to the same board. A one-request
        board (Greenhouse, Ashby, Lever) never waits. A Workday board of 2,630
        jobs is ~150 requests, so this adds ~75 seconds -- slow enough that one
        check cannot look like a burst, fast enough to finish well inside the
        worker's 15-minute visibility timeout;
      * `max_requests=300`: twice that 2,630-job board, so a tenant up to about
        5,000 jobs checks completely, and a runaway loop stops at a bounded cost.
        Hitting it makes the check `incomplete`, never `complete`;
      * `deadline_seconds=600`: ten minutes of wall clock. Also under the
        visibility timeout, so a slow check fails as `incomplete` rather than
        being reclaimed and run twice.
    """

    delay_seconds: float = 0.5
    max_requests: int = 300
    deadline_seconds: float = 600.0


class PoliteTransport:
    """Wraps a transport with a delay between requests, a request budget and a
    deadline. `sleep` and `clock` are injectable so tests neither wait nor race.
    """

    def __init__(
        self,
        inner: Transport,
        budget: RequestBudget,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._sleep = sleep
        self._clock = clock
        self._started = clock()
        self.requests = 0

    def get_json(self, url: str) -> HttpResponse:
        self._before_request()
        return self._inner.get_json(url)

    def post_json(self, url: str, body: Mapping[str, Any]) -> HttpResponse:
        self._before_request()
        return self._inner.post_json(url, body)

    def _before_request(self) -> None:
        if self.requests >= self._budget.max_requests:
            raise RequestBudgetExceeded("request_budget_exhausted")
        if self._clock() - self._started > self._budget.deadline_seconds:
            raise RequestBudgetExceeded("deadline_exceeded")
        if self.requests > 0 and self._budget.delay_seconds > 0:
            self._sleep(self._budget.delay_seconds)
        self.requests += 1
