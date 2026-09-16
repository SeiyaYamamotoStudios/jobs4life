"""Unit tests for app-level exception handling.

No database, no network: `create_app` never touches Postgres at construction
(the engine SQLAlchemy builds is lazy), and the identity provider is a stub so
Authlib's client is never built either.
"""

from __future__ import annotations

import logging
import re

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings

_REFERENCE = re.compile(r"quote this reference: ([0-9a-f]{8})\.")


class _StubGoogle:
    """Never called in these tests -- just enough to satisfy `create_app` so it
    does not build a real `AuthlibGoogleProvider`.
    """

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        raise NotImplementedError

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        raise NotImplementedError


def _client(settings: WebSettings) -> TestClient:
    app = create_app(settings, identity_provider=_StubGoogle())

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("a very specific and secret failure detail")

    # Starlette's ServerErrorMiddleware always re-raises after handing our
    # response to `send` (see app.py's docstring on `unhandled_exception`) --
    # raise_server_exceptions=False is what lets the test see that response
    # instead of the exception propagating into the test itself.
    return TestClient(app, raise_server_exceptions=False)


def test_an_unauthenticated_route_that_raises_returns_500_with_a_reference(
    settings: WebSettings, caplog: pytest.LogCaptureFixture
) -> None:
    """`/boom` here stands in for `/auth/google/callback`, which is not behind
    authentication -- the route needs no session, no cookie, nothing.
    """
    with caplog.at_level(logging.ERROR, logger="jfl_web.app"), _client(settings) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    match = _REFERENCE.search(response.text)
    assert match is not None, response.text
    reference = match.group(1)

    # Nothing about the failure itself reaches the body.
    assert "RuntimeError" not in response.text
    assert "a very specific and secret failure detail" not in response.text
    assert "Traceback" not in response.text
    assert "app.py" not in response.text
    assert "boom" not in response.text

    # The server log carries the same reference, next to the full traceback.
    assert reference in caplog.text
    assert "unhandled exception" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert "RuntimeError: a very specific and secret failure detail" in caplog.text
    assert "/boom" in caplog.text


def test_404_is_not_swallowed_by_the_new_handler(settings: WebSettings) -> None:
    """The `Exception` handler hooks `ServerErrorMiddleware`, which sits outside
    routing -- an `HTTPException` (404 here) is dispatched by the separate,
    inner `ExceptionMiddleware` and never reaches it. FastAPI's default JSON
    404 body proves that.
    """
    with _client(settings) as client:
        response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
