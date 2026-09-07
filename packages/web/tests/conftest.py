"""Fixtures for the web package's unit tests. No database, no network."""

from __future__ import annotations

import datetime as dt

import pytest
from jfl_core.crypto.envelope import MasterKey
from jfl_web.settings import WebSettings


@pytest.fixture
def settings() -> WebSettings:
    """Deployment-shaped settings: secure cookies, validation off.

    Validation is off because switching it on would have the settings form
    construct a real Anthropic client, which the root conftest guard blocks --
    correctly. The e2e test is the only place it runs.
    """
    return WebSettings(
        database_url="postgresql+psycopg://unused/unused",
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
        google_redirect_uri="https://example.invalid/auth/google/callback",
        oauth_state_secret="0" * 43,
        master_key=MasterKey.generate(),
        session_ttl=dt.timedelta(days=14),
        session_touch_after=dt.timedelta(minutes=5),
        insecure_cookies=False,
        validate_api_keys=False,
    )
