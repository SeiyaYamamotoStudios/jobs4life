"""Configuration is read once, at the boundary, and fails loudly when incomplete."""

from __future__ import annotations

import base64
import os

import pytest
from jfl_core.crypto.envelope import MasterKey
from jfl_web.settings import GOOGLE_SCOPES, ConfigError, WebSettings

_COMPLETE = {
    "JFL_DATABASE_URL": "postgresql+psycopg://jfl:jfl@localhost:5433/jfl",
    "JFL_GOOGLE_CLIENT_ID": "id.apps.googleusercontent.com",
    "JFL_GOOGLE_CLIENT_SECRET": "secret",
    "JFL_GOOGLE_REDIRECT_URI": "https://jobs4life.example/auth/google/callback",
    "JFL_OAUTH_STATE_SECRET": "s" * 43,
    "JFL_MASTER_KEY": base64.b64encode(b"\x07" * 32).decode(),
}


def test_reads_a_complete_environment() -> None:
    settings = WebSettings.from_env(_COMPLETE)
    assert settings.google_client_id == _COMPLETE["JFL_GOOGLE_CLIENT_ID"]
    assert settings.master_key.key_id == MasterKey.from_base64(_COMPLETE["JFL_MASTER_KEY"]).key_id


def test_names_every_missing_variable_at_once() -> None:
    """An operator setting this up should see the whole list, not one per restart."""
    with pytest.raises(ConfigError) as exc:
        WebSettings.from_env({"JFL_DATABASE_URL": _COMPLETE["JFL_DATABASE_URL"]})
    message = str(exc.value)
    for name in _COMPLETE:
        if name != "JFL_DATABASE_URL":
            assert name in message


def test_a_missing_master_key_is_never_invented() -> None:
    """A silently generated KEK would decrypt nothing stored before the restart,
    and would fail at the point of *use* rather than the point of misconfiguration.
    """
    env = {k: v for k, v in _COMPLETE.items() if k != "JFL_MASTER_KEY"}
    with pytest.raises(ConfigError) as exc:
        WebSettings.from_env(env)
    assert "JFL_MASTER_KEY" in str(exc.value)


def test_a_malformed_master_key_stops_startup() -> None:
    with pytest.raises(ConfigError, match="base64"):
        WebSettings.from_env({**_COMPLETE, "JFL_MASTER_KEY": "@@@not base64@@@"})


def test_secrets_are_not_in_the_repr() -> None:
    """A settings object is exactly the sort of thing that lands in a log line."""
    settings = WebSettings.from_env(_COMPLETE)
    text = repr(settings)
    assert "secret" not in text
    assert _COMPLETE["JFL_OAUTH_STATE_SECRET"] not in text
    assert _COMPLETE["JFL_MASTER_KEY"] not in text
    assert base64.b64encode(settings.master_key.material).decode() not in text
    # The non-secret half is still there, or the repr would be useless.
    assert settings.google_client_id in text


def test_host_prefixed_cookies_by_default() -> None:
    settings = WebSettings.from_env(_COMPLETE)
    assert settings.session_cookie_name.startswith("__Host-")
    assert settings.oauth_cookie_name.startswith("__Host-")
    assert not settings.insecure_cookies


def test_insecure_mode_drops_the_prefix_it_cannot_honour() -> None:
    """`__Host-` requires Secure, which requires HTTPS. Keeping the prefix over
    plain http would produce a cookie the browser silently refuses.
    """
    settings = WebSettings.from_env({**_COMPLETE, "JFL_INSECURE_COOKIES": "1"})
    assert not settings.session_cookie_name.startswith("__Host-")


def test_key_validation_defaults_on_and_is_skippable() -> None:
    assert WebSettings.from_env(_COMPLETE).validate_api_keys is True
    assert (
        WebSettings.from_env({**_COMPLETE, "JFL_VALIDATE_API_KEYS": "0"}).validate_api_keys is False
    )


def test_ttl_must_be_a_positive_integer() -> None:
    with pytest.raises(ConfigError):
        WebSettings.from_env({**_COMPLETE, "JFL_SESSION_TTL_HOURS": "nonsense"})
    with pytest.raises(ConfigError):
        WebSettings.from_env({**_COMPLETE, "JFL_SESSION_TTL_HOURS": "0"})


def test_scopes_are_exactly_the_three_and_never_gmail() -> None:
    """Restricted scopes need an annual CASA assessment; email intake is a
    forwarding address instead. This is a standing architectural constraint.
    """
    assert GOOGLE_SCOPES.split() == ["openid", "email", "profile"]
    assert "gmail" not in GOOGLE_SCOPES


def test_the_environment_is_read_in_exactly_one_module() -> None:
    """`from_env` is the only place `jfl_web` touches os.environ."""
    import pathlib

    import jfl_web

    root = pathlib.Path(jfl_web.__file__).parent
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if path.name != "settings.py" and "os.environ" in path.read_text()
    ]
    assert not offenders, f"environment read outside settings.py: {offenders}"
    assert "os.environ" in (root / "settings.py").read_text()
    assert os.environ is not None  # the import above is the thing under test
