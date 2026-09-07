"""Session tokens and cookie attributes."""

from __future__ import annotations

from fastapi import Response
from jfl_web.security import (
    clear_session_cookie,
    csrf_ok,
    hash_token,
    new_csrf_token,
    new_session_token,
    set_session_cookie,
)


def test_tokens_are_long_and_unique() -> None:
    tokens = {new_session_token() for _ in range(500)}
    assert len(tokens) == 500
    assert all(len(t) >= 43 for t in tokens)


def test_the_stored_hash_is_not_the_token() -> None:
    """A database read must not yield a usable session."""
    token = new_session_token()
    digest = hash_token(token)
    assert token not in digest
    assert len(digest) == 64
    assert hash_token(token) == digest  # deterministic, or lookup would never match
    assert hash_token(new_session_token()) != digest


def test_csrf_comparison_rejects_the_obvious_things() -> None:
    token = new_csrf_token()
    assert csrf_ok(token, token)
    assert not csrf_ok(None, token)
    assert not csrf_ok("", token)
    assert not csrf_ok(token[:-1], token)
    assert not csrf_ok(new_csrf_token(), token)


def _cookie_header(response: Response) -> str:
    return dict(response.headers).get("set-cookie", "")


def test_session_cookie_carries_the_host_prefix_attributes() -> None:
    response = Response()
    set_session_cookie(
        response, name="__Host-jfl_session", token="abc", max_age_seconds=60, insecure=False
    )
    header = _cookie_header(response)
    assert "__Host-jfl_session=abc" in header
    assert "HttpOnly" in header
    assert "Secure" in header
    assert "SameSite=lax" in header
    assert "Path=/" in header
    assert "Domain" not in header  # `__Host-` forbids it


def test_insecure_mode_omits_secure_and_nothing_else() -> None:
    response = Response()
    set_session_cookie(
        response, name="jfl_session_dev", token="abc", max_age_seconds=60, insecure=True
    )
    header = _cookie_header(response)
    assert "Secure" not in header
    assert "HttpOnly" in header


def test_clearing_matches_the_attributes_it_set() -> None:
    """A delete-cookie whose attributes differ leaves the original in place."""
    response = Response()
    clear_session_cookie(response, name="__Host-jfl_session", insecure=False)
    header = _cookie_header(response)
    assert "__Host-jfl_session=" in header
    assert "Path=/" in header
    assert "Secure" in header
