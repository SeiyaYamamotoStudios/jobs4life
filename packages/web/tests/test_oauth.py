"""Claim handling. Identity is the `sub`; everything else is presentation."""

from __future__ import annotations

import pytest
from jfl_web.oauth import OAuthError, _identity_from_claims


def test_takes_sub_as_the_identity() -> None:
    identity = _identity_from_claims(
        {"sub": "1078...", "email": "a@example.com", "email_verified": True, "name": "A Person"}
    )
    assert identity.sub == "1078..."
    assert identity.email == "a@example.com"
    assert identity.display_name == "A Person"


def test_a_missing_name_is_not_fatal() -> None:
    identity = _identity_from_claims({"sub": "s", "email": "a@example.com", "email_verified": True})
    assert identity.display_name is None


def test_an_unverified_email_is_refused() -> None:
    """`users.email` is unique, so an unverified address could squat one its real
    owner needs. Identity is the `sub`, so this is not takeover -- but it is a
    denial of service worth one line of code to prevent.
    """
    with pytest.raises(OAuthError, match="not verified"):
        _identity_from_claims({"sub": "s", "email": "a@example.com", "email_verified": False})


@pytest.mark.parametrize(
    "claims",
    [{}, {"sub": "s"}, {"email": "a@example.com", "email_verified": True}],
    ids=["empty", "no-email", "no-sub"],
)
def test_incomplete_claims_are_refused(claims: dict[str, object]) -> None:
    with pytest.raises(OAuthError):
        _identity_from_claims(claims)


class TestAuthlibErrorSurface:
    """The names this module imports out of Authlib must actually exist.

    `fetch_identity` imports its exception class lazily -- deliberately, so the
    test suite does not need Authlib's import chain -- which means a wrong name
    raises only during a real token exchange. It did: `BaseAppError` does not
    exist in Authlib 1.8, mypy does not resolve names through an untyped
    dependency, and every other test stubs the provider out, so the first thing
    to notice was a live Google sign-in returning 500. This test closes that gap
    by checking the contract against the installed Authlib rather than a stub.
    """

    def test_the_base_error_class_is_importable(self) -> None:
        from authlib.integrations.base_client.errors import AuthlibBaseError

        assert issubclass(AuthlibBaseError, Exception)

    def test_the_error_carries_the_attribute_the_handler_reads(self) -> None:
        from authlib.integrations.base_client.errors import AuthlibBaseError

        assert getattr(AuthlibBaseError(error="boom"), "error", None) == "boom"
