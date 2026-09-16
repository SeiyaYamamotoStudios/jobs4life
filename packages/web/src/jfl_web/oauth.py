"""Google sign-in: authorization code flow with PKCE, via Authlib.

Behind a small interface for two reasons. Authlib ships no type information, so
the untyped surface stops at this module; and the callback route is worth testing
without a live Google, which a stub implementation makes possible.

**Scopes are exactly `openid email profile`.** No Gmail scope, ever -- restricted
scopes require an annual CASA security assessment, and email intake is a
dedicated forwarding address instead (CLAUDE.md, architectural constraints).

**PKCE is on**: `code_challenge_method=S256`. The flow is server-side and holds a
client secret, so PKCE is not strictly required here; it costs one parameter and
removes an entire class of authorization-code interception, including the one
that matters in practice -- a code leaking through a redirect, a referer header,
or a proxy log.

**Identity is the `sub` claim.** Email is read for display and refreshed on every
login. It is never a lookup key: addresses are reassigned, in a workspace to
different people, and matching on one would eventually hand somebody another
person's account.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import Request, Response

from jfl_web.settings import GOOGLE_SCOPES, WebSettings

GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"


class OAuthError(RuntimeError):
    """The sign-in did not complete. Message is safe to show a user."""


@dataclass(frozen=True, slots=True)
class GoogleIdentity:
    sub: str
    email: str
    display_name: str | None


class GoogleIdentityProvider(Protocol):
    """What the auth routes need. Two calls, one per leg of the flow."""

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response: ...

    async def fetch_identity(self, request: Request) -> GoogleIdentity: ...


class AuthlibGoogleProvider:
    """The real thing. Authlib keeps the OAuth `state` and the PKCE verifier in
    `request.session` -- the short-lived signed cookie installed by
    `SessionMiddleware` in `app.py`, not the Postgres session.
    """

    def __init__(self, settings: WebSettings) -> None:
        # Imported here rather than at module scope so that a stub provider (and
        # therefore most of the test suite) does not need Authlib's import chain.
        from authlib.integrations.starlette_client import OAuth

        oauth = OAuth()
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            server_metadata_url=GOOGLE_METADATA_URL,
            client_kwargs={
                "scope": GOOGLE_SCOPES,
                "code_challenge_method": "S256",  # PKCE
            },
        )
        self._client: Any = oauth.google

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        response: Response = await self._client.authorize_redirect(request, redirect_uri)
        return response

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        # AuthlibBaseError, not BaseAppError: the latter does not exist in
        # Authlib 1.8 and the import sat inside this function, so it raised only
        # on a real token exchange -- every test stubs the provider out, and
        # mypy does not resolve names through an untyped dependency. The first
        # thing to exercise it was a live Google sign-in returning a 500.
        from authlib.integrations.base_client.errors import AuthlibBaseError

        try:
            token = await self._client.authorize_access_token(request)
        except AuthlibBaseError as exc:
            # Authlib's message can quote provider error codes but never the
            # client secret or the code; still, do not chain -- a traceback from
            # inside the client has the token exchange's request in scope.
            raise OAuthError(
                f"Google sign-in failed: {getattr(exc, 'error', None) or 'unknown error'}"
            ) from None

        claims = token.get("userinfo") or {}
        return _identity_from_claims(claims)


def _identity_from_claims(claims: dict[str, Any]) -> GoogleIdentity:
    sub = claims.get("sub")
    email = claims.get("email")
    if not sub or not email:
        raise OAuthError("Google did not return an identity for this account.")
    if not claims.get("email_verified", False):
        # Unverified addresses can be claimed by someone who does not control
        # them. Identity here is the `sub`, so this is not an account-takeover
        # risk, but the address is still shown back to the user on every page
        # as theirs -- refuse rather than display one they do not control.
        raise OAuthError("This Google account's email address is not verified.")
    name = claims.get("name")
    return GoogleIdentity(
        sub=str(sub),
        email=str(email),
        display_name=str(name) if name else None,
    )
