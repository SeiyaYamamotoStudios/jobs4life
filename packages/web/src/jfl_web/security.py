"""Session tokens, cookies and CSRF primitives. No storage, no framework state.

Three decisions live here.

**The session id is opaque and random, not a JWT.** This app holds other people's
Anthropic API keys, so revocation has to take effect on the next request. A
signed self-contained token cannot promise that without a revocation list, at
which point it is a database lookup wearing a costume.

**Postgres stores sha256 of the cookie value, never the value.** A database read
-- a backup, a dump, an over-broad SELECT in a debugging session -- then yields
nothing anyone can log in with. The hash is unsalted and uniterated on purpose:
the input is 256 bits of `secrets.token_urlsafe` entropy, so there is nothing to
brute force and a KDF here would only add latency to every request.

**The cookie carries the `__Host-` prefix.** The browser enforces what the
prefix promises: Secure, Path=/, and no Domain attribute -- so a subdomain, even
a compromised one, cannot write a cookie this app will read.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Response

# 32 bytes -> 43 url-safe characters. Well past any brute-force reach.
_TOKEN_BYTES = 32


def new_session_token() -> str:
    """The value that goes in the cookie. Never stored anywhere."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def new_csrf_token() -> str:
    """Per-session, stored in the clear because it must be rendered into a form.

    On its own it authenticates nothing: a forged request also needs the session
    cookie, which the same-origin rules and `SameSite=Lax` are what stop an
    attacker from getting.
    """
    return secrets.token_urlsafe(_TOKEN_BYTES)


def hash_token(token: str) -> str:
    """What Postgres stores. See the module docstring for why this is bare sha256."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def csrf_ok(submitted: str | None, expected: str) -> bool:
    """Constant-time comparison; a `==` here is a timing oracle on the token."""
    if not submitted:
        return False
    return secrets.compare_digest(submitted, expected)


def set_session_cookie(
    response: Response, *, name: str, token: str, max_age_seconds: int, insecure: bool
) -> None:
    """`__Host-`-compatible attributes: Secure, Path=/, no Domain.

    `SameSite=Lax` rather than Strict: Strict would drop the cookie on the
    redirect back from Google and the user would land logged out on the page
    that just logged them in.
    """
    response.set_cookie(
        key=name,
        value=token,
        max_age=max_age_seconds,
        httponly=True,
        secure=not insecure,
        samesite="lax",
        path="/",
        # domain deliberately unset -- `__Host-` forbids it, and setting one
        # would share the cookie with every subdomain.
    )


def clear_session_cookie(response: Response, *, name: str, insecure: bool) -> None:
    response.delete_cookie(key=name, httponly=True, secure=not insecure, samesite="lax", path="/")
