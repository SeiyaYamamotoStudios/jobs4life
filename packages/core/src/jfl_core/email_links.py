"""Signed tokens for the links in an email, which carry no login.

One purpose today: **unsubscribe**. The link in every digest has to work for a
person who is not signed in -- on a phone, in a mail client, a year later -- and
the `List-Unsubscribe-Post` header (RFC 8058) has a mail client POST to it with no
cookie at all. So the link authenticates itself: the token is the user id, a
purpose string and an HMAC-SHA256 over both under `JFL_EMAIL_LINK_SECRET`.

**Decisions.**

  * *The purpose is inside the MAC.* A token minted for one job can never be
    replayed for another that is added later, and the secret is dedicated to
    email links rather than shared with the OAuth state secret, so rotating one
    does not break the other.
  * *No expiry.* An unsubscribe link that stops working after a week is how mail
    gets reported as spam. The only thing the token can do is turn email off, for
    the one user named in it; leaking one costs its owner nothing they cannot
    re-enable from settings.
  * *Rotating the secret revokes every link already sent*, which is the lever if
    one ever matters. Old emails then fail with "this link is no longer valid" and
    the person can still switch the email off from Settings.
  * *Constant-time comparison*, and a malformed token is the same answer as a
    forged one: `None`.

Pure: no storage, no clock, no framework.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import uuid

UNSUBSCRIBE = "unsubscribe"
_VERSION = "v1"


def _mac(secret: str, purpose: str, user_id: uuid.UUID) -> bytes:
    message = f"{_VERSION}:{purpose}:{user_id}".encode()
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def sign_link_token(secret: str, purpose: str, user_id: uuid.UUID) -> str:
    if not secret:
        raise ValueError("an email link secret is required to sign a token")
    return f"{user_id.hex}.{_b64(_mac(secret, purpose, user_id))}"


def verify_link_token(secret: str, purpose: str, token: str) -> uuid.UUID | None:
    """The user the token was signed for, or None for anything else: forged,
    tampered, signed for another purpose or under another secret, or malformed.
    """
    if not secret or len(token) > 200:
        return None
    ident, dot, signature = token.partition(".")
    if not dot:
        return None
    try:
        user_id = uuid.UUID(hex=ident)
    except ValueError:
        return None
    if user_id.hex != ident:
        return None
    expected = _b64(_mac(secret, purpose, user_id))
    if not hmac.compare_digest(expected.encode("ascii"), signature.encode("utf-8", "replace")):
        return None
    return user_id
