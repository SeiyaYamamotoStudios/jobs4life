"""The emailed digest's controls (PLAN.md C9).

Screens:

  POST /settings/email/enable    -- opt in (CSRF, session)
  POST /settings/email/disable   -- opt out (CSRF, session)
  POST /settings/email/test      -- enqueue "Send a test email" (CSRF, session)
  GET  /email/unsubscribe/{token}  -- confirm page, no login
  POST /email/unsubscribe/{token}  -- unsubscribe, no login, no CSRF token

The toggle and the test button live on the existing settings page
(`routes/pages.py` renders the section); these are only the writes.

**Nothing is sent in a request.** "Send a test email" enqueues a `send_test_email`
task and redirects; the worker sends it. `enqueue_unique`, so pressing it five
times queues one.

**The unsubscribe link is the one place in the app that acts without a session or a
CSRF token, and it is scoped to exactly that.** The token is an HMAC over the user
id and the purpose (`jfl_core.email_links`), which is the authentication; the POST
route simply does not depend on `CsrfDep`, so the exemption cannot spread to any
other route by accident. Two routes, because link scanners and mail clients
prefetch GETs: **a GET never changes anything** -- it shows a page with a button --
and only the POST unsubscribes. The POST is also what a mail client sends for the
`List-Unsubscribe-Post: List-Unsubscribe=One-Click` header (RFC 8058), with a form
body this route ignores. Both answer the same for an unknown user as a known one,
so the link cannot be used to probe which ids exist, and both are served
`no-store` with `Referrer-Policy: no-referrer` because the token is in the URL.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.email_links import UNSUBSCRIBE, verify_link_token
from jfl_core.storage.digest import PostgresDigestRepository
from jfl_intake.digest import SEND_TEST_EMAIL_KIND

from jfl_web.deps import ConnDep, CsrfDep, DigestRepoDep, SessionDep, SettingsDep, TaskRepoDep
from jfl_web.templating import render

router = APIRouter()

_INVALID_LINK = (
    "This unsubscribe link is not valid any more. You can still switch the email "
    "off from Settings once you have signed in."
)


@router.post("/settings/email/enable")
def enable_digest(
    settings: SettingsDep, digest: DigestRepoDep, _csrf: CsrfDep, session: SessionDep
) -> Response:
    if not settings.mail_configured:
        return RedirectResponse("/settings?email=unavailable", status_code=303)
    digest.enable(now=dt.datetime.now(dt.UTC))
    return RedirectResponse("/settings?email=on", status_code=303)


@router.post("/settings/email/disable")
def disable_digest(digest: DigestRepoDep, _csrf: CsrfDep) -> Response:
    digest.disable()
    return RedirectResponse("/settings?email=off", status_code=303)


@router.post("/settings/email/test")
def send_test_email(
    settings: SettingsDep, tasks: TaskRepoDep, _csrf: CsrfDep, session: SessionDep
) -> Response:
    if not settings.mail_configured:
        return RedirectResponse("/settings?email=unavailable", status_code=303)
    tasks.enqueue_unique(kind=SEND_TEST_EMAIL_KIND, max_attempts=3)
    return RedirectResponse("/settings?email=test_queued", status_code=303)


def _private(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@router.get("/email/unsubscribe/{token}")
def unsubscribe_confirm(request: Request, token: str, settings: SettingsDep) -> Response:
    """Shows a button. Changes nothing -- see the module docstring."""
    secret = settings.email_link_secret
    if not secret or verify_link_token(secret, UNSUBSCRIBE, token) is None:
        return _private(render(request, "error.html", {"message": _INVALID_LINK}, status_code=400))
    return _private(render(request, "unsubscribe.html", {"token": token, "done": False}))


@router.post("/email/unsubscribe/{token}")
def unsubscribe(request: Request, token: str, settings: SettingsDep, conn: ConnDep) -> Response:
    """Authenticated by the token alone. No session, no CSRF token."""
    secret = settings.email_link_secret
    user_id = verify_link_token(secret, UNSUBSCRIBE, token) if secret else None
    if user_id is None:
        return _private(render(request, "error.html", {"message": _INVALID_LINK}, status_code=400))
    PostgresDigestRepository(conn, user_id).disable()
    return _private(render(request, "unsubscribe.html", {"token": token, "done": True}))
