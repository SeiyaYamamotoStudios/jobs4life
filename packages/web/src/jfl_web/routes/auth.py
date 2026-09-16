"""Sign in, sign out. Nothing else belongs in here.

The flow, in order:

  GET  /login                  -- a page with one button
  GET  /auth/google            -- Authlib mints `state` + PKCE verifier into the
                                  signed OAuth cookie, then redirects to Google
  GET  /auth/google/callback   -- Authlib checks `state`, exchanges the code with
                                  the verifier, verifies the id token; we take
                                  `sub`, upsert the user, mint a session row
  POST /logout                 -- deletes the row, then clears the cookie

Logout is a POST with a CSRF token because a GET logout is forgeable from any
page on the internet -- an annoyance rather than a breach, but the cheapest thing
in the world to get right.

Failures come back as a fixed error *code* in the query string, never as a
message. A message parameter reflected into the page is a phishing primitive: any
attacker could send someone a real login URL that says whatever they liked above
a real Google button.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from jfl_web.deps import (
    CsrfDep,
    IdentityProviderDep,
    SessionDep,
    SessionRepoDep,
    SettingsDep,
    UserRepoDep,
)
from jfl_web.oauth import OAuthError
from jfl_web.security import (
    clear_session_cookie,
    hash_token,
    new_csrf_token,
    new_session_token,
    set_session_cookie,
)
from jfl_web.templating import render

log = logging.getLogger(__name__)

router = APIRouter()

LOGIN_ERRORS = {
    "google": "Google sign-in did not complete. Please try again.",
    "deactivated": "That account is deactivated.",
}


@router.get("/login")
def login_page(request: Request) -> Response:
    code = request.query_params.get("error")
    return render(request, "login.html", {"error": LOGIN_ERRORS.get(code or "")})


@router.get("/auth/google")
async def start_google_login(
    request: Request,
    settings: SettingsDep,
    provider: IdentityProviderDep,
) -> Response:
    return await provider.authorize_redirect(request, settings.google_redirect_uri)


@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    settings: SettingsDep,
    sessions: SessionRepoDep,
    users: UserRepoDep,
    provider: IdentityProviderDep,
) -> Response:
    try:
        identity = await provider.fetch_identity(request)
    except OAuthError as exc:
        # Safe to log: OAuth error text carries provider error codes, never the
        # client secret, the authorization code, or the id token.
        log.warning("google sign-in failed: %s", exc)
        return _login_error("google")

    account = users.upsert_google_user(
        google_sub=identity.sub,
        email=identity.email,
        display_name=identity.display_name,
    )

    if not account.is_active:
        return _login_error("deactivated")

    token = new_session_token()
    now = dt.datetime.now(dt.UTC)
    sessions.create(
        user_id=account.id,
        token_hash=hash_token(token),
        csrf_token=new_csrf_token(),
        expires_at=now + settings.session_ttl,
        user_agent=request.headers.get("user-agent"),
    )

    response = RedirectResponse("/", status_code=303)
    set_session_cookie(
        response,
        name=settings.session_cookie_name,
        token=token,
        max_age_seconds=int(settings.session_ttl.total_seconds()),
        insecure=settings.insecure_cookies,
    )
    # The OAuth transaction cookie has done its job; leaving it set would keep a
    # spent `state` and PKCE verifier sitting in the browser.
    response.delete_cookie(settings.oauth_cookie_name, path="/")
    return response


@router.post("/logout")
def logout(
    request: Request,
    settings: SettingsDep,
    sessions: SessionRepoDep,
    session: SessionDep,
    _csrf: CsrfDep,
) -> Response:
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        # Delete the row first. Clearing only the cookie would leave a working
        # session behind for anyone who had already copied the value.
        sessions.delete(hash_token(token))
    response = RedirectResponse("/login", status_code=303)
    clear_session_cookie(
        response, name=settings.session_cookie_name, insecure=settings.insecure_cookies
    )
    return response


def _login_error(code: str) -> RedirectResponse:
    return RedirectResponse(f"/login?error={code}", status_code=303)
