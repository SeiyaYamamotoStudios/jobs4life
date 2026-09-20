"""The FastAPI application.

Server-rendered, no SPA, no JS build step, no CDN. Every asset is a file under
`static/`, which is also what keeps the page working with no third-party
requests at all.

Two cookies, and they do different jobs:

  * `__Host-jfl_session` -- an opaque id; the session itself is a Postgres row,
    so logging out revokes it server-side.
  * `__Host-jfl_oauth`   -- Starlette's signed `SessionMiddleware` cookie, alive
    only for the seconds between the redirect to Google and the callback. It
    holds the OAuth `state` and the PKCE verifier, which is where Authlib puts
    them. Short `max_age`, and the callback deletes it.

Configuration is read exactly once, at startup, by `WebSettings.from_env`. A
missing `JFL_MASTER_KEY` stops the process there rather than at the first user
who tries to save a key -- and the key is never generated as a fallback, because
a fresh one would silently orphan every credential already stored.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine
from starlette.middleware.sessions import SessionMiddleware

from jfl_web.deps import CsrfFailed, NotAuthenticated
from jfl_web.oauth import AuthlibGoogleProvider, GoogleIdentityProvider
from jfl_web.routes import (
    applications,
    auth,
    boards,
    changes,
    drafts,
    jobs,
    pages,
    profile,
    title_suggestions,
    tracking,
)
from jfl_web.settings import WebSettings
from jfl_web.templating import STATIC_DIR, render

log = logging.getLogger(__name__)

# The OAuth transaction is a round trip through Google, not a session. Five
# minutes is generous for it and short enough that a stale `state` cannot be
# replayed a day later.
_OAUTH_COOKIE_MAX_AGE = 300


def create_app(
    settings: WebSettings | None = None,
    *,
    identity_provider: GoogleIdentityProvider | None = None,
) -> FastAPI:
    """Build the app. Both arguments are injection points for tests; in
    deployment neither is passed and both come from the environment.
    """
    resolved = settings if settings is not None else WebSettings.from_env()

    if resolved.insecure_cookies:
        log.warning(
            "JFL_INSECURE_COOKIES is set: cookies are being issued without the "
            "Secure flag and without the __Host- prefix. Local development only."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(resolved.database_url, pool_pre_ping=True)
        app.state.engine = engine
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="jobs4life", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = resolved
    app.state.identity_provider = (
        identity_provider if identity_provider is not None else AuthlibGoogleProvider(resolved)
    )

    app.add_middleware(
        SessionMiddleware,
        secret_key=resolved.oauth_state_secret,
        session_cookie=resolved.oauth_cookie_name,
        max_age=_OAUTH_COOKIE_MAX_AGE,
        same_site="lax",
        https_only=not resolved.insecure_cookies,
        path="/",
    )

    app.include_router(auth.router)
    app.include_router(pages.router)
    app.include_router(applications.router)
    app.include_router(drafts.router)
    app.include_router(boards.router)
    app.include_router(jobs.router)
    app.include_router(changes.router)
    app.include_router(title_suggestions.router)
    app.include_router(tracking.router)
    app.include_router(profile.router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    _install_error_handlers(app)
    return app


def _install_error_handlers(app: FastAPI) -> None:
    async def not_authenticated(request: Request, exc: Exception) -> Response:
        """A signed-out browser gets the login page, not a JSON 401 body."""
        return RedirectResponse("/login", status_code=303)

    async def csrf_failed(request: Request, exc: Exception) -> Response:
        return render(request, "error.html", {"message": _CSRF_MESSAGE}, status_code=403)

    async def unhandled_exception(request: Request, exc: Exception) -> Response:
        """Everything else. Nothing about the failure -- type, message, module
        path, traceback -- reaches the browser; only a short reference does.

        This has to hold for unauthenticated routes too: the failure that
        motivated it was on `/auth/google/callback`, which runs before a
        session exists, and a stack through `envelope.py` named the crypto
        layer to whoever triggered it. Registering a handler for `Exception`
        (rather than `HTTPException` or a status code) hooks Starlette's
        `ServerErrorMiddleware`, which sits outside routing -- so this also
        covers a failure in middleware, not only inside a route -- and it
        never sees an `HTTPException`: those are dispatched by the separate,
        inner `ExceptionMiddleware` and reach FastAPI's own 404/redirect
        handling exactly as before.

        `ServerErrorMiddleware` re-raises after sending this response, which
        is by design (see its docstring) so a hosting server can log or a test
        client can raise -- in deployment this means uvicorn's own error log
        also sees the exception, without a reference, in addition to the line
        below. That is redundant but harmless: both land in the same stdout
        stream, and only the line below carries the reference tying it to what
        the user was told.
        """
        reference = secrets.token_hex(4)
        log.error(
            "unhandled exception (reference=%s, path=%s)",
            reference,
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return render(
            request,
            "error.html",
            {"message": _unhandled_message(reference)},
            status_code=500,
        )

    handlers: dict[type[Exception], Callable[[Request, Exception], object]] = {
        NotAuthenticated: not_authenticated,
        CsrfFailed: csrf_failed,
        Exception: unhandled_exception,
    }
    for exc_type, handler in handlers.items():
        app.add_exception_handler(exc_type, handler)  # type: ignore[arg-type]


_CSRF_MESSAGE = (
    "That form could not be verified -- it was probably left open too long. "
    "Reload the page and try again."
)


def _unhandled_message(reference: str) -> str:
    return (
        "Something went wrong on our end -- nothing you did caused this. "
        f"If it keeps happening, quote this reference: {reference}."
    )
