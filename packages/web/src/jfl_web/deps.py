"""Request-scoped wiring: the transaction, the session, the tenant-scoped repos.

The important line in this file is `credential_repo`: it builds the repository
from the *session's* user id, and nothing downstream can name a different one.
That is where structural tenancy is actually applied -- the base class makes a
per-call override inexpressible, and this makes the constructor argument come
from the cookie rather than from anything the request can influence.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from jfl_core.storage.accounts import (
    AuthenticatedSession,
    PostgresSessionRepository,
    PostgresUserRepository,
)
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from sqlalchemy.engine import Connection

from jfl_web.oauth import GoogleIdentityProvider
from jfl_web.security import csrf_ok, hash_token
from jfl_web.settings import WebSettings


class NotAuthenticated(Exception):
    """No live session. Handled as a redirect to the login page, not a 401 body."""


class CsrfFailed(Exception):
    """A state-changing request arrived without a matching CSRF token."""


def get_settings(request: Request) -> WebSettings:
    settings: WebSettings = request.app.state.settings
    return settings


def get_identity_provider(request: Request) -> GoogleIdentityProvider:
    provider: GoogleIdentityProvider = request.app.state.identity_provider
    return provider


def db_conn(request: Request) -> Iterator[Connection]:
    """One transaction per request, committed on a clean return.

    `engine.begin()` rolls back if the handler raises, so a half-written login --
    a user row created but no session -- cannot survive an error.
    """
    with request.app.state.engine.begin() as conn:
        yield conn


ConnDep = Annotated[Connection, Depends(db_conn)]
SettingsDep = Annotated[WebSettings, Depends(get_settings)]


def user_repo(conn: ConnDep) -> PostgresUserRepository:
    return PostgresUserRepository(conn)


UserRepoDep = Annotated[PostgresUserRepository, Depends(user_repo)]
IdentityProviderDep = Annotated[GoogleIdentityProvider, Depends(get_identity_provider)]


def session_repo(conn: ConnDep) -> PostgresSessionRepository:
    return PostgresSessionRepository(conn)


SessionRepoDep = Annotated[PostgresSessionRepository, Depends(session_repo)]


def optional_session(
    request: Request, sessions: SessionRepoDep, settings: SettingsDep
) -> AuthenticatedSession | None:
    """Resolve the cookie, and roll the expiry forward if it is getting stale."""
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    now = dt.datetime.now(dt.UTC)
    token_hash = hash_token(token)
    session = sessions.lookup(token_hash, now=now)
    if session is None:
        return None
    if now - session.last_seen_at >= settings.session_touch_after:
        sessions.touch(token_hash, now=now, expires_at=now + settings.session_ttl)
    return session


OptionalSessionDep = Annotated[AuthenticatedSession | None, Depends(optional_session)]


def require_session(session: OptionalSessionDep) -> AuthenticatedSession:
    if session is None:
        raise NotAuthenticated
    return session


SessionDep = Annotated[AuthenticatedSession, Depends(require_session)]


def credential_repo(session: SessionDep, conn: ConnDep) -> PostgresCredentialRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresCredentialRepository(conn, session.user.id)


CredentialRepoDep = Annotated[PostgresCredentialRepository, Depends(credential_repo)]


def application_repo(session: SessionDep, conn: ConnDep) -> PostgresApplicationRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresApplicationRepository(conn, session.user.id)


ApplicationRepoDep = Annotated[PostgresApplicationRepository, Depends(application_repo)]


def task_repo(session: SessionDep, conn: ConnDep) -> PostgresTaskRepository:
    """Bound to the signed-in user, and to no other. See the module docstring.

    The web side of the queue only ever enqueues and reads back; claiming is the
    worker's, through `PostgresTaskQueue`, which is cross-tenant by necessity.
    """
    return PostgresTaskRepository(conn, session.user.id)


TaskRepoDep = Annotated[PostgresTaskRepository, Depends(task_repo)]


async def require_csrf(request: Request, session: SessionDep) -> None:
    """Form token check for every state-changing route.

    The OAuth `state` parameter covers the login flow; it does nothing for the
    forms that follow, which is why this exists separately. `SameSite=Lax` alone
    is not the whole answer either -- it is a browser default that a browser is
    free to relax.
    """
    form = await request.form()
    submitted = form.get("csrf_token")
    if not isinstance(submitted, str) or not csrf_ok(submitted, session.csrf_token):
        raise CsrfFailed


CsrfDep = Annotated[None, Depends(require_csrf)]
