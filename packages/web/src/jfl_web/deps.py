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
from jfl_core.storage.application_questions import PostgresApplicationQuestionRepository
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.credentials import PostgresCredentialRepository
from jfl_core.storage.job_feed import PostgresJobFeedRepository
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_core.storage.title_suggestions import PostgresTitleSuggestionRepository
from jfl_core.storage.user_corpus import PostgresUserCorpusRepository
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


def score_repo(session: SessionDep, conn: ConnDep) -> PostgresScoreRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresScoreRepository(conn, session.user.id)


ScoreRepoDep = Annotated[PostgresScoreRepository, Depends(score_repo)]


def application_question_repo(
    session: SessionDep, conn: ConnDep
) -> PostgresApplicationQuestionRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresApplicationQuestionRepository(conn, session.user.id)


ApplicationQuestionRepoDep = Annotated[
    PostgresApplicationQuestionRepository, Depends(application_question_repo)
]


def task_repo(session: SessionDep, conn: ConnDep) -> PostgresTaskRepository:
    """Bound to the signed-in user, and to no other. See the module docstring.

    The web side of the queue only ever enqueues and reads back; claiming is the
    worker's, through `PostgresTaskQueue`, which is cross-tenant by necessity.
    """
    return PostgresTaskRepository(conn, session.user.id)


TaskRepoDep = Annotated[PostgresTaskRepository, Depends(task_repo)]


def board_repo(session: SessionDep, conn: ConnDep) -> PostgresBoardRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresBoardRepository(conn, session.user.id)


BoardRepoDep = Annotated[PostgresBoardRepository, Depends(board_repo)]


def job_filter_repo(session: SessionDep, conn: ConnDep) -> PostgresJobFilterRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresJobFilterRepository(conn, session.user.id)


JobFilterRepoDep = Annotated[PostgresJobFilterRepository, Depends(job_filter_repo)]


def job_feed_repo(session: SessionDep, conn: ConnDep) -> PostgresJobFeedRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresJobFeedRepository(conn, session.user.id)


JobFeedRepoDep = Annotated[PostgresJobFeedRepository, Depends(job_feed_repo)]


def profile_repo(session: SessionDep, conn: ConnDep) -> PostgresProfileRepository:
    """Bound to the signed-in user, and to no other. See the module docstring.

    One repository for the profile, not two: the 2026-09-21 redesign replaced
    B3a's three tables with a single append-only `profiles` row, so the screens
    and the scoring path read the same store through this.
    """
    return PostgresProfileRepository(conn, session.user.id)


ProfileRepoDep = Annotated[PostgresProfileRepository, Depends(profile_repo)]


def title_suggestion_repo(session: SessionDep, conn: ConnDep) -> PostgresTitleSuggestionRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresTitleSuggestionRepository(conn, session.user.id)


TitleSuggestionRepoDep = Annotated[
    PostgresTitleSuggestionRepository, Depends(title_suggestion_repo)
]


def sent_document_repo(session: SessionDep, conn: ConnDep) -> PostgresSentDocumentRepository:
    """Bound to the signed-in user, and to no other. See the module docstring.

    The sent-document store, which nothing grounding may reach -- see
    `jfl_core.storage.sent_documents`.
    """
    return PostgresSentDocumentRepository(conn, session.user.id)


SentDocumentRepoDep = Annotated[PostgresSentDocumentRepository, Depends(sent_document_repo)]


def candidate_fact_repo(session: SessionDep, conn: ConnDep) -> PostgresCandidateFactRepository:
    """Bound to the signed-in user, and to no other. See the module docstring."""
    return PostgresCandidateFactRepository(conn, session.user.id)


CandidateFactRepoDep = Annotated[PostgresCandidateFactRepository, Depends(candidate_fact_repo)]


def user_corpus_repo(session: SessionDep, conn: ConnDep) -> PostgresUserCorpusRepository:
    """Bound to the signed-in user, and to no other. See the module docstring.

    One user's hand-confirmed corpus text -- the write side only. Grounding
    still reads the corpus through `GroundingRepository`, which has no method
    that could reach another tenant's spans.
    """
    return PostgresUserCorpusRepository(conn, session.user.id)


UserCorpusRepoDep = Annotated[PostgresUserCorpusRepository, Depends(user_corpus_repo)]


def job_repo(conn: ConnDep) -> PostgresJobRepository:
    """Not tenant-bound at construction, unlike the repositories above -- see
    `tests/test_tenancy_enforcement.py`'s `_LEGACY_MODULES`, which names
    `jfl_core.storage.postgres` explicitly: it is the CLI-era storage layer,
    written before structural tenancy and taking `user_id` per call. B5's
    routes (`jfl_web.routes.drafts`) must therefore pass `session.user.id`
    explicitly on every call, the same discipline `jfl_worker.handlers.extraction`
    already follows for this same repository.
    """
    return PostgresJobRepository(conn)


JobRepoDep = Annotated[PostgresJobRepository, Depends(job_repo)]


def grounding_repo(conn: ConnDep) -> PostgresGroundingRepository:
    """Same caveat as `job_repo` above -- `user_id` goes on every call."""
    return PostgresGroundingRepository(conn)


GroundingRepoDep = Annotated[PostgresGroundingRepository, Depends(grounding_repo)]


def run_repo(conn: ConnDep) -> PostgresRunRepository:
    """Same caveat as `job_repo` above -- `user_id` goes on every call. Used by
    B5's drafting screen only to read back what a draft or a coverage check
    cost (`cost_for_trace`); nothing in the request path writes a `runs` row
    directly -- that happens in the worker, where the model is actually called.
    """
    return PostgresRunRepository(conn)


RunRepoDep = Annotated[PostgresRunRepository, Depends(run_repo)]


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
