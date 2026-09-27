"""The signed-in pages: who you are, and the API key form.

Deliberately thin. The application tracker is slice A5 and is not here -- this
slice is the shell that authenticates, remembers who you are, and holds your key.

The settings form is **write-only**: the page renders `sk-ant-...4f2a` from the
stored hint and a Replace control. There is no route, no template variable and no
repository method that could return the key itself, which is a stronger promise
than "the template happens not to print it".
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse, Response
from jfl_core.storage.credentials import ANTHROPIC_API_KEY

from jfl_web.credentials import (
    InvalidApiKeyError,
    normalise_submitted_key,
    store_api_key,
    validate_api_key,
)
from jfl_web.deps import (
    ApiKeyHealthRepoDep,
    CredentialRepoDep,
    CsrfDep,
    OptionalSessionDep,
    SessionDep,
    SettingsDep,
    TaskRepoDep,
)
from jfl_web.templating import render

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only. No database call, no auth, nothing about configuration."""
    return {"status": "ok"}


@router.get("/")
def home(request: Request, session: OptionalSessionDep) -> Response:
    if session is None:
        return RedirectResponse("/login", status_code=303)
    return render(request, "home.html", {"session": session, "user": session.user})


@router.get("/settings")
def settings_page(
    request: Request,
    session: SessionDep,
    credentials: CredentialRepoDep,
    settings: SettingsDep,
    health: ApiKeyHealthRepoDep,
) -> Response:
    return render(
        request,
        "settings.html",
        {
            "session": session,
            "user": session.user,
            "credential": credentials.summary(ANTHROPIC_API_KEY),
            "validates": settings.validate_api_keys,
            "saved": "saved" in request.query_params,
            "resumed": _resumed_count(request),
            "health": health.get(),
        },
    )


@router.post("/settings/api-key")
def save_api_key(
    request: Request,
    session: SessionDep,
    credentials: CredentialRepoDep,
    settings: SettingsDep,
    health: ApiKeyHealthRepoDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    api_key: Annotated[str, Form()],
) -> Response:
    """Set or replace the key. `api_key` lives as a local for a few statements
    and is never assigned to anything wider -- no logger, no template context, no
    exception argument.
    """
    try:
        key = normalise_submitted_key(api_key)
        if settings.validate_api_keys:
            validate_api_key(key)
    except InvalidApiKeyError as exc:
        return render(
            request,
            "settings.html",
            {
                "session": session,
                "user": session.user,
                "credential": credentials.summary(ANTHROPIC_API_KEY),
                "validates": settings.validate_api_keys,
                "error": str(exc),
                "health": health.get(),
            },
            status_code=400,
        )

    store_api_key(credentials, settings.master_key, key)
    # A new key is a fresh start: whatever the old one was refused for, this
    # one has not been refused yet. Clear the alert and wake the work that was
    # parked waiting on it, rather than leaving it to the next probe. If the
    # new key is refused too, the first task to try it says so again.
    now = dt.datetime.now(dt.UTC)
    health.mark_ok(now=now)
    resumed = tasks.resume_parked(now=now)
    # POST/redirect/GET: a refresh must not resubmit a credential.
    return RedirectResponse(f"/settings?saved=1&resumed={resumed}", status_code=303)


@router.post("/settings/api-key/retry")
def retry_parked_work(
    session: SessionDep,
    tasks: TaskRepoDep,
    _csrf: CsrfDep,
    next_path: Annotated[str, Form(alias="next")] = "/settings",
) -> Response:
    """ "Retry now": make this user's parked tasks due immediately.

    For the user who has just topped up and does not want to wait for the
    worker's next probe. Only parked tasks move (see
    `PostgresTaskRepository.resume_parked`), and only this user's. The health
    row is left alone: whether the account works is for the next call to say,
    not for a button press to assume.
    """
    tasks.resume_parked(now=dt.datetime.now(dt.UTC))
    return RedirectResponse(_local_path(next_path), status_code=303)


def _local_path(candidate: str) -> str:
    """Only a path on this site -- never an absolute or scheme-relative URL,
    which would make this form an open redirect."""
    if candidate.startswith("/") and not candidate.startswith("//") and "\\" not in candidate:
        return candidate
    return "/settings"


def _resumed_count(request: Request) -> int | None:
    raw = request.query_params.get("resumed")
    return int(raw) if raw is not None and raw.isdigit() else None
