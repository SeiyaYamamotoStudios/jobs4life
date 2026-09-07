"""The signed-in pages: who you are, and the API key form.

Deliberately thin. The application tracker is slice A5 and is not here -- this
slice is the shell that authenticates, remembers who you are, and holds your key.

The settings form is **write-only**: the page renders `sk-ant-...4f2a` from the
stored hint and a Replace control. There is no route, no template variable and no
repository method that could return the key itself, which is a stronger promise
than "the template happens not to print it".
"""

from __future__ import annotations

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
from jfl_web.deps import CredentialRepoDep, CsrfDep, OptionalSessionDep, SessionDep, SettingsDep
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
        },
    )


@router.post("/settings/api-key")
def save_api_key(
    request: Request,
    session: SessionDep,
    credentials: CredentialRepoDep,
    settings: SettingsDep,
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
            },
            status_code=400,
        )

    store_api_key(credentials, settings.master_key, key)
    # POST/redirect/GET: a refresh must not resubmit a credential.
    return RedirectResponse("/settings?saved=1", status_code=303)
