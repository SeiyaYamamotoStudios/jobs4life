"""The API-key alert: is this user's Anthropic account refusing calls, and how
much of their work is waiting on it?

Owner feedback: "there is no indication when the credits run out, or the API
isn't working". The worker now records that (`api_key_health`, written by
`jfl_worker.account`) and parks the work instead of failing it; this module
turns it into the banner on every signed-in page and into the "waiting on your
API key" line beside any row that is pending.

**Cheap, and at most once per request.** Templates reach it through the
`api_key_alert()` Jinja global. The first call reads the health row by primary
key -- one indexed lookup -- and only when that says blocked does it count the
parked tasks. The answer is kept on `request.state`, so a page with a banner
and six pending rows still asks once. It reads through the request's own
connection (`jfl_web.deps.db_conn` leaves it on `request.state`), so it sees
what this request has just written.

**Tenancy** is the repository's, as everywhere: both reads are built with the
session's user id and nothing else.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import jinja2
from fastapi import Request
from jfl_core.models import ApiKeyHealthStatus
from jfl_core.storage.accounts import AuthenticatedSession
from jfl_core.storage.api_key_health import PostgresApiKeyHealthRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from sqlalchemy.engine import Connection

# How often the worker probes a parked user's account -- the worker's
# `JFL_WORKER_PARK_DELAY` default. Only used in wording ("within about 15
# minutes"); the web does not read the worker's settings.
PROBE_MINUTES = 15

CONSOLE_BILLING_URL = "https://console.anthropic.com/settings/billing"
SETTINGS_URL = "/settings"

_HEADLINES: dict[ApiKeyHealthStatus, str] = {
    "ok": "",
    "credits_exhausted": "Your Anthropic credit balance has run out.",
    "invalid_key": "Anthropic is rejecting your API key.",
    "permission_denied": "Your Anthropic API key does not have access to the model.",
}

# What fixes it, in the user's terms. Credits: top up, and the work resumes on
# its own. A rejected key or a key without access: only a new key helps.
_ACTIONS: dict[ApiKeyHealthStatus, tuple[str, str]] = {
    "ok": ("", ""),
    "credits_exhausted": ("Top up at console.anthropic.com", CONSOLE_BILLING_URL),
    "invalid_key": ("Replace your key in Settings", SETTINGS_URL),
    "permission_denied": ("Replace your key in Settings", SETTINGS_URL),
}


@dataclass(frozen=True, slots=True)
class KeyAlert:
    status: ApiKeyHealthStatus
    since: dt.datetime
    waiting: int

    @property
    def headline(self) -> str:
        return _HEADLINES[self.status]

    @property
    def action_label(self) -> str:
        return _ACTIONS[self.status][0]

    @property
    def action_url(self) -> str:
        return _ACTIONS[self.status][1]

    @property
    def external(self) -> bool:
        return self.action_url.startswith("https://")

    @property
    def row_note(self) -> str:
        """The line beside a pending row, in place of "working on it"."""
        if self.status == "credits_exhausted":
            return (
                "Waiting on your API key: your Anthropic credit balance has run out. "
                "This runs on its own once you top up."
            )
        return (
            "Waiting on your API key: Anthropic is refusing it. "
            "This runs as soon as you save a working key."
        )

    @property
    def detail(self) -> str:
        """What is happening to the work, and when it will move."""
        if self.waiting == 0:
            waiting = "Nothing is queued right now, but new work will wait until this is fixed."
        elif self.waiting == 1:
            waiting = "1 task is waiting."
        else:
            waiting = f"{self.waiting} tasks are waiting."
        if self.status == "credits_exhausted":
            resume = (
                f" Nothing has been lost: it resumes on its own within about {PROBE_MINUTES} "
                "minutes of topping up, or straight away with Retry now."
            )
        else:
            resume = " Nothing has been lost: it resumes as soon as you save a working key."
        return waiting + resume


def key_alert(conn: Connection, session: AuthenticatedSession) -> KeyAlert | None:
    """The alert for this user, or None when their key is fine (or unknown)."""
    health = PostgresApiKeyHealthRepository(conn, session.user.id).get()
    if health is None or not health.blocked:
        return None
    waiting = PostgresTaskRepository(conn, session.user.id).count_parked()
    return KeyAlert(status=health.status, since=health.since, waiting=waiting)


_UNSET = object()


def request_key_alert(request: Request, session: AuthenticatedSession | None) -> KeyAlert | None:
    """`key_alert` for this request, computed once and cached on
    `request.state`. None without a signed-in session or a request connection.
    """
    cached = getattr(request.state, "key_alert", _UNSET)
    if cached is not _UNSET:
        return cached  # type: ignore[return-value]
    conn = getattr(request.state, "db_conn", None)
    alert = None
    if session is not None and isinstance(conn, Connection):
        alert = key_alert(conn, session)
    request.state.key_alert = alert
    return alert


@jinja2.pass_context
def api_key_alert(ctx: jinja2.runtime.Context) -> KeyAlert | None:
    """Jinja global: `{% set alert = api_key_alert() %}`. Reads `request` and
    `session` from the template context, which every signed-in page passes.
    """
    request: Any = ctx.get("request")
    session: Any = ctx.get("session")
    if request is None or not isinstance(session, AuthenticatedSession):
        return None
    return request_key_alert(request, session)
