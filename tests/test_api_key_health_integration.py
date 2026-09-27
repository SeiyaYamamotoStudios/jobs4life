"""The API-key alert, against a live Postgres: the health row's transitions, the
parked-task count and "Retry now", and the banner on the signed-in pages.

The owner's feedback this answers: "there is no indication when the credits run
out, or the API isn't working", and "will it die and never recover?". The
worker side (a refused call parks the task and marks the key) is exercised in
`test_scoring_worker_integration.py`; this file is the storage and the screens.

Marked `integration`. No Anthropic call anywhere: the worker is never run and
`validate_api_keys` is off. Google is stubbed as in the other web tests.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import uuid
from collections.abc import Iterator

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import sessions as sessions_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.model_api import AccountBlock, parked_note
from jfl_core.models import ApiKeyHealth
from jfl_core.storage.api_key_health import PostgresApiKeyHealthRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.security import hash_token
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

T0 = dt.datetime(2026, 9, 27, 9, 0, tzinfo=dt.UTC)
KEY = "sk-ant-api03-cccccccccccccccccccccccccccccccccccccccc3333"


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")


@pytest.fixture(scope="module")
def engine(database_url: str) -> Iterator[Engine]:
    created = create_engine(database_url)
    yield created
    created.dispose()


@pytest.fixture
def user(engine: Engine) -> Iterator[uuid.UUID]:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users_table).values(id=uid, email=f"{uid}@test.invalid"))
    try:
        yield uid
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id == uid))


other_user = user


def health_of(engine: Engine, user_id: uuid.UUID) -> ApiKeyHealth | None:
    with engine.begin() as conn:
        return PostgresApiKeyHealthRepository(conn, user_id).get()


def add_task(
    engine: Engine,
    user_id: uuid.UUID,
    *,
    kind: str = "score_application",
    last_error: str | None = None,
    scheduled_at: dt.datetime | None = None,
    status: str = "pending",
) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=kind, payload={}, scheduled_at=scheduled_at
        )
        conn.execute(
            update(tasks_table)
            .where(tasks_table.c.id == task.id)
            .values(last_error=last_error, status=status)
        )
    return task.id


def scheduled_at(engine: Engine, task_id: uuid.UUID) -> dt.datetime:
    with engine.begin() as conn:
        return conn.execute(
            select(tasks_table.c.scheduled_at).where(tasks_table.c.id == task_id)
        ).scalar_one()


# --------------------------------------------------------------------------
# The health row
# --------------------------------------------------------------------------


def test_no_row_means_nothing_has_said_otherwise(engine: Engine, user: uuid.UUID) -> None:
    assert health_of(engine, user) is None
    with engine.begin() as conn:
        # Clearing a state that was never set changes nothing, and says so.
        assert PostgresApiKeyHealthRepository(conn, user).mark_ok(now=T0) is False
    assert health_of(engine, user) is None


def test_a_block_is_recorded_and_since_only_moves_when_the_status_does(
    engine: Engine, user: uuid.UUID
) -> None:
    later = T0 + dt.timedelta(minutes=15)
    later_still = T0 + dt.timedelta(minutes=30)
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user).mark_blocked("credits_exhausted", now=T0)
    first = health_of(engine, user)
    assert first is not None and first.status == "credits_exhausted" and first.blocked
    assert first.since == T0 and first.checked_at == T0

    # The next probe finds the same thing: "since" stays, "checked" moves --
    # "ran out three hours ago" must not reset every fifteen minutes.
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user).mark_blocked("credits_exhausted", now=later)
    again = health_of(engine, user)
    assert again is not None and again.since == T0 and again.checked_at == later

    # A different reason is a different state, with its own "since".
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user).mark_blocked("invalid_key", now=later_still)
    moved = health_of(engine, user)
    assert moved is not None and moved.status == "invalid_key" and moved.since == later_still


def test_a_successful_call_clears_it_once(engine: Engine, user: uuid.UUID) -> None:
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user).mark_blocked("permission_denied", now=T0)
    with engine.begin() as conn:
        assert PostgresApiKeyHealthRepository(conn, user).mark_ok(now=T0) is True
        # Already ok: the UPDATE matches nothing, which is the cheap common case.
        assert PostgresApiKeyHealthRepository(conn, user).mark_ok(now=T0) is False
    cleared = health_of(engine, user)
    assert cleared is not None and cleared.status == "ok" and not cleared.blocked


def test_health_is_per_user(engine: Engine, user: uuid.UUID, other_user: uuid.UUID) -> None:
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user).mark_blocked("credits_exhausted", now=T0)
    assert health_of(engine, other_user) is None


def test_the_database_refuses_a_status_outside_the_closed_set(
    engine: Engine, user: uuid.UUID
) -> None:
    from jfl_core.db.tables import api_key_health
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert(api_key_health).values(user_id=user, status="it said something"))


# --------------------------------------------------------------------------
# Parked tasks: the count and "Retry now"
# --------------------------------------------------------------------------


def test_only_this_users_pending_parked_tasks_are_counted_and_resumed(
    engine: Engine, user: uuid.UUID, other_user: uuid.UUID
) -> None:
    park_at = T0 + dt.timedelta(minutes=15)
    note = parked_note("credits_exhausted")
    parked = add_task(engine, user, last_error=note, scheduled_at=park_at)
    parked_too = add_task(
        engine, user, kind="generate_cv_draft", last_error=note, scheduled_at=park_at
    )
    # Not parked: a transient retry, a finished task that was once parked, and
    # another user's parked work.
    retrying = add_task(
        engine, user, last_error="GenerateError: rate_limited: slow down", scheduled_at=park_at
    )
    add_task(engine, user, last_error=note, status="succeeded")
    theirs = add_task(engine, other_user, last_error=note, scheduled_at=park_at)

    with engine.begin() as conn:
        assert PostgresTaskRepository(conn, user).count_parked() == 2
        assert PostgresTaskRepository(conn, other_user).count_parked() == 1

    with engine.begin() as conn:
        assert PostgresTaskRepository(conn, user).resume_parked(now=T0) == 2

    assert scheduled_at(engine, parked) == T0
    assert scheduled_at(engine, parked_too) == T0
    assert scheduled_at(engine, retrying) == park_at
    assert scheduled_at(engine, theirs) == park_at


# --------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.identity is None:
            raise OAuthError("no identity set")
        return self.identity


@pytest.fixture
def settings(database_url: str) -> WebSettings:
    return WebSettings(
        database_url=database_url,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
        google_redirect_uri="https://testserver/auth/google/callback",
        oauth_state_secret="0" * 43,
        master_key=MasterKey.generate(),
        session_ttl=dt.timedelta(days=14),
        session_touch_after=dt.timedelta(minutes=5),
        insecure_cookies=False,
        validate_api_keys=False,
    )


@pytest.fixture
def subs() -> list[str]:
    return []


@pytest.fixture
def client(settings: WebSettings, engine: Engine, subs: list[str]) -> Iterator[TestClient]:
    google = StubGoogle()
    app = create_app(settings, identity_provider=google)
    app.state.test_google = google
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client
    with engine.begin() as conn:
        conn.execute(delete(users_table).where(users_table.c.google_sub.in_(subs)))


def sign_in(client: TestClient, subs: list[str]) -> uuid.UUID:
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}", email=f"{uuid.uuid4()}@test.invalid", display_name="T"
    )
    subs.append(identity.sub)
    client.app.state.test_google.identity = identity  # type: ignore[attr-defined]
    assert client.get("/auth/google/callback").status_code == 200
    token = client.cookies.get("__Host-jfl_session")
    assert token is not None
    engine = client.app.state.engine  # type: ignore[attr-defined]
    with engine.begin() as conn:
        user_id: uuid.UUID = conn.execute(
            select(sessions_table.c.user_id).where(sessions_table.c.token_hash == hash_token(token))
        ).scalar_one()
    return user_id


def csrf(client: TestClient, path: str = "/settings") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None
    return match.group(1)


def block(engine: Engine, user_id: uuid.UUID, status: AccountBlock = "credits_exhausted") -> None:
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user_id).mark_blocked(status, now=T0)


def test_no_banner_while_the_key_is_fine(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    assert 'class="key-alert"' not in client.get("/applications").text
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user_id).mark_blocked("credits_exhausted", now=T0)
        PostgresApiKeyHealthRepository(conn, user_id).mark_ok(now=T0)
    assert 'class="key-alert"' not in client.get("/applications").text


def test_run_out_of_credits_shows_on_every_page_with_what_is_waiting(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    block(engine, user_id)
    for _ in range(4):
        add_task(
            engine,
            user_id,
            last_error=parked_note("credits_exhausted"),
            scheduled_at=T0 + dt.timedelta(minutes=15),
        )

    for path in ("/", "/applications", "/jobs", "/settings", "/profile"):
        page = client.get(path).text
        assert 'class="key-alert"' in page, path
        assert "credit balance has run out" in page, path
        assert "4 tasks are waiting" in page, path
        assert "console.anthropic.com" in page, path
        assert "Retry now" in page, path


def test_a_rejected_key_points_at_settings_not_at_billing(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    block(engine, user_id, "invalid_key")
    page = client.get("/applications").text
    assert "rejecting your API key" in page
    assert 'href="/settings"' in page
    assert "Replace your key in Settings" in page
    # With nothing queued there is nothing to retry, but the banner still says
    # new work will wait.
    assert "Retry now" not in page
    assert "new work will wait" in page


def test_one_users_alert_is_not_anothers(
    client: TestClient, engine: Engine, user: uuid.UUID, subs: list[str]
) -> None:
    block(engine, user)
    sign_in(client, subs)
    assert 'class="key-alert"' not in client.get("/applications").text


def test_retry_now_makes_the_parked_work_due_and_returns_you_where_you_were(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    block(engine, user_id)
    far = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=15)
    task_id = add_task(
        engine, user_id, last_error=parked_note("credits_exhausted"), scheduled_at=far
    )

    response = client.post(
        "/settings/api-key/retry",
        data={"csrf_token": csrf(client, "/applications"), "next": "/applications"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/applications"
    assert scheduled_at(engine, task_id) <= dt.datetime.now(dt.UTC)
    # The alert stays: whether the account works is for the next call to say.
    health = health_of(engine, user_id)
    assert health is not None and health.status == "credits_exhausted"


def test_retry_now_needs_the_csrf_token(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    far = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=15)
    task_id = add_task(
        engine, user_id, last_error=parked_note("credits_exhausted"), scheduled_at=far
    )
    response = client.post(
        "/settings/api-key/retry", data={"csrf_token": "wrong"}, follow_redirects=False
    )
    assert response.status_code in (400, 403)
    assert scheduled_at(engine, task_id) == far


def test_retry_now_never_redirects_off_site(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    sign_in(client, subs)
    for target in ("https://evil.example/", "//evil.example/x", "settings"):
        response = client.post(
            "/settings/api-key/retry",
            data={"csrf_token": csrf(client), "next": target},
            follow_redirects=False,
        )
        assert response.headers["location"] == "/settings", target


def test_saving_a_new_key_clears_the_alert_and_wakes_the_waiting_work(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    block(engine, user_id, "invalid_key")
    far = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=15)
    task_id = add_task(engine, user_id, last_error=parked_note("invalid_key"), scheduled_at=far)

    response = client.post(
        "/settings/api-key",
        data={"csrf_token": csrf(client), "api_key": KEY},
        follow_redirects=False,
    )
    assert response.status_code == 303

    health = health_of(engine, user_id)
    assert health is not None and health.status == "ok"
    assert scheduled_at(engine, task_id) <= dt.datetime.now(dt.UTC)

    page = client.get(response.headers["location"]).text
    assert 'class="key-alert"' not in page
    assert "1 waiting task will now run on it." in page


def test_the_settings_page_shows_the_keys_health(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    user_id = sign_in(client, subs)
    client.post("/settings/api-key", data={"csrf_token": csrf(client), "api_key": KEY})
    with engine.begin() as conn:
        conn.execute(
            delete(tasks_table).where(tasks_table.c.user_id == user_id)
        )  # nothing queued, to keep the page about the key
    block(engine, user_id)
    page = client.get("/settings").text
    assert "<dt>Status</dt>" in page
    assert "credit balance run out" in page

    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, user_id).mark_ok(now=T0)
    assert "working" in client.get("/settings").text


def test_a_pending_score_says_it_is_waiting_on_the_key(
    client: TestClient, engine: Engine, subs: list[str]
) -> None:
    """The per-row twin of the banner: a panel that would say "Scoring…" says
    what it is actually waiting for."""
    from jfl_core.storage.scores import PostgresScoreRepository

    user_id = sign_in(client, subs)
    ad = "Engineering Manager at Northwind. Five years of Python."
    response = client.post(
        "/applications",
        data={"csrf_token": csrf(client, "/applications/new"), "job_ad": ad, "url": ""},
        follow_redirects=False,
    )
    application_id = uuid.UUID(response.headers["location"].rsplit("/", 1)[1])
    with engine.begin() as conn:
        PostgresScoreRepository(conn, user_id).create_pending(application_id)

    page = client.get(f"/applications/{application_id}").text
    assert "Waiting on your API key" not in page

    block(engine, user_id)
    page = client.get(f"/applications/{application_id}").text
    assert "Waiting on your API key" in page
