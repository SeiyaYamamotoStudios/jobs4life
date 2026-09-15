"""Suggested title expansions, through the real routes against a live Postgres
-- slice C7a. Marked `integration`; needs `docker compose up -d` and
`alembic upgrade head`. Same stub Google provider, sign-in helper and CSRF
scraping as `test_jobs_web_integration.py`.

No model is ever called here: the enqueue path is checked by inspecting the
`tasks` table, never by letting a task run, and every `title_suggestions` row
this file needs already `done` or `pending` is written directly through the
repository.
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
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import users as users_table
from jfl_core.models import SuggestedTitle
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_core.storage.title_suggestions import PostgresTitleSuggestionRepository
from jfl_intake.normalise import normalise
from jfl_web.app import create_app
from jfl_web.jobfilter import MAX_FILTER_TEXT
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"
SUGGEST_TITLES_KIND = "suggest_titles"


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        assert self.identity is not None
        return self.identity


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")


@pytest.fixture(scope="module")
def engine(database_url: str) -> Iterator[Engine]:
    created = create_engine(database_url)
    yield created
    created.dispose()


@pytest.fixture
def master_key() -> MasterKey:
    return MasterKey.generate()


@pytest.fixture
def settings(database_url: str, master_key: MasterKey) -> WebSettings:
    return WebSettings(
        database_url=database_url,
        google_client_id="test-client-id",
        google_client_secret="test-client-secret",
        google_redirect_uri="https://testserver/auth/google/callback",
        oauth_state_secret="0" * 43,
        master_key=master_key,
        session_ttl=dt.timedelta(days=14),
        session_touch_after=dt.timedelta(minutes=5),
        insecure_cookies=False,
        validate_api_keys=False,
    )


@pytest.fixture
def google() -> StubGoogle:
    return StubGoogle()


@pytest.fixture
def subs() -> list[str]:
    return []


@pytest.fixture
def client(
    settings: WebSettings, google: StubGoogle, engine: Engine, subs: list[str]
) -> Iterator[TestClient]:
    app = create_app(settings, identity_provider=google)
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client
    with engine.begin() as conn:
        conn.execute(delete(users_table).where(users_table.c.google_sub.in_(subs)))


def sign_in(client: TestClient, google: StubGoogle, subs: list[str], engine: Engine) -> uuid.UUID:
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}", email=f"{uuid.uuid4()}@test.invalid", display_name="T"
    )
    subs.append(identity.sub)
    google.identity = identity
    assert client.get("/auth/google/callback").status_code == 200
    with engine.begin() as conn:
        user_id = conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()
    return user_id


def csrf(client: TestClient, path: str = "/jobs") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, FAKE_KEY, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=FAKE_KEY[-4:],
        )


def enqueued_suggest_tasks(engine: Engine, user_id: uuid.UUID) -> list[object]:
    with engine.begin() as conn:
        return PostgresTaskRepository(conn, user_id).list_tasks(kind=SUGGEST_TITLES_KIND)


def seed_done_suggestion(
    engine: Engine, user_id: uuid.UUID, phrase: str, titles: list[SuggestedTitle]
) -> uuid.UUID:
    with engine.begin() as conn:
        row = PostgresTitleSuggestionRepository(conn, user_id).create_pending(
            phrase=phrase, phrase_key=normalise(phrase)
        )
        assert row is not None
        PostgresTitleSuggestionRepository(conn, user_id).mark_done(row.id, titles)
    return row.id


def set_filter(engine: Engine, user_id: uuid.UUID, title_includes: str) -> None:
    with engine.begin() as conn:
        PostgresJobFilterRepository(conn, user_id).save_filter(
            workplaces=[], title_includes=title_includes, title_excludes="", location=""
        )


def get_filter_includes(engine: Engine, user_id: uuid.UUID) -> str:
    with engine.begin() as conn:
        return PostgresJobFilterRepository(conn, user_id).get_filter().title_includes


# --------------------------------------------------------------------------
# Enqueueing from save_filter
# --------------------------------------------------------------------------


def test_saving_a_filter_with_a_new_phrase_enqueues_once_and_not_again_on_resave(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine, master_key: MasterKey
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)

    response = client.post(
        "/jobs/filter",
        data={
            "csrf_token": csrf(client),
            "title_includes": "engineering manager",
            "title_excludes": "",
            "location": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    tasks = enqueued_suggest_tasks(engine, user_id)
    assert len(tasks) == 1

    with engine.begin() as conn:
        row = PostgresTitleSuggestionRepository(conn, user_id).get_by_phrase_key(
            normalise("engineering manager")
        )
    assert row is not None
    assert str(row.id) == tasks[0].payload["suggestion_id"]

    # Re-saving the same phrase must not enqueue a second call.
    client.post(
        "/jobs/filter",
        data={
            "csrf_token": csrf(client),
            "title_includes": "engineering manager",
            "title_excludes": "",
            "location": "",
        },
        follow_redirects=False,
    )
    assert len(enqueued_suggest_tasks(engine, user_id)) == 1


def test_no_api_key_means_nothing_is_enqueued(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = client.post(
        "/jobs/filter",
        data={
            "csrf_token": csrf(client),
            "title_includes": "engineering manager",
            "title_excludes": "",
            "location": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert enqueued_suggest_tasks(engine, user_id) == []
    with engine.begin() as conn:
        row = PostgresTitleSuggestionRepository(conn, user_id).get_by_phrase_key(
            normalise("engineering manager")
        )
    assert row is None

    # The panel says to add a key, rather than showing nothing.
    page = client.get("/jobs").text
    assert "Add your API key" in page


# --------------------------------------------------------------------------
# Accepting suggestions
# --------------------------------------------------------------------------


def test_accept_appends_only_titles_actually_offered(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    set_filter(engine, user_id, "engineering manager")
    suggestion_id = seed_done_suggestion(
        engine,
        user_id,
        "engineering manager",
        [
            SuggestedTitle(title="Senior Engineering Manager", gloss="a step up"),
            SuggestedTitle(title="Engineering Lead", gloss=""),
        ],
    )

    response = client.post(
        f"/jobs/filter/titles/{suggestion_id}/accept",
        data={
            "csrf_token": csrf(client),
            "title": ["Senior Engineering Manager", "Not An Offered Title"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    includes = get_filter_includes(engine, user_id)
    assert "Senior Engineering Manager" in includes
    assert "Not An Offered Title" not in includes
    assert "engineering manager" in includes


def test_accept_rejects_rather_than_truncates_when_over_the_limit(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    near_limit = "a" * (MAX_FILTER_TEXT - 10)
    set_filter(engine, user_id, near_limit)
    suggestion_id = seed_done_suggestion(
        engine,
        user_id,
        near_limit,
        [SuggestedTitle(title="A Much Longer Suggested Title", gloss="")],
    )

    response = client.post(
        f"/jobs/filter/titles/{suggestion_id}/accept",
        data={"csrf_token": csrf(client), "title": ["A Much Longer Suggested Title"]},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert get_filter_includes(engine, user_id) == near_limit


# --------------------------------------------------------------------------
# Dismissing
# --------------------------------------------------------------------------


def test_dismiss_hides_the_row(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    set_filter(engine, user_id, "engineering manager")
    suggestion_id = seed_done_suggestion(
        engine, user_id, "engineering manager", [SuggestedTitle(title="Engineering Lead", gloss="")]
    )

    response = client.post(
        f"/jobs/filter/titles/{suggestion_id}/dismiss",
        data={"csrf_token": csrf(client)},
        follow_redirects=False,
    )
    assert response.status_code == 303

    with engine.begin() as conn:
        row = PostgresTitleSuggestionRepository(conn, user_id).get(suggestion_id)
    assert row is not None and row.dismissed_at is not None

    # The standalone polling route treats a dismissed row as gone.
    assert client.get(f"/jobs/filter/titles/{suggestion_id}").status_code == 404


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_another_users_suggestion_id_is_404_everywhere(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    owner_id = sign_in(client, google, subs, engine)
    set_filter(engine, owner_id, "engineering manager")
    suggestion_id = seed_done_suggestion(
        engine,
        owner_id,
        "engineering manager",
        [SuggestedTitle(title="Engineering Lead", gloss="")],
    )

    # A second sign-in on the same client replaces the session cookie.
    sign_in(client, google, subs, engine)

    assert client.get(f"/jobs/filter/titles/{suggestion_id}").status_code == 404
    assert (
        client.post(
            f"/jobs/filter/titles/{suggestion_id}/accept",
            data={"csrf_token": csrf(client), "title": ["Engineering Lead"]},
            follow_redirects=False,
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/jobs/filter/titles/{suggestion_id}/dismiss",
            data={"csrf_token": csrf(client)},
            follow_redirects=False,
        ).status_code
        == 404
    )
