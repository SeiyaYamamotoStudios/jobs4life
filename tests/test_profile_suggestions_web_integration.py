"""The CV-read settings panel through the real `/profile` routes against a live
Postgres. Marked `integration`; needs `docker compose up -d` and
`alembic upgrade head`.

**No model call happens here and none may.** The route enqueues a task and
returns; the root `conftest.py` guard is what would raise if anything in a
request reached the API. What these tests defend:

* pressing the button enqueues **one** task, and pressing it again while a run
  is in flight enqueues none;
* a user with no CV uploaded is told so and pointed at where CVs are uploaded,
  rather than being charged to find out;
* a proposal is **never applied silently** -- it lands on the profile when, and
  only when, the user accepts it, with a stance they chose and, for a level,
  words they typed;
* a setting the user has already stated **wins**: a conflicting accept is
  refused unless they say to replace it;
* rejecting is remembered;
* a run belonging to another account is a 404, not a read;
* every state-changing route needs CSRF.
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
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import setting_key
from jfl_core.models import ProposedSetting
from jfl_core.profile import Constraint, Disciplines, Profile, location_value
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_core.storage.profile_suggestions import PostgresProfileSuggestionRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

CV = "Jo Smith\nNorthwind, London, 2021-2024\nRan platform engineering across three squads.\n"


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
        return conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()


def csrf(client: TestClient, path: str = "/profile") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def post(client: TestClient, path: str, **fields: object) -> Response:
    return client.post(path, data={"csrf_token": csrf(client), **fields}, follow_redirects=False)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, FAKE_KEY, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=FAKE_KEY[-4:],
        )


def upload_cv(engine: Engine, user_id: uuid.UUID) -> None:
    with engine.begin() as conn:
        PostgresSentDocumentRepository(conn, user_id).add_cv(filename="cv.md", text=CV)


def a_proposal(kind: str, *values: str) -> ProposedSetting:
    return ProposedSetting(
        kind=kind,  # type: ignore[arg-type]
        key=setting_key(kind, list(values)),
        values=list(values),
        source_lines=["Ran platform engineering across three squads."],
    )


def a_finished_run(
    engine: Engine, user_id: uuid.UUID, proposals: list[ProposedSetting]
) -> uuid.UUID:
    """A run the worker has already answered. Written through the real
    repository, because what these tests exercise is the screen, not the call.
    """
    with engine.begin() as conn:
        repo = PostgresProfileSuggestionRepository(conn, user_id)
        row = repo.create_pending(trace_id=uuid.uuid4())
        repo.mark_done(row.id, proposals, cv_count=1)
    return row.id


def stored_profile(engine: Engine, user_id: uuid.UUID) -> Profile:
    with engine.begin() as conn:
        return PostgresProfileRepository(conn, user_id).current()


def save_profile(engine: Engine, user_id: uuid.UUID, profile: Profile) -> None:
    with engine.begin() as conn:
        PostgresProfileRepository(conn, user_id).save(profile)


def queued(engine: Engine, user_id: uuid.UUID) -> int:
    with engine.begin() as conn:
        return conn.execute(
            select(func.count())
            .select_from(tasks_table)
            .where(
                tasks_table.c.user_id == user_id,
                tasks_table.c.kind == "suggest_profile_settings",
            )
        ).scalar_one()


# -- starting a run -----------------------------------------------------------


def test_pressing_the_button_enqueues_exactly_one_task(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)
    upload_cv(engine, user_id)

    assert post(client, "/profile/suggestions").status_code == 303
    assert queued(engine, user_id) == 1


def test_pressing_it_again_while_one_is_running_enqueues_nothing(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    """One press, one call. Their money, not ours."""
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)
    upload_cv(engine, user_id)

    post(client, "/profile/suggestions")
    post(client, "/profile/suggestions")
    assert queued(engine, user_id) == 1


def test_no_cv_uploaded_says_so_and_points_at_where_to_upload_one(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)

    response = post(client, "/profile/suggestions")
    assert response.status_code == 303
    assert "suggest=no_cvs" in response.headers["location"]
    assert queued(engine, user_id) == 0

    body = client.get("/profile?suggest=no_cvs").text
    assert "nothing to read" in text_of(body)
    assert 'href="/background"' in body


def test_the_empty_state_shows_on_a_plain_visit_with_no_cvs(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    body = client.get("/profile").text
    assert "No CV uploaded yet" in text_of(body)
    assert 'href="/background"' in body


def test_no_api_key_enqueues_nothing_and_says_where_to_add_one(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    upload_cv(engine, user_id)

    response = post(client, "/profile/suggestions")
    assert "suggest=needs_key" in response.headers["location"]
    assert queued(engine, user_id) == 0
    assert "/settings" in client.get("/profile?suggest=needs_key").text


# -- the panel ----------------------------------------------------------------


def test_a_pending_run_polls_itself(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)
    upload_cv(engine, user_id)
    post(client, "/profile/suggestions")

    body = client.get("/profile").text
    assert 'hx-trigger="every 3s"' in body
    assert "/profile/suggestions/" in body


def test_the_poll_route_renders_the_same_panel(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    run_id = a_finished_run(engine, user_id, [a_proposal("discipline", "platform engineering")])

    body = client.get(f"/profile/suggestions/{run_id}").text
    assert "platform engineering" in body
    # Every proposal shows what in the CV suggested it.
    assert "Ran platform engineering across three squads." in body


def test_the_panel_says_what_the_run_cost(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    a_finished_run(engine, user_id, [a_proposal("discipline", "platform engineering")])
    assert "on your own key" in text_of(client.get("/profile").text)


# -- accepting ----------------------------------------------------------------


def test_accepting_a_discipline_puts_it_on_the_profile(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("discipline", "platform engineering")
    run_id = a_finished_run(engine, user_id, [proposal])

    response = post(client, f"/profile/suggestions/{run_id}/{proposal.key}/accept")
    assert response.status_code == 303
    assert stored_profile(engine, user_id).disciplines.practises == ["platform engineering"]


def test_nothing_reaches_the_profile_before_a_button_is_pressed(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    a_finished_run(engine, user_id, [a_proposal("discipline", "platform engineering")])

    client.get("/profile")
    assert stored_profile(engine, user_id).is_empty


def test_a_location_needs_a_stance_and_is_refused_without_one(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("location", "London")
    run_id = a_finished_run(engine, user_id, [proposal])

    response = post(client, f"/profile/suggestions/{run_id}/{proposal.key}/accept", places="London")
    assert response.status_code == 400
    assert stored_profile(engine, user_id).constraint("location") is None


def test_accepting_a_location_records_the_order_the_user_left(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("location", "London", "Bristol")
    run_id = a_finished_run(engine, user_id, [proposal])

    post(
        client,
        f"/profile/suggestions/{run_id}/{proposal.key}/accept",
        stance="must",
        places="Bristol\nLondon",
    )
    constraint = stored_profile(engine, user_id).constraint("location")
    assert constraint is not None
    assert constraint.stance == "must"
    assert constraint.value["places"] == ["Bristol", "London"]


def test_a_level_is_never_recorded_from_the_observation(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """`level_floor` is the user's choice, so the box is empty and an empty box
    is refused rather than filled in with what the CV happened to describe.
    """
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("level", "has been operating at engineering-manager level")
    run_id = a_finished_run(engine, user_id, [proposal])

    response = post(client, f"/profile/suggestions/{run_id}/{proposal.key}/accept", stance="must")
    assert response.status_code == 400
    assert stored_profile(engine, user_id).constraint("level_floor") is None

    post(
        client,
        f"/profile/suggestions/{run_id}/{proposal.key}/accept",
        stance="must",
        level_text="Engineering manager or above",
    )
    constraint = stored_profile(engine, user_id).constraint("level_floor")
    assert constraint is not None
    assert constraint.value["text"] == "Engineering manager or above"


# -- already stated wins -------------------------------------------------------


def test_a_setting_the_user_already_stated_is_shown_as_a_conflict(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    save_profile(
        engine,
        user_id,
        Profile(
            constraints=[
                Constraint(kind="location", stance="must", value=location_value(["Leeds"]))
            ]
        ),
    )
    proposal = a_proposal("location", "London")
    a_finished_run(engine, user_id, [proposal])

    body = client.get("/profile").text
    assert "already says something else" in text_of(body)
    assert "Leeds" in body


def test_a_conflicting_accept_changes_nothing_unless_the_user_says_to(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    save_profile(engine, user_id, Profile(disciplines=Disciplines(not_practised=["frontend"])))
    proposal = a_proposal("discipline", "frontend")
    run_id = a_finished_run(engine, user_id, [proposal])

    response = post(client, f"/profile/suggestions/{run_id}/{proposal.key}/accept")
    assert response.status_code == 400
    profile = stored_profile(engine, user_id)
    assert profile.disciplines.not_practised == ["frontend"]
    assert profile.disciplines.practises == []


def test_a_conflict_the_user_chose_to_resolve_is_applied(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    save_profile(engine, user_id, Profile(disciplines=Disciplines(not_practised=["frontend"])))
    proposal = a_proposal("discipline", "frontend")
    run_id = a_finished_run(engine, user_id, [proposal])

    post(client, f"/profile/suggestions/{run_id}/{proposal.key}/accept", replace="yes")
    profile = stored_profile(engine, user_id)
    assert profile.disciplines.practises == ["frontend"]
    assert profile.disciplines.not_practised == []


def test_a_proposal_that_agrees_adds_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    save_profile(
        engine, user_id, Profile(disciplines=Disciplines(practises=["Platform Engineering"]))
    )
    proposal = a_proposal("discipline", "platform engineering")
    run_id = a_finished_run(engine, user_id, [proposal])

    assert "already say this" in text_of(client.get("/profile").text)
    post(client, f"/profile/suggestions/{run_id}/{proposal.key}/accept")
    assert stored_profile(engine, user_id).disciplines.practises == ["Platform Engineering"]


# -- rejecting -----------------------------------------------------------------


def test_rejecting_writes_nothing_and_is_remembered(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("discipline", "platform engineering")
    run_id = a_finished_run(engine, user_id, [proposal])

    assert post(client, f"/profile/suggestions/{run_id}/{proposal.key}/reject").status_code == 303
    assert stored_profile(engine, user_id).is_empty
    # Off the screen, and recorded on the run so the next one never re-offers it.
    assert f'id="suggestion-{proposal.key}"' not in client.get("/profile").text
    with engine.begin() as conn:
        answered = PostgresProfileSuggestionRepository(conn, user_id).answered_keys()
    assert proposal.key in answered


def test_answering_the_same_proposal_twice_is_a_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("discipline", "platform engineering")
    run_id = a_finished_run(engine, user_id, [proposal])

    post(client, f"/profile/suggestions/{run_id}/{proposal.key}/reject")
    second = post(client, f"/profile/suggestions/{run_id}/{proposal.key}/reject")
    assert second.status_code == 404


# -- tenancy and CSRF ----------------------------------------------------------


def test_another_accounts_run_is_a_404_not_a_read(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    settings: WebSettings,
) -> None:
    other = TestClient(
        create_app(settings, identity_provider=google), base_url="https://testserver"
    )
    with other:
        other_id = sign_in(other, google, subs, engine)
        proposal = a_proposal("discipline", "platform engineering")
        theirs = a_finished_run(engine, other_id, [proposal])

    sign_in(client, google, subs, engine)
    assert client.get(f"/profile/suggestions/{theirs}").status_code == 404
    assert post(client, f"/profile/suggestions/{theirs}/{proposal.key}/accept").status_code == 404
    assert post(client, f"/profile/suggestions/{theirs}/{proposal.key}/reject").status_code == 404


def test_every_state_changing_route_needs_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    proposal = a_proposal("discipline", "platform engineering")
    run_id = a_finished_run(engine, user_id, [proposal])

    for path in (
        "/profile/suggestions",
        f"/profile/suggestions/{run_id}/{proposal.key}/accept",
        f"/profile/suggestions/{run_id}/{proposal.key}/reject",
    ):
        response = client.post(path, data={}, follow_redirects=False)
        assert response.status_code == 403, path
    assert stored_profile(engine, user_id).is_empty
