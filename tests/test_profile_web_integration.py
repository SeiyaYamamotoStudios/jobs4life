"""The /profile page through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_jobs_web_integration.py`. No network and no model call.
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
from jfl_core.db.tables import users as users_table
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration


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


def csrf(client: TestClient, path: str = "/profile") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def post(client: TestClient, path: str, **fields: object) -> Response:
    data: dict[str, object] = {"csrf_token": csrf(client), **fields}
    return client.post(path, data=data, follow_redirects=False)


# -- access -------------------------------------------------------------------


def test_signed_out_profile_redirects_to_login(client: TestClient) -> None:
    response = client.get("/profile", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_the_nav_links_to_profile(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    page = client.get("/profile").text
    assert 'href="/profile">Profile</a>' in page


def test_saving_a_section_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post(
        "/profile/trajectory",
        data={"csrf_token": "wrong", "trajectory": "somewhere director-shaped"},
    )
    assert response.status_code == 403
    match = re.search(r'id="trajectory"[^>]*>([^<]*)</textarea>', client.get("/profile").text)
    assert match is not None
    assert match.group(1).strip() == ""


# -- an all-blank save is valid, and unanswered questions read "not stated" ---


def test_a_fresh_profile_shows_not_stated_everywhere(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    text = text_of(client.get("/profile").text)
    assert text.count("Not stated.") >= 10  # one per unanswered simple question


def test_saving_an_all_blank_hard_gates_section_is_valid(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = post(client, "/profile/hard-gates")
    assert response.status_code == 303
    assert response.headers["location"] == "/profile?saved=1#hard-gates"
    assert "Not stated." in client.get("/profile").text


# -- round trip, structured tickboxes, and re-rendering -----------------------


def test_hard_gates_round_trip_verbatim_text_and_structured_ticks(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = post(
        client,
        "/profile/hard-gates",
        location_commute="London, up to one office day a month",
        level=["em", "above_em"],
        levels="EM or a small step up",
        comp_amount="120000",
        comp_currency="gbp",
        comp_floor="Base + bonus + a small equity grant",
        contract_type=["permanent"],
        notice_period="One month",
    )
    assert response.status_code == 303

    html = client.get("/profile").text
    assert "London, up to one office day a month" in html
    assert re.search(r'name="level" value="em"\s+checked', html)
    assert re.search(r'name="level" value="above_em"\s+checked', html)
    assert not re.search(r'name="level" value="ic"\s+checked', html)
    assert re.search(r'value="120000"', html)
    assert re.search(r'value="GBP"', html)
    assert "Saved" in html  # the answer_note timestamp, not "Not stated."


def test_discipline_custom_bucket_is_kept_and_removable(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    post(client, "/profile/discipline", discipline_custom="Developer relations")
    html = client.get("/profile").text
    assert re.search(r'name="discipline" value="Developer relations"\s+checked', html)

    # Unticking the custom bucket (by not resubmitting its value) removes it.
    post(client, "/profile/discipline")
    html_after = client.get("/profile").text
    assert not re.search(r'name="discipline" value="Developer relations"', html_after)


# -- objectives: separate records --------------------------------------------


def test_objectives_are_saved_as_separate_slots(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = post(
        client,
        "/profile/objectives",
        objective_1="More comp",
        evidence_1="Offer at or above floor",
        objective_2="More scope",
        evidence_2="Org of 40+",
    )
    assert response.status_code == 303
    html = client.get("/profile").text
    assert "More comp" in html and "Offer at or above floor" in html
    assert "More scope" in html and "Org of 40+" in html


# -- ruled-out: dated, kept, reopen never deletes -----------------------------


def test_ruled_out_entries_are_dated_and_reopen_keeps_the_entry(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    post(client, "/profile/ruled-out", decision_text="Not considering Acme again")
    html = client.get("/profile").text
    assert "Not considering Acme again" in html
    assert "Ruled out" in html
    assert "Reopened" not in html

    match = re.search(r'action="(/profile/ruled-out/[0-9a-f-]+/reopen)"', html)
    assert match is not None
    response = post(client, match.group(1))
    assert response.status_code == 303

    html_after = client.get("/profile").text
    assert "Not considering Acme again" in html_after  # never deleted
    assert "Reopened" in html_after


# -- tenancy: user A cannot read or write user B's profile via the web -------


def test_a_users_profile_answers_are_never_shown_to_another_user(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    post(client, "/profile/place", employer_deal_breakers="No recent layoffs, please")

    # A second signed-in identity, same browser client, fresh session.
    sign_in(client, google, subs, engine)
    html = client.get("/profile").text
    assert "No recent layoffs, please" not in html
