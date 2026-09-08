"""Slice A5/A6 end to end against a live Postgres: the application tracker's
screens, htmx wiring, CSRF, and tenancy through the actual routes rather than
the repository directly.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Google is stubbed, same pattern as test_web_auth_integration.py -- what is
being tested here is everything downstream of a signed-in session.

No Anthropic API calls anywhere: nothing in this file constructs a client, and
`validate_api_keys` is off, same as the auth integration tests.
"""

from __future__ import annotations

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
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration


class StubGoogle:
    """Stands in for `AuthlibGoogleProvider`."""

    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None
        self.error: str | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.error is not None:
            raise OAuthError(self.error)
        assert self.identity is not None, "the test must set an identity first"
        return self.identity


@pytest.fixture(scope="module")
def database_url() -> str:
    return os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")


@pytest.fixture(scope="module")
def engine(database_url: str) -> Iterator[Engine]:
    created = create_engine(database_url)
    yield created
    created.dispose()


@pytest.fixture(scope="module")
def master_key() -> MasterKey:
    return MasterKey.generate()


@pytest.fixture
def settings(database_url: str, master_key: MasterKey) -> WebSettings:
    import datetime as dt

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


def sign_in(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    *,
    sub: str | None = None,
    email: str | None = None,
    name: str | None = "Test Person",
) -> GoogleIdentity:
    identity = GoogleIdentity(
        sub=sub or f"test-sub-{uuid.uuid4()}",
        email=email or f"{uuid.uuid4()}@test.invalid",
        display_name=name,
    )
    subs.append(identity.sub)
    google.identity = identity
    response = client.get("/auth/google/callback")
    assert response.status_code == 200, response.text
    return identity


def _csrf(client: TestClient, path: str = "/applications/new") -> str:
    """Scrape the token out of a rendered form, the way a browser would."""
    page = client.get(path).text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None, f"no CSRF token found on {path}"
    return match.group(1)


def _add_application(
    client: TestClient,
    *,
    title: str,
    employer: str = "",
    url: str = "",
    source: str = "",
    notes: str = "",
    job_ad: str = "",
) -> str:
    """Submit the add form; returns the new application's id from the redirect."""
    response = client.post(
        "/applications",
        data={
            "csrf_token": _csrf(client),
            "title": title,
            "employer": employer,
            "url": url,
            "source": source,
            "notes": notes,
            "job_ad": job_ad,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    location = response.headers["location"]
    return location.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------
# The primary screen -- the reason the slice exists
# --------------------------------------------------------------------------


def test_a_signed_out_visitor_is_redirected_to_login(client: TestClient) -> None:
    response = client.get("/applications", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_add_then_it_appears_in_the_list(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Platform Engineer", employer="Acme")

    page = client.get("/applications").text
    assert "Platform Engineer" in page
    assert "Acme" in page
    assert f"/applications/{app_id}" in page


def test_new_applications_default_to_interested_and_open_the_timeline(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Backend Engineer")

    page = client.get(f"/applications/{app_id}").text
    assert "interested" in page.lower()
    # One timeline entry: the opening "added" event.
    assert page.count("timeline-status") == 1


def test_a_blank_title_is_rejected_without_creating_a_row(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    response = client.post(
        "/applications",
        data={"csrf_token": _csrf(client), "title": "   "},
    )
    assert response.status_code == 400
    assert "required" in response.text.lower()
    assert "No applications yet" in client.get("/applications").text


def test_a_pasted_job_ad_is_stored_and_never_shown_as_extracted_fields(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    """The form explicitly says nothing is parsed -- prove the detail page
    does not silently start showing extracted employer/title from the ad.
    """
    sign_in(client, google, subs)
    ad_text = "Senior Platform Engineer at Acme. Own the deployment pipeline."
    app_id = _add_application(client, title="My own title", job_ad=ad_text)

    page = client.get(f"/applications/{app_id}").text
    assert "My own title" in page
    assert "not yet extracted" in page.lower()


# --------------------------------------------------------------------------
# Status changes: htmx from the list, plain submit from the detail page --
# and either way, the event log grows rather than being overwritten.
# --------------------------------------------------------------------------


def test_changing_status_from_the_detail_page_redirects_and_grows_the_timeline(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="SRE")

    response = client.post(
        f"/applications/{app_id}/status",
        data={
            "csrf_token": _csrf(client),
            "to_status": "applied",
            "note": "Submitted via referral",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/applications/{app_id}"

    page = client.get(f"/applications/{app_id}").text
    assert page.count("timeline-status") == 2  # "added" + this transition
    assert "Submitted via referral" in page
    assert "applied" in page.lower()


def test_changing_status_over_htmx_returns_just_the_row_and_updates_the_list(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    """The acceptance shape for "the list updates without a full page
    reload": an htmx request gets back a `<tr>` fragment, not a redirect and
    not a full page.
    """
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Data Engineer")

    response = client.post(
        f"/applications/{app_id}/status",
        data={"csrf_token": _csrf(client), "to_status": "screening"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert response.text.strip().startswith("<tr")
    assert "<html" not in response.text.lower()
    assert f'id="application-{app_id}"' in response.text
    assert "Data Engineer" in response.text

    # And it really did change, not just the fragment.
    page = client.get(f"/applications/{app_id}").text
    assert "screening" in page.lower()


def test_two_status_changes_leave_two_events_not_one_overwritten(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Engineering Manager")

    client.post(
        f"/applications/{app_id}/status",
        data={"csrf_token": _csrf(client), "to_status": "applied"},
    )
    client.post(
        f"/applications/{app_id}/status",
        data={"csrf_token": _csrf(client), "to_status": "interviewing"},
    )

    page = client.get(f"/applications/{app_id}").text
    # "added" + "applied" + "interviewing" == three entries survive.
    assert page.count("timeline-status") == 3


def test_status_and_notes_changes_require_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Needs CSRF")

    status_response = client.post(
        f"/applications/{app_id}/status", data={"to_status": "applied", "csrf_token": "wrong"}
    )
    assert status_response.status_code == 403

    notes_response = client.post(
        f"/applications/{app_id}/notes", data={"notes": "sneaky", "csrf_token": "wrong"}
    )
    assert notes_response.status_code == 403

    page = client.get(f"/applications/{app_id}").text
    assert "interested" in page.lower()
    assert "sneaky" not in page


def test_notes_can_be_edited(client: TestClient, google: StubGoogle, subs: list[str]) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Notes test")

    response = client.post(
        f"/applications/{app_id}/notes",
        data={"csrf_token": _csrf(client), "notes": "Called the recruiter, waiting to hear back."},
        follow_redirects=False,
    )
    assert response.status_code == 303

    page = client.get(f"/applications/{app_id}").text
    assert "Called the recruiter, waiting to hear back." in page


# --------------------------------------------------------------------------
# Timestamps: A6 -- absolute and relative, together, everywhere
# --------------------------------------------------------------------------


def test_timeline_entries_show_absolute_and_relative_time(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client, title="Timestamp check")

    page = client.get(f"/applications/{app_id}").text
    assert "just now" in page  # relative half, for a freshly created row
    # Absolute half: a weekday abbreviation is present somewhere near a year.
    assert re.search(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{1,2} \w{3} \d{4}\b", page)


# --------------------------------------------------------------------------
# Tenancy: the acceptance criterion from PLAN.md, through the actual routes
# --------------------------------------------------------------------------


def test_two_users_cannot_see_each_others_applications_in_the_list(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    _add_application(client, title="Alice's role")
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    page = client.get("/applications").text
    assert "Alice's role" not in page
    assert "No applications yet" in page


def test_a_guessed_application_url_returns_not_found_not_someone_elses_data(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    alice_app_id = _add_application(client, title="Alice's confidential role", notes="secret notes")
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    response = client.get(f"/applications/{alice_app_id}")
    assert response.status_code == 404
    assert "Alice's confidential role" not in response.text
    assert "secret notes" not in response.text


def test_a_second_user_cannot_change_status_on_the_first_users_application(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    alice_app_id = _add_application(client, title="Alice's role")
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    response = client.post(
        f"/applications/{alice_app_id}/status",
        data={"csrf_token": _csrf(client), "to_status": "rejected"},
    )
    assert response.status_code == 404


def test_a_second_user_cannot_edit_notes_on_the_first_users_application(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    alice_app_id = _add_application(client, title="Alice's role")
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    response = client.post(
        f"/applications/{alice_app_id}/notes",
        data={"csrf_token": _csrf(client), "notes": "tampered"},
    )
    assert response.status_code == 404
