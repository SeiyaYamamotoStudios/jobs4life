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
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
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
    title: str = "Platform Engineer",
    job_ad: str = "",
    url: str = "",
) -> str:
    """Submit the add form; returns the new application's id from the redirect.

    Slice B3 turned the form into a paste box, so `title` is no longer a field:
    it is the first line of the ad, which is where the provisional title comes
    from. Tests that want a particular title on the row say it here and it goes
    in at the top of the pasted text, which is what a real paste looks like.
    """
    ad = job_ad or f"{title}\n\nWe are hiring. You will own the deployment pipeline."
    response = client.post(
        "/applications",
        data={"csrf_token": _csrf(client), "job_ad": ad, "url": url},
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
    app_id = _add_application(client, title="Platform Engineer at Acme")

    page = client.get("/applications").text
    # The provisional title, taken off the top of the ad -- no model has run.
    assert "Platform Engineer at Acme" in page
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


def test_a_blank_ad_is_rejected_without_creating_a_row(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    """The ad is the only required field now, and it is the whole input."""
    sign_in(client, google, subs)
    response = client.post(
        "/applications",
        data={"csrf_token": _csrf(client), "job_ad": "   "},
    )
    assert response.status_code == 400
    assert "paste the job ad" in response.text.lower()
    assert "No applications yet" in client.get("/applications").text


def test_a_rejected_submission_keeps_what_was_typed(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    """Losing a pasted ad to a validation error is the small insult that stops
    a tool being used.
    """
    sign_in(client, google, subs)
    response = client.post(
        "/applications",
        data={
            "csrf_token": _csrf(client),
            "job_ad": "Staff Engineer at Acme",
            "url": "acme.example/jobs/1",
        },
    )
    assert response.status_code == 400
    assert "http://" in response.text
    assert "Staff Engineer at Acme" in response.text
    assert "acme.example/jobs/1" in response.text


def test_the_pasted_ad_gives_a_provisional_title_that_says_it_is_provisional(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    """No model has run at this point, so the title is a line off the top of
    the ad -- and the page says so rather than presenting it as read.
    """
    sign_in(client, google, subs)
    app_id = _add_application(
        client,
        job_ad="# Senior Platform Engineer\n\nAcme. Own the deployment pipeline.",
    )

    page = client.get(f"/applications/{app_id}").text
    # The markdown heading marks are stripped; the words are not.
    assert "Senior Platform Engineer" in page
    assert "provisional title" in page.lower()
    assert "Reading the job ad" in page


# --------------------------------------------------------------------------
# B3: the paste box enqueues the read, and the request path calls no model
# --------------------------------------------------------------------------


def _extraction_tasks(engine: Engine, application_id: str) -> list[Any]:
    """Every queued task naming this application, whoever owns it."""
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(
                    tasks_table.c.kind,
                    tasks_table.c.status,
                    tasks_table.c.payload,
                    tasks_table.c.user_id,
                ).where(tasks_table.c.payload["application_id"].astext == application_id)
            ).all()
        )


def test_pasting_an_ad_enqueues_the_read_and_calls_no_model(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Fast input, slow processing. The POST returns a redirect, so nothing in
    the request path constructed an Anthropic client -- the root conftest guard
    replaces `anthropic.Anthropic` with something that raises, and a 303 is
    proof it was never called.

    The payload carries one id. Not the ad text, which is already stored once in
    `jobs.raw_text`, and above all not a credential.
    """
    sign_in(client, google, subs)
    app_id = _add_application(client, job_ad="Staff Engineer\n\nAcme. Build things.")

    tasks = _extraction_tasks(engine, app_id)
    assert len(tasks) == 1
    assert tasks[0].kind == "extract_job_ad"
    assert tasks[0].status == "pending"
    assert tasks[0].payload == {"application_id": app_id}

    page = client.get(f"/applications/{app_id}").text
    assert "Reading the job ad" in page
    # Polling is set up, and points at the fragment route.
    assert f'hx-get="/applications/{app_id}/extraction"' in page
    assert 'hx-trigger="every 3s"' in page


def test_the_extraction_fragment_stops_polling_once_it_is_no_longer_pending(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The poll stops by virtue of what came back, not by anything cancelling
    it: the trigger is on the fragment and only while it is pending.
    """
    sign_in(client, google, subs)
    app_id = _add_application(client, job_ad="Staff Engineer\n\nAcme.")

    pending = client.get(f"/applications/{app_id}/extraction")
    assert pending.status_code == 200
    assert pending.text.strip().startswith("<section")
    assert "<html" not in pending.text.lower()
    assert "hx-trigger" in pending.text

    with engine.begin() as conn:
        conn.execute(
            tasks_table.update()
            .where(tasks_table.c.payload["application_id"].astext == app_id)
            .values(status="failed")
        )
        from jfl_core.db.tables import applications as applications_table

        conn.execute(
            applications_table.update()
            .where(applications_table.c.id == uuid.UUID(app_id))
            .values(extraction_status="failed", extraction_error_code="no_api_key")
        )

    settled = client.get(f"/applications/{app_id}/extraction").text
    assert "hx-trigger" not in settled
    # And the failure is actionable, not merely honest.
    assert "API key" in settled
    assert 'href="/settings"' in settled


def test_a_finished_extraction_renders_its_result_and_stops_polling(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The success path through the template.

    The worker half is proved in `tests/test_extraction_integration.py`; what
    this proves is that the page can render what it wrote. The state is set
    directly rather than by running the worker, so this test costs nothing and
    constructs no client.
    """
    import uuid as uuid_module

    from jfl_core.db.tables import applications as applications_table
    from jfl_core.db.tables import job_requirements as job_requirements_table
    from jfl_core.db.tables import jobs as jobs_table

    sign_in(client, google, subs)
    app_id = _add_application(client, job_ad="Staff Engineer\n\nAcme, London.")

    with engine.begin() as conn:
        row = conn.execute(
            select(applications_table.c.job_id, applications_table.c.user_id).where(
                applications_table.c.id == uuid_module.UUID(app_id)
            )
        ).one()
        conn.execute(
            jobs_table.update()
            .where(jobs_table.c.id == row.job_id)
            .values(employer="Acme Corp", title="Staff Engineer", location="London")
        )
        conn.execute(
            job_requirements_table.insert().values(
                id=uuid_module.uuid4(),
                user_id=row.user_id,
                job_id=row.job_id,
                ordinal=0,
                text="5+ years of Python",
                necessity="essential",
            )
        )
        conn.execute(
            applications_table.update()
            .where(applications_table.c.id == uuid_module.UUID(app_id))
            .values(extraction_status="done", employer="Acme Corp")
        )

    page = client.get(f"/applications/{app_id}").text
    assert "Acme Corp" in page
    assert "London" in page
    assert "5+ years of Python" in page
    assert "essential" in page
    # Nothing left to poll for.
    assert "hx-trigger" not in page
    # And a re-read is offered as a deliberate act, with its cost named.
    assert "Read it again" in page
    assert "billed to your own API key" in page


def test_a_second_user_cannot_enqueue_extraction_on_the_first_users_application(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The re-read button costs money -- the account whose key would pay for it
    must be the account that owns the application.
    """
    sign_in(client, google, subs)
    alice_app_id = _add_application(client, job_ad="Alice's role\n\nConfidential.")
    before = _extraction_tasks(engine, alice_app_id)
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    response = client.post(
        f"/applications/{alice_app_id}/extract", data={"csrf_token": _csrf(client)}
    )
    assert response.status_code == 404
    # No extra task, and certainly not one owned by the second user.
    assert _extraction_tasks(engine, alice_app_id) == before

    # The fragment route leaks nothing either.
    assert client.get(f"/applications/{alice_app_id}/extraction").status_code == 404


def test_a_re_read_is_explicit_and_requires_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Extraction is billed to the user's own key, so a re-run happens because
    a person pressed something -- never because a page was loaded.
    """
    sign_in(client, google, subs)
    app_id = _add_application(client, job_ad="Staff Engineer\n\nAcme.")

    # Loading the detail page repeatedly enqueues nothing.
    client.get(f"/applications/{app_id}")
    client.get(f"/applications/{app_id}")
    assert len(_extraction_tasks(engine, app_id)) == 1

    assert (
        client.post(f"/applications/{app_id}/extract", data={"csrf_token": "wrong"}).status_code
        == 403
    )
    assert len(_extraction_tasks(engine, app_id)) == 1

    response = client.post(
        f"/applications/{app_id}/extract",
        data={"csrf_token": _csrf(client)},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert len(_extraction_tasks(engine, app_id)) == 2


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
    alice_app_id = _add_application(
        client, job_ad="Alice's confidential role\n\nSecret duties at a named employer."
    )
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    response = client.get(f"/applications/{alice_app_id}")
    assert response.status_code == 404
    assert "Alice's confidential role" not in response.text
    assert "Secret duties" not in response.text


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
