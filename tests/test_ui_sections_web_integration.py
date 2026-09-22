"""Collapsible sections against a live Postgres: what starts open, what forces
itself open, how a change marker appears and clears, and that a user's own
choice beats the default on their next visit.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Google is stubbed, same pattern as `tests/test_scoring_web_integration.py`.

No Anthropic API calls anywhere: nothing here constructs a client, the worker is
never run, and `validate_api_keys` is off. Every state these tests need is
written straight to the database, which is also what keeps them free.

**These tests assert on markup rather than on a rendered page**, because there
is no browser here. That is a real limit and it is worth naming: what is proved
is that the server emits `<details>` with the right `open` attribute, the right
summary and the right marker. What is not proved is that a browser paints it --
see `docs/ui-sections.md`.
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
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import job_requirements as job_requirements_table
from jfl_core.db.tables import ui_section_states as section_states_table
from jfl_core.db.tables import users as users_table
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

AD = "Engineering Manager at Northwind. Five years of Python. On site in Manchester."


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


def sign_in(client: TestClient, google: StubGoogle, subs: list[str]) -> uuid.UUID:
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}",
        email=f"{uuid.uuid4()}@test.invalid",
        display_name="Test Person",
    )
    subs.append(identity.sub)
    google.identity = identity
    response = client.get("/auth/google/callback")
    assert response.status_code == 200, response.text
    assert client.get("/applications").status_code == 200
    return _signed_in_user_id(client)


def _signed_in_user_id(client: TestClient) -> uuid.UUID:
    from jfl_core.db.tables import sessions as sessions_table
    from jfl_web.security import hash_token

    token = client.cookies.get("__Host-jfl_session")
    assert token is not None
    engine = client.app.state.engine  # type: ignore[attr-defined]
    with engine.begin() as conn:
        return conn.execute(
            select(sessions_table.c.user_id).where(sessions_table.c.token_hash == hash_token(token))
        ).scalar_one()


def _csrf(client: TestClient, path: str = "/applications/new") -> str:
    page = client.get(path).text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None, f"no CSRF token found on {path}"
    return match.group(1)


def add_application(client: TestClient) -> uuid.UUID:
    response = client.post(
        "/applications",
        data={"csrf_token": _csrf(client), "job_ad": AD, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return uuid.UUID(response.headers["location"].rsplit("/", 1)[1])


def finish_the_ad(engine: Engine, application_id: uuid.UUID, requirements: int = 3) -> None:
    """A read ad with requirements against it -- the worker's job, done here so
    the page can be rendered without a model call.
    """
    with engine.begin() as conn:
        row = conn.execute(
            select(applications_table.c.job_id, applications_table.c.user_id).where(
                applications_table.c.id == application_id
            )
        ).one()
        for ordinal in range(requirements):
            conn.execute(
                job_requirements_table.insert().values(
                    id=uuid.uuid4(),
                    user_id=row.user_id,
                    job_id=row.job_id,
                    ordinal=ordinal,
                    text=f"Requirement {ordinal}",
                    necessity="essential" if ordinal == 0 else "desirable",
                )
            )
        conn.execute(
            applications_table.update()
            .where(applications_table.c.id == application_id)
            .values(extraction_status="done", extracted_at=dt.datetime.now(dt.UTC))
        )


# --------------------------------------------------------------------------
# Reading the markup
# --------------------------------------------------------------------------


def section_tag(page: str, key: str) -> str:
    """The opening `<details ...>` tag for one section, as the server wrote it."""
    needle = f'data-section="{key}"'
    assert needle in page, f"no section {key!r} on this page"
    at = page.index(needle)
    start = page.rindex("<details", 0, at)
    return page[start : page.index(">", at) + 1]


def is_open(page: str, key: str) -> bool:
    return re.search(r"\sopen\s*>$", section_tag(page, key)) is not None


def summary_line(page: str, key: str) -> str:
    """Everything between this section's `<summary>` and `</summary>`."""
    tag = section_tag(page, key)
    start = page.index(tag) + len(tag)
    return page[start : page.index("</summary>", start)]


def toggle(client: TestClient, key: str, *, open_: bool, default_open: bool = True) -> None:
    """What `static/sections.js` posts when a `<details>` fires `toggle`."""
    response = client.post(
        "/ui/sections",
        data={
            "csrf_token": _csrf(client),
            "section": key,
            "open": "true" if open_ else "false",
            "default_open": "true" if default_open else "false",
        },
    )
    assert response.status_code == 204, response.text


# --------------------------------------------------------------------------
# The defaults
# --------------------------------------------------------------------------


def test_a_read_ad_folds_behind_its_counts_and_status_stays_open(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The owner's complaint, in one assertion: an application whose essentials
    and desirables filled the screen now opens on something actionable.
    """
    sign_in(client, google, subs)
    application_id = add_application(client)
    finish_the_ad(engine, application_id)

    page = client.get(f"/applications/{application_id}").text
    assert not is_open(page, "application.ad")
    # A folded section still says what is inside it.
    assert "3 requirements" in summary_line(page, "application.ad")
    assert "1 essential" in summary_line(page, "application.ad")
    # Status and timeline are the agreed always-open pair.
    assert is_open(page, "application.status")
    assert is_open(page, "application.timeline")


def test_an_unread_ad_stays_open_because_it_needs_attention(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    application_id = add_application(client)
    page = client.get(f"/applications/{application_id}").text
    assert is_open(page, "application.ad")


def test_a_pending_run_forces_its_section_open_over_a_stored_choice(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The one place the user's own choice does not win.

    A pending run is transient state that needs watching, and the choice to fold
    the panel away was made about a different situation. Anything else would
    leave someone staring at a collapsed panel while the thing they paid for
    runs behind it.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_the_ad(engine, application_id)

    # A finished score, so the panel is not forced open by being absent.
    with engine.begin() as conn:
        repo = PostgresScoreRepository(conn, user_id)
        row = repo.create_pending(application_id)
        repo.mark_done(
            row.id,
            could_get_score=6,
            could_get_assessment="Three roles evidence the platform work.",
            want_it_score=3,
            want_it_assessment="The commute breaks what you said you would travel.",
            constraint_verdicts=[],
            objective_verdicts=[],
            hard_gate_breaches=[],
            levers=[],
            not_stated=[],
            model="claude-opus-5",
            cost_usd=None,
            trace_id=uuid.uuid4(),
        )

    toggle(client, "application.score", open_=False)
    assert not is_open(client.get(f"/applications/{application_id}").text, "application.score")

    # Now ask for another score. The run is pending, so the panel comes back.
    client.post(
        f"/applications/{application_id}/score",
        data={"csrf_token": _csrf(client, f"/applications/{application_id}")},
        follow_redirects=False,
    )
    page = client.get(f"/applications/{application_id}").text
    assert is_open(page, "application.score")


# --------------------------------------------------------------------------
# Remembering the choice, and recording when it disagrees
# --------------------------------------------------------------------------


def test_a_users_own_choice_beats_the_default_next_visit(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    application_id = add_application(client)
    finish_the_ad(engine, application_id)

    # Timeline defaults open; the ad defaults folded. Reverse both.
    toggle(client, "application.timeline", open_=False, default_open=True)
    toggle(client, "application.ad", open_=True, default_open=False)

    page = client.get(f"/applications/{application_id}").text
    assert not is_open(page, "application.timeline")
    assert is_open(page, "application.ad")


def test_going_against_the_default_is_counted(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """ "We will have to track if people go against this." A default everybody
    undoes should be answerable with a query, not noticed eventually.
    """
    user_id = sign_in(client, google, subs)
    toggle(client, "application.timeline", open_=False, default_open=True)
    toggle(client, "application.timeline", open_=True, default_open=True)
    toggle(client, "application.timeline", open_=False, default_open=True)

    with engine.begin() as conn:
        row = conn.execute(
            select(
                section_states_table.c.is_open,
                section_states_table.c.toggles,
                section_states_table.c.against_default,
            ).where(
                section_states_table.c.user_id == user_id,
                section_states_table.c.section_key == "application.timeline",
            )
        ).one()
    assert row.is_open is False
    assert row.toggles == 3
    # Two of the three disagreed with the default; the middle one agreed.
    assert row.against_default == 2


def test_one_users_choices_are_invisible_to_another(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    toggle(client, "application.timeline", open_=False)

    second = sign_in(client, google, subs)
    application_id = add_application(client)
    page = client.get(f"/applications/{application_id}").text
    assert is_open(page, "application.timeline"), "the second user got the first user's choice"
    with engine.begin() as conn:
        assert (
            conn.execute(
                select(section_states_table.c.id).where(section_states_table.c.user_id == second)
            ).first()
            is None
        )


def test_a_toggle_without_a_csrf_token_records_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    response = client.post(
        "/ui/sections",
        data={"section": "application.timeline", "open": "false", "default_open": "true"},
    )
    assert response.status_code == 403
    with engine.begin() as conn:
        assert (
            conn.execute(
                select(section_states_table.c.id).where(section_states_table.c.user_id == user_id)
            ).first()
            is None
        )


def test_a_key_that_is_not_a_section_key_is_ignored(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The route validates the key's *shape*, never a list of known keys -- a
    per-draft section is named after the draft. Prose is not a shape.
    """
    user_id = sign_in(client, google, subs)
    response = client.post(
        "/ui/sections",
        data={
            "csrf_token": _csrf(client),
            "section": "Robert'); DROP TABLE ui_section_states;--",
            "open": "false",
            "default_open": "true",
        },
    )
    assert response.status_code == 204
    with engine.begin() as conn:
        assert (
            conn.execute(
                select(section_states_table.c.id).where(section_states_table.c.user_id == user_id)
            ).first()
            is None
        )


# --------------------------------------------------------------------------
# The change marker
# --------------------------------------------------------------------------


def test_the_change_marker_appears_on_something_new_and_clears_when_opened(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The mechanism the owner asked for: knowing that something has changed
    inside a section that is folded away.
    """
    sign_in(client, google, subs)
    application_id = add_application(client)
    finish_the_ad(engine, application_id)
    detail = f"/applications/{application_id}"

    # Nothing is marked before the section has ever been looked at: without a
    # watermark there is no honest answer to "new since when".
    assert "section-marker" not in client.get(detail).text

    toggle(client, "application.timeline", open_=False, default_open=True)
    assert "1 new" not in summary_line(client.get(detail).text, "application.timeline")

    # A status change writes a timeline event -- something new, behind the fold.
    client.post(
        f"{detail}/status",
        data={"csrf_token": _csrf(client, detail), "to_status": "applied", "note": ""},
        follow_redirects=False,
    )
    page = client.get(detail).text
    assert not is_open(page, "application.timeline")
    assert "1 new" in summary_line(page, "application.timeline")

    # Opening it is what clears the marker -- and an open section never carries
    # one anyway, because its contents are already on screen.
    toggle(client, "application.timeline", open_=True, default_open=True)
    page = client.get(detail).text
    assert is_open(page, "application.timeline")
    assert "1 new" not in summary_line(page, "application.timeline")

    # And folding it again does not bring the marker back for something already
    # seen.
    toggle(client, "application.timeline", open_=False, default_open=True)
    assert "1 new" not in summary_line(client.get(detail).text, "application.timeline")


# --------------------------------------------------------------------------
# With JavaScript switched off
# --------------------------------------------------------------------------


def test_a_folded_section_still_carries_its_whole_content(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The `<details>` fallback, asserted the only way it can be without a
    browser: the content of a folded section is in the HTML, reachable by the
    browser's own disclosure behaviour and by its in-page find, with no script
    involved.
    """
    sign_in(client, google, subs)
    application_id = add_application(client)
    finish_the_ad(engine, application_id)

    page = client.get(f"/applications/{application_id}").text
    assert not is_open(page, "application.ad")
    # Folded, and every requirement is still served.
    assert "Requirement 0" in page
    assert "Requirement 2" in page
    # Native disclosure, not a script-driven one: no `hidden`, no `hx-` trigger
    # on the element itself, and a `<summary>` for the keyboard to land on.
    tag = section_tag(page, "application.ad")
    assert "hidden" not in tag
    assert "hx-" not in tag
    assert '<summary class="section-summary">' in page
    # The stylesheet and the enhancement script are separate files, so blocking
    # the script cannot take the layout with it.
    assert "sections.js" in page


def test_the_profile_folds_the_sections_that_are_already_answered(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A section you have not stated anything in is the one that still wants
    you, so it stays open; an answered one folds behind what it holds.
    """
    sign_in(client, google, subs)
    page = client.get("/profile").text
    assert is_open(page, "profile.disciplines")

    response = client.post(
        "/profile/disciplines",
        data={
            "csrf_token": _csrf(client, "/profile"),
            "practises": "engineering management\nplatform engineering",
            "not_this": "frontend",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    # The redirect carries the anchor as a query parameter as well, so the
    # section a save has just landed on is not folded away under the user.
    assert "open=disciplines" in response.headers["location"]
    assert is_open(client.get("/profile?saved=1&open=disciplines").text, "profile.disciplines")

    # On a later visit with no anchor, it folds behind its counts.
    page = client.get("/profile").text
    assert not is_open(page, "profile.disciplines")
    assert "2 practised" in summary_line(page, "profile.disciplines")
    assert "1 ruled out" in summary_line(page, "profile.disciplines")
