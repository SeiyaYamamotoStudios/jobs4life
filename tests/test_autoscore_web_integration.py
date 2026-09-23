"""The screens around automatic scoring, against a live Postgres.

  * what adding costs is said before anything is pressed -- on the add form and
    beside "Track as application" -- and with no key stored nothing is queued
    and the page says why;
  * the score panel says "scoring starts by itself" while the ad is read, and
    "retrying" -- not an error -- while a failed attempt has a retry queued;
  * the applications list shows both numbers on every row, never a third,
    sorts by one column at a time from its header -- each score by its own
    axis alone -- and archives from the row behind a confirm.

No model call anywhere: the root conftest guard would raise if one were made.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import re
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import CheckPlan, ObservedJob
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_web.app import create_app
from jfl_web.jobads import EXTRACTION_RETRYING_NOTE
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.scores import (
    ADD_COST_NOTE,
    AWAITS_AD_NOTE,
    NO_KEY_ADD_NOTE,
    NO_KEY_TRACK_NOTE,
    RETRYING_NOTE,
    TRACK_COST_NOTE,
)
from jfl_web.settings import WebSettings
from markupsafe import escape
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"
AD = "Engineering Manager\n\nNorthwind. Five years of Python."


def rendered(text: str) -> str:
    return str(escape(text))


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.identity is None:
            raise OAuthError("no identity")
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
def google() -> StubGoogle:
    return StubGoogle()


@pytest.fixture
def subs() -> list[str]:
    return []


@pytest.fixture
def client(
    database_url: str, master_key: MasterKey, google: StubGoogle, engine: Engine, subs: list[str]
) -> Iterator[TestClient]:
    settings = WebSettings(
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


def store_key(engine: Engine, master_key: MasterKey, user_id: uuid.UUID) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, FAKE_KEY, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=FAKE_KEY[-4:],
        )


def csrf(client: TestClient, path: str = "/applications/new") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None
    return match.group(1)


def add_application(client: TestClient, ad: str = AD) -> uuid.UUID:
    response = client.post(
        "/applications",
        data={"csrf_token": csrf(client), "job_ad": ad, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return uuid.UUID(response.headers["location"].rsplit("/", 1)[1])


def tasks_for(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return list(conn.execute(select(tasks_table).where(tasks_table.c.user_id == user_id)).all())


def finish_score(
    engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID, *, could: int, want: int | None
) -> None:
    with engine.begin() as conn:
        repo = PostgresScoreRepository(conn, user_id)
        row = repo.create_pending(application_id)
        repo.mark_done(
            row.id,
            could_get_score=could,
            could_get_assessment="assessed",
            want_it_score=want,
            want_it_assessment="assessed",
            constraint_verdicts=[],
            objective_verdicts=[],
            hard_gate_breaches=[],
            levers=[],
            not_stated=[],
            model="claude-opus-5",
            cost_usd=None,
            trace_id=uuid.uuid4(),
        )


def pending_score(
    engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID, *, retrying: bool = False
) -> uuid.UUID:
    with engine.begin() as conn:
        repo = PostgresScoreRepository(conn, user_id)
        row = repo.create_pending(application_id)
        if retrying:
            repo.note_retry(row.id, "model_error")
    return row.id


def row_html(page: str, application_id: uuid.UUID) -> str:
    start = page.index(f'<tr id="application-{application_id}"')
    return page[start : page.index("</tr>", start)]


def cell(row: str, css_class: str) -> str:
    match = re.search(rf'<td class="col-score {css_class}"[^>]*>(.*?)</td>', row, re.DOTALL)
    assert match is not None, f"no {css_class} cell in row"
    return match.group(1)


# --------------------------------------------------------------------------
# What adding costs, said before it is pressed
# --------------------------------------------------------------------------


def test_the_add_form_names_the_calls_adding_triggers(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine, master_key: MasterKey
) -> None:
    store_key(engine, master_key, sign_in(client, google, subs, engine))
    page = client.get("/applications/new").text
    assert rendered(ADD_COST_NOTE) in page
    assert "up to three model calls" in page


def test_with_no_key_the_form_says_nothing_will_run(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    page = client.get("/applications/new").text
    assert rendered(NO_KEY_ADD_NOTE) in page
    assert rendered(ADD_COST_NOTE) not in page
    assert 'href="/settings"' in page


def test_with_no_key_adding_queues_nothing_and_the_page_says_why(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = add_application(client)

    assert tasks_for(engine, user_id) == []
    page = client.get(f"/applications/{application_id}").text
    assert "This needs your own Anthropic API key" in page
    assert 'href="/settings"' in page
    with engine.begin() as conn:
        assert PostgresScoreRepository(conn, user_id).latest(application_id) is None


def test_with_a_key_adding_queues_the_read_and_the_score_panel_waits_for_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine, master_key: MasterKey
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, master_key, user_id)
    application_id = add_application(client)

    assert [t.kind for t in tasks_for(engine, user_id)] == ["extract_job_ad"]
    panel = client.get(f"/applications/{application_id}/score").text
    assert rendered(AWAITS_AD_NOTE) in panel
    # Polls for the chained score, and offers no button to press meanwhile.
    assert f'hx-get="/applications/{application_id}/score"' in panel
    assert "Score this application" not in panel


def test_the_track_button_names_the_calls_or_says_nothing_will_run(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine, master_key: MasterKey
) -> None:
    user_id = sign_in(client, google, subs, engine)
    day0 = dt.datetime(2026, 9, 1, 6, 0, tzinfo=dt.UTC)
    with engine.begin() as conn:
        boards = PostgresBoardRepository(conn, user_id)
        board = boards.add_board(
            platform="greenhouse",
            board_url="https://boards.greenhouse.io/acme",
            board_key={"token": "acme"},
            label="Acme",
        )
        observed = ObservedJob(
            external_id="1",
            title="Platform Engineer",
            location="London",
            url="https://job-boards.greenhouse.io/acme/jobs/1",
            fingerprint=fingerprint("Platform Engineer", "London"),
        )
        state = boards.lock_check_state(
            board.id, observed_external_ids=["1"], closed_since=day0 - REPOST_WINDOW
        )
        assert state is not None
        plan: CheckPlan = plan_check(
            state,
            FetchResult(status="complete", jobs=(observed,), expected_total=1),
            observed_at=day0,
        )
        boards.apply_check_plan(plan, started_at=day0 - dt.timedelta(minutes=1), finished_at=day0)

    no_key = client.get(f"/boards/{board.id}").text
    assert "Track as application" in no_key
    assert rendered(NO_KEY_TRACK_NOTE) in no_key

    store_key(engine, master_key, user_id)
    with_key = client.get(f"/boards/{board.id}").text
    assert rendered(TRACK_COST_NOTE) in with_key


# --------------------------------------------------------------------------
# Retrying is not failing
# --------------------------------------------------------------------------


def test_a_retrying_score_says_so_and_is_not_an_error(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = add_application(client)
    pending_score(engine, user_id, application_id, retrying=True)

    panel = client.get(f"/applications/{application_id}/score").text
    assert rendered(RETRYING_NOTE) in panel
    assert "Scoring failed" not in panel
    assert 'class="error"' not in panel
    assert 'hx-trigger="every 5s"' in panel  # still polling for the retry


def test_an_exhausted_score_renders_as_failed_with_re_score(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = add_application(client)
    score_id = pending_score(engine, user_id, application_id)
    with engine.begin() as conn:
        PostgresScoreRepository(conn, user_id).mark_failed(score_id, "model_error")

    panel = client.get(f"/applications/{application_id}/score").text
    assert "Scoring failed" in panel
    assert rendered(RETRYING_NOTE) not in panel
    assert "Re-score" in panel
    assert "hx-trigger" not in panel


def test_a_retrying_read_says_so_and_is_not_an_error(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine, master_key: MasterKey
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, master_key, user_id)
    application_id = add_application(client)
    with engine.begin() as conn:
        PostgresApplicationRepository(conn, user_id).note_extraction_retry(
            application_id, "model_error"
        )

    panel = client.get(f"/applications/{application_id}/extraction").text
    assert rendered(EXTRACTION_RETRYING_NOTE) in panel
    assert 'role="alert"' not in panel
    assert "hx-trigger" in panel


# --------------------------------------------------------------------------
# The list: both numbers, never a third
# --------------------------------------------------------------------------


def test_each_row_shows_both_numbers_labelled_and_no_third(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id, could=7, want=2)

    page = client.get("/applications").text
    row = row_html(page, application_id)
    assert "<strong>7</strong>" in cell(row, "col-could-get")
    assert "<strong>2</strong>" in cell(row, "col-want-it")
    # Exactly two numbers out of ten on the row -- nothing that could be a
    # sum, an average or an "overall".
    assert re.findall(r"<strong>(\d+)</strong>", row) == ["7", "2"]
    assert row.count("/10") == 2
    # Labelled for the stacked phone card, one label per axis.
    assert 'data-label="Could I get this"' in row
    assert 'data-label="Do I want this"' in row
    assert "Could I get this" in page and "Do I want this" in page


def test_unscored_scoring_and_retrying_rows_say_so(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    unscored = add_application(client, "Unscored\n\nA.")
    scoring = add_application(client, "Scoring\n\nB.")
    retrying = add_application(client, "Retrying\n\nC.")
    pending_score(engine, user_id, scoring)
    pending_score(engine, user_id, retrying, retrying=True)

    page = client.get("/applications").text
    assert "not scored" in cell(row_html(page, unscored), "col-could-get")
    assert "scoring…" in cell(row_html(page, scoring), "col-could-get")
    assert "retrying" in cell(row_html(page, retrying), "col-want-it")
    assert "<strong>" not in row_html(page, scoring)


def _order(client: TestClient, query: str, ids: tuple[uuid.UUID, ...]) -> list[uuid.UUID]:
    page = client.get(f"/applications{query}").text
    return [i for _, i in sorted((page.index(f'id="application-{i}"'), i) for i in ids)]


def _headers(page: str) -> dict[str, tuple[str | None, str]]:
    """{header label: (aria-sort or None, the header link's href)}."""
    thead = page[page.index("<thead>") : page.index("</thead>")]
    found = {}
    for attrs, href, inner in re.findall(
        r'<th scope="col" class="sortable"([^>]*)><a href="([^"]+)">(.*?)</a></th>', thead, re.S
    ):
        label = re.sub(r"<[^>]+>|[▲▼↕]", "", inner).strip()
        aria = re.search(r'aria-sort="([^"]+)"', attrs)
        found[label] = (aria.group(1) if aria else None, html.unescape(href))
    return found


def test_the_list_sorts_by_one_axis_at_a_time(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Love it, will not get it (a) against would get it, dislike it (b): the
    two orders are opposite, which is exactly what a blend would hide -- and
    unscored stays last whichever way either axis runs."""
    user_id = sign_in(client, google, subs, engine)
    a = add_application(client, "Role A\n\nA.")
    b = add_application(client, "Role B\n\nB.")
    unscored = add_application(client, "Role C\n\nC.")
    finish_score(engine, user_id, a, could=3, want=9)
    finish_score(engine, user_id, b, could=8, want=1)
    ids = (a, b, unscored)

    assert _order(client, "?sort=could_get", ids) == [b, a, unscored]
    assert _order(client, "?sort=want_it", ids) == [a, b, unscored]
    assert _order(client, "?sort=could_get&dir=asc", ids) == [a, b, unscored]
    assert _order(client, "?sort=want_it&dir=asc", ids) == [b, a, unscored]
    # An unknown order is ignored rather than invented: the default, newest first.
    response = client.get("/applications?sort=overall&dir=asc")
    assert response.status_code == 200
    assert _order(client, "?sort=overall&dir=asc", ids) == [unscored, b, a]


def test_every_column_header_is_a_sort_link_that_reverses_when_active(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    add_application(client, "Role A\n\nA.")

    default = _headers(client.get("/applications").text)
    assert list(default) == [
        "Title",
        "Employer",
        "Status",
        "Could I get this",
        "Do I want this",
        "Updated",
    ]
    # One header carries aria-sort: the active one.
    assert [k for k, (aria, _) in default.items() if aria] == ["Updated"]
    assert default["Updated"] == ("descending", "/applications?sort=updated&dir=asc")
    assert default["Title"] == (None, "/applications?sort=title")
    assert default["Could I get this"] == (None, "/applications?sort=could_get")
    assert default["Do I want this"] == (None, "/applications?sort=want_it")

    by_could = _headers(client.get("/applications?sort=could_get").text)
    assert by_could["Could I get this"] == ("descending", "/applications?sort=could_get&dir=asc")
    assert by_could["Updated"] == (None, "/applications")
    reversed_ = _headers(client.get("/applications?sort=could_get&dir=asc").text)
    assert reversed_["Could I get this"] == ("ascending", "/applications?sort=could_get")

    # The separate "Sort:" row is gone; the headers are the control.
    page = client.get("/applications").text
    assert "Sort:" not in page and 'aria-label="Sort applications"' not in page
    assert page.count('class="sort-arrow"') == 1  # the visible arrow, on the active column


def test_sorting_keeps_the_status_filter_and_filtering_keeps_the_sort(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    add_application(client, "Role A\n\nA.")

    page = client.get("/applications?status=interested&sort=want_it&dir=asc").text
    headers = _headers(page)
    assert headers["Do I want this"] == (
        "ascending",
        "/applications?status=interested&sort=want_it",
    )
    assert headers["Title"][1] == "/applications?status=interested&sort=title"
    nav = page[page.index('aria-label="Filter applications by status"') :]
    nav = html.unescape(nav[: nav.index("</nav>")])
    assert 'href="/applications?status=applied&sort=want_it&dir=asc"' in nav
    assert 'href="/applications?sort=want_it&dir=asc"' in nav  # "All" keeps it too


def test_a_status_change_over_htmx_returns_the_row_with_both_scores(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id, could=6, want=4)

    response = client.post(
        f"/applications/{application_id}/status",
        data={"csrf_token": csrf(client), "to_status": "applied"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert re.findall(r"<strong>(\d+)</strong>", response.text) == ["6", "4"]


# --------------------------------------------------------------------------
# Archive from the list, behind a confirm
# --------------------------------------------------------------------------


def test_archive_on_the_list_sits_behind_a_confirm_disclosure(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    application_id = add_application(client, "Archive Me\n\nA.")

    row = row_html(client.get("/applications").text, application_id)
    details = re.search(r'<details class="confirm-remove row-archive">(.*?)</details>', row, re.S)
    assert details is not None, "the archive control is not inside a disclosure"
    # A visible label, not a bare disclosure triangle, and the confirm
    # button inside the same cell.
    assert '<summary class="row-archive-toggle">Archive…</summary>' in details.group(1)
    assert "Yes, archive it" in details.group(1)
    actions = re.search(r'<td class="col-actions">(.*?)</td>', row, re.S)
    assert actions is not None and details.group(0) in actions.group(1)
    assert f'action="/applications/{application_id}/archive"' in details.group(1)
    # The only archive form on the row is the one inside the disclosure.
    assert row.count(f'action="/applications/{application_id}/archive"') == 1
    assert "hx-" not in details.group(1)  # works with JavaScript off


def test_archiving_from_the_list_needs_csrf_and_shows_the_looked_up_title(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    application_id = add_application(client, "Archive Me\n\nA.")

    assert (
        client.post(
            f"/applications/{application_id}/archive", data={"csrf_token": "wrong"}
        ).status_code
        == 403
    )
    assert f'id="application-{application_id}"' in client.get("/applications").text

    response = client.post(
        f"/applications/{application_id}/archive",
        data={"csrf_token": csrf(client, "/applications")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/applications?just_archived={application_id}"
    page = client.get(response.headers["location"]).text
    assert 'Archived "Archive Me"' in page
    assert f'id="application-{application_id}"' not in page


def test_another_user_cannot_archive_from_their_list(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    application_id = add_application(client, "Alice's Role\n\nA.")
    client.post("/logout", data={"csrf_token": csrf(client, "/applications")})

    sign_in(client, google, subs, engine)
    response = client.post(
        f"/applications/{application_id}/archive",
        data={"csrf_token": csrf(client, "/applications")},
    )
    assert response.status_code == 404
