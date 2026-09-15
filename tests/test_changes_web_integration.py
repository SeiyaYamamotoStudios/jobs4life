"""The /changes feed through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_jobs_web_integration.py`. Board history is written the way the worker
writes it -- `lock_check_state`, the pure `plan_check`, `apply_check_plan` -- from
hand-built `FetchResult`s, so no request leaves the machine and no model is called.

The route reads the real clock. Rather than give production code a clock to
inject, "a day later" is simulated by moving this user's stored times back:
`first_seen_at` on their marks and `last_looked_at` on their feed state. That is
the same thing to the visibility rule, which only ever compares those times with
now.
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
from jfl_core.db.tables import job_feed_marks, job_feed_state
from jfl_core.db.tables import users as users_table
from jfl_core.models import ObservedJob, Workplace
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select, update
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


def sign_out(client: TestClient) -> None:
    client.post("/logout", data={"csrf_token": csrf(client, "/jobs")})


def csrf(client: TestClient, path: str = "/changes") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def text_of(html: str) -> str:
    """The page's visible text with tags stripped and whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def dismiss_ids(html: str) -> list[str]:
    return re.findall(r'action="/changes/([0-9a-f-]{36})/dismiss"', html)


def job(ext: str, title: str, workplace: Workplace = "remote") -> ObservedJob:
    return ObservedJob(
        external_id=ext,
        title=title,
        location="London, UK",
        url=f"https://example.invalid/jobs/{ext}",
        fingerprint=fingerprint(title, "London, UK"),
        workplace=workplace,
        locations=("London, UK",),
    )


BASELINE = [
    job("a1", "Engineering Manager, Platform"),
    job("a2", "Staff Engineer"),
    job("a3", "Engineering Manager, Data"),
]


def apply(
    engine: Engine, user_id: uuid.UUID, board_id: uuid.UUID, result: FetchResult, at: dt.datetime
) -> None:
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        state = repo.lock_check_state(
            board_id,
            observed_external_ids=[j.external_id for j in result.jobs],
            closed_since=at - REPOST_WINDOW,
        )
        assert state is not None
        repo.apply_check_plan(
            plan_check(state, result, observed_at=at), started_at=at, finished_at=at
        )


def complete(jobs: list[ObservedJob]) -> FetchResult:
    return FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))


def watched_board(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    """A board whose baseline (a1-a3) was taken two days ago."""
    with engine.begin() as conn:
        board = PostgresBoardRepository(conn, user_id).add_board(
            platform="greenhouse",
            board_url=f"https://example.invalid/acme-{uuid.uuid4().hex[:6]}",
            board_key={"token": f"acme-{uuid.uuid4().hex[:8]}"},
            label="Acme",
        )
    now = dt.datetime.now(dt.UTC)
    apply(engine, user_id, board.id, complete(BASELINE), now - dt.timedelta(days=2))
    return board.id


def later_check(engine: Engine, user_id: uuid.UUID, board_id: uuid.UUID) -> None:
    """An hour ago: a new EM job and a new sales job; `a2` gone."""
    jobs = [
        BASELINE[0],
        BASELINE[2],
        job("n1", "Engineering Manager, Inference"),
        job("n2", "Account Executive"),
    ]
    apply(
        engine, user_id, board_id, complete(jobs), dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    )


def age_the_feed(engine: Engine, user_id: uuid.UUID, by: dt.timedelta) -> None:
    """Move this user's stored feed times back -- as if `by` had passed."""
    with engine.begin() as conn:
        conn.execute(
            update(job_feed_marks)
            .where(job_feed_marks.c.user_id == user_id)
            .values(first_seen_at=job_feed_marks.c.first_seen_at - by)
        )
        conn.execute(
            update(job_feed_state)
            .where(job_feed_state.c.user_id == user_id)
            .values(last_looked_at=job_feed_state.c.last_looked_at - by)
        )


# -- access ---------------------------------------------------------------------------


def test_signed_out_changes_redirects_to_login(client: TestClient) -> None:
    response = client.get("/changes", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_the_nav_lists_changes_between_jobs_and_boards(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    page = client.get("/changes").text
    jobs_at = page.index('href="/jobs">Jobs<')
    changes_at = page.index('href="/changes">Changes<')
    assert jobs_at < changes_at < page.index('href="/boards">Boards<')


# -- what is news ---------------------------------------------------------------------


def test_a_baseline_produces_no_changes(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    watched_board(engine, user_id)
    text = text_of(client.get("/changes").text)
    assert "0 changes matching your filter of 0" in text
    assert "Nothing has changed on your boards in the last 7 days" in text
    assert "Engineering Manager, Platform" not in text


def test_a_failed_check_produces_no_changes(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    partial = FetchResult(
        status="incomplete", jobs=(job("x1", "Engineering Manager, Ghost"),), error_code="timeout"
    )
    apply(engine, user_id, board_id, partial, dt.datetime.now(dt.UTC) - dt.timedelta(hours=1))
    text = text_of(client.get("/changes").text)
    assert "0 changes matching your filter of 0" in text
    assert "Ghost" not in text


def test_changes_after_the_baseline_show_and_a_second_visit_within_24_hours_keeps_them(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)

    html = client.get("/changes").text
    text = text_of(html)
    assert "3 changes matching your filter of 3" in text
    assert "First visit" in text
    assert "new Engineering Manager, Inference" in text
    assert "new Account Executive" in text
    assert "gone Staff Engineer" in text
    assert "first seen by you" not in text
    assert 'href="https://example.invalid/jobs/n1"' in html
    assert len(dismiss_ids(html)) == 3  # every row can be dismissed

    age_the_feed(engine, user_id, dt.timedelta(hours=23))
    text = text_of(client.get("/changes").text)
    assert "3 changes matching your filter of 3" in text
    assert "You last looked" in text
    assert text.count("first seen by you") == 3


def test_changes_are_gone_24_hours_after_first_seen(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    client.get("/changes")

    age_the_feed(engine, user_id, dt.timedelta(hours=24, minutes=1))
    text = text_of(client.get("/changes").text)
    assert "0 changes matching your filter of 0" in text
    assert "Nothing has changed on your boards since you last looked" in text
    assert "Inference" not in text


def test_an_event_that_happened_before_the_last_look_is_not_news(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    with engine.begin() as conn:  # looked a minute ago, but never saw these rows
        conn.execute(
            job_feed_state.insert().values(
                id=uuid.uuid4(),
                user_id=user_id,
                last_looked_at=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1),
            )
        )
    assert "0 changes matching your filter of 0" in text_of(client.get("/changes").text)


# -- dismissal -------------------------------------------------------------------------


def test_dismissing_one_change_removes_only_that_row(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    html = client.get("/changes").text
    match = re.search(
        r'Staff Engineer.*?action="/changes/([0-9a-f-]{36})/dismiss"', html, flags=re.DOTALL
    )
    assert match is not None

    response = client.post(
        f"/changes/{match.group(1)}/dismiss",
        data={"csrf_token": csrf(client)},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/changes?status=dismissed"
    text = text_of(client.get("/changes?status=dismissed").text)
    assert "Dismissed." in text
    assert "Staff Engineer" not in text
    assert "2 changes matching your filter of 2" in text


def test_dismiss_all_clears_what_was_shown(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    token = csrf(client)  # this GET is the view that shows the three changes

    response = client.post(
        "/changes/dismiss-all", data={"csrf_token": token}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/changes?status=dismissed_all"
    assert "0 changes matching your filter of 0" in text_of(client.get("/changes").text)


def test_dismissing_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    (first, *_) = dismiss_ids(client.get("/changes").text)
    assert client.post(f"/changes/{first}/dismiss", data={"csrf_token": "wrong"}).status_code == 403
    assert client.post("/changes/dismiss-all", data={"csrf_token": "wrong"}).status_code == 403
    assert "3 changes matching your filter of 3" in text_of(client.get("/changes").text)


def test_another_users_mark_cannot_be_dismissed_or_seen(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    alices = dismiss_ids(client.get("/changes").text)

    sign_out(client)
    sign_in(client, google, subs, engine)
    text = text_of(client.get("/changes").text)
    assert "Inference" not in text
    assert "You aren't watching any boards yet" in text
    token = csrf(client)
    for mark_id in alices:
        response = client.post(f"/changes/{mark_id}/dismiss", data={"csrf_token": token})
        assert response.status_code == 404
    assert client.post("/changes/dismiss-all", data={"csrf_token": token}).status_code == 200

    with engine.begin() as conn:
        dismissed = (
            conn.execute(
                select(job_feed_marks.c.dismissed_at).where(job_feed_marks.c.user_id == user_id)
            )
            .scalars()
            .all()
        )
    assert len(dismissed) == 3 and set(dismissed) == {None}


# -- the filter -----------------------------------------------------------------------


def test_the_saved_filter_applies_and_the_page_says_how_many_it_left_out(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)
    response = client.post(
        "/jobs/filter",
        data={"csrf_token": csrf(client, "/jobs"), "title_includes": "engineering manager"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    html = client.get("/changes").text
    text = text_of(html)
    assert "1 change matching your filter of 3" in text
    assert "Engineering Manager, Inference" in text
    assert "Account Executive" not in text
    assert "Staff Engineer" not in text
    assert len(dismiss_ids(html)) == 1

    with engine.begin() as conn:  # only the displayed change was marked as seen
        marked = (
            conn.execute(select(job_feed_marks.c.kind).where(job_feed_marks.c.user_id == user_id))
            .scalars()
            .all()
        )
    assert marked == ["new"]


def test_a_change_can_be_tracked_from_the_feed_except_for_a_job_that_has_gone(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The feed is where changes are read, so it is where they have to be
    actionable -- the same Track button /jobs shows. A job that has gone has no
    posting left to read, so it gets no button.
    """
    user_id = sign_in(client, google, subs, engine)
    board_id = watched_board(engine, user_id)
    later_check(engine, user_id, board_id)

    html = client.get("/changes").text
    track_ids = re.findall(r'action="/jobs/([0-9a-f-]+)/track"', html)
    assert len(track_ids) == 2  # the two new jobs, not the one that went

    response = client.post(
        f"/jobs/{track_ids[0]}/track",
        data={"csrf_token": csrf(client)},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "Tracked" in client.get("/changes").text
