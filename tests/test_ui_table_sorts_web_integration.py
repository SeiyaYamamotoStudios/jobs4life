"""Persisted table sort against a live Postgres: a header click is saved and
applied, a plain visit reads back what was saved, a stale saved key falls
back to the default silently, and one user's choice never reaches another's.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Follows the sibling web integration files' pattern (`StubGoogle`, sign-in,
CSRF scraping); each table's fixture data is seeded the same way its own
integration test file seeds it (`test_boards_web_integration.py`,
`test_jobs_web_integration.py`, `test_changes_web_integration.py`,
`test_cv_intake_web_integration.py`), so the sort itself is what is under
test here, not each screen's own mechanics.

No Anthropic API call and no HTTP request to a board's own site: applications
are added unread (no stored API key, so nothing is queued), boards' history is
written the way the worker writes it from hand-built `FetchResult`s, and CVs
are pasted as plain text, which stores immediately with no read queued.
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
from jfl_core.db.tables import ui_table_sorts as sorts_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import ObservedJob
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None
        self.error: str | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.error is not None:
            raise OAuthError(self.error)
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
        return conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()


def csrf(client: TestClient, path: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def order_of(page: str, needles: list[str]) -> list[str]:
    """Which of `needles` appears first, second, ... in `page`."""
    return [n for _, n in sorted((page.index(n), n) for n in needles if n in page)]


def saved_sort(engine: Engine, user_id: uuid.UUID, table_key: str) -> tuple[str, str] | None:
    with engine.begin() as conn:
        row = conn.execute(
            select(sorts_table.c.sort_key, sorts_table.c.direction).where(
                sorts_table.c.user_id == user_id, sorts_table.c.table_key == table_key
            )
        ).first()
    return None if row is None else (row.sort_key, row.direction)


# -- applications -----------------------------------------------------------


def add_application(client: TestClient, ad: str) -> uuid.UUID:
    response = client.post(
        "/applications",
        data={"csrf_token": csrf(client, "/applications/new"), "job_ad": ad, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return uuid.UUID(response.headers["location"].rsplit("/", 1)[1])


def test_applications_sort_persists_across_requests(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    add_application(client, "Zebra Role\n\nSome text.")
    add_application(client, "Alpha Role\n\nSome text.")

    # A header click: title ascending.
    clicked = client.get("/applications?sort=title&dir=asc").text
    assert order_of(clicked, ["Alpha Role", "Zebra Role"]) == ["Alpha Role", "Zebra Role"]
    assert saved_sort(engine, user_id, "applications") == ("title", "asc")

    # A plain visit -- no `?sort=` -- reads back what was saved.
    plain = client.get("/applications").text
    assert order_of(plain, ["Alpha Role", "Zebra Role"]) == ["Alpha Role", "Zebra Role"]


def test_a_stale_saved_sort_key_falls_back_to_the_default_silently(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A column a screen has since dropped must not error -- the page renders
    with its own default order instead."""
    user_id = sign_in(client, google, subs, engine)
    add_application(client, "First Role\n\nSome text.")
    with engine.begin() as conn:
        conn.execute(
            insert(sorts_table).values(
                id=uuid.uuid4(),
                user_id=user_id,
                table_key="applications",
                sort_key="no_longer_a_column",
                direction="asc",
            )
        )
    response = client.get("/applications")
    assert response.status_code == 200
    assert "First Role" in response.text


def test_one_users_sort_is_invisible_to_another(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    client.get("/applications?sort=title&dir=asc")

    second_user = sign_in(client, google, subs, engine)
    add_application(client, "Only Role\n\nSome text.")
    page = client.get("/applications").text
    assert "Only Role" in page
    assert saved_sort(engine, second_user, "applications") is None


# -- boards -------------------------------------------------------------------


def seed_board(engine: Engine, user_id: uuid.UUID, label: str) -> uuid.UUID:
    with engine.begin() as conn:
        board = PostgresBoardRepository(conn, user_id).add_board(
            platform="greenhouse",
            board_url=f"https://example.invalid/{label}-{uuid.uuid4().hex[:6]}",
            board_key={"token": f"{label}-{uuid.uuid4().hex[:8]}"},
            label=label,
        )
    return board.id


def test_boards_sort_persists_across_requests(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    seed_board(engine, user_id, "Zeta Corp")
    seed_board(engine, user_id, "Acme Corp")

    clicked = client.get("/boards?sort=board&dir=asc").text
    assert order_of(clicked, ["Acme Corp", "Zeta Corp"]) == ["Acme Corp", "Zeta Corp"]
    assert saved_sort(engine, user_id, "boards") == ("board", "asc")

    plain = client.get("/boards").text
    assert order_of(plain, ["Acme Corp", "Zeta Corp"]) == ["Acme Corp", "Zeta Corp"]


# -- jobs -----------------------------------------------------------------------


def job(ext: str, title: str) -> ObservedJob:
    return ObservedJob(
        external_id=ext,
        title=title,
        location="London, UK",
        url=f"https://example.invalid/jobs/{ext}",
        fingerprint=fingerprint(title, "London, UK"),
        workplace="remote",
        locations=("London, UK",),
    )


def seed_open_jobs(engine: Engine, user_id: uuid.UUID, jobs: list[ObservedJob]) -> None:
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        board = repo.add_board(
            platform="greenhouse",
            board_url=f"https://example.invalid/board-{uuid.uuid4().hex[:6]}",
            board_key={"token": uuid.uuid4().hex[:8]},
            label="Board",
        )
        at = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
        state = repo.lock_check_state(
            board.id,
            observed_external_ids=[j.external_id for j in jobs],
            closed_since=at - REPOST_WINDOW,
        )
        assert state is not None
        result = FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))
        repo.apply_check_plan(
            plan_check(state, result, observed_at=at), started_at=at, finished_at=at
        )


def test_jobs_sort_persists_across_requests(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    seed_open_jobs(engine, user_id, [job("j1", "Zebra Job"), job("j2", "Alpha Job")])

    clicked = client.get("/jobs?sort=job&dir=asc").text
    assert order_of(clicked, ["Alpha Job", "Zebra Job"]) == ["Alpha Job", "Zebra Job"]
    assert saved_sort(engine, user_id, "jobs") == ("job", "asc")

    plain = client.get("/jobs").text
    assert order_of(plain, ["Alpha Job", "Zebra Job"]) == ["Alpha Job", "Zebra Job"]


# -- changes --------------------------------------------------------------------


def test_changes_sort_persists_across_requests(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        board = repo.add_board(
            platform="greenhouse",
            board_url=f"https://example.invalid/board-{uuid.uuid4().hex[:6]}",
            board_key={"token": uuid.uuid4().hex[:8]},
            label="Board",
        )
    baseline_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        state = repo.lock_check_state(
            board.id, observed_external_ids=[], closed_since=baseline_at - REPOST_WINDOW
        )
        assert state is not None
        repo.apply_check_plan(
            plan_check(
                state,
                FetchResult(status="complete", jobs=(), expected_total=0),
                observed_at=baseline_at,
            ),
            started_at=baseline_at,
            finished_at=baseline_at,
        )
    later_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    jobs = [job("c1", "Zebra Change"), job("c2", "Alpha Change")]
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        state = repo.lock_check_state(
            board.id,
            observed_external_ids=[j.external_id for j in jobs],
            closed_since=later_at - REPOST_WINDOW,
        )
        assert state is not None
        result = FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))
        repo.apply_check_plan(
            plan_check(state, result, observed_at=later_at),
            started_at=later_at,
            finished_at=later_at,
        )

    clicked = client.get("/changes?sort=job&dir=asc").text
    assert order_of(clicked, ["Alpha Change", "Zebra Change"]) == ["Alpha Change", "Zebra Change"]
    assert saved_sort(engine, user_id, "changes") == ("job", "asc")

    plain = client.get("/changes").text
    assert order_of(plain, ["Alpha Change", "Zebra Change"]) == ["Alpha Change", "Zebra Change"]


# -- cvs (background) ------------------------------------------------------------


def seed_cv(engine: Engine, user_id: uuid.UUID, title: str, text: str) -> None:
    """Direct to the repository, with an explicit `title` -- unlike
    `/background/paste`, which never sets one, so the CV would otherwise
    display (and sort) by its internal content-hashed path rather than a name
    a test can predict."""
    with engine.begin() as conn:
        PostgresSentDocumentRepository(conn, user_id).add_cv(
            filename=f"{title}.txt", text=text, title=title
        )


def test_cvs_sort_persists_across_requests(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    seed_cv(engine, user_id, "Zebra CV", "Some CV text, long enough to store.")
    seed_cv(engine, user_id, "Alpha CV", "Some other CV text, long enough to store.")

    clicked = client.get("/background?sort=cv&dir=asc").text
    assert order_of(clicked, ["Alpha CV", "Zebra CV"]) == ["Alpha CV", "Zebra CV"]
    assert saved_sort(engine, user_id, "cvs") == ("cv", "asc")

    plain = client.get("/background").text
    assert order_of(plain, ["Alpha CV", "Zebra CV"]) == ["Alpha CV", "Zebra CV"]
