"""The /corpus upload page through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_profile_web_integration.py`. No network and no model call: the upload
stores bytes and queues a task, and the queue is not drained here.
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
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_web.app import create_app
from jfl_web.corpus import MAX_CV_CHARS
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

CV_ONE = "# Jane Doe\n\n## Acme Ltd\n\n- Led a team of 8.\n"
CV_TWO = "# Jane Doe\n\n## Northwind\n\n- Wrote the scheduler.\n"


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
        return conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()


def csrf(client: TestClient, path: str = "/corpus") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def upload(client: TestClient, *files: tuple[str, str]) -> Response:
    return client.post(
        "/corpus/upload",
        data={"csrf_token": csrf(client)},
        files=[(("files"), (name, body.encode("utf-8"), "text/markdown")) for name, body in files],
        follow_redirects=False,
    )


def cvs_of(engine: Engine, user_id: uuid.UUID) -> list[object]:
    with engine.begin() as conn:
        return list(PostgresSentDocumentRepository(conn, user_id).list_cvs())


def tasks_of(engine: Engine, user_id: uuid.UUID) -> list[object]:
    with engine.begin() as conn:
        return list(PostgresTaskRepository(conn, user_id).list_tasks(kind="extract_cv_facts"))


# -- the page ------------------------------------------------------------------


def test_the_page_says_plainly_that_pdfs_are_not_read_yet(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    body = text_of(client.get("/corpus").text)
    assert "PDF" in body
    assert ".md" in body and ".txt" in body


def test_signed_out_users_get_the_login_page(client: TestClient) -> None:
    response = client.get("/corpus", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# -- uploading -----------------------------------------------------------------


def test_several_cvs_upload_at_once_and_each_queues_one_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("cv-2024.md", CV_ONE), ("cv-2022.txt", CV_TWO))

    assert response.status_code == 303
    assert response.headers["location"] == "/corpus?added=2&already=0"
    assert len(cvs_of(engine, user_id)) == 2
    assert len(tasks_of(engine, user_id)) == 2


def test_the_cv_is_stored_exactly_as_written(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    upload(client, ("cv.md", CV_ONE))

    with engine.begin() as conn:
        repo = PostgresSentDocumentRepository(conn, user_id)
        stored = repo.list_cvs()[0]
        assert repo.cv_text(stored.id) == CV_ONE


def test_re_uploading_the_same_cv_queues_nothing_new(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A second read is a second charge on the user's own key."""
    user_id = sign_in(client, google, subs, engine)
    upload(client, ("cv.md", CV_ONE))

    again = upload(client, ("cv.md", CV_ONE))

    assert again.status_code == 303
    assert again.headers["location"] == "/corpus?added=0&already=1"
    assert len(cvs_of(engine, user_id)) == 1
    assert len(tasks_of(engine, user_id)) == 1


def test_a_pdf_is_refused_with_a_message_rather_than_stored(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("cv.pdf", "%PDF-1.4 not really"))

    assert response.status_code == 400
    assert "not a plain-text file" in text_of(response.text)
    assert cvs_of(engine, user_id) == []
    assert tasks_of(engine, user_id) == []


def test_an_oversized_cv_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("huge.md", "x" * (MAX_CV_CHARS + 1)))

    assert response.status_code == 400
    assert "past the" in text_of(response.text)
    assert cvs_of(engine, user_id) == []


def test_one_bad_file_stores_none_of_the_batch(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A partial upload the user has to reason about is worse than doing it
    again.
    """
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("good.md", CV_ONE), ("bad.docx", CV_TWO))

    assert response.status_code == 400
    assert cvs_of(engine, user_id) == []


def test_uploading_nothing_says_so(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post(
        "/corpus/upload", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 400
    assert "Choose at least one file" in text_of(response.text)


def test_an_upload_without_a_csrf_token_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/corpus/upload",
        files=[("files", ("cv.md", CV_ONE.encode("utf-8"), "text/markdown"))],
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert cvs_of(engine, user_id) == []


# -- pasting -------------------------------------------------------------------


def test_a_pasted_cv_is_stored_and_queued(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = client.post(
        "/corpus/paste",
        data={"csrf_token": csrf(client), "name": "2024 CV", "cv_text": CV_ONE},
        follow_redirects=False,
    )

    assert response.status_code == 303
    stored = cvs_of(engine, user_id)
    assert len(stored) == 1
    assert len(tasks_of(engine, user_id)) == 1


def test_an_empty_paste_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/corpus/paste",
        data={"csrf_token": csrf(client), "name": "", "cv_text": "   "},
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert cvs_of(engine, user_id) == []


# -- tenancy -------------------------------------------------------------------


def test_another_users_cv_is_not_on_this_users_page(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    alice = sign_in(client, google, subs, engine)
    upload(client, ("alice-cv.md", CV_ONE))
    assert "alice-cv.md" in client.get("/corpus").text

    bob = sign_in(client, google, subs, engine)  # a second sign-in replaces the cookie
    assert bob != alice

    page = client.get("/corpus").text
    assert "alice-cv.md" not in page
    assert cvs_of(engine, bob) == []
    assert len(cvs_of(engine, alice)) == 1


def test_a_failed_read_is_explained_on_the_page(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The page renders a code from the closed set as a sentence, with the link
    that fixes it -- the worker never writes prose, so this is where the words
    come from.
    """
    user_id = sign_in(client, google, subs, engine)
    upload(client, ("cv.md", CV_ONE))
    with engine.begin() as conn:
        repo = PostgresSentDocumentRepository(conn, user_id)
        repo.fail_extraction(repo.list_cvs()[0].id, "no_api_key")

    body = text_of(client.get("/corpus").text)
    assert "Failed" in body
    assert "your own Anthropic API key" in body
    assert "Add an API key" in body


def test_a_finished_read_shows_its_fact_count(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    upload(client, ("cv.md", CV_ONE))
    with engine.begin() as conn:
        repo = PostgresSentDocumentRepository(conn, user_id)
        repo.finish_extraction(repo.list_cvs()[0].id, facts_proposed=7)

    body = text_of(client.get("/corpus").text)
    assert "Read" in body
    assert "7" in body
