"""The /background upload page through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_profile_web_integration.py`. No network and no model call: the upload
stores bytes and queues a task, and the queue is not drained here.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import re
import uuid
from collections.abc import Iterator
from html import unescape as html_unescape

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.cv_limits import MAX_CV_READ_CHARS
from jfl_core.db.tables import users as users_table
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_web.app import create_app
from jfl_web.corpus import MAX_CV_CHARS, MAX_FILE_BYTES
from jfl_web.corpus import MAX_FILES_PER_UPLOAD as MAX_FILES
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

CV_ONE = "# Jane Doe\n\n## Acme Ltd\n\n- Led a team of 8.\n"
CV_TWO = "# Jane Doe\n\n## Northwind\n\n- Wrote the scheduler.\n"

# Long enough to clear the no-text-layer floor, so these tests are about the
# review step rather than about scan detection.
PDF_LINES = [
    "Jane Doe",
    "Engineering Manager | London, UK",
    "Acme Ltd, Engineering Manager Nov 2021 - Present",
    "- Led a platform team of eight engineers through a migration off a",
    "  monolith, with the on-call rota and the hiring budget.",
    "- Owned the pricing service end to end, including its error budget.",
    "- Delivered a rewrite of the settlement pipeline in twelve weeks.",
    "Northwind, Senior Engineer Feb 2017 - Oct 2021",
    "- Wrote the scheduler that replaced a nightly batch job.",
    "- Reviewed the cross-border payments design and its failure modes.",
]


def make_pdf(lines: list[str] | None = None) -> bytes:
    """A PDF with a real text layer, built here rather than committed."""
    from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in lines if lines is not None else PDF_LINES:
        c.drawString(72, y, line)
        y -= 14
    c.save()
    return buf.getvalue()


def make_scan() -> bytes:
    """Marks on the page, no text layer: what a scanner or a camera produces."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 780
    for _ in range(40):
        c.rect(72, y, 400, 6, fill=1, stroke=0)
        y -= 14
    c.save()
    return buf.getvalue()


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


def csrf(client: TestClient, path: str = "/background") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def upload(client: TestClient, *files: tuple[str, str | bytes]) -> Response:
    return client.post(
        "/background/upload",
        data={"csrf_token": csrf(client)},
        files=[
            (
                "files",
                (
                    name,
                    body.encode("utf-8") if isinstance(body, str) else body,
                    "application/pdf" if name.lower().endswith(".pdf") else "text/markdown",
                ),
            )
            for name, body in files
        ],
        follow_redirects=False,
    )


def review_fields(html: str) -> tuple[list[str], list[str]]:
    """(filenames, texts) as the review form would post them."""
    names = re.findall(r'name="filenames" value="([^"]*)"', html)
    texts = [
        html_unescape(body)
        for body in re.findall(r'name="texts"[^>]*>(.*?)</textarea>', html, re.S)
    ]
    return names, texts


def confirm(client: TestClient, html: str, *, texts: list[str] | None = None) -> Response:
    names, extracted = review_fields(html)
    return client.post(
        "/background/confirm",
        data={
            "csrf_token": csrf(client),
            "filenames": names,
            "texts": texts if texts is not None else extracted,
        },
        follow_redirects=False,
    )


def cvs_of(engine: Engine, user_id: uuid.UUID) -> list[object]:
    with engine.begin() as conn:
        return list(PostgresSentDocumentRepository(conn, user_id).list_cvs())


def tasks_of(engine: Engine, user_id: uuid.UUID) -> list[object]:
    with engine.begin() as conn:
        return list(PostgresTaskRepository(conn, user_id).list_tasks(kind="extract_cv_facts"))


# -- the page ------------------------------------------------------------------


def test_the_page_names_the_formats_that_work(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    body = text_of(client.get("/background").text)
    assert "PDF" in body
    assert ".md" in body and ".txt" in body
    assert "shown back to you" in body


def test_signed_out_users_get_the_login_page(client: TestClient) -> None:
    response = client.get("/background", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# -- uploading -----------------------------------------------------------------


def test_several_cvs_upload_at_once_and_each_queues_one_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("cv-2024.md", CV_ONE), ("cv-2022.txt", CV_TWO))

    assert response.status_code == 303
    assert response.headers["location"] == "/background?added=2&already=0"
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
    assert again.headers["location"] == "/background?added=0&already=1"
    assert len(cvs_of(engine, user_id)) == 1
    assert len(tasks_of(engine, user_id)) == 1


def test_a_damaged_pdf_is_refused_with_a_message_rather_than_stored(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("cv.pdf", b"%PDF-1.4 not really"))

    assert response.status_code == 400
    assert "could not be opened as a PDF" in text_of(response.text)
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
    assert "PDF" in text_of(response.text)
    assert cvs_of(engine, user_id) == []


def test_uploading_nothing_says_so(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post(
        "/background/upload", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 400
    assert "Choose at least one file" in text_of(response.text)


def test_an_upload_without_a_csrf_token_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/background/upload",
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
        "/background/paste",
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
        "/background/paste",
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
    assert "alice-cv.md" in client.get("/background").text

    bob = sign_in(client, google, subs, engine)  # a second sign-in replaces the cookie
    assert bob != alice

    page = client.get("/background").text
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

    body = text_of(client.get("/background").text)
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

    body = text_of(client.get("/background").text)
    assert "Read" in body
    assert "7" in body


# -- PDF, and the review step that made it possible to accept one --------------


def test_a_pdf_upload_stores_nothing_until_it_is_confirmed(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The protection is not a better extractor. It is that nothing reaches the
    store, and no read is charged to the user's key, until they have seen the
    text that will be quoted back to them as their own words.
    """
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("cv.pdf", make_pdf()))

    assert response.status_code == 200
    body = text_of(response.text)
    assert "Check what was read" in body
    assert "Nothing has been stored yet" in body
    assert "Jane Doe" in response.text
    assert cvs_of(engine, user_id) == []
    assert tasks_of(engine, user_id) == []


def test_confirming_stores_the_pdf_text_and_queues_one_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    review = upload(client, ("cv.pdf", make_pdf())).text

    done = confirm(client, review)

    assert done.status_code == 303
    assert done.headers["location"] == "/background?added=1&already=0"
    stored = cvs_of(engine, user_id)
    assert len(stored) == 1
    assert len(tasks_of(engine, user_id)) == 1


def test_the_text_the_user_edited_is_what_is_stored_verbatim(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The whole point of the screen: a mis-read line the author fixed must be
    the line on file, or every later quotation is still wrong.
    """
    user_id = sign_in(client, google, subs, engine)
    review = upload(client, ("cv.pdf", make_pdf())).text
    corrected = "Jane Doe\n\nLed a platform team of eight engineers.\n"

    confirm(client, review, texts=[corrected])

    with engine.begin() as conn:
        repo = PostgresSentDocumentRepository(conn, user_id)
        assert repo.cv_text(repo.list_cvs()[0].id) == corrected


def test_a_renamed_cv_keeps_the_name_the_user_typed(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    upload(client, ("cv.pdf", make_pdf()))
    # The review form posts the name back, so the user may change it there.
    client.post(
        "/background/confirm",
        data={
            "csrf_token": csrf(client),
            "filenames": ["2021 CV, platform roles"],
            "texts": ["Jane Doe\n\nLed a platform team of eight.\n"],
        },
        follow_redirects=False,
    )
    with engine.begin() as conn:
        stored = PostgresSentDocumentRepository(conn, user_id).list_cvs()[0]
    assert "2021 CV, platform roles" in stored.path


def test_a_scanned_pdf_is_refused_with_the_scan_message(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Detected, rather than producing three characters of garbage and carrying
    on. No OCR in this slice, and the message says so.
    """
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("scanned.pdf", make_scan()))

    assert response.status_code == 400
    body = text_of(response.text)
    assert "scanned.pdf" in body
    assert "scan" in body
    assert "does not read images" in body
    assert cvs_of(engine, user_id) == []
    assert tasks_of(engine, user_id) == []


def test_one_scan_in_a_batch_stores_none_of_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("good.pdf", make_pdf()), ("scanned.pdf", make_scan()))

    assert response.status_code == 400
    assert cvs_of(engine, user_id) == []


def test_a_batch_with_a_pdf_in_it_reviews_every_file_in_the_batch(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """One upload with two different outcomes is worse than one extra screen."""
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("cv.md", CV_ONE), ("cv.pdf", make_pdf()))

    assert response.status_code == 200
    names, texts = review_fields(response.text)
    assert names == ["cv.md", "cv.pdf"]
    assert texts[0] == CV_ONE
    assert cvs_of(engine, user_id) == []


def test_confirming_twice_stores_one_cv_and_queues_one_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    review = upload(client, ("cv.pdf", make_pdf())).text
    confirm(client, review)

    again = confirm(client, review)

    assert again.headers["location"] == "/background?added=0&already=1"
    assert len(cvs_of(engine, user_id)) == 1
    assert len(tasks_of(engine, user_id)) == 1


def test_confirming_without_a_csrf_token_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    upload(client, ("cv.pdf", make_pdf()))

    response = client.post(
        "/background/confirm",
        data={"filenames": ["cv.pdf"], "texts": ["Led a team of eight engineers."]},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert cvs_of(engine, user_id) == []


def test_a_confirm_with_mismatched_lists_stores_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = client.post(
        "/background/confirm",
        data={"csrf_token": csrf(client), "filenames": ["a.pdf", "b.pdf"]},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert cvs_of(engine, user_id) == []


def test_an_emptied_review_box_is_refused_rather_than_stored_blank(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    review = upload(client, ("cv.pdf", make_pdf())).text

    response = confirm(client, review, texts=["   \n  "])

    assert response.status_code == 400
    assert cvs_of(engine, user_id) == []


def test_one_users_review_cannot_be_confirmed_into_anothers_account(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The review text travels in the form, so the tenancy question is whether
    a confirm lands anywhere but the session that posts it. It cannot: the
    repository is constructed with the signed-in user's id and takes no
    override.
    """
    alice = sign_in(client, google, subs, engine)
    review = upload(client, ("alice.pdf", make_pdf())).text

    bob = sign_in(client, google, subs, engine)  # a second sign-in replaces the cookie
    assert bob != alice
    assert confirm(client, review).status_code == 303

    assert cvs_of(engine, alice) == []
    assert len(tasks_of(engine, alice)) == 0
    assert len(cvs_of(engine, bob)) == 1


# -- the bigger limits ---------------------------------------------------------


def test_a_cv_longer_than_one_model_call_is_stored_whole_and_said_to_be_part_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Silently truncating would make the facts proposed from the first hundred
    thousand characters look like the facts in the whole document.
    """
    user_id = sign_in(client, google, subs, engine)
    body = "Led a platform team of eight engineers.\n" * 4_000
    assert MAX_CV_READ_CHARS < len(body) <= MAX_CV_CHARS

    upload(client, ("long.md", body))

    with engine.begin() as conn:
        repo = PostgresSentDocumentRepository(conn, user_id)
        assert repo.cv_text(repo.list_cvs()[0].id) == body

    page = text_of(client.get("/background").text)
    assert "Stored in full" in page
    assert f"{MAX_CV_READ_CHARS:,}" in page


def test_a_cv_under_the_read_ceiling_says_nothing_about_being_part_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    upload(client, ("cv.md", CV_ONE))
    assert "Stored in full" not in text_of(client.get("/background").text)


def test_more_files_than_the_limit_are_refused_by_count(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(
        client, *[(f"cv-{n}.md", f"# CV {n}\n\n- Led a team.\n") for n in range(MAX_FILES + 1)]
    )

    assert response.status_code == 400
    assert "at a time is the limit" in text_of(response.text)
    assert cvs_of(engine, user_id) == []


def test_a_single_file_past_the_byte_ceiling_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)

    response = upload(client, ("big.pdf", b"%PDF-1.4" + b"0" * MAX_FILE_BYTES))

    assert response.status_code == 400
    assert "one file" in text_of(response.text)
    assert cvs_of(engine, user_id) == []
