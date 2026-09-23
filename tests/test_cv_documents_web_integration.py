"""The generated CV document on the CV page, and the CV header settings on
/profile -- through the real routes against a live Postgres. Marked
`integration`; needs `alembic upgrade head`.

No model is ever called: "Check my edits" is checked by inspecting the `tasks`
table, and the handler itself is run with `check_text` and the key loader
replaced. Every fixture is fictional.
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
from jfl_core.crypto.envelope import MasterKey
from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvRole, CvSkill
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import Task
from jfl_core.storage.cv_documents import CvDocumentVersion, PostgresCvDocumentRepository
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_web.app import create_app
from jfl_web.cv_pdf import render_cv_pdf
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from jfl_worker.handlers import cv_edits_check
from jfl_worker.registry import TaskContext
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

CHECK_KIND = "check_cv_edits"


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


def csrf(client: TestClient, path: str = "/applications") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def add_application(client: TestClient, engine: Engine, employer: str = "Fictional Freight") -> str:
    response = client.post(
        "/applications",
        data={
            "csrf_token": csrf(client, "/applications/new"),
            "job_ad": "Engineering Manager\n\nFictional Freight. Lead the platform team.",
            "url": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    app_id = response.headers["location"].rsplit("/", 1)[-1]
    with engine.begin() as conn:
        conn.execute(
            update(applications_table)
            .where(applications_table.c.id == uuid.UUID(app_id))
            .values(employer=employer)
        )
    return app_id


def sample_doc(**changes: Any) -> CvDocument:
    doc = CvDocument(
        template="modern",
        header=CvHeader(name="Robin Example", tagline="Engineering Manager", contact=["Bristol"]),
        summary=[CvLine(text="Engineering leader for platform teams.", verdict="framing")],
        skills=[
            CvSkill(
                label="Platform Delivery",
                text=CvLine(text="Ran the build farm for four teams.", verdict="supported"),
            )
        ],
        roles=[
            CvRole(
                title="Engineering Manager",
                employer="Harbour Logistics",
                location="Bristol",
                dates="Jan 2021 – Present",
                descriptor="Freight routing software.",
                bullets=[
                    CvLine(text="Led a team of eight engineers.", verdict="supported"),
                    CvLine(
                        text="Cut cloud costs by 40%.",
                        verdict="unsupported",
                        note="No figure in the facts.",
                    ),
                    CvLine(text="Owned the incident process.", verdict="review"),
                ],
            )
        ],
        education=[CvLine(text="BSc Computing, Example University, 2008", origin="fact")],
        interests=["Sea swimming", "Chess"],
    )
    return doc.model_copy(update=changes)


def store(engine: Engine, user_id: uuid.UUID, app_id: str, doc: CvDocument) -> CvDocumentVersion:
    with engine.begin() as conn:
        version = PostgresCvDocumentRepository(conn, user_id).add_version(
            uuid.UUID(app_id), doc, status="generated", trace_id=uuid.uuid4()
        )
    assert version is not None
    return version


def versions(engine: Engine, user_id: uuid.UUID, app_id: str) -> list[CvDocumentVersion]:
    with engine.begin() as conn:
        return PostgresCvDocumentRepository(conn, user_id).list_versions(uuid.UUID(app_id))


def check_tasks(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table).where(
                    tasks_table.c.user_id == user_id, tasks_table.c.kind == CHECK_KIND
                )
            ).all()
        )


def page(client: TestClient, app_id: str) -> str:
    response = client.get(f"/applications/{app_id}/drafts")
    assert response.status_code == 200
    return response.text


def edit_form(client: TestClient, app_id: str) -> dict[str, str]:
    """Every field the edit page renders, with its current value."""
    text = client.get(f"/applications/{app_id}/cv/edit").text
    fields: dict[str, str] = {}
    for name, value in re.findall(
        r'<textarea [^>]*name="([^"]+)"[^>]*>(.*?)</textarea>', text, re.S
    ):
        fields[name] = html.unescape(value)
    for tag in re.findall(r"<input [^>]*>", text):
        name = re.search(r'name="([^"]+)"', tag)
        value = re.search(r'value="([^"]*)"', tag)
        if name:
            fields[name.group(1)] = html.unescape(value.group(1)) if value else ""
    return fields


# -- the CV header settings on /profile ---------------------------------------


def test_header_settings_save_and_read_back(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/profile/cv-header",
        data={
            "csrf_token": csrf(client, "/profile"),
            "cv_name": "Robin Example",
            "cv_tagline": "Engineering Manager | Platform",
            "cv_phone": "+44 7700 900000",
            "cv_email": "robin@example.test",
            "cv_location": "Bristol, UK",
            "link_label_1": "",
            "link_url_1": "github.com/robin-example",
            "interests": "Sea swimming\n\nChess",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/profile?saved=1&open=cv-header")

    with engine.begin() as conn:
        profile = PostgresProfileRepository(conn, user_id).current()
    assert profile.cv_header.name == "Robin Example"
    assert profile.cv_header.links[0].url == "https://github.com/robin-example"
    assert profile.cv_header.links[0].label == "github.com/robin-example"
    assert profile.interests == ["Sea swimming", "Chess"]

    text = client.get("/profile?saved=1&open=cv-header").text
    assert "settings, not claims" in text
    assert "robin@example.test" in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cv_email", "not-an-email"),
        ("link_url_1", "javascript:alert(1)"),
        ("cv_phone", "call me"),
    ],
)
def test_header_settings_are_validated_and_nothing_is_stored(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    field: str,
    value: str,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    data = {"csrf_token": csrf(client, "/profile"), "cv_name": "Robin", field: value}
    if field == "link_url_1":
        data["link_label_1"] = "Mine"
    response = client.post("/profile/cv-header", data=data, follow_redirects=False)
    assert response.status_code == 400
    assert 'role="alert"' in response.text
    with engine.begin() as conn:
        assert PostgresProfileRepository(conn, user_id).history() == []


def test_a_link_label_without_an_address_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post(
        "/profile/cv-header",
        data={"csrf_token": csrf(client, "/profile"), "link_label_1": "GitHub"},
    )
    assert response.status_code == 400
    assert "needs a URL" in response.text


def test_header_settings_never_reach_the_facts_or_the_check(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    client.post(
        "/profile/cv-header",
        data={
            "csrf_token": csrf(client, "/profile"),
            "cv_name": "Robin Example",
            "cv_email": "robin@example.test",
            "interests": "Sea swimming",
        },
    )
    with engine.begin() as conn:
        spans = conn.execute(
            select(spans_table.c.text).where(spans_table.c.user_id == user_id)
        ).all()
        tasks = conn.execute(select(tasks_table).where(tasks_table.c.user_id == user_id)).all()
    assert spans == []
    assert tasks == []


def test_header_post_needs_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post("/profile/cv-header", data={"cv_name": "X"})
    assert response.status_code == 403


# -- the CV as it reads, and its check ----------------------------------------


def test_the_preview_shows_every_section_and_the_check_beside_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    text = page(client, app_id)

    # Both templates are previewed, from render_cv_html, in iframes.
    assert text.count("<iframe") == 2
    srcdocs = [html.unescape(s) for s in re.findall(r'srcdoc="([^"]*)"', text)]
    for words in (
        "Robin Example",
        "Engineering leader for platform teams.",
        "What I Bring",
        "Ran the build farm for four teams.",
        "Professional Experience",
        "Harbour Logistics",
        "Jan 2021 – Present",
        "Cut cloud costs by 40%.",
        "Education &amp; Credentials",
        "BSc Computing, Example University, 2008",
        "Personal Interests",
        "Sea swimming",
    ):
        assert all(words in doc for doc in srcdocs), words
    # No gate annotation reaches the preview.
    assert all("No figure in the facts." not in doc for doc in srcdocs)

    # The check, in the page's four marks, flagged first, with the next step.
    assert "Needs your attention (2)" in text
    assert "Not supported" in text and "Check this" in text
    assert "No figure in the facts." in text
    assert "Edit this line" in text
    # Framing reads as not checked, never as supported -- and so does the
    # role's descriptor, which the claim gate is never sent (it describes the
    # employer, not the person), so it must not look as if it passed.
    assert "Not checked (2)" in text
    assert "Freight routing software." in text
    assert "Describes the employer, not you, so it is never checked." in text


def test_the_warning_appears_and_the_download_still_works(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    version = store(engine, user_id, app_id, sample_doc())
    text = page(client, app_id)
    assert "2 lines aren&#39;t backed by your confirmed facts." in text
    assert "It still downloads" in text

    pdf = client.get(f"/applications/{app_id}/cv/{version.id}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert (
        pdf.headers["content-disposition"]
        == 'attachment; filename="Robin_Example_CV_Fictional_Freight.pdf"'
    )
    # Exactly the stored version, rendered -- nothing regenerated or added.
    assert pdf.content == render_cv_pdf(version.document)
    assert b"No figure in the facts." not in pdf.content

    txt = client.get(f"/applications/{app_id}/cv/{version.id}/text")
    assert txt.status_code == 200
    assert txt.headers["content-type"].startswith("text/plain")
    assert 'filename="Robin_Example_CV_Fictional_Freight.txt"' in txt.headers["content-disposition"]
    assert "Cut cloud costs by 40%." in txt.text
    assert "No figure in the facts." not in txt.text
    assert "Not supported" not in txt.text


def test_no_warning_when_every_line_is_backed(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    doc = sample_doc()
    doc.roles[0].bullets = [doc.roles[0].bullets[0]]
    store(engine, user_id, app_id, doc)
    text = page(client, app_id)
    assert "backed by your confirmed facts." not in text
    assert "It still downloads" not in text


# -- editing ------------------------------------------------------------------


def test_editing_a_line_writes_a_new_version_marked_as_the_users(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    first = store(engine, user_id, app_id, sample_doc())

    form = edit_form(client, app_id)
    assert form["base_version"] == str(first.id)
    # Facts are shown on the edit page but never as fields.
    assert not any(k.endswith((".title", ".employer", ".dates", ".location")) for k in form)
    assert not any(k.startswith(("education.", "header", "interests.")) for k in form)
    form["roles.0.bullets.1"] = "Cut cloud costs."
    form["skills_heading"] = "Skills"
    response = client.post(f"/applications/{app_id}/cv/edit", data=form, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].endswith("?cv=saved#cv-document")

    stored = versions(engine, user_id, app_id)
    assert [v.status for v in stored] == ["edited", "generated"]
    assert stored[1].id == first.id
    latest = stored[0].document
    bullet = latest.roles[0].bullets[1]
    assert (bullet.text, bullet.origin, bullet.verdict, bullet.note) == (
        "Cut cloud costs.",
        "user",
        None,
        "",
    )
    assert latest.skills_heading == "Skills"
    # Untouched lines keep their verdicts; the first version is unchanged.
    assert latest.roles[0].bullets[0].verdict == "supported"
    assert stored[1].document == first.document

    text = page(client, app_id)
    assert "Not checked since you edited it." in text
    assert "1 line you edited hasn&#39;t been checked." in text
    assert "Check my edits" in text
    assert "on your own API key" in text


def test_a_crafted_post_changing_a_fact_is_refused_and_nothing_is_stored(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    for field, value in (
        ("roles.0.title", "Chief Technology Officer"),
        ("roles.0.dates", "Jan 2015 – Present"),
        ("roles.0.employer", "Somewhere Grander"),
        ("education.0", "PhD, Example University"),
        ("header.name", "Someone Else"),
    ):
        form = edit_form(client, app_id)
        form["roles.0.bullets.1"] = "Also changed."
        form[field] = value
        response = client.post(f"/applications/{app_id}/cv/edit", data=form)
        assert response.status_code == 400, field
        assert "can&#39;t be changed here" in response.text
    assert len(versions(engine, user_id, app_id)) == 1


def test_an_unchanged_submission_saves_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    response = client.post(
        f"/applications/{app_id}/cv/edit", data=edit_form(client, app_id), follow_redirects=False
    )
    assert response.headers["location"].endswith("?cv=unchanged#cv-document")
    assert len(versions(engine, user_id, app_id)) == 1


def test_an_edit_against_an_older_version_is_not_saved(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    form = edit_form(client, app_id)
    store(engine, user_id, app_id, sample_doc(template="classic"))
    form["summary.0"] = "Changed."
    response = client.post(f"/applications/{app_id}/cv/edit", data=form, follow_redirects=False)
    assert response.headers["location"].endswith("/cv/edit?stale=1")
    assert len(versions(engine, user_id, app_id)) == 2


# -- "Check my edits" ---------------------------------------------------------


def test_check_my_edits_enqueues_once_with_only_the_edited_lines(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    form = edit_form(client, app_id)
    form["roles.0.bullets.1"] = "Cut cloud costs."
    form["summary.0"] = "Engineering leader."
    client.post(f"/applications/{app_id}/cv/edit", data=form)
    latest = versions(engine, user_id, app_id)[0]

    for _ in range(2):
        response = client.post(
            f"/applications/{app_id}/cv/check",
            data={"csrf_token": csrf(client)},
            follow_redirects=False,
        )
        assert response.headers["location"].endswith("?cv=checking#cv-document")
    tasks = check_tasks(engine, user_id)
    assert len(tasks) == 1
    assert tasks[0].payload == {
        "application_id": app_id,
        "version_id": str(latest.id),
        "paths": ["summary.0", "roles.0.bullets.1"],
    }
    assert "Checking your edits now" in page(client, app_id)


def test_check_my_edits_with_nothing_edited_queues_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    response = client.post(
        f"/applications/{app_id}/cv/check",
        data={"csrf_token": csrf(client)},
        follow_redirects=False,
    )
    assert response.headers["location"].endswith("?cv=nothing#cv-document")
    assert check_tasks(engine, user_id) == []


class _FakeOutput:
    def __init__(self, sentences: list[Any]) -> None:
        self.sentences = sentences

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        # The handler stores the gate's raw output on the `checked` version.
        return {"sentences": [vars(s) for s in self.sentences]}


class _FakeSentence:
    def __init__(self, text: str, kind: str, verdict: str | None, note: str = "") -> None:
        self.text, self.kind, self.verdict, self.evidence_note = text, kind, verdict, note


def test_the_check_handler_writes_verdicts_as_a_new_version(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    store(engine, user_id, app_id, sample_doc())
    form = edit_form(client, app_id)
    form["roles.0.bullets.1"] = "Cut cloud costs."
    client.post(f"/applications/{app_id}/cv/edit", data=form)
    client.post(f"/applications/{app_id}/cv/check", data={"csrf_token": csrf(client)})
    row = check_tasks(engine, user_id)[0]
    task = Task(
        id=row.id,
        user_id=row.user_id,
        kind=row.kind,
        payload=row.payload,
        status="running",
        attempts=1,
        max_attempts=row.max_attempts,
        scheduled_at=row.scheduled_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )

    given: list[str] = []

    def fake_check_text(request: Any, grounding: Any, runs: Any, text: str) -> _FakeOutput:
        given.append(text)
        return _FakeOutput([_FakeSentence("Cut cloud costs.", "claim", "review", "Partly.")])

    monkeypatch.setattr(cv_edits_check, "load_api_key", lambda repo, key: "sk-test-fake")
    monkeypatch.setattr(cv_edits_check, "check_text", fake_check_text)
    handler = cv_edits_check.build_check_cv_edits(master_key=MasterKey.generate(), model="m")
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    handler(ctx)

    assert given == ["Cut cloud costs."]  # the edited line only
    stored = versions(engine, user_id, app_id)
    assert [v.status for v in stored][:2] == ["checked", "edited"]
    assert stored[0].trace_id == task.id
    bullet = stored[0].document.roles[0].bullets[1]
    assert (bullet.verdict, bullet.note, bullet.origin) == ("review", "Partly.", "user")

    # A redelivery does nothing.
    assert handler(ctx)["skipped"] == "already checked"
    assert len(versions(engine, user_id, app_id)) == 3


# -- template, header, versions -----------------------------------------------


def test_template_switch_previews_both_and_saves_a_new_version(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    first = store(engine, user_id, app_id, sample_doc())
    text = page(client, app_id)
    assert 'value="classic"' in text and 'value="modern" checked' in text
    assert "cv-frame-modern is-current" in text

    response = client.post(
        f"/applications/{app_id}/cv/template",
        data={"csrf_token": csrf(client), "template": "classic", "base_version": str(first.id)},
        follow_redirects=False,
    )
    assert response.headers["location"].endswith("?cv=template#cv-document")
    stored = versions(engine, user_id, app_id)
    assert stored[0].document.template == "classic"
    assert stored[0].document.model_copy(update={"template": "modern"}) == first.document
    assert "cv-frame-classic is-current" in page(client, app_id)

    bad = client.post(
        f"/applications/{app_id}/cv/template",
        data={"csrf_token": csrf(client), "template": "garish", "base_version": str(stored[0].id)},
    )
    assert bad.status_code == 200
    assert len(versions(engine, user_id, app_id)) == 2


def test_header_refresh_takes_the_profile_settings(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    first = store(engine, user_id, app_id, sample_doc())
    client.post(
        "/profile/cv-header",
        data={
            "csrf_token": csrf(client, "/profile"),
            "cv_email": "robin@example.test",
            "interests": "Bouldering",
        },
    )
    client.post(
        f"/applications/{app_id}/cv/header",
        data={"csrf_token": csrf(client), "base_version": str(first.id)},
    )
    latest = versions(engine, user_id, app_id)[0].document
    assert latest.header.name == "Robin Example"  # kept: the profile left it blank
    assert latest.header.contact == ["robin@example.test"]
    assert latest.interests == ["Bouldering"]


def test_older_versions_stay_listed_and_downloadable(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    first = store(engine, user_id, app_id, sample_doc())
    store(engine, user_id, app_id, sample_doc(template="classic"))
    text = page(client, app_id)
    assert "Earlier versions of this CV" in text
    assert "Version 1" in text
    assert f"/cv/{first.id}/pdf" in text
    # Folded by default, and still carrying its content.
    section = re.search(r'<details[^>]*data-section="cv.versions"[^>]*>', text)
    assert section is not None
    assert not re.search(r"\sopen\b", section.group(0).replace("data-default-open", ""))
    old = client.get(f"/applications/{app_id}/cv/{first.id}/pdf")
    assert old.status_code == 200
    assert old.content == render_cv_pdf(first.document)


# -- tenancy and CSRF ---------------------------------------------------------


def test_another_users_cv_404s_on_every_route(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    alice_id = sign_in(client, google, subs, engine)
    alice_app = add_application(client, engine)
    alice_version = store(engine, alice_id, alice_app, sample_doc())

    bob_id = sign_in(client, google, subs, engine)
    token = csrf(client)
    base = f"/applications/{alice_app}/cv"
    assert client.get(f"{base}/edit").status_code == 404
    assert client.get(f"{base}/{alice_version.id}/pdf").status_code == 404
    assert client.get(f"{base}/{alice_version.id}/text").status_code == 404
    assert client.post(f"{base}/edit", data={"csrf_token": token}).status_code == 404
    assert (
        client.post(
            f"{base}/template",
            data={
                "csrf_token": token,
                "template": "classic",
                "base_version": str(alice_version.id),
            },
        ).status_code
        == 404
    )
    assert client.post(f"{base}/header", data={"csrf_token": token}).status_code == 404
    assert client.post(f"{base}/check", data={"csrf_token": token}).status_code == 404
    assert check_tasks(engine, bob_id) == []
    assert len(versions(engine, alice_id, alice_app)) == 1

    # Bob's own application cannot reach Alice's version by its id either.
    bob_app = add_application(client, engine)
    assert client.get(f"/applications/{bob_app}/cv/{alice_version.id}/pdf").status_code == 404
    with engine.begin() as conn:
        written = PostgresCvDocumentRepository(conn, bob_id).add_version(
            uuid.UUID(alice_app), sample_doc(), status="edited", trace_id=None
        )
    assert written is None
    assert len(versions(engine, alice_id, alice_app)) == 1


def test_every_cv_post_needs_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client, engine)
    version = store(engine, user_id, app_id, sample_doc())
    base = f"/applications/{app_id}/cv"
    for path, data in (
        ("edit", {"base_version": str(version.id), "summary.0": "Changed."}),
        ("template", {"base_version": str(version.id), "template": "classic"}),
        ("header", {"base_version": str(version.id)}),
        ("check", {}),
    ):
        assert client.post(f"{base}/{path}", data=data).status_code == 403, path
        bad = {**data, "csrf_token": "not-the-token"}
        assert client.post(f"{base}/{path}", data=bad).status_code == 403, path
    assert len(versions(engine, user_id, app_id)) == 1
    assert check_tasks(engine, user_id) == []
