"""NEXT.md's task 4, through the real routes against a live Postgres: adding a
question, "check my answer", "draft one for me", both equally reachable, and
how a checked answer's verdicts render. Marked `integration`; needs
`docker compose up -d` and `alembic upgrade head`. Same stub Google provider,
sign-in helper and CSRF scraping as `test_applications_web_integration.py`.

No model is ever called here: the enqueue path is checked by inspecting the
`tasks` table, never by letting a task run -- same rule
`test_title_suggestions_web_integration.py` follows. Rendering of a *finished*
answer (verdicts, framing as NOT CHECKED, the assessment) is tested by seeding
a `done` row directly through the repository, never through a live call.
"""

from __future__ import annotations

import datetime as dt
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
from jfl_core.storage.application_questions import PostgresApplicationQuestionRepository
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


def sign_in(client: TestClient, google: StubGoogle, subs: list[str]) -> GoogleIdentity:
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}", email=f"{uuid.uuid4()}@test.invalid", display_name="T"
    )
    subs.append(identity.sub)
    google.identity = identity
    response = client.get("/auth/google/callback")
    assert response.status_code == 200, response.text
    return identity


def _user_id(engine: Engine, identity: GoogleIdentity) -> uuid.UUID:
    with engine.begin() as conn:
        return conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()


def _csrf(client: TestClient, path: str) -> str:
    page = client.get(path).text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None, f"no CSRF token found on {path}"
    return match.group(1)


def _add_application(client: TestClient, *, title: str = "Platform Engineer") -> str:
    ad = f"{title}\n\nWe are hiring. You will own the deployment pipeline."
    response = client.post(
        "/applications",
        data={"csrf_token": _csrf(client, "/applications/new"), "job_ad": ad, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def _add_question(client: TestClient, application_id: str, question_text: str) -> None:
    response = client.post(
        f"/applications/{application_id}/questions",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "question_text": question_text,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def _question_id(engine: Engine, user_id: uuid.UUID, application_id: str) -> uuid.UUID:
    with engine.begin() as conn:
        questions = PostgresApplicationQuestionRepository(conn, user_id).list_questions(
            uuid.UUID(application_id)
        )
    assert len(questions) == 1
    return questions[0].id


def _question_tasks(engine: Engine, kind: str) -> list[Any]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table.c.kind, tasks_table.c.payload, tasks_table.c.user_id).where(
                    tasks_table.c.kind == kind
                )
            ).all()
        )


# --------------------------------------------------------------------------
# Adding a question
# --------------------------------------------------------------------------


def test_add_question_then_it_appears_with_both_buttons(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why do you want to work here?")

    page = client.get(f"/applications/{app_id}").text
    assert "Why do you want to work here?" in page
    assert f"/applications/{app_id}/questions/" in page
    assert "/check" in page
    assert "/draft" in page
    assert "Check my answer" in page
    assert "Draft one for me" in page


def test_adding_a_question_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client)
    response = client.post(
        f"/applications/{app_id}/questions",
        data={"csrf_token": "wrong", "question_text": "Why this role?"},
    )
    assert response.status_code == 403


def test_a_blank_question_is_rejected(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client)
    response = client.post(
        f"/applications/{app_id}/questions",
        data={"csrf_token": _csrf(client, f"/applications/{app_id}"), "question_text": "   "},
    )
    assert response.status_code == 400


def test_adding_a_question_to_another_users_application_is_404(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    app_id = _add_application(client)

    sign_in(client, google, subs)  # a fresh session replaces the cookie
    response = client.post(
        f"/applications/{app_id}/questions",
        data={"csrf_token": _csrf(client, "/applications/new"), "question_text": "Why this role?"},
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# "Check my answer" and "Draft one for me" -- both equally reachable
# --------------------------------------------------------------------------


def test_check_my_answer_creates_a_pending_row_and_enqueues_a_task(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    response = client.post(
        f"/applications/{app_id}/questions/{question_id}/check",
        data={
            "csrf_token": _csrf(client, f"/applications/{app_id}"),
            "answer_text": "Because I care about reliability.",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    with engine.begin() as conn:
        answers = PostgresApplicationQuestionRepository(conn, user_id).list_answers(question_id)
    assert len(answers) == 1
    assert answers[0].kind == "user"
    assert answers[0].status == "pending"
    assert answers[0].answer_text == "Because I care about reliability."

    tasks = _question_tasks(engine, "check_application_answer")
    assert any(t.payload.get("answer_id") == str(answers[0].id) for t in tasks)

    # The page shows it as in flight, and polls for the result.
    page = client.get(f"/applications/{app_id}").text
    assert "Checking your answer" in page

    # The standalone fragment `_application_question.html`'s polling row
    # requests -- same content, same "in flight" state.
    poll_response = client.get(f"/applications/{app_id}/questions/{question_id}")
    assert poll_response.status_code == 200
    assert "Checking your answer" in poll_response.text
    assert f'hx-get="/applications/{app_id}/questions/{question_id}"' in poll_response.text


def test_draft_one_for_me_creates_a_pending_row_and_enqueues_a_task(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    response = client.post(
        f"/applications/{app_id}/questions/{question_id}/draft",
        data={"csrf_token": _csrf(client, f"/applications/{app_id}")},
        follow_redirects=False,
    )
    assert response.status_code == 303

    with engine.begin() as conn:
        answers = PostgresApplicationQuestionRepository(conn, user_id).list_answers(question_id)
    assert len(answers) == 1
    assert answers[0].kind == "draft"
    assert answers[0].status == "pending"
    assert answers[0].answer_text == ""

    tasks = _question_tasks(engine, "draft_application_answer")
    assert any(t.payload.get("answer_id") == str(answers[0].id) for t in tasks)

    page = client.get(f"/applications/{app_id}").text
    assert "Drafting an answer" in page


def test_check_answer_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    response = client.post(
        f"/applications/{app_id}/questions/{question_id}/check",
        data={"csrf_token": "wrong", "answer_text": "An answer."},
    )
    assert response.status_code == 403

    with engine.begin() as conn:
        assert PostgresApplicationQuestionRepository(conn, user_id).list_answers(question_id) == []


def test_draft_answer_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    response = client.post(
        f"/applications/{app_id}/questions/{question_id}/draft", data={"csrf_token": "wrong"}
    )
    assert response.status_code == 403


def test_a_blank_typed_answer_creates_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    response = client.post(
        f"/applications/{app_id}/questions/{question_id}/check",
        data={"csrf_token": _csrf(client, f"/applications/{app_id}"), "answer_text": "   "},
        follow_redirects=False,
    )
    assert response.status_code == 303

    with engine.begin() as conn:
        assert PostgresApplicationQuestionRepository(conn, user_id).list_answers(question_id) == []


# --------------------------------------------------------------------------
# Rendering a finished answer -- verdicts, framing as NOT CHECKED
# --------------------------------------------------------------------------


def test_verdicts_render_with_framing_as_not_checked_never_supported(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    with engine.begin() as conn:
        repo = PostgresApplicationQuestionRepository(conn, user_id)
        answer = repo.create_user_answer(question_id, "I have owned reliability for five years.")
        assert answer is not None
        repo.mark_done(
            answer.id,
            answer_text=None,
            gate_result={
                "sentences": [
                    {
                        "index": 1,
                        "kind": "claim",
                        "verdict": "unsupported",
                        "drift_label": "invented_quantity",
                        "cited_span_ids": [],
                        "evidence_note": "no such tenure documented",
                        "rule_flags": [],
                        "text": "I have owned reliability for five years.",
                    },
                    {
                        "index": 2,
                        "kind": "framing",
                        "verdict": "supported",
                        "drift_label": "framing",
                        "cited_span_ids": [],
                        "evidence_note": "",
                        "rule_flags": [],
                        "text": "It is work I find meaningful.",
                    },
                ]
            },
            assessment={
                "assessment": "Addresses motivation directly.",
                "gaps": "No example given.",
            },
            model="claude-opus-5",
            trace_id=uuid.uuid4(),
        )

    page = client.get(f"/applications/{app_id}").text
    assert "UNSUPPORTED" in page
    assert "NOT CHECKED" in page
    # The framing sentence must never render as SUPPORTED -- see CLAUDE.md's
    # "Known to be wrong": framing is the one path nothing ever checks.
    assert "SUPPORTED\n" not in page.upper().replace(">", ">\n")
    assert "Addresses motivation directly." in page
    assert "No example given." in page


def test_both_buttons_stay_present_after_a_finished_answer(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The page advises, it never prescribes: a finished check does not hide
    "draft one for me", and vice versa.
    """
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    with engine.begin() as conn:
        repo = PostgresApplicationQuestionRepository(conn, user_id)
        answer = repo.create_user_answer(question_id, "Because I care.")
        assert answer is not None
        repo.mark_done(
            answer.id,
            answer_text=None,
            gate_result={"sentences": []},
            assessment={"assessment": "Fine.", "gaps": ""},
            model="claude-opus-5",
            trace_id=uuid.uuid4(),
        )

    page = client.get(f"/applications/{app_id}").text
    assert f"/applications/{app_id}/questions/{question_id}/check" in page
    assert f"/applications/{app_id}/questions/{question_id}/draft" in page


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_another_users_question_is_404_on_every_route(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    identity = sign_in(client, google, subs)
    user_id = _user_id(engine, identity)
    app_id = _add_application(client)
    _add_question(client, app_id, "Why this role?")
    question_id = _question_id(engine, user_id, app_id)

    sign_in(client, google, subs)  # a fresh session replaces the cookie

    assert client.get(f"/applications/{app_id}/questions/{question_id}").status_code == 404
    assert (
        client.post(
            f"/applications/{app_id}/questions/{question_id}/check",
            data={"csrf_token": _csrf(client, "/applications/new"), "answer_text": "x"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/applications/{app_id}/questions/{question_id}/draft",
            data={"csrf_token": _csrf(client, "/applications/new")},
        ).status_code
        == 404
    )
