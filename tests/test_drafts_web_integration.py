"""B5's drafting screen, through the real routes against a live Postgres --
`jfl_web.routes.drafts`. Marked `integration`; needs `docker compose up -d`
and `alembic upgrade head`. Same stub Google provider, sign-in helper and CSRF
scraping as `test_title_suggestions_web_integration.py`.

No model is ever called here: the enqueue path is checked by inspecting the
`tasks` table, never by letting a task run. Coverage, requirements and drafts
this file needs are written directly through the repositories, the same way
`test_title_suggestions_web_integration.py` writes `title_suggestions` rows
directly rather than running the worker.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import uuid
from collections.abc import Iterator
from decimal import Decimal

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import requirement_id
from jfl_core.models import Draft, JobRequirement, RequirementCoverage, RunRecord
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.postgres import PostgresJobRepository, PostgresRunRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

GENERATE_COVERAGE_KIND = "generate_coverage"
GENERATE_CV_DRAFT_KIND = "generate_cv_draft"


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


def csrf(client: TestClient, path: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def add_application(
    client: TestClient, *, job_ad: str = "Senior Engineer\n\nAcme. Own the deployment pipeline."
) -> str:
    response = client.post(
        "/applications",
        data={"csrf_token": csrf(client, "/applications/new"), "job_ad": job_ad, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return response.headers["location"].rsplit("/", 1)[-1]


def get_job_id(engine: Engine, user_id: uuid.UUID, application_id: str) -> uuid.UUID:
    with engine.begin() as conn:
        detail = PostgresApplicationRepository(conn, user_id).get_application(
            uuid.UUID(application_id)
        )
    assert detail is not None and detail.application.job_id is not None
    return detail.application.job_id


def add_requirements(
    engine: Engine, user_id: uuid.UUID, job_id: uuid.UUID, texts: list[str]
) -> list[JobRequirement]:
    with engine.begin() as conn:
        job_repo = PostgresJobRepository(conn)
        requirements = [
            JobRequirement(
                id=requirement_id(job_id, text),
                user_id=user_id,
                job_id=job_id,
                ordinal=i,
                text=text,
                necessity="essential",
            )
            for i, text in enumerate(texts)
        ]
        job_repo.replace_requirements(user_id, job_id, requirements)
    return requirements


def add_coverage(engine: Engine, user_id: uuid.UUID, requirements: list[JobRequirement]) -> None:
    with engine.begin() as conn:
        job_repo = PostgresJobRepository(conn)
        for requirement in requirements:
            job_repo.record_coverage(
                RequirementCoverage(
                    user_id=user_id,
                    requirement_id=requirement.id,
                    trace_id=uuid.uuid4(),
                    status="evidenced",
                    cited_span_ids=[],
                    evidence_note="Traces to the corpus.",
                )
            )


def enqueued_tasks(engine: Engine, user_id: uuid.UUID, kind: str) -> list[object]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table).where(
                    tasks_table.c.user_id == user_id, tasks_table.c.kind == kind
                )
            ).all()
        )


CITED_SPAN_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CITED_FACT_TEXT = "Led the platform team at Acme: eight engineers, hiring and on-call."


def add_cited_fact(engine: Engine, user_id: uuid.UUID) -> None:
    """The span the fake gate result cites, so the page can show its words.

    Without it the citation resolves to nothing, which is the *other* case the
    screen has to handle -- a fact the user has since changed.
    """
    with engine.begin() as conn:
        conn.execute(
            insert(spans_table).values(
                id=CITED_SPAN_ID,
                user_id=user_id,
                document_id=None,
                provenance="adjudicated",
                kind="paragraph",
                section_path=None,
                ordinal=0,
                text=CITED_FACT_TEXT,
                content_hash="0" * 64,
                char_start=None,
                char_end=None,
                retired_at=None,
            )
        )


def mark_task_succeeded(engine: Engine, task_id: uuid.UUID) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(tasks_table).where(tasks_table.c.id == task_id).values(status="succeeded")
        )


_GATE_RESULT = {
    "sentences": [
        {
            "index": 1,
            "kind": "title",
            "verdict": None,
            "drift_label": None,
            "cited_span_ids": [],
            "evidence_note": "Document title: not checked against the corpus.",
            "rule_flags": [],
            "text": "CV bullets for Acme",
        },
        {
            "index": 2,
            "kind": "claim",
            "verdict": "supported",
            "drift_label": "supported",
            "cited_span_ids": ["11111111-1111-1111-1111-111111111111"],
            "evidence_note": "Traces cleanly to the corpus.",
            "rule_flags": [],
            "text": "Led the platform team at Acme.",
        },
        {
            "index": 3,
            "kind": "claim",
            "verdict": "review",
            "drift_label": "scope_inflation",
            "cited_span_ids": [],
            "evidence_note": "The corpus does not state the team's size.",
            "rule_flags": [],
            "text": "Managed a large engineering organisation.",
        },
        {
            "index": 4,
            "kind": "claim",
            "verdict": "unsupported",
            "drift_label": "invented_quantity",
            "cited_span_ids": [],
            "evidence_note": "The corpus names no such figure.",
            "rule_flags": [],
            "text": "Cut costs by 40%.",
        },
        {
            "index": 5,
            "kind": "framing",
            "verdict": "supported",
            "drift_label": "framing",
            "cited_span_ids": [],
            "evidence_note": "Framing is never checked.",
            "rule_flags": [],
            "text": "Motivated by a desire to build lasting systems.",
        },
    ]
}


def add_draft(
    engine: Engine,
    user_id: uuid.UUID,
    job_id: uuid.UUID,
    *,
    trace_id: uuid.UUID,
    cost: Decimal | None = Decimal("0.4123"),
) -> Draft:
    draft = Draft(
        user_id=user_id,
        job_id=job_id,
        kind="cv_bullets",
        text="# CV bullets for Acme\n\nLed the platform team at Acme.",
        gate_result=_GATE_RESULT,
        trace_id=trace_id,
    )
    with engine.begin() as conn:
        PostgresJobRepository(conn).record_draft(draft)
        if cost is not None:
            PostgresRunRepository(conn).record(
                RunRecord(
                    user_id=user_id,
                    trace_id=trace_id,
                    component="generate",
                    stage="draft",
                    model="claude-opus-5",
                    cost_usd=cost,
                    outcome="ok",
                    started_at=dt.datetime.now(dt.UTC),
                )
            )
    return draft


# --------------------------------------------------------------------------
# Prerequisites are explicit, never silent
# --------------------------------------------------------------------------


def test_a_signed_out_visitor_is_redirected_to_login(client: TestClient) -> None:
    random_id = uuid.uuid4()
    response = client.get(f"/applications/{random_id}/drafts", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_no_requirements_yet_offers_no_generate_buttons(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The ad has been pasted but never read -- no requirements, so nothing to
    draft against, and no coverage button either (there is nothing to check).
    """
    sign_in(client, google, subs, engine)
    app_id = add_application(client)

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "has not been read yet" in page.lower()
    assert 'action="/applications/' not in page or "/drafts/coverage" not in page
    assert "Generate a CV" not in page


def test_requirements_with_no_coverage_offers_the_coverage_button(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "5+ years of Python" in page
    assert "has not been checked" in page.lower()
    assert f'action="/applications/{app_id}/drafts/coverage"' in page
    assert "Generate a CV" not in page


def test_coverage_present_offers_the_generate_buttons(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    requirements = add_requirements(engine, user_id, job_id, ["5+ years of Python"])
    add_coverage(engine, user_id, requirements)

    page = client.get(f"/applications/{app_id}/drafts").text
    assert "evidenced" in page.lower()
    assert "Generate a CV" in page
    assert "Generate a cover letter" in page
    assert f'action="/applications/{app_id}/drafts"' in page


# --------------------------------------------------------------------------
# The buttons enqueue, once, and redirect to the polling view
# --------------------------------------------------------------------------


def test_check_coverage_enqueues_once_and_redirects_with_the_task_id(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    response = client.post(
        f"/applications/{app_id}/drafts/coverage",
        data={"csrf_token": csrf(client, f"/applications/{app_id}/drafts")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith(f"/applications/{app_id}/drafts?task=")

    tasks = enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND)
    assert len(tasks) == 1
    assert tasks[0].payload == {"job_id": str(job_id)}

    # Loading the page again enqueues nothing further.
    client.get(location)
    assert len(enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND)) == 1


def test_generate_a_draft_enqueues_once_and_redirects_with_the_task_id(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    requirements = add_requirements(engine, user_id, job_id, ["5+ years of Python"])
    add_coverage(engine, user_id, requirements)

    response = client.post(
        f"/applications/{app_id}/drafts",
        data={
            "csrf_token": csrf(client, f"/applications/{app_id}/drafts"),
            "kind": "cv_bullets",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith(f"/applications/{app_id}/drafts?task=")

    tasks = enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND)
    assert len(tasks) == 1
    assert tasks[0].payload == {"application_id": app_id, "kind": "cv_bullets"}


def test_coverage_and_draft_posts_require_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    assert (
        client.post(
            f"/applications/{app_id}/drafts/coverage", data={"csrf_token": "wrong"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/applications/{app_id}/drafts", data={"csrf_token": "wrong", "kind": "cv_bullets"}
        ).status_code
        == 403
    )
    assert enqueued_tasks(engine, user_id, GENERATE_COVERAGE_KIND) == []
    assert enqueued_tasks(engine, user_id, GENERATE_CV_DRAFT_KIND) == []


# --------------------------------------------------------------------------
# The polling fragment: pending, then a rendered result
# --------------------------------------------------------------------------


def test_the_poll_panel_shows_progress_then_the_draft_with_verdicts_citations_and_cost(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    requirements = add_requirements(engine, user_id, job_id, ["5+ years of Python"])
    add_coverage(engine, user_id, requirements)
    add_cited_fact(engine, user_id)

    response = client.post(
        f"/applications/{app_id}/drafts",
        data={
            "csrf_token": csrf(client, f"/applications/{app_id}/drafts"),
            "kind": "cv_bullets",
        },
        follow_redirects=False,
    )
    location = response.headers["location"]
    task_id = uuid.UUID(location.rsplit("task=", 1)[-1])

    pending = client.get(location)
    assert pending.status_code == 200
    assert "takes about a minute" in pending.text
    assert f'hx-get="/applications/{app_id}/drafts/tasks/{task_id}"' in pending.text
    assert 'hx-trigger="every 3s"' in pending.text

    mark_task_succeeded(engine, task_id)
    add_draft(engine, user_id, job_id, trace_id=task_id, cost=Decimal("0.4123"))

    settled = client.get(location)
    assert settled.status_code == 200
    assert "hx-trigger" not in settled.text

    # Every verdict word appears somewhere, and framing/title render as NOT
    # CHECKED rather than SUPPORTED -- the one rule this screen exists to keep.
    assert "SUPPORTED" in settled.text
    assert "REVIEW" in settled.text
    assert "UNSUPPORTED" in settled.text
    # Two verdict badges (the title, and the framing sentence) plus one mention
    # in the legend explaining what NOT CHECKED means.
    assert settled.text.count("NOT CHECKED") == 3
    assert "Led the platform team at Acme." in settled.text
    assert "Motivated by a desire to build lasting systems." in settled.text

    # A citation shows the fact's own words, never its id: a UUID tells the
    # reader nothing about whether the sentence is actually supported.
    assert CITED_FACT_TEXT in settled.text
    assert str(CITED_SPAN_ID) not in settled.text

    # The per-run cost.
    assert "$0.4123" in settled.text

    # And the same fragment renders inline on the full page too.
    full_page = client.get(f"/applications/{app_id}/drafts?task={task_id}").text
    assert "SUPPORTED" in full_page
    assert "$0.4123" in full_page


def test_a_failed_task_shows_a_plain_sentence_never_the_raw_error(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    app_id = add_application(client)
    job_id = get_job_id(engine, user_id, app_id)
    add_requirements(engine, user_id, job_id, ["5+ years of Python"])

    response = client.post(
        f"/applications/{app_id}/drafts/coverage",
        data={"csrf_token": csrf(client, f"/applications/{app_id}/drafts")},
        follow_redirects=False,
    )
    location = response.headers["location"]
    task_id = uuid.UUID(location.rsplit("task=", 1)[-1])

    with engine.begin() as conn:
        conn.execute(
            update(tasks_table)
            .where(tasks_table.c.id == task_id)
            .values(
                status="failed",
                last_error="PermanentTaskError: coverage generation failed permanently: no_api_key",
            )
        )

    page = client.get(location)
    assert "own Anthropic API key" in page.text
    assert 'href="/settings"' in page.text
    assert "PermanentTaskError" not in page.text
    assert "no_api_key" not in page.text


def test_an_unknown_or_cross_user_task_id_shows_nothing_special_inline(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """`?task=` naming a task that does not belong to this user is treated as
    "nothing to show" on the main page -- it must never leak another user's
    task, and it must never 500.
    """
    sign_in(client, google, subs, engine)
    app_id = add_application(client)

    page = client.get(f"/applications/{app_id}/drafts?task={uuid.uuid4()}")
    assert page.status_code == 200
    assert "hx-trigger" not in page.text


def test_the_standalone_fragment_route_404s_for_an_unknown_task(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    app_id = add_application(client)

    response = client.get(f"/applications/{app_id}/drafts/tasks/{uuid.uuid4()}")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_another_users_application_404s_on_every_route(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    alice_app_id = add_application(client, job_ad="Alice's role\n\nAcme.")

    sign_in(client, google, subs, engine)  # Bob
    assert client.get(f"/applications/{alice_app_id}/drafts").status_code == 404
    assert (
        client.post(
            f"/applications/{alice_app_id}/drafts/coverage",
            data={"csrf_token": csrf(client, "/applications")},
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/applications/{alice_app_id}/drafts",
            data={"csrf_token": csrf(client, "/applications"), "kind": "cv_bullets"},
        ).status_code
        == 404
    )
    assert (
        client.get(f"/applications/{alice_app_id}/drafts/tasks/{uuid.uuid4()}").status_code == 404
    )
