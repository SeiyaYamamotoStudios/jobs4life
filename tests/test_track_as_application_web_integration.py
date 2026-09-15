"""Slice C7's "Track as application" button, and the paste-box fallback it
offers when a description could not be fetched, through the real routes
against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_boards_web_integration.py` and `test_jobs_web_integration.py`. Board
history is seeded the way the worker writes it -- `lock_check_state`, the pure
`plan_check`, `apply_check_plan` -- with a hand-built `FetchResult`, so no
request ever leaves the machine.

No Anthropic API call and no HTTP request to a board's own site anywhere in
this file: `POST /jobs/{id}/track` only writes a row and enqueues a task, and
`POST /applications/{id}/ad` is the same paste-and-enqueue path
`test_applications_web_integration.py` already exercises for a fresh
application.
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
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import ObservedJob
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
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


def csrf(client: TestClient, path: str = "/jobs") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def seed_board_job(
    engine: Engine, user_id: uuid.UUID, *, label: str = "Acme", title: str = "Senior Engineer"
) -> uuid.UUID:
    """A watched board with one open job, seeded the way the worker writes it."""
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        board = repo.add_board(
            platform="greenhouse",
            board_url=f"https://example.invalid/{label}-{uuid.uuid4().hex[:6]}",
            board_key={"token": f"{label}-{uuid.uuid4().hex[:8]}"},
            label=label,
        )
        observed = ObservedJob(
            external_id="1",
            title=title,
            location="London, UK",
            url="https://example.invalid/jobs/1",
            fingerprint=fingerprint(title, "London, UK"),
        )
        at = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
        state = repo.lock_check_state(
            board.id, observed_external_ids=["1"], closed_since=at - REPOST_WINDOW
        )
        assert state is not None
        result = FetchResult(status="complete", jobs=(observed,), expected_total=1)
        repo.apply_check_plan(
            plan_check(state, result, observed_at=at), started_at=at, finished_at=at
        )
        (job,) = repo.list_jobs(board.id)
    return job.id


def application_id_from_redirect(response: Response) -> str:
    location = response.headers["location"]
    assert location.startswith("/applications/")
    return location.rsplit("/", 1)[-1]


def application_row(engine: Engine, application_id: str) -> object:
    with engine.begin() as conn:
        return conn.execute(
            select(applications_table).where(applications_table.c.id == uuid.UUID(application_id))
        ).one()


def fetch_description_tasks(engine: Engine, application_id: str) -> list[object]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table.c.status, tasks_table.c.payload).where(
                    tasks_table.c.kind == "fetch_job_description",
                    tasks_table.c.payload["application_id"].astext == application_id,
                )
            ).all()
        )


# --------------------------------------------------------------------------
# Signed out
# --------------------------------------------------------------------------


def test_a_signed_out_visitor_is_redirected_to_login(client: TestClient) -> None:
    response = client.post(
        f"/jobs/{uuid.uuid4()}/track", data={"csrf_token": "x"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# --------------------------------------------------------------------------
# Tracking a job
# --------------------------------------------------------------------------


def test_tracking_a_job_creates_an_application_and_enqueues_the_fetch(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    job_id = seed_board_job(engine, user_id, label="Acme Co", title="Staff Engineer")

    response = client.post(
        f"/jobs/{job_id}/track", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 303
    application_id = application_id_from_redirect(response)

    row = application_row(engine, application_id)
    assert row.title == "Staff Engineer"  # the employer's own words
    assert row.title_is_provisional is False
    assert row.employer == "Acme Co"  # the board's label
    assert row.source == "Watched board"
    assert row.extraction_status == "pending"
    assert str(row.board_job_id) == str(job_id)
    assert row.job_id is None  # nothing fetched yet -- that is the worker's job

    tasks = fetch_description_tasks(engine, application_id)
    assert len(tasks) == 1
    assert tasks[0].status == "pending"
    assert tasks[0].payload == {"application_id": application_id}


def test_tracking_the_same_job_twice_redirects_to_the_existing_application(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    job_id = seed_board_job(engine, user_id)

    first = client.post(
        f"/jobs/{job_id}/track", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    first_id = application_id_from_redirect(first)

    second = client.post(
        f"/jobs/{job_id}/track", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    assert second.status_code == 303
    assert application_id_from_redirect(second) == first_id

    # Exactly one application and one fetch task -- the second click created
    # neither.
    with engine.begin() as conn:
        apps = conn.execute(
            select(applications_table.c.id).where(
                applications_table.c.user_id == user_id,
                applications_table.c.board_job_id == job_id,
            )
        ).all()
    assert len(apps) == 1
    assert len(fetch_description_tasks(engine, first_id)) == 1


def test_tracking_an_archived_applications_job_creates_a_fresh_one(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    job_id = seed_board_job(engine, user_id)

    first = client.post(
        f"/jobs/{job_id}/track", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    first_id = application_id_from_redirect(first)
    with engine.begin() as conn:
        PostgresApplicationRepository(conn, user_id).archive(uuid.UUID(first_id))

    second = client.post(
        f"/jobs/{job_id}/track", data={"csrf_token": csrf(client)}, follow_redirects=False
    )
    second_id = application_id_from_redirect(second)
    assert second_id != first_id

    with engine.begin() as conn:
        apps = conn.execute(
            select(applications_table.c.id, applications_table.c.archived_at).where(
                applications_table.c.user_id == user_id,
                applications_table.c.board_job_id == job_id,
            )
        ).all()
    assert len(apps) == 2


def test_tracking_an_unknown_job_is_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post(f"/jobs/{uuid.uuid4()}/track", data={"csrf_token": csrf(client)})
    assert response.status_code == 404


def test_tracking_another_users_job_is_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    owner_id = sign_in(client, google, subs, engine)
    job_id = seed_board_job(engine, owner_id)

    # Sign in as someone else in the same client session.
    other_id = sign_in(client, google, subs, engine)
    assert other_id != owner_id
    response = client.post(f"/jobs/{job_id}/track", data={"csrf_token": csrf(client)})
    assert response.status_code == 404

    with engine.begin() as conn:
        apps = conn.execute(
            select(applications_table.c.id).where(applications_table.c.board_job_id == job_id)
        ).all()
    assert apps == []


# --------------------------------------------------------------------------
# The paste-box fallback after a failed fetch
# --------------------------------------------------------------------------


def _tracked_application_with_failed_fetch(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    """What the database looks like after `fetch_job_description` gives up:
    a board-tracked application, no ad text, `failed` / `description_unavailable`.
    """
    job_id = seed_board_job(engine, user_id)
    with engine.begin() as conn:
        apps = PostgresApplicationRepository(conn, user_id)
        application = apps.create_application(
            title="Staff Engineer",
            source="Watched board",
            extraction_status="pending",
            board_job_id=job_id,
        )
        apps.fail_extraction(application.id, "description_unavailable")
    return application.id


def test_the_detail_page_offers_a_paste_box_after_a_failed_fetch(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = _tracked_application_with_failed_fetch(engine, user_id)

    page = client.get(f"/applications/{application_id}").text
    assert "read this job" in page.lower() and "description from the board" in page.lower()
    assert f'action="/applications/{application_id}/ad"' in page


def test_pasting_an_ad_after_a_failed_fetch_attaches_it_and_queues_extraction(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = _tracked_application_with_failed_fetch(engine, user_id)

    response = client.post(
        f"/applications/{application_id}/ad",
        data={
            "csrf_token": csrf(client, f"/applications/{application_id}"),
            "job_ad": "Staff Engineer at Acme. Own the deployment pipeline.",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/applications/{application_id}"

    row = application_row(engine, str(application_id))
    assert row.job_id is not None
    assert row.extraction_status == "pending"
    assert row.extraction_error_code is None

    with engine.begin() as conn:
        extract_tasks = conn.execute(
            select(tasks_table.c.status).where(
                tasks_table.c.kind == "extract_job_ad",
                tasks_table.c.payload["application_id"].astext == str(application_id),
            )
        ).all()
    assert len(extract_tasks) == 1
    assert extract_tasks[0].status == "pending"


def test_pasting_a_blank_ad_is_rejected_without_attaching_anything(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    application_id = _tracked_application_with_failed_fetch(engine, user_id)

    response = client.post(
        f"/applications/{application_id}/ad",
        data={
            "csrf_token": csrf(client, f"/applications/{application_id}"),
            "job_ad": "   ",
        },
    )
    assert response.status_code == 400
    row = application_row(engine, str(application_id))
    assert row.job_id is None
    assert row.extraction_status == "failed"


def test_pasting_an_ad_against_an_unknown_application_is_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post(
        f"/applications/{uuid.uuid4()}/ad",
        data={"csrf_token": csrf(client), "job_ad": "Some role."},
    )
    assert response.status_code == 404
