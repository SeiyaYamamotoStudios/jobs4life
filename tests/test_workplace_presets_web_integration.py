"""The workplace presets on /jobs, and the board's "hybrid too heavy" setting,
through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper, CSRF scraping and worker-shaped
seeding as `test_jobs_web_integration.py`, so no request leaves the machine and
no model is called.
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
from jfl_core.models import BoardPlatform, ObservedJob, Workplace
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.job_filters import PostgresJobFilterRepository
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


def sign_out(client: TestClient) -> None:
    client.post("/logout", data={"csrf_token": csrf(client)})


def csrf(client: TestClient, path: str = "/jobs") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def job(
    ext: str,
    title: str,
    workplace: Workplace,
    locations: tuple[str, ...],
    label: str | None = None,
) -> ObservedJob:
    location = "; ".join(locations)
    return ObservedJob(
        external_id=ext,
        title=title,
        location=location,
        url=f"https://example.invalid/jobs/{ext}",
        fingerprint=fingerprint(title, location),
        workplace=workplace,
        workplace_label=label,
        locations=locations,
    )


def seed_board(
    engine: Engine,
    user_id: uuid.UUID,
    *,
    platform: BoardPlatform,
    label: str,
    jobs: list[ObservedJob],
) -> uuid.UUID:
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        board = repo.add_board(
            platform=platform,
            board_url=f"https://example.invalid/{label}-{uuid.uuid4().hex[:6]}",
            board_key={"token": f"{label}-{uuid.uuid4().hex[:8]}"},
            label=label,
        )
        at = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
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
    return board.id


def save_filter(client: TestClient, mode: str, **fields: object) -> Response:
    data: dict[str, object] = {
        "csrf_token": csrf(client),
        "workplace_mode": mode,
        "title_includes": "engineering manager",
        **fields,
    }
    return client.post("/jobs/filter", data=data, follow_redirects=False)


@pytest.fixture
def seeded(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> dict[str, uuid.UUID]:
    """Anthropic on Greenhouse, shaped like the live board on 2026-09-15, and
    Acme on Ashby, whose hybrid comes from an enum with no label.
    """
    user_id = sign_in(client, google, subs, engine)
    anthropic = [
        job("a1", "Engineering Manager, Strict", "remote", ("Remote, US",), "Remote"),
        job(
            "a2",
            "Engineering Manager, Compute",
            "onsite",
            ("Remote-Friendly (Travel-Required)", "Canada"),
            "On-Site",
        ),
        job(
            "a3",
            "Engineering Manager, Industries",
            "hybrid",
            ("New York City, NY",),
            "Hybrid (Travel-Required)",
        ),
        job("a4", "Engineering Manager, Office", "onsite", ("London, UK",), "On-Site"),
    ]
    acme = [job("b1", "Engineering Manager, Acme", "hybrid", ("Berlin, Germany",))]
    return {
        "user": user_id,
        "anthropic": seed_board(
            engine, user_id, platform="greenhouse", label="Anthropic", jobs=anthropic
        ),
        "acme": seed_board(engine, user_id, platform="ashby", label="Acme", jobs=acme),
    }


# -- saving a mode ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["remote_only", "remote_friendly", "custom"])
def test_each_mode_saves_and_is_rechecked_on_the_form(
    client: TestClient, seeded: dict[str, uuid.UUID], engine: Engine, mode: str
) -> None:
    response = save_filter(client, mode, workplace=["hybrid"])
    assert response.status_code == 303 and response.headers["location"] == "/jobs?status=saved"
    html = client.get("/jobs").text
    assert re.search(rf'name="workplace_mode" value="{mode}"\s+checked', html)
    # The custom ticks are kept whatever the mode, so Custom comes back as it was.
    assert re.search(r'name="workplace" value="hybrid"\s+checked', html)
    with engine.begin() as conn:
        saved = PostgresJobFilterRepository(conn, seeded["user"]).get_filter()
    assert saved.workplace_mode == mode and saved.workplaces == ["hybrid"]


def test_a_mode_outside_the_choices_is_refused_and_nothing_is_saved(
    client: TestClient, seeded: dict[str, uuid.UUID], engine: Engine
) -> None:
    save_filter(client, "remote_only")
    response = save_filter(client, "mostly_remote", title_includes="changed")
    assert response.status_code == 400
    assert "mostly_remote" not in response.text  # never echoed back
    with engine.begin() as conn:
        saved = PostgresJobFilterRepository(conn, seeded["user"]).get_filter()
    assert saved.workplace_mode == "remote_only" and saved.title_includes == "engineering manager"


def test_saving_a_mode_requires_csrf(client: TestClient, seeded: dict[str, uuid.UUID]) -> None:
    response = client.post(
        "/jobs/filter", data={"csrf_token": "wrong", "workplace_mode": "remote_only"}
    )
    assert response.status_code == 403


def test_the_form_says_the_checkboxes_apply_only_under_custom(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    text = text_of(client.get("/jobs").text)
    assert "Used only when Custom is chosen" in text
    assert text.index("Remote only") < text.index("Remote friendly") < text.index("Custom")


# -- what each mode shows ---------------------------------------------------------------------


def test_remote_only_shows_only_strict_remote(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    save_filter(client, "remote_only")
    text = text_of(client.get("/jobs").text)
    assert "Engineering Manager, Strict" in text
    for hidden in ("Compute", "Industries", "Office", "Acme"):
        assert f"Engineering Manager, {hidden}" not in text
    assert "1 matching of 5 open jobs across 2 boards" in text
    assert "— days not stated" not in text


def test_remote_only_keeps_a_remote_job_whose_posting_says_remote_friendly(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Owner ruling, 2026-09-16: the structured field says Remote, the location
    says Remote-Friendly. Remote only keeps it, with the words shown on the row.
    """
    user_id = sign_in(client, google, subs, engine)
    seed_board(
        engine,
        user_id,
        platform="greenhouse",
        label="Anthropic",
        jobs=[
            job(
                "r1",
                "Engineering Manager, Remote-Friendly",
                "remote",
                ("Remote-Friendly (Travel-Required)", "San Francisco, CA"),
                "Remote",
            ),
            job("r2", "Engineering Manager, Compute", "onsite", ("Remote-Friendly",), "On-Site"),
        ],
    )
    save_filter(client, "remote_only")
    text = text_of(client.get("/jobs").text)
    assert "Engineering Manager, Remote-Friendly" in text
    assert "Remote The posting says Remote-Friendly" in text
    # Words alone never bring in a job whose field is not remote.
    assert "Engineering Manager, Compute" not in text
    assert "1 matching of 2 open jobs" in text


def test_remote_friendly_shows_hybrid_badged_and_the_conflict_on_the_row(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    save_filter(client, "remote_friendly")
    text = text_of(client.get("/jobs").text)
    for shown in ("Strict", "Compute", "Industries", "Acme"):
        assert f"Engineering Manager, {shown}" in text
    assert "Engineering Manager, Office" not in text

    # Employer labels verbatim, with what the evidence says beside them.
    assert "On-Site Listed On-Site · the posting also says Remote-Friendly" in text
    assert "Hybrid (Travel-Required) — days not stated" in text
    assert "Hybrid — days not stated" in text  # Acme's enum, no label
    assert "4 matching of 5 open jobs across 2 boards." in text
    assert "2 of them are hybrid with days not stated." in text


def test_the_hybrid_count_is_said_only_under_remote_friendly(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    save_filter(client, "custom", workplace=["hybrid"])
    text = text_of(client.get("/jobs").text)
    assert "Hybrid — days not stated" in text  # the row still says so
    assert "hybrid with days not stated." not in text


def test_an_exception_note_replaces_days_not_stated(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    save_filter(client, "remote_friendly")
    board_id = seeded["acme"]
    response = client.post(
        f"/boards/{board_id}/exceptions",
        data={
            "csrf_token": csrf(client, f"/boards/{board_id}"),
            "workplace": ["hybrid"],
            "note": "~1 day a week in Berlin",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    text = text_of(client.get("/jobs").text)
    assert "hybrid, per your Acme exception: ~1 day a week in Berlin" in text
    assert "Hybrid (Travel-Required) — days not stated" in text  # Anthropic's is unchanged
    assert "1 of them is hybrid with days not stated." in text


# -- the board's hybrid-too-heavy setting ----------------------------------------------------------


def set_too_heavy(client: TestClient, board_id: uuid.UUID, setting: str) -> Response:
    return client.post(
        f"/boards/{board_id}/hybrid",
        data={"csrf_token": csrf(client, f"/boards/{board_id}"), "setting": setting},
        follow_redirects=False,
    )


def test_a_board_marked_too_heavy_contributes_only_plain_remote_to_remote_friendly(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    board_id = seeded["anthropic"]
    detail = text_of(client.get(f"/boards/{board_id}").text)
    assert "Hybrid here is more than about a day a week — leave it out of remote friendly" in detail
    assert "Currently: included" in detail

    save_filter(client, "remote_friendly")
    response = set_too_heavy(client, board_id, "too_heavy")
    assert response.status_code == 303
    assert response.headers["location"] == f"/boards/{board_id}?status=setting_saved"
    assert "Currently: left out" in text_of(client.get(f"/boards/{board_id}").text)

    text = text_of(client.get("/jobs").text)
    assert "Engineering Manager, Strict" in text
    assert "Engineering Manager, Compute" not in text
    assert "Engineering Manager, Industries" not in text
    assert "Engineering Manager, Acme" in text  # another board's hybrid is unaffected
    assert "2 matching of 5 open jobs" in text

    set_too_heavy(client, board_id, "include")
    assert "Engineering Manager, Industries" in text_of(client.get("/jobs").text)


def test_the_hybrid_setting_rejects_values_outside_its_choices_and_requires_csrf(
    client: TestClient, seeded: dict[str, uuid.UUID], engine: Engine
) -> None:
    board_id = seeded["anthropic"]
    assert set_too_heavy(client, board_id, "sometimes").status_code == 400
    forged = client.post(
        f"/boards/{board_id}/hybrid", data={"csrf_token": "wrong", "setting": "too_heavy"}
    )
    assert forged.status_code == 403
    with engine.begin() as conn:
        board = PostgresBoardRepository(conn, seeded["user"]).get_board(board_id)
    assert board is not None and board.hybrid_too_heavy is False


def test_another_users_board_setting_is_a_404_and_unchanged(
    client: TestClient,
    seeded: dict[str, uuid.UUID],
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
) -> None:
    board_id = seeded["anthropic"]
    sign_out(client)
    sign_in(client, google, subs, engine)
    response = client.post(
        f"/boards/{board_id}/hybrid",
        data={"csrf_token": csrf(client), "setting": "too_heavy"},
    )
    assert response.status_code == 404
    with engine.begin() as conn:
        board = PostgresBoardRepository(conn, seeded["user"]).get_board(board_id)
    assert board is not None and board.hybrid_too_heavy is False
