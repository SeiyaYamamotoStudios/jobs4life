"""The /jobs page, the board filter setting and board exceptions, through the real
routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_boards_web_integration.py`. Board history is seeded the way the worker
writes it -- `lock_check_state`, the pure `plan_check`, `apply_check_plan` -- with
hand-built `FetchResult`s, so no request ever leaves the machine and no model is
called.
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
from jfl_intake.adapters.base import FetchResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.normalise import fingerprint
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

NOTE = "Accepts ~25% in office — 1 day a week in London"


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
    """The page's visible text with tags stripped and whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def job(
    ext: str,
    title: str,
    workplace: Workplace = "unknown",
    locations: tuple[str, ...] = ("London, UK",),
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
    jobs: list[ObservedJob] | None,
) -> uuid.UUID:
    """A watched board, with one complete check of `jobs` -- or none if None."""
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, user_id)
        board = repo.add_board(
            platform=platform,
            board_url=f"https://example.invalid/{label}-{uuid.uuid4().hex[:6]}",
            board_key={"token": f"{label}-{uuid.uuid4().hex[:8]}"},
            label=label,
        )
        if jobs is not None:
            _check(repo, board.id, jobs, dt.datetime.now(dt.UTC) - dt.timedelta(days=2))
    return board.id


def _check(
    repo: PostgresBoardRepository, board_id: uuid.UUID, jobs: list[ObservedJob], at: dt.datetime
) -> None:
    state = repo.lock_check_state(
        board_id,
        observed_external_ids=[j.external_id for j in jobs],
        closed_since=at - REPOST_WINDOW,
    )
    assert state is not None
    result = FetchResult(status="complete", jobs=tuple(jobs), expected_total=len(jobs))
    repo.apply_check_plan(plan_check(state, result, observed_at=at), started_at=at, finished_at=at)


def anthropic_jobs() -> list[ObservedJob]:
    return [
        job("a1", "Engineering Manager, Inference", "onsite", ("London, UK",), "On-Site"),
        job("a2", "Engineering Manager, Remote", "remote", ("Remote-Friendly, US",), "Remote"),
        job("a3", "Sales Engineering Manager", "onsite", ("London, UK",), "On-Site"),
        job("a4", "Engineering Manager, Unstated", "unknown", ("Sydney, Australia",)),
    ]


def acme_jobs() -> list[ObservedJob]:
    return [
        job("b1", "Manager, Engineering", "remote", ("Berlin, Germany",)),
        job("b2", "Software Engineer", "remote", ("Berlin, Germany",)),
        job("b3", "Engineering Manager, Hybrid", "hybrid", ("London, UK",)),
        job("b4", "Engineering Manager, Not stated", "unknown", ("Paris, France",)),
    ]


def save_filter(client: TestClient, **fields: object) -> Response:
    data: dict[str, object] = {"csrf_token": csrf(client), **fields}
    return client.post("/jobs/filter", data=data, follow_redirects=False)


@pytest.fixture
def seeded(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> dict[str, uuid.UUID]:
    """Anthropic on Greenhouse (includes unstated by default), Acme on Ashby
    (does not), and Pending, never checked. Filter: remote, "engineering manager".
    """
    user_id = sign_in(client, google, subs, engine)
    ids = {
        "user": user_id,
        "anthropic": seed_board(
            engine, user_id, platform="greenhouse", label="Anthropic", jobs=anthropic_jobs()
        ),
        "acme": seed_board(engine, user_id, platform="ashby", label="Acme", jobs=acme_jobs()),
        "pending": seed_board(engine, user_id, platform="lever", label="Pending", jobs=None),
    }
    response = save_filter(client, workplace=["remote"], title_includes="engineering manager")
    assert response.status_code == 303 and response.headers["location"] == "/jobs?status=saved"
    return ids


# -- access ---------------------------------------------------------------------------


def test_signed_out_jobs_redirects_to_login(client: TestClient) -> None:
    response = client.get("/jobs", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_the_nav_lists_jobs_before_boards(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    page = client.get("/jobs").text
    assert page.index('href="/jobs">Jobs<') < page.index('href="/boards">Boards<')


def test_saving_the_filter_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    response = client.post("/jobs/filter", data={"csrf_token": "wrong", "title_includes": "x"})
    assert response.status_code == 403
    assert 'value="x"' not in client.get("/jobs").text


# -- aggregation and the filter -----------------------------------------------------------


def test_jobs_aggregates_across_boards_and_applies_the_saved_filter(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    html = client.get("/jobs").text
    text = text_of(html)

    for shown in (
        "Engineering Manager, Remote",
        "Manager, Engineering",
        "Engineering Manager, Unstated",
    ):
        assert shown in text
    for hidden in (
        "Engineering Manager, Inference",  # on-site
        "Sales Engineering Manager",  # on-site
        "Software Engineer",  # title
        "Engineering Manager, Hybrid",  # hybrid
        "Engineering Manager, Not stated",  # unstated on a board that hides them
    ):
        assert hidden not in text

    assert (
        "3 matching (1 with workplace not stated, included by board setting)"
        " of 8 open jobs across 2 boards" in text
    )
    assert 'value="engineering manager"' in html  # the saved filter, re-rendered
    assert re.search(r'name="workplace" value="remote"\s+checked', html)
    assert "seen since" in text.lower()
    assert "posted" not in text.lower()

    # Each row links to its board. (Ordering is covered by the repository test.)
    assert f'href="/boards/{seeded["anthropic"]}"' in html
    assert f'href="/boards/{seeded["acme"]}"' in html
    assert "workplace not stated" in text


def test_the_hidden_unstated_count_is_stated_with_a_one_click_show(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    text = text_of(client.get("/jobs").text)
    assert "1 more job would match but its workplace isn't stated" in text
    assert 'href="/jobs?show_unstated=1"' in client.get("/jobs").text

    shown = text_of(client.get("/jobs?show_unstated=1").text)
    assert "Engineering Manager, Not stated" in shown
    assert "4 matching" in shown and "of 8 open jobs across 2 boards" in shown
    assert "would match but" not in shown


def test_a_board_with_no_complete_check_is_named_not_silently_absent(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    html = client.get("/jobs").text
    text = text_of(html)
    assert "first check pending: Pending" in text
    assert f'href="/boards/{seeded["pending"]}"' in html


def test_a_held_board_is_named(
    client: TestClient, seeded: dict[str, uuid.UUID], engine: Engine
) -> None:
    with engine.begin() as conn:
        repo = PostgresBoardRepository(conn, seeded["user"])
        _check(repo, seeded["acme"], [], dt.datetime.now(dt.UTC) - dt.timedelta(days=1))
        board = repo.get_board(seeded["acme"])
        assert board is not None and board.held_check_id is not None
    text = text_of(client.get("/jobs").text)
    assert "as of their last applied check: Acme" in text
    assert "Manager, Engineering" in text  # a held check changes no job's state


def test_the_rendered_list_is_capped_and_says_so(
    client: TestClient, seeded: dict[str, uuid.UUID], monkeypatch: pytest.MonkeyPatch
) -> None:
    import jfl_web.routes.jobs as jobs_routes

    monkeypatch.setattr(jobs_routes, "JOBS_PAGE_CAP", 2)
    text = text_of(client.get("/jobs").text)
    assert "Showing the first 2 of 3 matching jobs" in text


def test_the_boards_page_shows_n_of_m_match(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    text = text_of(client.get("/boards").text)
    assert "2 of 4 match" in text  # Anthropic: the remote EM, and the unstated EM
    assert "1 of 4 match" in text  # Acme


def test_an_empty_filter_matches_every_open_job(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    save_filter(client)
    text = text_of(client.get("/jobs").text)
    assert "8 matching" in text and "of 8 open jobs across 2 boards" in text


# -- the per-board include-unstated setting ------------------------------------------------


def test_include_unstated_defaults_by_platform_and_can_be_overridden(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    anthropic = text_of(client.get(f"/boards/{seeded['anthropic']}").text)
    assert "Currently: included" in anthropic
    acme = text_of(client.get(f"/boards/{seeded['acme']}").text)
    assert "Currently: hidden and counted" in acme

    token = csrf(client, f"/boards/{seeded['anthropic']}")
    response = client.post(
        f"/boards/{seeded['anthropic']}/unstated",
        data={"csrf_token": token, "setting": "hide"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    text = text_of(client.get("/jobs").text)
    assert "Engineering Manager, Unstated" not in text
    assert "2 more jobs would match but their workplace isn't stated" in text

    client.post(
        f"/boards/{seeded['acme']}/unstated",
        data={"csrf_token": csrf(client), "setting": "include"},
    )
    text = text_of(client.get("/jobs").text)
    assert "Engineering Manager, Not stated" in text
    assert "1 more job would match" in text

    client.post(
        f"/boards/{seeded['anthropic']}/unstated",
        data={"csrf_token": csrf(client), "setting": "default"},
    )
    assert "Currently: included" in text_of(client.get(f"/boards/{seeded['anthropic']}").text)


def test_the_setting_rejects_values_outside_its_choices(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    response = client.post(
        f"/boards/{seeded['anthropic']}/unstated",
        data={"csrf_token": csrf(client), "setting": "sometimes"},
    )
    assert response.status_code == 400


# -- board exceptions ---------------------------------------------------------------------------


def add_london_exception(client: TestClient, board_id: uuid.UUID) -> Response:
    return client.post(
        f"/boards/{board_id}/exceptions",
        data={
            "csrf_token": csrf(client, f"/boards/{board_id}"),
            "workplace": ["onsite", "hybrid"],
            "location": "london",
            "note": NOTE,
        },
        follow_redirects=False,
    )


def test_an_onsite_london_exception_shows_the_employers_label_and_the_owners_reason(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    save_filter(
        client, workplace=["remote"], title_includes="engineering manager", title_excludes="sales"
    )
    response = add_london_exception(client, seeded["anthropic"])
    assert response.status_code == 303
    assert response.headers["location"] == f"/boards/{seeded['anthropic']}?status=exception_added"

    html = client.get("/jobs").text
    text = text_of(html)
    assert "Engineering Manager, Inference" in text
    assert f"On-Site included by your Anthropic exception: {NOTE}" in text
    assert "Sales Engineering Manager" not in text  # title excludes still apply
    assert "Engineering Manager, Hybrid" not in text  # Acme's; the exception is Anthropic's
    assert "4 matching (1 via board exceptions, 1 with workplace not stated" in text
    assert "of 8 open jobs across 2 boards" in text

    detail = client.get(f"/boards/{seeded['anthropic']}").text
    assert f'value="{NOTE}"' in detail


def test_exceptions_can_be_edited_and_removed(
    client: TestClient, seeded: dict[str, uuid.UUID], engine: Engine
) -> None:
    board_id = seeded["anthropic"]
    add_london_exception(client, board_id)
    detail = client.get(f"/boards/{board_id}").text
    match = re.search(rf"/boards/{board_id}/exceptions/([0-9a-f-]{{36}})/remove", detail)
    assert match is not None
    exception_id = match.group(1)

    response = client.post(
        f"/boards/{board_id}/exceptions/{exception_id}",
        data={
            "csrf_token": csrf(client, f"/boards/{board_id}"),
            "workplace": ["hybrid"],
            "location": "london",
            "note": "hybrid only",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    text = text_of(client.get("/jobs").text)
    assert "Engineering Manager, Inference" not in text  # on-site no longer included

    response = client.post(
        f"/boards/{board_id}/exceptions/{exception_id}/remove",
        data={"csrf_token": csrf(client, f"/boards/{board_id}")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "hybrid only" not in client.get(f"/boards/{board_id}").text


def test_adding_an_exception_requires_csrf(
    client: TestClient, seeded: dict[str, uuid.UUID]
) -> None:
    response = client.post(
        f"/boards/{seeded['anthropic']}/exceptions",
        data={"csrf_token": "wrong", "workplace": ["onsite"], "note": "nope"},
    )
    assert response.status_code == 403
    assert "nope" not in client.get(f"/boards/{seeded['anthropic']}").text


# -- tenancy -------------------------------------------------------------------------------------


def test_another_user_never_sees_the_filter_jobs_or_exceptions_and_cannot_edit_them(
    client: TestClient,
    seeded: dict[str, uuid.UUID],
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
) -> None:
    board_id = seeded["anthropic"]
    add_london_exception(client, board_id)
    detail = client.get(f"/boards/{board_id}").text
    match = re.search(rf"/boards/{board_id}/exceptions/([0-9a-f-]{{36}})/remove", detail)
    assert match is not None
    exception_id = match.group(1)

    sign_out(client)
    sign_in(client, google, subs, engine)

    html = client.get("/jobs").text
    text = text_of(html)
    assert 'value="engineering manager"' not in html  # the saved filter is not bob's
    assert "Engineering Manager, Remote" not in text
    assert "0 matching of 0 open jobs across 0 boards" in text
    assert "Pending" not in text

    token = csrf(client)
    assert client.get(f"/boards/{board_id}").status_code == 404
    edit = client.post(
        f"/boards/{board_id}/exceptions/{exception_id}",
        data={"csrf_token": token, "workplace": ["remote"], "note": "hijacked"},
    )
    assert edit.status_code == 404
    remove = client.post(
        f"/boards/{board_id}/exceptions/{exception_id}/remove", data={"csrf_token": token}
    )
    assert remove.status_code == 404
    add = client.post(
        f"/boards/{board_id}/exceptions", data={"csrf_token": token, "note": "mine now"}
    )
    assert add.status_code == 404
    setting = client.post(
        f"/boards/{board_id}/unstated", data={"csrf_token": token, "setting": "hide"}
    )
    assert setting.status_code == 404

    with engine.begin() as conn:
        from jfl_core.storage.job_filters import PostgresJobFilterRepository

        owners = PostgresJobFilterRepository(conn, seeded["user"])
        (kept,) = owners.list_exceptions(board_id)
        assert kept.note == NOTE and kept.workplaces == ["hybrid", "onsite"]
        board = PostgresBoardRepository(conn, seeded["user"]).get_board(board_id)
        assert board is not None and board.include_unstated_workplace is None
