"""The boards screen against a live Postgres: adding a board, the list, "check
now", stop watching, and tenancy through the actual routes.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Follows `tests/test_applications_web_integration.py`'s pattern exactly -- same
stub Google provider, same sign-in helper, same CSRF-scraping helper -- since
this screen and the application tracker share every one of those mechanics.

No Anthropic API call anywhere in this file (nothing here reads a job ad), and
no HTTP request to a board's own site either: `detect_board` is pattern
matching against the pasted URL and `add_board` is a database write, so the
root conftest's network-socket guard would fail loudly if either path ever
tried to reach the internet -- it never gets the chance to.
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
from jfl_core.db.tables import board_checks as checks_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.db.tables import watched_boards as watched_boards_table
from jfl_intake.scheduling import CHECK_BOARD_KIND
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration


class StubGoogle:
    """Stands in for `AuthlibGoogleProvider`."""

    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None
        self.error: str | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.error is not None:
            raise OAuthError(self.error)
        assert self.identity is not None, "the test must set an identity first"
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
    import datetime as dt

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


def sign_in(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    *,
    sub: str | None = None,
    email: str | None = None,
    name: str | None = "Test Person",
) -> GoogleIdentity:
    identity = GoogleIdentity(
        sub=sub or f"test-sub-{uuid.uuid4()}",
        email=email or f"{uuid.uuid4()}@test.invalid",
        display_name=name,
    )
    subs.append(identity.sub)
    google.identity = identity
    response = client.get("/auth/google/callback")
    assert response.status_code == 200, response.text
    return identity


def _csrf(client: TestClient, path: str = "/boards") -> str:
    """Scrape the token out of a rendered form, the way a browser would."""
    page = client.get(path).text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None, f"no CSRF token found on {path}"
    return match.group(1)


def _add_board(client: TestClient, *, url: str) -> Any:
    return client.post(
        "/boards",
        data={"csrf_token": _csrf(client), "url": url},
        follow_redirects=False,
    )


def _board_id_from_list(client: TestClient, url: str) -> str:
    """A board's id, read off the rendered list by its detail link. The list
    route is the only surface this screen offers for finding an id -- add_board
    deliberately redirects to the bare list rather than leaking the id into the
    URL, so a test recovers it the way a person reading the page would.
    """
    page = client.get("/boards").text
    for match in re.finditer(r"/boards/([0-9a-f-]{36})", page):
        return match.group(1)
    raise AssertionError(f"no board link found on /boards after adding {url}")


def _check_tasks(engine: Engine, board_id: str) -> list[Any]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table.c.status, tasks_table.c.payload).where(
                    tasks_table.c.kind == CHECK_BOARD_KIND,
                    tasks_table.c.payload["board_id"].astext == board_id,
                )
            ).all()
        )


# --------------------------------------------------------------------------
# Signed out
# --------------------------------------------------------------------------


def test_a_signed_out_visitor_is_redirected_to_login(client: TestClient) -> None:
    response = client.get("/boards", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


# --------------------------------------------------------------------------
# Adding a board
# --------------------------------------------------------------------------


def test_adding_a_valid_board_creates_it_and_enqueues_exactly_one_check(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """No HTTP request is made in the request path: `add_board` is a database
    write and `detect_board` is pattern matching, so there is nothing here for
    the root conftest's socket guard to catch -- which is the point. The
    baseline check itself only ever runs in the worker.
    """
    sign_in(client, google, subs)
    url = f"https://boards.greenhouse.io/acme-{uuid.uuid4().hex[:8]}"

    response = _add_board(client, url=url)
    assert response.status_code == 303
    assert response.headers["location"] == "/boards?status=added"

    board_id = _board_id_from_list(client, url)
    tasks = _check_tasks(engine, board_id)
    assert len(tasks) == 1
    assert tasks[0].status == "pending"
    assert tasks[0].payload == {"board_id": board_id}

    page = client.get("/boards").text
    assert "Greenhouse" in page


def test_a_linkedin_url_is_refused_and_creates_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    response = _add_board(client, url="https://www.linkedin.com/jobs/view/12345")
    assert response.status_code == 400
    assert "LinkedIn" in response.text
    assert "does not scrape" in response.text.lower() or "not supported" in response.text.lower()

    with engine.begin() as conn:
        count = conn.execute(
            select(watched_boards_table.c.id).where(
                watched_boards_table.c.board_url == "https://www.linkedin.com/jobs/view/12345"
            )
        ).all()
    assert count == []


def test_an_unsupported_site_is_refused_and_lists_supported_platforms(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    response = _add_board(client, url="https://example.com/careers")
    assert response.status_code == 400
    assert "Greenhouse" in response.text
    assert "Workable" in response.text


def test_a_malformed_url_says_so(client: TestClient, google: StubGoogle, subs: list[str]) -> None:
    sign_in(client, google, subs)
    response = _add_board(client, url="not a url")
    assert response.status_code == 400
    assert "not a valid url" in response.text.lower()


def test_adding_the_same_board_twice_does_not_duplicate_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    url = f"https://jobs.ashbyhq.com/acme-{uuid.uuid4().hex[:8]}"

    first = _add_board(client, url=url)
    assert first.headers["location"] == "/boards?status=added"
    second = _add_board(client, url=url)
    assert second.headers["location"] == "/boards?status=already_watched"

    with engine.begin() as conn:
        rows = conn.execute(
            select(watched_boards_table.c.id).where(watched_boards_table.c.board_url == url)
        ).all()
    assert len(rows) == 1

    board_id = _board_id_from_list(client, url)
    assert len(_check_tasks(engine, board_id)) == 1


# --------------------------------------------------------------------------
# The list: a board with no history yet is honest about that
# --------------------------------------------------------------------------


def test_a_board_with_no_complete_check_shows_first_check_pending_not_zero_jobs(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    url = f"https://jobs.lever.co/acme-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=url)

    page = client.get("/boards").text
    assert "first check pending" in page
    # And not presented as a real, empty count -- "0" appearing anywhere on
    # the page is not itself proof of the bug, so check the shape instead:
    # the open-jobs cell for this fresh board must be the pending phrase.
    board_id = _board_id_from_list(client, url)
    detail = client.get(f"/boards/{board_id}").text
    assert "first check pending" in detail
    assert "not an empty board" in detail


# --------------------------------------------------------------------------
# Check now
# --------------------------------------------------------------------------


def test_check_now_enqueues_once_and_a_second_press_does_not_duplicate(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    url = f"https://ats.rippling.com/acme-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=url)
    board_id = _board_id_from_list(client, url)
    # Adding already queued the baseline check.
    assert len(_check_tasks(engine, board_id)) == 1

    # Pressing "check now" while that task is still pending must not queue a
    # second one.
    response = client.post(
        f"/boards/{board_id}/check",
        data={"csrf_token": _csrf(client), "next": "list"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/boards?status=already_queued"
    assert len(_check_tasks(engine, board_id)) == 1

    # Once the pending task has settled, a fresh press queues a new one.
    with engine.begin() as conn:
        conn.execute(
            tasks_table.update()
            .where(
                tasks_table.c.kind == CHECK_BOARD_KIND,
                tasks_table.c.payload["board_id"].astext == board_id,
            )
            .values(status="succeeded")
        )
    response = client.post(
        f"/boards/{board_id}/check",
        data={"csrf_token": _csrf(client), "next": "detail"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/boards/{board_id}?status=queued"
    assert len(_check_tasks(engine, board_id)) == 2

    # And a second press while that new one is pending is a no-op again.
    client.post(
        f"/boards/{board_id}/check",
        data={"csrf_token": _csrf(client), "next": "list"},
    )
    assert len(_check_tasks(engine, board_id)) == 2


def test_check_now_requires_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    url = f"https://ats.rippling.com/acme-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=url)
    board_id = _board_id_from_list(client, url)

    response = client.post(f"/boards/{board_id}/check", data={"csrf_token": "wrong"})
    assert response.status_code == 403
    assert len(_check_tasks(engine, board_id)) == 1


# --------------------------------------------------------------------------
# Check all boards now
# --------------------------------------------------------------------------


def _board_id_by_url(engine: Engine, url: str) -> str:
    """Look the board's id up directly, rather than through
    `_board_id_from_list` -- that helper returns whichever detail link comes
    first on the page, which is fine with one board in play but wrong as soon
    as several are, as this file's "check all" tests have several.
    """
    with engine.begin() as conn:
        row = conn.execute(
            select(watched_boards_table.c.id).where(watched_boards_table.c.board_url == url)
        ).one()
    return str(row.id)


def _settle(engine: Engine, board_id: str) -> None:
    """Move every pending/running check task for one board to `succeeded`, the
    way the worker would once it finished -- so the next enqueue attempt is
    not skipped as already-queued.
    """
    with engine.begin() as conn:
        conn.execute(
            tasks_table.update()
            .where(
                tasks_table.c.kind == CHECK_BOARD_KIND,
                tasks_table.c.payload["board_id"].astext == board_id,
            )
            .values(status="succeeded")
        )


def test_check_all_queues_one_per_board_and_skips_ones_already_queued(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    url_a = f"https://boards.greenhouse.io/acme-{uuid.uuid4().hex[:8]}"
    url_b = f"https://jobs.lever.co/acme-{uuid.uuid4().hex[:8]}"
    url_c = f"https://jobs.ashbyhq.com/acme-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=url_a)
    _add_board(client, url=url_b)
    _add_board(client, url=url_c)
    board_a = _board_id_by_url(engine, url_a)
    board_b = _board_id_by_url(engine, url_b)
    board_c = _board_id_by_url(engine, url_c)

    # Each add already queued its baseline check. Settle two of them and
    # leave the third's baseline still pending, so "check all" must queue two
    # fresh checks and skip the one still mid-flight.
    _settle(engine, board_a)
    _settle(engine, board_b)
    assert len(_check_tasks(engine, board_c)) == 1  # still pending from add_board

    response = client.post(
        "/boards/check-all", data={"csrf_token": _csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/boards?status=check_all&queued=2&skipped=1"

    assert len(_check_tasks(engine, board_a)) == 2
    assert len(_check_tasks(engine, board_b)) == 2
    assert len(_check_tasks(engine, board_c)) == 1  # not double-queued

    page = client.get(response.headers["location"]).text
    assert "Queued 2 checks" in page
    assert "1 already checking, skipped" in page


def test_a_recheck_shows_the_checking_badge_next_to_the_last_result(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A board's *first* check queued is reported as "first check pending" --
    covered above. This is the other case the brief asks for: a board that
    already has a completed check, re-checked through "check all" (or, as
    here, the equivalent single "check now" -- both go through the same
    `enqueue_board_check`), shows a "checking" badge next to its last result
    rather than silently dropping that result or claiming it is still the
    same, stale status.
    """
    sign_in(client, google, subs)
    url = f"https://boards.greenhouse.io/acme-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=url)
    board_id = _board_id_by_url(engine, url)
    _settle(engine, board_id)  # the baseline task is done...

    # ...but nothing here runs the worker, so simulate what a completed
    # baseline check leaves behind: one `board_checks` row, and the board
    # pointed at it. Real work (`apply_check_plan`) is out of scope for a web
    # test; only the rows this screen reads need to exist.
    with engine.begin() as conn:
        user_id = conn.execute(
            select(watched_boards_table.c.user_id).where(watched_boards_table.c.id == board_id)
        ).scalar_one()
        check_id = uuid.uuid4()
        now = dt.datetime.now(dt.UTC)
        conn.execute(
            checks_table.insert().values(
                id=check_id,
                user_id=user_id,
                board_id=board_id,
                started_at=now,
                finished_at=now,
                status="complete",
                jobs_seen=0,
                expected_total=0,
                is_baseline=True,
            )
        )
        conn.execute(
            watched_boards_table.update()
            .where(watched_boards_table.c.id == board_id)
            .values(last_check_id=check_id, baseline_check_id=check_id)
        )

    response = client.post(
        f"/boards/{board_id}/check",
        data={"csrf_token": _csrf(client), "next": "list"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    page = client.get("/boards").text
    assert 'status-checking">checking' in page
    assert "Last check: complete" in page


def test_check_all_requires_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    url = f"https://ats.rippling.com/acme-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=url)
    board_id = _board_id_from_list(client, url)
    _settle(engine, board_id)

    response = client.post("/boards/check-all", data={"csrf_token": "wrong"})
    assert response.status_code == 403
    assert len(_check_tasks(engine, board_id)) == 1


def test_check_all_with_no_boards_queues_nothing(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    # No boards watched, so the button itself is not on the page.
    assert "Check all boards now" not in client.get("/boards").text

    response = client.post(
        "/boards/check-all", data={"csrf_token": _csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/boards?status=check_all&queued=0&skipped=0"
    assert "No boards to check" in client.get(response.headers["location"]).text


def test_check_all_touches_only_this_users_boards(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    alice_url = f"https://jobs.ashbyhq.com/alice-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=alice_url)
    alice_board_id = _board_id_from_list(client, alice_url)
    _settle(engine, alice_board_id)
    before = _check_tasks(engine, alice_board_id)

    client.post("/logout", data={"csrf_token": _csrf(client)})
    sign_in(client, google, subs)
    bob_url = f"https://jobs.smartrecruiters.com/Bob{uuid.uuid4().hex[:8]}"
    _add_board(client, url=bob_url)
    bob_board_id = _board_id_from_list(client, bob_url)
    _settle(engine, bob_board_id)

    response = client.post(
        "/boards/check-all", data={"csrf_token": _csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/boards?status=check_all&queued=1&skipped=0"

    # Bob's press queued exactly his own board -- Alice's is untouched.
    assert _check_tasks(engine, alice_board_id) == before
    assert len(_check_tasks(engine, bob_board_id)) == 2


# --------------------------------------------------------------------------
# Stop watching
# --------------------------------------------------------------------------


def test_stop_watching_removes_the_board(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    url = f"https://{uuid.uuid4().hex[:8]}.breezy.hr"
    _add_board(client, url=url)
    board_id = _board_id_from_list(client, url)

    response = client.post(
        f"/boards/{board_id}/remove",
        data={"csrf_token": _csrf(client)},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/boards?status=removed"

    assert client.get(f"/boards/{board_id}").status_code == 404
    assert url not in client.get("/boards").text


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_another_users_board_id_returns_404_for_view_check_and_remove(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    alice_url = f"https://jobs.ashbyhq.com/alice-{uuid.uuid4().hex[:8]}"
    _add_board(client, url=alice_url)
    alice_board_id = _board_id_from_list(client, alice_url)
    before = _check_tasks(engine, alice_board_id)

    client.post("/logout", data={"csrf_token": _csrf(client)})
    sign_in(client, google, subs)

    assert client.get(f"/boards/{alice_board_id}").status_code == 404

    csrf = _csrf(client)
    check_response = client.post(
        f"/boards/{alice_board_id}/check", data={"csrf_token": csrf, "next": "list"}
    )
    assert check_response.status_code == 404
    assert _check_tasks(engine, alice_board_id) == before

    remove_response = client.post(f"/boards/{alice_board_id}/remove", data={"csrf_token": csrf})
    assert remove_response.status_code == 404

    # Alice's board survived every attempt, and never leaked into Bob's list.
    assert alice_url not in client.get("/boards").text


def test_two_users_do_not_see_each_others_boards_in_the_list(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    alice_url = f"https://jobs.smartrecruiters.com/Alice{uuid.uuid4().hex[:8]}"
    _add_board(client, url=alice_url)
    client.post("/logout", data={"csrf_token": _csrf(client)})

    sign_in(client, google, subs)
    page = client.get("/boards").text
    assert alice_url not in page
    assert "No boards watched yet" in page
