"""Profile questions 15 and 16 reach the corpus. Nothing else on /profile does.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider and sign-in helper as `test_profile_web_integration.py`.
No network and no model call anywhere in this path -- that is one of the things
being asserted.

The distinction this file defends is the reason the two questions are on that
page at all. Everything else there is a **preference**: what the user wants, what
they will not accept. Those must never become evidence -- a tool that quietly
turned "I want more scope" into a corpus fact would be citing an aspiration back
at its author as though it were something they had done. 15 and 16 are
**claims about the person**, so they are stored verbatim, in the corpus, and are
cited.
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
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import users as users_table
from jfl_core.profile_questions import CORPUS_SECTIONS
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


def csrf(client: TestClient) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/profile").text)
    assert match is not None
    return match.group(1)


def post(client: TestClient, path: str, **fields: object) -> Response:
    return client.post(path, data={"csrf_token": csrf(client), **fields}, follow_redirects=False)


def live_spans(engine: Engine, user_id: uuid.UUID) -> list[tuple[str, str]]:
    with engine.begin() as conn:
        rows = conn.execute(
            select(spans_table.c.section_path, spans_table.c.text).where(
                spans_table.c.user_id == user_id,
                spans_table.c.retired_at.is_(None),
            )
        ).all()
    return [(row.section_path, row.text) for row in rows]


DEPTH = "Deep in JVM platform work. Kubernetes is exposure only -- I have never run it in anger."
GAPS = "Terraform keeps coming up and I have only read it."


# -- 15 and 16 do reach the corpus -------------------------------------------


def test_depth_and_gaps_are_recorded_in_the_corpus_verbatim(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = post(client, "/profile/depth-and-gaps", depth_genuine=DEPTH, recurring_gaps=GAPS)
    assert response.status_code == 303

    assert sorted(live_spans(engine, user_id)) == sorted(
        [(CORPUS_SECTIONS["depth_genuine"], DEPTH), (CORPUS_SECTIONS["recurring_gaps"], GAPS)]
    )
    # And they are still profile answers, rendered back on the page.
    assert DEPTH in client.get("/profile").text


def test_the_page_says_which_answers_become_corpus(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A user cannot consent to something the page does not tell them. The
    corpus-bound section has to be marked as such in plain words.
    """
    sign_in(client, google, subs, engine)
    page = client.get("/profile").text
    assert 'id="depth-and-gaps"' in page
    assert "corpus" in page.lower()


def test_re_answering_supersedes_the_earlier_statement(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A statement the user has replaced must stop grounding claims. Its span is
    retired, not deleted, so anything already citing it still resolves.
    """
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/depth-and-gaps", depth_genuine=DEPTH)
    post(client, "/profile/depth-and-gaps", depth_genuine="Actually my depth is in data.")

    live = live_spans(engine, user_id)
    assert live == [(CORPUS_SECTIONS["depth_genuine"], "Actually my depth is in data.")]
    with engine.begin() as conn:
        total = conn.execute(select(spans_table.c.id).where(spans_table.c.user_id == user_id)).all()
    assert len(total) == 2  # the old one is retired, not removed


def test_clearing_the_answer_clears_the_corpus_text(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/depth-and-gaps", depth_genuine=DEPTH, recurring_gaps=GAPS)
    post(client, "/profile/depth-and-gaps", depth_genuine="", recurring_gaps=GAPS)

    assert live_spans(engine, user_id) == [(CORPUS_SECTIONS["recurring_gaps"], GAPS)]


def test_saving_depth_and_gaps_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/profile/depth-and-gaps", data={"csrf_token": "wrong", "depth_genuine": DEPTH}
    )
    assert response.status_code == 403
    assert live_spans(engine, user_id) == []


# -- everything else on the page does not -------------------------------------


def test_no_other_profile_answer_reaches_the_corpus(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Every other section of /profile, saved with real text, leaves the corpus
    empty. A preference that became evidence would be the tool citing an
    aspiration back at its author as something they had done.
    """
    user_id = sign_in(client, google, subs, engine)

    post(
        client,
        "/profile/hard-gates",
        location_commute="London, one office day a month",
        levels="EM or a small step up",
        comp_floor="Base plus bonus",
        notice_period="One month",
        right_to_work="British citizen",
        categorical_no="No on-call rota of one",
    )
    post(client, "/profile/discipline", disciplines="Platform, not frontend")
    post(
        client,
        "/profile/objectives",
        objective_1="More scope",
        evidence_1="An org of 40+",
    )
    post(client, "/profile/trajectory", trajectory="Director inside two years")
    post(client, "/profile/place", employer_deal_breakers="No PE-owned employers")
    post(client, "/profile/tells", warning_signs="'Wear many hats'")
    post(client, "/profile/ruled-out", decision_text="Not considering Acme again")

    assert live_spans(engine, user_id) == []

    # ...and the page really did save them, so this is not passing by accident.
    page = client.get("/profile").text
    assert "Director inside two years" in page
    assert "No PE-owned employers" in page
