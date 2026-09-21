"""The profile's self-assessment reaches the corpus. Nothing else on it does.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider and sign-in helper as `test_profile_web_integration.py`.
No network and no model call anywhere in this path -- that is one of the things
being asserted.

The distinction this file defends is the reason the two questions are on that
page at all. Everything else there is a **preference**: what the user wants, what
they will not accept. Those must never become evidence -- a tool that quietly
turned "I want more scope" into a corpus fact would be citing an aspiration back
at its author as though it were something they had done. Questions 15 and 16 --
where your depth is genuine, and the gaps that keep coming up -- are **claims
about the person**, so they are stored verbatim, in the corpus, and are cited.

The section headings are still the retired `profile_questions.CORPUS_SECTIONS`,
deliberately: the shape of the page changed on 2026-09-21, but the corpus
section a statement is filed under did not, so a user who answered under the old
page has their statement superseded rather than duplicated.
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
from jfl_core.profile import CORPUS_SECTIONS
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
    """(section, text) for every statement of this user's now grounding.

    Bullets only. 15 and 16 are stored the way every corpus fact is -- as a
    bullet in the user's corpus markdown (`jfl_core.corpus_source`, the one
    write path) -- so the document also carries heading spans for its title and
    for each section. Those are structure, not statements about the person, and
    what these tests are about is which of the user's own words became corpus.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            select(spans_table.c.section_path, spans_table.c.text)
            .where(
                spans_table.c.user_id == user_id,
                spans_table.c.kind == "bullet",
                spans_table.c.retired_at.is_(None),
            )
            .order_by(spans_table.c.ordinal)
        ).all()
    return [(row.section_path, row.text) for row in rows]


DEPTH = "Deep in JVM platform work. Kubernetes is exposure only -- I have never run it in anger."
GAPS = "Terraform keeps coming up and I have only read it."


# -- the self-assessment does reach the corpus --------------------------------


def test_the_self_assessment_is_recorded_in_the_corpus_verbatim(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = post(client, "/profile/self-assessment", depth_genuine=DEPTH, recurring_gaps=GAPS)
    assert response.status_code == 303

    assert sorted(live_spans(engine, user_id)) == sorted(
        [(CORPUS_SECTIONS["depth_genuine"], DEPTH), (CORPUS_SECTIONS["recurring_gaps"], GAPS)]
    )
    # And it is still on the profile, rendered back on the page.
    assert DEPTH in client.get("/profile").text


def test_the_page_says_which_answers_become_corpus(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A user cannot consent to something the page does not tell them. The
    corpus-bound section has to be marked as such in plain words, and the rest
    of the page has to say that it is not.
    """
    sign_in(client, google, subs, engine)
    page = client.get("/profile").text
    assert 'id="self-assessment"' in page
    assert "corpus" in page.lower()
    assert "not preferences" in page


def test_re_answering_supersedes_the_earlier_statement(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A statement the user has replaced must stop grounding claims. Its span is
    retired, not deleted, so anything already citing it still resolves.
    """
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/self-assessment", depth_genuine=DEPTH)
    post(client, "/profile/self-assessment", depth_genuine="Actually my depth is in data.")

    live = live_spans(engine, user_id)
    assert live == [(CORPUS_SECTIONS["depth_genuine"], "Actually my depth is in data.")]
    with engine.begin() as conn:
        total = conn.execute(
            select(spans_table.c.id).where(
                spans_table.c.user_id == user_id,
                spans_table.c.kind == "bullet",
            )
        ).all()
    assert len(total) == 2  # the old one is retired, not removed


def test_clearing_the_answer_clears_the_corpus_text(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/self-assessment", depth_genuine=DEPTH, recurring_gaps=GAPS)
    post(client, "/profile/self-assessment", depth_genuine="", recurring_gaps=GAPS)

    assert live_spans(engine, user_id) == [(CORPUS_SECTIONS["recurring_gaps"], GAPS)]


def test_saving_the_self_assessment_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/profile/self-assessment", data={"csrf_token": "wrong", "depth_genuine": DEPTH}
    )
    assert response.status_code == 403
    assert live_spans(engine, user_id) == []


# -- everything else on the page does not -------------------------------------


def test_no_constraint_or_preference_reaches_the_corpus(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Every other section of /profile, saved with real text, leaves the corpus
    empty. A preference that became evidence would be the tool citing an
    aspiration back at its author as something they had done.
    """
    user_id = sign_in(client, google, subs, engine)

    post(
        client,
        "/profile/constraints",
        **{
            "stance-location": "must",
            "places-location": "London",
            "note-location": "One office day a month at most",
            "stance-comp_floor": "must",
            "comp-guaranteed": "120000",
            "stance-categorical_no": "never",
            "text-categorical_no": "No on-call rota of one",
        },
    )
    post(client, "/profile/capabilities", label="Running a platform group")
    post(
        client,
        "/profile/disciplines",
        practises="platform engineering",
        not_this="frontend",
    )
    post(
        client,
        "/profile/objectives",
        objective_1="More scope",
        evidence_1="An org of 40+",
    )

    assert live_spans(engine, user_id) == []

    # ...and the page really did save them, so this is not passing by accident.
    page = client.get("/profile").text
    assert "One office day a month at most" in page
    assert "Running a platform group" in page
