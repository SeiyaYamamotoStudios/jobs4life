"""The fact-confirmation screen through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_profile_web_integration.py`. No network and no model call: the facts these
tests confirm are written straight through the repository, standing in for what
an extraction pass would have proposed.

What is actually being defended here, beyond "the buttons work":

* an **edit** stores the user's words, and the model's proposal never reaches
  the corpus;
* a fact with an unanswered **probe** cannot be swept up by the per-role
  control, because the unstated number is the whole difference between a
  supported claim and `scope_inflation`;
* a per-role confirm touches **only that role**, and only what was on screen;
* a **rejected** fact is kept, its corpus span stops grounding anything, and it
  can be brought back.
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
from jfl_core.db.tables import sent_documents as sent_documents_table
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import content_hash
from jfl_core.models import CandidateFact
from jfl_core.storage.candidate_facts import (
    PROBE_JOIN,
    PostgresCandidateFactRepository,
    ProposedFact,
    corpus_section,
)
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


def csrf(client: TestClient, path: str = "/corpus/facts") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def post(client: TestClient, path: str, **fields: object) -> Response:
    data: dict[str, object] = {"csrf_token": csrf(client), **fields}
    return client.post(path, data=data, follow_redirects=False)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


# -- fixtures standing in for an extraction pass ------------------------------


def a_cv(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    """A row in the sent-document store -- form, never truth. Nothing in it is
    grounding; the facts below are proposals about it, not evidence.
    """
    doc_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            sent_documents_table.insert().values(
                id=doc_id,
                user_id=user_id,
                kind="cv",
                path=f"upload:{doc_id}",
                content_hash=content_hash(str(doc_id)),
                text="Northwind. Led the platform team. Cut deploy time.",
            )
        )
    return doc_id


def propose(
    engine: Engine, user_id: uuid.UUID, doc_id: uuid.UUID, *facts: ProposedFact
) -> list[uuid.UUID]:
    with engine.begin() as conn:
        repo = PostgresCandidateFactRepository(conn, user_id)
        repo.add_proposed(list(facts))
        return [f.id for f in repo.list_facts()]


def fact(
    doc_id: uuid.UUID,
    role: str,
    text: str,
    *,
    source: str | None = None,
    probe: str | None = None,
) -> ProposedFact:
    return ProposedFact(
        sent_document_id=doc_id,
        role_label=role,
        role_key=role.lower().replace(" ", "-"),
        source_line=source or text,
        fact_text=text,
        fingerprint=content_hash(f"{role}|{text}"),
        probe=probe,
    )


def _facts_of(engine: Engine, user_id: uuid.UUID) -> list[CandidateFact]:
    with engine.begin() as conn:
        return list(PostgresCandidateFactRepository(conn, user_id).list_facts())


def live_corpus_texts(engine: Engine, user_id: uuid.UUID) -> list[str]:
    with engine.begin() as conn:
        rows = conn.execute(
            select(spans_table.c.text).where(
                spans_table.c.user_id == user_id,
                spans_table.c.retired_at.is_(None),
            )
        ).all()
    return [row.text for row in rows]


# -- access and the empty state ----------------------------------------------


def test_signed_out_is_sent_to_login(client: TestClient) -> None:
    response = client.get("/corpus/facts", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_before_any_cv_the_page_points_at_the_upload(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    page = client.get("/corpus/facts").text
    assert "Nothing to check yet" in page
    assert 'href="/corpus/cvs"' in page


def test_confirming_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(engine, user_id, doc, fact(doc, "Northwind", "Ran the platform team"))

    response = client.post(
        f"/corpus/facts/{fact_id}/confirm",
        data={"csrf_token": "wrong", "fact_text": "Ran the platform team"},
    )
    assert response.status_code == 403
    assert live_corpus_texts(engine, user_id) == []


# -- listing ------------------------------------------------------------------


def test_facts_are_grouped_by_role_with_the_cv_line_beside_them(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    propose(
        engine,
        user_id,
        doc,
        fact(doc, "Northwind", "Ran the platform team", source="Platform lead, Northwind"),
        fact(doc, "Contoso", "Wrote the pricing service", source="Built pricing at Contoso"),
    )
    page = client.get("/corpus/facts").text
    body = text_of(page)

    assert "Northwind" in body and "Contoso" in body
    assert "Platform lead, Northwind" in body  # the CV's own words, shown
    assert "Built pricing at Contoso" in body
    assert "0 confirmed" in body and "2 still to check" in body
    # Roles keep CV order, not alphabetical order.
    assert body.index("Northwind") < body.index("Contoso")


# -- confirming ---------------------------------------------------------------


def test_confirming_as_written_records_the_proposal_verbatim(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(engine, user_id, doc, fact(doc, "Northwind", "Ran the platform team"))

    response = post(client, f"/corpus/facts/{fact_id}/confirm", fact_text="Ran the platform team")
    assert response.status_code == 303
    assert live_corpus_texts(engine, user_id) == ["Ran the platform team"]
    assert "1 confirmed" in text_of(client.get("/corpus/facts").text)


def test_an_edit_stores_the_users_words_and_never_the_models(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The point of the whole screen. The model proposed "owned"; the user says
    "maintained". Only "maintained" may become corpus -- a tool that quietly
    kept the stronger wording would be manufacturing the drift it exists to
    measure.
    """
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(engine, user_id, doc, fact(doc, "Northwind", "Owned the pricing platform"))

    post(client, f"/corpus/facts/{fact_id}/confirm", fact_text="Maintained the pricing platform")

    texts = live_corpus_texts(engine, user_id)
    assert texts == ["Maintained the pricing platform"]
    assert not any("Owned" in t for t in texts)
    assert "Maintained the pricing platform" in client.get("/corpus/facts").text


def test_a_probe_answer_is_stored_with_the_fact(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(
        engine,
        user_id,
        doc,
        fact(doc, "Northwind", "Led the platform team", probe="Led how many people?"),
    )
    assert "Led how many people?" in client.get("/corpus/facts").text

    post(
        client,
        f"/corpus/facts/{fact_id}/confirm",
        fact_text="Led the platform team",
        probe_answer="Nine engineers across two squads",
    )
    assert live_corpus_texts(engine, user_id) == [
        f"Led the platform team{PROBE_JOIN}Nine engineers across two squads"
    ]


def test_an_unanswered_probe_blocks_a_single_confirm(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(
        engine,
        user_id,
        doc,
        fact(doc, "Northwind", "Led the platform team", probe="Led how many people?"),
    )
    response = post(client, f"/corpus/facts/{fact_id}/confirm", fact_text="Led the platform team")
    assert response.status_code == 400
    assert live_corpus_texts(engine, user_id) == []


# -- per-role confirm ---------------------------------------------------------


def test_confirming_a_role_leaves_every_other_role_alone(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    propose(
        engine,
        user_id,
        doc,
        fact(doc, "Northwind", "Ran the platform team"),
        fact(doc, "Northwind", "Cut deploy time"),
        fact(doc, "Contoso", "Wrote the pricing service"),
    )
    response = client.post(
        "/corpus/facts/roles/northwind/confirm",
        data={
            "csrf_token": csrf(client),
            "fact_id": [str(f.id) for f in _facts_of(engine, user_id) if f.role_key == "northwind"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    texts = sorted(live_corpus_texts(engine, user_id))
    assert texts == ["Cut deploy time", "Ran the platform team"]
    assert "Wrote the pricing service" not in texts

    body = text_of(client.get("/corpus/facts").text)
    assert "2 confirmed" in body and "1 still to check" in body


def test_a_role_confirm_skips_a_fact_whose_probe_is_unanswered(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    propose(
        engine,
        user_id,
        doc,
        fact(doc, "Northwind", "Cut deploy time"),
        fact(doc, "Northwind", "Led the platform team", probe="Led how many people?"),
    )
    page = client.get("/corpus/facts").text
    # The page itself only offers the unblocked fact to the bulk control.
    form = page[page.index('action="/corpus/facts/roles/northwind/confirm"') :]
    submitted = re.findall(r'name="fact_id" value="([0-9a-f-]+)"', form)
    assert len(submitted) == 1

    response = client.post(
        "/corpus/facts/roles/northwind/confirm",
        data={"csrf_token": csrf(client), "fact_id": submitted},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert live_corpus_texts(engine, user_id) == ["Cut deploy time"]


def test_a_role_confirm_ignores_a_fact_id_from_another_role(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The submitted ids are narrowed by role server-side, so a stale or
    tampered form cannot reach across roles.
    """
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    propose(
        engine,
        user_id,
        doc,
        fact(doc, "Northwind", "Ran the platform team"),
        fact(doc, "Contoso", "Wrote the pricing service"),
    )
    every_id = [str(f.id) for f in _facts_of(engine, user_id)]
    response = client.post(
        "/corpus/facts/roles/northwind/confirm",
        data={"csrf_token": csrf(client), "fact_id": every_id},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert live_corpus_texts(engine, user_id) == ["Ran the platform team"]


# -- rejecting, and bringing a fact back --------------------------------------


def test_a_rejected_fact_stays_visible_and_can_be_brought_back(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(engine, user_id, doc, fact(doc, "Northwind", "Ran the platform team"))

    assert post(client, f"/corpus/facts/{fact_id}/reject").status_code == 303
    body = text_of(client.get("/corpus/facts").text)
    assert "not true as written" in body
    assert "Ran the platform team" in body  # kept, never deleted
    assert live_corpus_texts(engine, user_id) == []

    assert post(client, f"/corpus/facts/{fact_id}/restore").status_code == 303
    assert "1 still to check" in text_of(client.get("/corpus/facts").text)


def test_rejecting_a_confirmed_fact_retires_its_corpus_span(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A statement the user withdraws must stop grounding claims immediately --
    and the span itself is retired, never deleted, so anything already citing it
    still resolves.
    """
    user_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, user_id)
    (fact_id,) = propose(engine, user_id, doc, fact(doc, "Northwind", "Ran the platform team"))
    post(client, f"/corpus/facts/{fact_id}/confirm", fact_text="Ran the platform team")
    assert live_corpus_texts(engine, user_id) == ["Ran the platform team"]

    post(client, f"/corpus/facts/{fact_id}/reject")
    assert live_corpus_texts(engine, user_id) == []
    with engine.begin() as conn:
        still_there = conn.execute(
            select(spans_table.c.retired_at).where(
                spans_table.c.user_id == user_id,
                spans_table.c.section_path == corpus_section("Northwind"),
            )
        ).all()
    assert len(still_there) == 1 and still_there[0].retired_at is not None


# -- tenancy ------------------------------------------------------------------


def test_another_users_fact_is_a_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    other_id = sign_in(client, google, subs, engine)
    doc = a_cv(engine, other_id)
    (their_fact,) = propose(engine, other_id, doc, fact(doc, "Northwind", "Ran the platform team"))
    client.post("/logout", data={"csrf_token": csrf(client, "/corpus/facts")})

    sign_in(client, google, subs, engine)
    response = post(client, f"/corpus/facts/{their_fact}/confirm", fact_text="Mine now")
    assert response.status_code == 404
    assert post(client, f"/corpus/facts/{their_fact}/reject").status_code == 404
    assert live_corpus_texts(engine, other_id) == []
