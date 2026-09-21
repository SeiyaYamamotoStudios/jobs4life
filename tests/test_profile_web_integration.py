"""The rebuilt `/profile` through the real routes against a live Postgres.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Same stub Google provider, sign-in helper and CSRF scraping as
`test_jobs_web_integration.py`. No network and no model call anywhere -- that
last part is one of the things being asserted, since nothing on this page has
any business calling a model.

What these tests are really defending, section by section:

* a value typed with no stance is **refused**, not filed under a guess;
* a capability's depth comes from behavioural answers, and an unanswered one
  stays "not stated" rather than rounding down;
* a row seeded from a confirmed CV fact is a proposal until the user saves it,
  and its evidence comes from the corpus rather than from the form;
* an identical save writes no version, so the history stays a record of changes
  rather than of page loads.

`test_profile_corpus_web_integration.py` holds the other half: that the
self-assessment reaches the corpus and that nothing else on the page does.
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
from jfl_core.db.tables import profiles as profiles_table
from jfl_core.db.tables import sent_documents as sent_documents_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import content_hash
from jfl_core.models import ProposedFact
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, func, select
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


def csrf(client: TestClient, path: str = "/profile") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def post(client: TestClient, path: str, **fields: object) -> Response:
    data: dict[str, object] = {"csrf_token": csrf(client), **fields}
    return client.post(path, data=data, follow_redirects=False)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def versions(engine: Engine, user_id: uuid.UUID) -> int:
    with engine.begin() as conn:
        return conn.execute(
            select(func.count())
            .select_from(profiles_table)
            .where(profiles_table.c.user_id == user_id)
        ).scalar_one()


def stored(engine: Engine, user_id: uuid.UUID) -> object:
    with engine.begin() as conn:
        return PostgresProfileRepository(conn, user_id).current()


def a_confirmed_fact(engine: Engine, user_id: uuid.UUID, text: str) -> uuid.UUID:
    """One CV fact, confirmed by its owner -- the only kind a capability row may
    be seeded from. Goes in through the real repository, so the span it produces
    is an ordinary corpus span rather than a fixture.
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
                text=text,
            )
        )
        repo = PostgresCandidateFactRepository(conn, user_id)
        repo.add_proposed(
            [
                ProposedFact(
                    sent_document_id=doc_id,
                    role_label="Northwind -- Engineering Manager",
                    role_key="northwind-engineering-manager",
                    source_line=text,
                    fact_text=text,
                    fingerprint=content_hash(text),
                )
            ]
        )
        fact = repo.list_facts()[-1]
        confirmed = repo.confirm(fact.id)
        assert confirmed is not None
        return confirmed.id


# -- access -------------------------------------------------------------------


def test_signed_out_profile_redirects_to_login(client: TestClient) -> None:
    response = client.get("/profile", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_signed_out_history_redirects_to_login(client: TestClient) -> None:
    response = client.get("/profile/history", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_saving_a_section_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = client.post(
        "/profile/constraints",
        data={"stance-location": "must", "places-location": "London"},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert versions(engine, user_id) == 0


@pytest.mark.parametrize(
    "path",
    [
        "/profile/constraints",
        "/profile/capabilities",
        "/profile/disciplines",
        "/profile/objectives",
        "/profile/self-assessment",
    ],
)
def test_every_section_save_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine, path: str
) -> None:
    user_id = sign_in(client, google, subs, engine)
    assert client.post(path, data={}, follow_redirects=False).status_code == 403
    assert versions(engine, user_id) == 0


# -- the empty state ----------------------------------------------------------


def test_an_untouched_profile_states_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    page = text_of(client.get("/profile").text)
    assert "Nothing saved yet" in page
    # Every constraint's stance select sits on "Not stated" and no value is
    # invented for it.
    assert "Not stated" in page


# -- 1. constraints -----------------------------------------------------------


def test_locations_are_an_ordered_list(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = post(
        client,
        "/profile/constraints",
        **{"stance-location": "must", "places-location": "London\nBristol"},
    )
    assert response.status_code == 303
    profile = stored(engine, user_id)
    constraint = profile.constraint("location")  # type: ignore[attr-defined]
    assert constraint.stance == "must"
    assert constraint.value == {"places": ["London", "Bristol"]}
    assert "London\nBristol" in client.get("/profile").text


def test_comp_carries_guaranteed_and_headline_separately(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(
        client,
        "/profile/constraints",
        **{
            "stance-comp_floor": "must",
            "comp-guaranteed": "120000",
            "comp-headline": "145000",
            "comp-ccy": "GBP",
            "note-comp_floor": "base + pension, ignoring equity",
        },
    )
    constraint = stored(engine, user_id).constraint("comp_floor")  # type: ignore[attr-defined]
    assert constraint.value == {"ccy": "GBP", "guaranteed": 120000, "headline": 145000}
    assert constraint.note == "base + pension, ignoring equity"
    page = text_of(client.get("/profile").text)
    assert "a headline number is not an offer" in page


def test_a_value_with_no_stance_is_refused_rather_than_guessed(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = post(client, "/profile/constraints", **{"places-location": "London"})
    assert response.status_code == 400
    assert "must, a nice-to-have or a never" in text_of(response.text)
    assert versions(engine, user_id) == 0


def test_a_stance_the_page_never_offered_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    response = post(
        client, "/profile/constraints", **{"stance-location": "maybe", "places-location": "London"}
    )
    assert response.status_code == 400
    assert versions(engine, user_id) == 0


def test_resaving_an_unchanged_section_writes_no_version(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fields = {"stance-notice": "must", "text-notice": "Three months"}
    post(client, "/profile/constraints", **fields)
    assert versions(engine, user_id) == 1
    post(client, "/profile/constraints", **fields)
    assert versions(engine, user_id) == 1


def test_saving_one_section_leaves_the_others_alone(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/constraints", **{"stance-notice": "must", "text-notice": "Three months"})
    post(client, "/profile/disciplines", practises="engineering management")
    profile = stored(engine, user_id)
    assert profile.constraint("notice").value == {"text": "Three months"}  # type: ignore[attr-defined]
    assert profile.disciplines.practises == ["engineering management"]  # type: ignore[attr-defined]


# -- 2. capabilities ----------------------------------------------------------


def test_a_confirmed_fact_is_proposed_as_an_untiered_capability(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    a_confirmed_fact(engine, user_id, "Ran the FX pricing platform")
    page = text_of(client.get("/profile").text)
    assert "Ran the FX pricing platform" in page
    assert "Not stated" in page
    assert "Proposed from a fact you confirmed" in page
    # Proposing it is not claiming it: nothing is written until the user saves.
    assert versions(engine, user_id) == 0


def test_a_proposed_capability_can_be_tiered_by_its_answers(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    a_confirmed_fact(engine, user_id, "Ran the FX pricing platform")
    key = re.search(r'id="capability-([0-9a-f]+)"', client.get("/profile").text)
    assert key is not None
    response = post(
        client,
        f"/profile/capabilities/{key.group(1)}",
        hands_on="yes",
        production="yes",
        interest="want_more",
    )
    assert response.status_code == 303
    capability = stored(engine, user_id).capabilities[0]  # type: ignore[attr-defined]
    assert capability.tier == "production_depth"
    assert capability.interest == "want_more"
    # The tier was never submitted; the evidence was never submitted either.
    assert capability.source == "cv_fact"
    assert len(capability.evidence) == 1
    page = text_of(client.get("/profile").text)
    assert "Production depth" in page
    assert "Evidence" in page


def test_an_unanswered_capability_stays_not_stated(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Answering "I did it myself" and stopping does not round down to
    `working`. The question that separates the top two tiers is unanswered, and
    a guess that happens to look modest is still a guess.
    """
    user_id = sign_in(client, google, subs, engine)
    a_confirmed_fact(engine, user_id, "Ran the FX pricing platform")
    key = re.search(r'id="capability-([0-9a-f]+)"', client.get("/profile").text)
    assert key is not None
    post(client, f"/profile/capabilities/{key.group(1)}", hands_on="yes")
    capability = stored(engine, user_id).capabilities[0]  # type: ignore[attr-defined]
    assert capability.tier is None


def test_a_tier_answer_the_page_never_offered_is_refused(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    a_confirmed_fact(engine, user_id, "Ran the FX pricing platform")
    key = re.search(r'id="capability-([0-9a-f]+)"', client.get("/profile").text)
    assert key is not None
    response = post(
        client, f"/profile/capabilities/{key.group(1)}", hands_on="expert", production="yes"
    )
    assert response.status_code == 400
    assert versions(engine, user_id) == 0


def test_a_capability_nobody_confirmed_says_it_is_only_a_claim(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/capabilities", label="Incident command")
    page = text_of(client.get("/profile").text)
    assert "Incident command" in page
    assert "Claimed, not yet evidenced" in page
    assert stored(engine, user_id).capabilities[0].source == "user"  # type: ignore[attr-defined]


def test_a_capability_of_another_account_cannot_be_tiered(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    first = sign_in(client, google, subs, engine)
    post(client, "/profile/capabilities", label="Incident command")
    key = re.search(r'id="capability-([0-9a-f]+)"', client.get("/profile").text)
    assert key is not None

    second = sign_in(client, google, subs, engine)
    response = post(
        client, f"/profile/capabilities/{key.group(1)}", hands_on="yes", production="yes"
    )
    assert response.status_code == 404
    assert versions(engine, second) == 0
    assert stored(engine, first).capabilities[0].tier is None  # type: ignore[attr-defined]


# -- 3. disciplines -----------------------------------------------------------


def test_disciplines_keep_a_not_this_list(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(
        client,
        "/profile/disciplines",
        practises="engineering management\nplatform engineering",
        not_this="frontend",
    )
    disciplines = stored(engine, user_id).disciplines  # type: ignore[attr-defined]
    assert disciplines.practises == ["engineering management", "platform engineering"]
    assert disciplines.not_practised == ["frontend"]
    assert "frontend" in client.get("/profile").text


# -- 4. objectives ------------------------------------------------------------


def test_objectives_are_ranked_and_the_page_says_never_weighted(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(
        client,
        "/profile/objectives",
        objective_1="Bigger scope",
        evidence_1="A department, not a squad",
        objective_3="Less travel",
    )
    objectives = stored(engine, user_id).objectives  # type: ignore[attr-defined]
    assert [(o.rank, o.text) for o in objectives] == [(1, "Bigger scope"), (3, "Less travel")]
    assert objectives[0].evidence_of_delivery == "A department, not a squad"
    assert "ranked and never weighted" in text_of(client.get("/profile").text)


# -- 5. self-assessment (corpus side lives in the sibling test file) ----------


def test_the_self_assessment_is_kept_on_the_profile_too(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(
        client,
        "/profile/self-assessment",
        depth_genuine="Depth is genuine in payments; Kubernetes is exposure only.",
        recurring_gaps="Formal data modelling.",
    )
    assessment = stored(engine, user_id).self_assessment  # type: ignore[attr-defined]
    assert assessment.depth_genuine.startswith("Depth is genuine in payments")
    assert assessment.recurring_gaps == "Formal data modelling."


# -- tenancy ------------------------------------------------------------------


def test_another_users_profile_is_invisible(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    first = sign_in(client, google, subs, engine)
    # Deliberately not a phrase the page itself uses as a placeholder -- the
    # assertion below is "this user's words are absent", and a placeholder that
    # happened to match would make it pass for the wrong reason.
    post(client, "/profile/disciplines", practises="mainframe wrangling")
    post(client, "/profile/capabilities", label="Mainframe failover drills")

    second = sign_in(client, google, subs, engine)
    page = client.get("/profile").text
    assert "mainframe wrangling" not in page
    assert "Mainframe failover drills" not in page
    assert "Nothing saved yet" in text_of(client.get("/profile/history").text)
    assert versions(engine, second) == 0
    assert versions(engine, first) == 2


# -- history ------------------------------------------------------------------


def test_history_lists_versions_newest_first(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    post(client, "/profile/capabilities", label="Incident command")
    post(client, "/profile/capabilities", label="Cost attribution")
    assert versions(engine, user_id) == 2

    body = text_of(client.get("/profile/history").text)
    assert "Current" in body
    # Newest first: the version holding both rows is listed above the one that
    # held only the first. Comparing the labels would not show this -- the newer
    # version contains both of them.
    assert "2 capabilities, 0 with a depth stated" in body
    assert "1 capability, 0 with a depth stated" in body
    assert body.index("2 capabilities") < body.index("1 capability,")


def test_history_is_empty_before_anything_is_saved(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs, engine)
    assert "Nothing saved yet" in text_of(client.get("/profile/history").text)
