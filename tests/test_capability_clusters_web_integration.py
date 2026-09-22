"""The capability-clustering panel through the real `/profile` routes against a
live Postgres. Marked `integration`; needs `docker compose up -d` and
`alembic upgrade head`.

**No model call happens here and none may.** The route enqueues a task and
returns; the root `conftest.py` guard is what would raise if anything in a
request reached the API. What these tests defend:

* pressing the button enqueues **one** task, and pressing it again while a run
  is in flight enqueues none;
* a user with no confirmed facts is told so and pointed at where facts come
  from, rather than being charged to find out;
* a proposal is **never applied silently** -- it lands on the profile when, and
  only when, the user accepts it, under the name they chose;
* a run belonging to another account is a 404, not a read;
* every state-changing route needs CSRF.
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
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import sent_documents as sent_documents_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import content_hash
from jfl_core.models import ProposedCapability, ProposedFact
from jfl_core.profile import Capability, Profile, capability_key
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.capability_clusters import PostgresCapabilityClusterRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"


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
        return conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == identity.sub)
        ).scalar_one()


def csrf(client: TestClient, path: str = "/profile") -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text)
    assert match is not None, f"no CSRF token on {path}"
    return match.group(1)


def post(client: TestClient, path: str, **fields: object) -> Response:
    return client.post(path, data={"csrf_token": csrf(client), **fields}, follow_redirects=False)


def text_of(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, FAKE_KEY, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=FAKE_KEY[-4:],
        )


def a_confirmed_fact(engine: Engine, user_id: uuid.UUID, text: str) -> uuid.UUID:
    """One CV fact, confirmed by its owner -- the only kind a capability may
    cite. Goes in through the real repository, so the span it produces is an
    ordinary corpus span rather than a fixture.
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
        confirmed = repo.confirm(repo.list_facts()[-1].id)
        assert confirmed is not None
        return confirmed.id


def a_finished_run(
    engine: Engine,
    user_id: uuid.UUID,
    proposals: list[ProposedCapability],
    *,
    unclustered: list[uuid.UUID] | None = None,
) -> uuid.UUID:
    """A run the worker has already answered. Written through the real
    repository, because what these tests exercise is the screen, not the call.
    """
    with engine.begin() as conn:
        repo = PostgresCapabilityClusterRepository(conn, user_id)
        row = repo.create_pending(trace_id=uuid.uuid4())
        repo.mark_done(
            row.id,
            proposals,
            fact_count=len(proposals),
            unclustered_fact_ids=unclustered or [],
        )
    return row.id


def stored_profile(engine: Engine, user_id: uuid.UUID) -> Profile:
    with engine.begin() as conn:
        return PostgresProfileRepository(conn, user_id).current()


def queued(engine: Engine, user_id: uuid.UUID) -> int:
    with engine.begin() as conn:
        return conn.execute(
            select(func.count())
            .select_from(tasks_table)
            .where(
                tasks_table.c.user_id == user_id,
                tasks_table.c.kind == "cluster_capabilities",
            )
        ).scalar_one()


# -- starting a run -----------------------------------------------------------


def test_pressing_the_button_enqueues_exactly_one_task(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)
    a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")

    assert post(client, "/profile/capabilities/cluster").status_code == 303
    assert queued(engine, user_id) == 1


def test_pressing_it_again_while_one_is_running_enqueues_nothing(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    """One press, one call. Their money, not ours."""
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)
    a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")

    post(client, "/profile/capabilities/cluster")
    post(client, "/profile/capabilities/cluster")
    assert queued(engine, user_id) == 1


def test_no_confirmed_facts_says_so_and_points_at_where_facts_come_from(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)

    response = post(client, "/profile/capabilities/cluster")
    assert response.status_code == 303
    assert "cluster=no_facts" in response.headers["location"]
    assert queued(engine, user_id) == 0

    body = client.get("/profile?cluster=no_facts").text
    assert "nothing to group" in text_of(body)
    assert "/corpus/facts" in body


def test_no_api_key_enqueues_nothing_and_says_where_to_add_one(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")

    response = post(client, "/profile/capabilities/cluster")
    assert "cluster=needs_key" in response.headers["location"]
    assert queued(engine, user_id) == 0
    assert "/settings" in client.get("/profile?cluster=needs_key").text


# -- the panel ----------------------------------------------------------------


def test_a_pending_run_polls_itself(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    master_key: MasterKey,
) -> None:
    user_id = sign_in(client, google, subs, engine)
    store_key(engine, user_id, master_key)
    a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    post(client, "/profile/capabilities/cluster")

    body = client.get("/profile").text
    assert 'hx-trigger="every 3s"' in body
    assert "/profile/capabilities/cluster/" in body


def test_the_poll_route_renders_the_same_panel(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    cluster_id = a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[fact_id])],
    )

    body = client.get(f"/profile/capabilities/cluster/{cluster_id}").text
    assert "FX pricing platforms" in body
    assert "Rebuilt the FX pricing platform." in body


def test_another_accounts_run_is_a_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    first = sign_in(client, google, subs, engine)
    cluster_id = a_finished_run(engine, first, [ProposedCapability(label="FX pricing platforms")])

    sign_in(client, google, subs, engine)  # a second account, same client
    key = capability_key("FX pricing platforms")
    assert client.get(f"/profile/capabilities/cluster/{cluster_id}").status_code == 404
    accept = post(client, f"/profile/capabilities/cluster/{cluster_id}/{key}/accept")
    assert accept.status_code == 404


def test_facts_the_run_did_not_place_are_shown_rather_than_dropped(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    placed = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    loose = a_confirmed_fact(engine, user_id, "Wrote the incident policy.")
    a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[placed])],
        unclustered=[loose],
    )

    body = text_of(client.get("/profile").text)
    assert "did not place" in body
    assert "Wrote the incident policy." in body


# -- accepting, renaming, rejecting -------------------------------------------


def test_a_proposal_is_not_on_the_profile_until_it_is_accepted(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[fact_id])],
    )
    client.get("/profile")

    labels = [c.label for c in stored_profile(engine, user_id).capabilities]
    assert "FX pricing platforms" not in labels


def test_accepting_puts_it_on_the_profile_with_its_evidence(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    with engine.begin() as conn:
        span_id = PostgresCandidateFactRepository(conn, user_id).get_fact(fact_id).span_id  # type: ignore[union-attr]
    cluster_id = a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[fact_id], span_ids=[span_id])],
    )

    key = capability_key("FX pricing platforms")
    assert (
        post(client, f"/profile/capabilities/cluster/{cluster_id}/{key}/accept").status_code == 303
    )

    saved = [
        c for c in stored_profile(engine, user_id).capabilities if c.label == "FX pricing platforms"
    ]
    assert len(saved) == 1
    assert saved[0].source == "clustered"
    assert saved[0].evidence == [span_id]
    assert saved[0].tier is None, "a proposal never arrives carrying a depth nobody chose"


def test_a_rename_is_stored_exactly_as_typed(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    cluster_id = a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[fact_id])],
    )

    key = capability_key("FX pricing platforms")
    post(
        client,
        f"/profile/capabilities/cluster/{cluster_id}/{key}/accept",
        label="FX rates and pricing",
    )

    labels = [c.label for c in stored_profile(engine, user_id).capabilities]
    assert "FX rates and pricing" in labels
    assert "FX pricing platforms" not in labels


def test_accepting_never_overwrites_a_row_the_user_tiered(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    with engine.begin() as conn:
        PostgresProfileRepository(conn, user_id).save(
            Profile(
                capabilities=[
                    Capability(label="FX pricing platforms", tier="working", source="user")
                ]
            )
        )
    cluster_id = a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="fx  PRICING platforms", fact_ids=[fact_id])],
    )

    key = capability_key("fx  PRICING platforms")
    post(client, f"/profile/capabilities/cluster/{cluster_id}/{key}/accept")

    rows = stored_profile(engine, user_id).capabilities
    assert len(rows) == 1
    assert rows[0].label == "FX pricing platforms"
    assert rows[0].tier == "working"
    assert rows[0].source == "user"


def test_an_accepted_proposal_is_not_offered_again(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    cluster_id = a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[fact_id])],
    )
    key = capability_key("FX pricing platforms")
    post(client, f"/profile/capabilities/cluster/{cluster_id}/{key}/accept")

    panel = client.get(f"/profile/capabilities/cluster/{cluster_id}").text
    assert "Add it to my profile" not in panel


def test_rejecting_writes_nothing_to_the_profile_and_keeps_the_fact(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    fact_id = a_confirmed_fact(engine, user_id, "Rebuilt the FX pricing platform.")
    cluster_id = a_finished_run(
        engine,
        user_id,
        [ProposedCapability(label="FX pricing platforms", fact_ids=[fact_id])],
    )

    key = capability_key("FX pricing platforms")
    assert (
        post(client, f"/profile/capabilities/cluster/{cluster_id}/{key}/reject").status_code == 303
    )

    labels = [c.label for c in stored_profile(engine, user_id).capabilities]
    assert "FX pricing platforms" not in labels
    with engine.begin() as conn:
        fact = PostgresCandidateFactRepository(conn, user_id).get_fact(fact_id)
    assert fact is not None and fact.state == "confirmed", (
        "rejecting a grouping is rejecting a label, never retracting a fact"
    )


def test_an_unknown_proposal_key_is_a_404(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    cluster_id = a_finished_run(engine, user_id, [ProposedCapability(label="FX pricing")])
    assert (
        post(client, f"/profile/capabilities/cluster/{cluster_id}/deadbeef/accept").status_code
        == 404
    )


# -- CSRF ---------------------------------------------------------------------


def test_every_clustering_post_requires_csrf(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs, engine)
    cluster_id = a_finished_run(engine, user_id, [ProposedCapability(label="FX pricing")])
    key = capability_key("FX pricing")
    for path in (
        "/profile/capabilities/cluster",
        f"/profile/capabilities/cluster/{cluster_id}/{key}/accept",
        f"/profile/capabilities/cluster/{cluster_id}/{key}/reject",
    ):
        assert client.post(path, data={}, follow_redirects=False).status_code == 403
    assert queued(engine, user_id) == 0
