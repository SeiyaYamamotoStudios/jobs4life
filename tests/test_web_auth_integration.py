"""Slice A end to end against a live Postgres: sign-in, sessions, tenancy,
credential custody.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.

Google is stubbed. What is being tested here is everything on *this* side of the
callback -- which user row a `sub` resolves to, what a session is, what a second
user can reach -- and none of that gets more true for having gone through a real
consent screen.

`TestClient` runs over `https://testserver` so the `__Host-` cookie is exercised
as it is in deployment rather than being switched off for the test.

Every test mints its own Google `sub`; teardown deletes the users it created and
the FK cascade takes their sessions and credentials with them.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey, unseal
from jfl_core.db.tables import sessions as sessions_table
from jfl_core.db.tables import users as users_table
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_web.app import create_app
from jfl_web.credentials import store_api_key
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Connection, Engine

pytestmark = pytest.mark.integration

KEY_A = "sk-ant-api03-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1111"
KEY_B = "sk-ant-api03-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2222"


class StubGoogle:
    """Stands in for `AuthlibGoogleProvider`. The identity it returns is set by
    the test, which is the only thing a real Google would contribute here.
    """

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
        # Off: validation constructs a real Anthropic client, which the root
        # conftest guard blocks in anything not marked `e2e`. See
        # tests/test_gate_e2e.py for the pattern when a real call is wanted.
        validate_api_keys=False,
    )


@pytest.fixture
def google() -> StubGoogle:
    return StubGoogle()


@pytest.fixture
def subs() -> list[str]:
    """Google `sub` values minted by a test, cleaned up afterwards."""
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


def user_id_for(engine: Engine, sub: str) -> uuid.UUID:
    with engine.connect() as conn:
        row = conn.execute(select(users_table.c.id).where(users_table.c.google_sub == sub)).one()
    return uuid.UUID(str(row.id))


# --------------------------------------------------------------------------
# A1/A2 -- identity and sessions
# --------------------------------------------------------------------------


def test_a_signed_out_visitor_gets_the_login_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Sign in with Google" in response.text


def test_signing_in_creates_a_user_keyed_on_sub(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    identity = sign_in(client, google, subs)
    with engine.connect() as conn:
        row = conn.execute(
            select(users_table.c.google_sub, users_table.c.email).where(
                users_table.c.google_sub == identity.sub
            )
        ).one()
    assert row.email == identity.email
    assert client.get("/").text.count(identity.email) >= 1


def test_the_same_sub_with_a_new_email_is_the_same_account(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    """The reason identity is the `sub`. A person changes their address; they do
    not become a new user, and their applications do not vanish.
    """
    first = sign_in(client, google, subs)
    original_id = user_id_for(engine, first.sub)

    client.post("/logout", data={"csrf_token": _csrf(client)})
    sign_in(client, google, subs, sub=first.sub, email="renamed@test.invalid", name="Renamed")

    assert user_id_for(engine, first.sub) == original_id
    with engine.connect() as conn:
        row = conn.execute(
            select(users_table.c.email, users_table.c.display_name).where(
                users_table.c.id == original_id
            )
        ).one()
    assert row.email == "renamed@test.invalid"
    assert row.display_name == "Renamed"


def test_a_new_sub_never_inherits_an_account_by_email(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    """The failure keying on email would eventually cause: an address is
    reassigned, a different person signs in, and gets somebody's career history.
    Identity stays the `sub`, so the stranger gets their own new account
    instead -- never the original one.
    """
    first = sign_in(client, google, subs)
    original_id = user_id_for(engine, first.sub)
    client.post("/logout", data={"csrf_token": _csrf(client)})

    stranger = f"test-sub-{uuid.uuid4()}"
    subs.append(stranger)
    google.identity = GoogleIdentity(sub=stranger, email=first.email, display_name="Someone Else")
    response = client.get("/auth/google/callback")

    assert response.status_code == 200  # signed in, not bounced to login
    with engine.connect() as conn:
        stranger_id = conn.execute(
            select(users_table.c.id).where(users_table.c.google_sub == stranger)
        ).one()
        # A new, different account -- not the original one, and not refused.
        assert stranger_id.id != original_id
        # And the original account is untouched.
        assert conn.execute(select(users_table.c.id).where(users_table.c.id == original_id)).one()


def test_two_users_with_the_same_email_can_both_exist_and_sign_in(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    """`users.email` carries no uniqueness constraint (migration
    6b3ce06d7b4e): email is presentation-only, so a reassigned
    address must not lock its new owner out. Both accounts stay independently
    resolvable by their own `sub`, and both can sign in.
    """
    shared_email = f"{uuid.uuid4()}@test.invalid"
    first = sign_in(client, google, subs, email=shared_email)
    first_id = user_id_for(engine, first.sub)
    assert "Signed in" in client.get("/").text
    client.post("/logout", data={"csrf_token": _csrf(client)})

    second = sign_in(client, google, subs, email=shared_email)
    second_id = user_id_for(engine, second.sub)
    assert "Signed in" in client.get("/").text

    assert first_id != second_id
    with engine.connect() as conn:
        rows = conn.execute(
            select(users_table.c.id, users_table.c.email).where(
                users_table.c.id.in_([first_id, second_id])
            )
        ).all()
    assert {row.id for row in rows} == {first_id, second_id}
    assert all(row.email == shared_email for row in rows)

    # Signing back into the first account still resolves to the same row.
    client.post("/logout", data={"csrf_token": _csrf(client)})
    sign_in(client, google, subs, sub=first.sub, email=shared_email)
    assert user_id_for(engine, first.sub) == first_id


def test_the_session_cookie_is_host_prefixed_and_opaque(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    identity = sign_in(client, google, subs)
    cookie = client.cookies.get("__Host-jfl_session")
    assert cookie is not None

    with engine.connect() as conn:
        rows = conn.execute(
            select(sessions_table.c.token_hash).where(
                sessions_table.c.user_id == user_id_for(engine, identity.sub)
            )
        ).all()
    assert len(rows) == 1
    # The stored value is a hash: what is in the database cannot be replayed.
    assert rows[0].token_hash != cookie
    assert cookie not in rows[0].token_hash


def test_logging_out_deletes_the_row_not_just_the_cookie(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    identity = sign_in(client, google, subs)
    user_id = user_id_for(engine, identity.sub)
    stolen = client.cookies.get("__Host-jfl_session")
    assert stolen is not None

    response = client.post("/logout", data={"csrf_token": _csrf(client)})
    assert "Sign in with Google" in response.text

    with engine.connect() as conn:
        assert (
            conn.execute(
                select(sessions_table.c.id).where(sessions_table.c.user_id == user_id)
            ).first()
            is None
        )

    # A copy of the cookie taken before logout is now worthless -- which is the
    # whole reason this is a database row and not a JWT.
    client.cookies.set("__Host-jfl_session", stolen)
    assert "Sign in with Google" in client.get("/").text


def test_a_forged_session_cookie_is_not_a_session(client: TestClient) -> None:
    client.cookies.set("__Host-jfl_session", "not-a-real-token")
    assert "Sign in with Google" in client.get("/").text


def test_state_changing_routes_require_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    assert client.post("/logout", data={}).status_code == 403
    assert client.post("/logout", data={"csrf_token": "wrong"}).status_code == 403
    # Still signed in.
    assert "Signed in" in client.get("/").text


def test_the_api_key_form_requires_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    response = client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": "wrong"})
    assert response.status_code == 403
    assert "sk-ant" not in response.text


# --------------------------------------------------------------------------
# A4 -- credential custody
# --------------------------------------------------------------------------


def test_a_key_is_stored_and_never_rendered_back(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str], master_key: MasterKey
) -> None:
    identity = sign_in(client, google, subs)
    response = client.post(
        "/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)}
    )
    assert response.status_code == 200
    assert "Key saved" in response.text

    page = client.get("/settings").text
    assert KEY_A not in page
    assert KEY_A[-8:] not in page  # not even most of the tail
    assert "sk-ant-…1111" in page  # the four-character hint, and no more

    # It really is stored, and it really is the key that was typed.
    user_id = user_id_for(engine, identity.sub)
    with engine.connect() as conn:
        repo = PostgresCredentialRepository(conn, user_id)
        sealed = repo.load_sealed(ANTHROPIC_API_KEY)
    assert sealed is not None
    assert unseal(master_key, sealed, user_id=user_id, provider=ANTHROPIC_API_KEY) == KEY_A


def test_replacing_a_key_reseals_it_with_a_fresh_nonce(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str], master_key: MasterKey
) -> None:
    identity = sign_in(client, google, subs)
    user_id = user_id_for(engine, identity.sub)

    client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)})
    with engine.connect() as conn:
        first = PostgresCredentialRepository(conn, user_id).load_sealed(ANTHROPIC_API_KEY)

    client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)})
    with engine.connect() as conn:
        second = PostgresCredentialRepository(conn, user_id).load_sealed(ANTHROPIC_API_KEY)

    assert first is not None and second is not None
    # Same plaintext, different ciphertext: a fresh nonce every time.
    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert unseal(master_key, second, user_id=user_id, provider=ANTHROPIC_API_KEY) == KEY_A


def test_the_plaintext_key_is_nowhere_in_the_row(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    identity = sign_in(client, google, subs)
    client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)})

    from jfl_core.db.tables import user_credentials

    with engine.connect() as conn:
        row = conn.execute(
            select(user_credentials).where(
                user_credentials.c.user_id == user_id_for(engine, identity.sub)
            )
        ).one()
    blob = repr(tuple(row)).encode()
    assert KEY_A.encode() not in blob
    assert KEY_A[:-4].encode() not in blob


# --------------------------------------------------------------------------
# A3 -- tenancy, proved rather than reviewed
# --------------------------------------------------------------------------


def test_two_users_cannot_reach_each_others_credentials(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str], master_key: MasterKey
) -> None:
    """The acceptance criterion from PLAN.md, at the repository layer.

    Both repositories run on the *same connection*, so nothing but the id fixed
    at construction separates them -- no per-request filter, no session, no
    middleware. If scoping were a `WHERE` clause someone had to remember, this is
    where it would show.
    """
    alice = sign_in(client, google, subs)
    client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)})
    client.post("/logout", data={"csrf_token": _csrf(client)})

    bob = sign_in(client, google, subs)
    client.post("/settings/api-key", data={"api_key": KEY_B, "csrf_token": _csrf(client)})

    alice_id = user_id_for(engine, alice.sub)
    bob_id = user_id_for(engine, bob.sub)
    assert alice_id != bob_id

    with engine.connect() as conn:
        alice_repo = PostgresCredentialRepository(conn, alice_id)
        bob_repo = PostgresCredentialRepository(conn, bob_id)

        alice_summary = alice_repo.summary(ANTHROPIC_API_KEY)
        bob_summary = bob_repo.summary(ANTHROPIC_API_KEY)
        assert alice_summary is not None and bob_summary is not None
        assert alice_summary.key_hint == "1111"
        assert bob_summary.key_hint == "2222"

        alice_sealed = alice_repo.load_sealed(ANTHROPIC_API_KEY)
        bob_sealed = bob_repo.load_sealed(ANTHROPIC_API_KEY)
        assert alice_sealed is not None and bob_sealed is not None
        assert alice_sealed.ciphertext != bob_sealed.ciphertext

        # Bob's repository returns Bob's key and there is no argument that would
        # make it return Alice's.
        assert unseal(master_key, bob_sealed, user_id=bob_id, provider=ANTHROPIC_API_KEY) == KEY_B

        # And even holding Alice's ciphertext, Bob's identity cannot open it:
        # the AAD binds it to her row.
        from jfl_core.crypto.envelope import SecretUnsealError

        with pytest.raises(SecretUnsealError):
            unseal(master_key, alice_sealed, user_id=bob_id, provider=ANTHROPIC_API_KEY)

        # Deleting through Bob's repository leaves Alice's row alone.
        bob_repo.delete(ANTHROPIC_API_KEY)
        assert bob_repo.summary(ANTHROPIC_API_KEY) is None
        assert alice_repo.summary(ANTHROPIC_API_KEY) is not None


def test_a_second_user_sees_only_their_own_settings_page(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    alice = sign_in(client, google, subs)
    client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)})
    client.post("/logout", data={"csrf_token": _csrf(client)})

    bob = sign_in(client, google, subs)
    page = client.get("/settings").text
    assert "No key stored yet" in page
    assert "1111" not in page
    assert alice.email not in page
    assert bob.email in client.get("/").text


def test_a_deleted_user_takes_their_sessions_and_credentials_with_them(
    client: TestClient, google: StubGoogle, engine: Engine, subs: list[str]
) -> None:
    """`ON DELETE CASCADE`, checked rather than assumed: a credential surviving
    the account it belongs to is ciphertext nobody is accountable for.
    """
    from jfl_core.db.tables import user_credentials

    identity = sign_in(client, google, subs)
    client.post("/settings/api-key", data={"api_key": KEY_A, "csrf_token": _csrf(client)})
    user_id = user_id_for(engine, identity.sub)

    with engine.begin() as conn:
        conn.execute(delete(users_table).where(users_table.c.id == user_id))
    with engine.connect() as conn:
        assert _count(conn, sessions_table, user_id) == 0
        assert _count(conn, user_credentials, user_id) == 0


def test_store_api_key_uses_the_repositorys_own_user(
    engine: Engine, master_key: MasterKey, client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    """There is no user argument to `store_api_key` to get wrong: the repository
    it is handed already knows whose row it is writing.
    """
    identity = sign_in(client, google, subs)
    user_id = user_id_for(engine, identity.sub)
    with engine.begin() as conn:
        repo = PostgresCredentialRepository(conn, user_id)
        assert store_api_key(repo, master_key, KEY_B) == "2222"
        assert repo.user_id == user_id


def _csrf(client: TestClient) -> str:
    """Scrape the token out of a rendered form -- the same way a browser gets it."""
    import re

    page = client.get("/settings").text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None, "no CSRF token in the rendered page"
    return match.group(1)


def _count(conn: Connection, table: Any, user_id: uuid.UUID) -> int:
    from sqlalchemy import func

    return int(
        conn.execute(
            select(func.count()).select_from(table).where(table.c.user_id == user_id)
        ).scalar_one()
    )
