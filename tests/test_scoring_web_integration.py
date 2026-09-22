"""Slice B4's screens against a live Postgres: the "Score this application"
action, the panel it renders, its htmx polling, CSRF and tenancy -- through the
actual routes rather than the repository directly.

Marked `integration`; needs `docker compose up -d` and `alembic upgrade head`.
Google is stubbed, same pattern as `test_applications_web_integration.py`.

No Anthropic API calls anywhere: nothing here constructs a client, the worker
is never run, and `validate_api_keys` is off. What the POST does is create a
row and enqueue a task; what the GET does is render what is stored.
"""

from __future__ import annotations

import datetime as dt
import decimal
import os
import re
import uuid
from collections.abc import Iterator

import pytest
from fastapi import Request
from fastapi.responses import RedirectResponse, Response
from fastapi.testclient import TestClient
from jfl_core.crypto.envelope import MasterKey
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import (
    ConstraintVerdict,
    HardGateBreach,
    NotStated,
    ObjectiveVerdict,
    ScoreLever,
)
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.scores import (
    COULD_GET_LABEL,
    NO_WANT_IT_SCORE,
    SILENCE_NOTE,
    UNMEASURED,
    WANT_IT_LABEL,
    WANT_IT_SUBTITLE,
)
from jfl_web.settings import WebSettings
from markupsafe import escape
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

AD = "Engineering Manager at Northwind. Five years of Python. On site in Manchester."


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


def sign_in(client: TestClient, google: StubGoogle, subs: list[str]) -> uuid.UUID:
    identity = GoogleIdentity(
        sub=f"test-sub-{uuid.uuid4()}",
        email=f"{uuid.uuid4()}@test.invalid",
        display_name="Test Person",
    )
    subs.append(identity.sub)
    google.identity = identity
    response = client.get("/auth/google/callback")
    assert response.status_code == 200, response.text
    page = client.get("/applications")
    assert page.status_code == 200
    return _signed_in_user_id(client)


def _signed_in_user_id(client: TestClient) -> uuid.UUID:
    """Whose session this client holds, read out of the app's own engine."""
    from jfl_core.db.tables import sessions as sessions_table
    from jfl_web.security import hash_token

    token = client.cookies.get("__Host-jfl_session")
    assert token is not None
    engine = client.app.state.engine  # type: ignore[attr-defined]
    with engine.begin() as conn:
        return conn.execute(
            select(sessions_table.c.user_id).where(sessions_table.c.token_hash == hash_token(token))
        ).scalar_one()


def rendered(text: str) -> str:
    """The same text as Jinja puts it on the page -- autoescaping turns the
    apostrophe in "the claim gate's over-claim rate" into an entity, and an
    assertion against the raw constant would fail for the wrong reason.
    """
    return str(escape(text))


def _csrf(client: TestClient, path: str = "/applications/new") -> str:
    page = client.get(path).text
    match = re.search(r'name="csrf_token" value="([^"]+)"', page)
    assert match is not None, f"no CSRF token found on {path}"
    return match.group(1)


def add_application(client: TestClient) -> uuid.UUID:
    response = client.post(
        "/applications",
        data={"csrf_token": _csrf(client), "job_ad": AD, "url": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    return uuid.UUID(response.headers["location"].rsplit("/", 1)[1])


def score_tasks(engine: Engine, user_id: uuid.UUID) -> list[uuid.UUID]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table.c.id).where(
                    tasks_table.c.user_id == user_id,
                    tasks_table.c.kind == "score_application",
                )
            ).scalars()
        )


def finish_score(
    engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID, **kw: object
) -> uuid.UUID:
    """A completed run, written straight to the repository -- the worker's job,
    done here so the page can be rendered without a model call.
    """
    with engine.begin() as conn:
        repo = PostgresScoreRepository(conn, user_id)
        row = repo.create_pending(application_id)
        defaults: dict[str, object] = {
            "could_get_score": 6,
            "could_get_assessment": "Three roles evidence the platform work.",
            "want_it_score": 3,
            "want_it_assessment": "The commute breaks what you said you would travel.",
            "constraint_verdicts": [
                ConstraintVerdict(
                    kind="workplace",
                    stance="must",
                    label="working arrangement -- one day a week at most",
                    verdict="contradicted",
                    note="On site five days a week in Manchester.",
                ),
                ConstraintVerdict(
                    kind="comp_floor",
                    stance="nice",
                    label="lowest package",
                    verdict="silent",
                    note="The ad does not mention pay.",
                ),
            ],
            "objective_verdicts": [
                ObjectiveVerdict(
                    rank=1,
                    objective="Back to hands-on work",
                    verdict="partial",
                    note="Unlikely here.",
                )
            ],
            "hard_gate_breaches": [
                HardGateBreach(gate="location", breach="On site five days a week in Manchester.")
            ],
            "levers": [
                ScoreLever(
                    fact_text="Ran a team of 12",
                    role_label="Northwind",
                    would_move_to=8,
                    note="Covers the headcount requirement.",
                )
            ],
            "not_stated": [
                NotStated(question_key="capabilities", wording="What you can do, and at what depth")
            ],
            "model": "claude-opus-5",
            "cost_usd": decimal.Decimal("0.4231"),
            "trace_id": uuid.uuid4(),
        }
        defaults.update(kw)
        repo.mark_done(row.id, **defaults)  # type: ignore[arg-type]
    return row.id


# --------------------------------------------------------------------------
# Asking for a score
# --------------------------------------------------------------------------


def test_the_action_enqueues_exactly_one_run(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)

    response = client.post(
        f"/applications/{application_id}/score",
        data={"csrf_token": _csrf(client, f"/applications/{application_id}")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/applications/{application_id}"
    assert len(score_tasks(engine, user_id)) == 1

    with engine.begin() as conn:
        latest = PostgresScoreRepository(conn, user_id).latest(application_id)
    assert latest is not None and latest.status == "pending"


def test_pressing_it_again_while_one_is_in_flight_does_not_buy_a_second_call(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Two presses on a page that says "scoring" would be two charges for one
    answer.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    token = _csrf(client, f"/applications/{application_id}")

    for _ in range(2):
        client.post(
            f"/applications/{application_id}/score",
            data={"csrf_token": token},
            follow_redirects=False,
        )

    assert len(score_tasks(engine, user_id)) == 1
    with engine.begin() as conn:
        assert len(PostgresScoreRepository(conn, user_id).history(application_id)) == 1


def test_rescoring_a_finished_run_enqueues_a_new_one_and_keeps_the_old(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    first = finish_score(engine, user_id, application_id, could_get_score=4, want_it_score=9)

    client.post(
        f"/applications/{application_id}/score",
        data={"csrf_token": _csrf(client, f"/applications/{application_id}")},
        follow_redirects=False,
    )

    assert len(score_tasks(engine, user_id)) == 1
    with engine.begin() as conn:
        history = PostgresScoreRepository(conn, user_id).history(application_id)
    assert [h.id for h in history][0] == first
    assert len(history) == 2
    assert history[0].could_get_score == 4, "the earlier run was overwritten"


def test_the_action_needs_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)

    response = client.post(f"/applications/{application_id}/score", data={})
    assert response.status_code == 403
    assert score_tasks(engine, user_id) == []


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------


def test_the_panel_offers_the_action_and_the_unmeasured_label_before_any_run(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    application_id = add_application(client)

    page = client.get(f"/applications/{application_id}").text
    assert "Score this application" in page
    assert rendered(UNMEASURED) in page
    assert "billed to you" in page or "your own API key" in page


def test_the_panel_shows_both_numbers_and_both_paragraphs(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    page = client.get(f"/applications/{application_id}").text

    assert COULD_GET_LABEL in page
    assert WANT_IT_LABEL in page
    assert ">6<" in page and ">3<" in page
    assert "Three roles evidence the platform work." in page
    assert "The commute breaks what you said you would travel." in page
    # Unmeasured, on screen, in those words.
    assert rendered(UNMEASURED) in page
    # Per-run cost, because the user paid for it with their own key.
    assert "$0.4231" in page


def test_a_breached_hard_gate_is_shown_in_plain_words(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Never folded silently into the number above it."""
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    page = client.get(f"/applications/{application_id}").text
    # "Must-haves", not "hard gates": the screens say what the user typed on
    # /profile, and "gate" is this codebase's word, not theirs.
    assert "Your must-haves this ad breaks" in page
    assert "On site five days a week in Manchester." in page


def test_an_unevidenced_capability_is_offered_as_a_lever_never_as_evidence(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """docs/profile-schema.md: a tier with no evidence is a claim, not a fact.
    The panel says which it is.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(
        engine,
        user_id,
        application_id,
        levers=[
            ScoreLever(
                fact_text="Kubernetes",
                claim_kind="capability",
                tier="working",
                would_move_to=8,
                note="Covers requirement 2.",
            )
        ],
    )

    page = " ".join(client.get(f"/applications/{application_id}").text.split())
    assert "Kubernetes" in page
    assert "your profile, claimed at working, no evidence" in page


def test_an_unconfirmed_cv_claim_is_offered_as_a_lever(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    page = " ".join(client.get(f"/applications/{application_id}").text.split())
    assert "Ran a team of 12" in page
    # The honest form of it: not credited, but named, with what it would be worth.
    assert "not evidence yet and did not count towards" in page
    assert "moves from 6 to 8." in page
    assert "Covers the headcount requirement." in page


def test_an_unfilled_profile_section_is_shown_as_not_stated(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    page = client.get(f"/applications/{application_id}").text
    assert "Not stated" in page
    assert "What you can do, and at what depth" in page


def test_every_constraint_is_listed_and_the_silences_read_as_questions(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The silences are the product: a constraint the ad says nothing about is
    a question to ask at interview, and the panel has to say so rather than
    quietly leaving it out.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    page = " ".join(client.get(f"/applications/{application_id}").text.split())
    assert "What the ad says about what you asked for" in page
    assert rendered(SILENCE_NOTE) in page
    assert "lowest package" in page
    assert "silent" in page and "the ad does not say -- ask" in page
    assert "The ad does not mention pay." in page


def test_the_number_is_shown_with_what_it_was_derived_from(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """A derived number is only honest if what it was derived from is printed
    beside it -- otherwise "3/10" reads as a prediction.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    page = " ".join(client.get(f"/applications/{application_id}").text.split())
    assert rendered(WANT_IT_SUBTITLE) in page
    assert "Across what you said matters:" in page
    assert "1 not mentioned" in page
    assert "A must-have or a never is broken" in page


def test_a_run_over_an_empty_profile_shows_no_second_number_at_all(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """Not a 1. Nothing was measured, so nothing is claimed."""
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(
        engine,
        user_id,
        application_id,
        want_it_score=None,
        constraint_verdicts=[],
        objective_verdicts=[],
        hard_gate_breaches=[],
    )

    page = " ".join(client.get(f"/applications/{application_id}").text.split())
    assert rendered(NO_WANT_IT_SCORE) in page


def test_the_fragment_polls_only_while_a_run_is_pending(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    client.post(
        f"/applications/{application_id}/score",
        data={"csrf_token": _csrf(client, f"/applications/{application_id}")},
        follow_redirects=False,
    )

    pending = client.get(f"/applications/{application_id}/score")
    assert pending.status_code == 200
    assert 'hx-trigger="every 5s"' in pending.text

    finish_score(engine, user_id, application_id)
    done = client.get(f"/applications/{application_id}/score").text
    assert "hx-trigger" not in done
    assert COULD_GET_LABEL in done


def test_a_failed_run_says_what_to_do_about_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    with engine.begin() as conn:
        repo = PostgresScoreRepository(conn, user_id)
        repo.mark_failed(repo.create_pending(application_id).id, "no_api_key")

    page = client.get(f"/applications/{application_id}").text
    assert "your own Anthropic API key" in page
    assert 'href="/settings"' in page


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_another_users_application_is_a_404_for_both_routes(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    sign_in(client, google, subs)
    theirs = add_application(client)
    client.get("/auth/logout")

    sign_in(client, google, subs)
    token = _csrf(client)

    assert client.get(f"/applications/{theirs}/score").status_code == 404
    posted = client.post(f"/applications/{theirs}/score", data={"csrf_token": token})
    assert posted.status_code == 404


def test_another_users_score_is_never_rendered_on_your_page(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    first_user = sign_in(client, google, subs)
    theirs = add_application(client)
    finish_score(engine, first_user, theirs)
    client.get("/auth/logout")

    sign_in(client, google, subs)
    mine = add_application(client)
    page = client.get(f"/applications/{mine}").text
    assert "Three roles evidence the platform work." not in page
    assert "Not scored yet." in page
