"""The whole pushback loop through the real routes, against a live Postgres.

Marked `integration`; needs a migrated database. Google is stubbed, same
pattern as `test_scoring_web_integration.py`. No Anthropic call anywhere:
nothing here constructs a client, the worker is never run, and the
classification a real run would produce is written straight to the repository
so the screens can be exercised without spending anything.

What this pins is the loop as a person walks it: push back, see the
classification, correct it, and see -- on the page -- what changed, what did
not, and what would. Plus the guarantees that are the whole point: the stored
score is never rewritten, a claim that the tool has underrated you moves
nothing until a fact is confirmed, the third correction on one dimension stops
arguing and offers a comparison, and the drift meter is on the screen where the
person is when they are about to push back again.
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
from jfl_core.db.tables import application_scores as scores_table
from jfl_core.db.tables import score_pushbacks as pushbacks_table
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import ConstraintVerdict, ObjectiveVerdict
from jfl_core.pushback import COULD_GET_OVERALL, WANT_OVERALL
from jfl_core.storage.pushbacks import PostgresPushbackRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.settings import WebSettings
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

AD = "Staff Platform Engineer at Acme. Org-wide platform ownership. On site in Leeds."
WORDS = "That's wrong, I ran the whole platform for two years."


class StubGoogle:
    def __init__(self) -> None:
        self.identity: GoogleIdentity | None = None
        self.error: str | None = None

    async def authorize_redirect(self, request: Request, redirect_uri: str) -> Response:
        return RedirectResponse("/auth/google/callback", status_code=303)

    async def fetch_identity(self, request: Request) -> GoogleIdentity:
        if self.error is not None:
            raise OAuthError(self.error)
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
    assert client.get("/auth/google/callback").status_code == 200
    assert client.get("/applications").status_code == 200
    return _signed_in_user_id(client)


def _signed_in_user_id(client: TestClient) -> uuid.UUID:
    from jfl_core.db.tables import sessions as sessions_table
    from jfl_web.security import hash_token

    token = client.cookies.get("__Host-jfl_session")
    assert token is not None
    engine = client.app.state.engine  # type: ignore[attr-defined]
    with engine.begin() as conn:
        return conn.execute(
            select(sessions_table.c.user_id).where(sessions_table.c.token_hash == hash_token(token))
        ).scalar_one()


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


def finish_score(
    engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID, **kw: object
) -> uuid.UUID:
    """A completed run written straight to the repository -- the worker's job,
    done here so the page renders without a model call.
    """
    with engine.begin() as conn:
        repo = PostgresScoreRepository(conn, user_id)
        row = repo.create_pending(application_id)
        defaults: dict[str, object] = {
            "could_get_score": 4,
            "could_get_assessment": "The ad asks for org-wide platform ownership.",
            "want_it_score": 5,
            "want_it_assessment": "Two of what you said matters are evidenced.",
            "constraint_verdicts": [
                ConstraintVerdict(
                    kind="comp_floor",
                    stance="nice",
                    label="lowest package",
                    verdict="silent",
                    note="The ad does not mention pay.",
                )
            ],
            "objective_verdicts": [
                ObjectiveVerdict(
                    rank=1, objective="Back to hands-on work", verdict="partial", note="Maybe."
                )
            ],
            "hard_gate_breaches": [],
            "levers": [],
            "not_stated": [],
            "model": "claude-opus-5",
            "cost_usd": None,
            "trace_id": uuid.uuid4(),
        }
        defaults.update(kw)
        repo.mark_done(row.id, **defaults)  # type: ignore[arg-type]
    return row.id


def push_back(
    client: TestClient,
    application_id: uuid.UUID,
    *,
    axis: str = "want",
    dimension: str = WANT_OVERALL,
    direction: str = "up",
    text: str = WORDS,
    points: str = "1",
) -> Response:
    return client.post(
        f"/applications/{application_id}/pushback",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "axis": axis,
            "dimension": dimension,
            "direction": direction,
            "user_text": text,
            "points": points,
        },
        follow_redirects=False,
    )


def latest_pushback(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        return conn.execute(
            select(pushbacks_table.c.id)
            .where(pushbacks_table.c.user_id == user_id)
            .order_by(pushbacks_table.c.created_at.desc())
            .limit(1)
        ).scalar_one()


def classify_as(engine: Engine, user_id: uuid.UUID, pushback_id: uuid.UUID, kind: str) -> None:
    """What the worker's cheap model call would have written. Nothing is
    applied by it -- that is the whole point of the confirmation step.
    """
    with engine.begin() as conn:
        PostgresPushbackRepository(conn, user_id).set_classification(
            pushback_id,
            classification=kind,  # type: ignore[arg-type]
            new_information=True,
            note="Reads as a statement about what you want.",
        )


def apply_as(
    client: TestClient,
    pushback_id: uuid.UUID,
    application_id: uuid.UUID,
    kind: str,
    new_information: str = "yes",
) -> Response:
    return client.post(
        f"/pushbacks/{pushback_id}/apply",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "classification": kind,
            "new_information": new_information,
        },
        follow_redirects=False,
    )


def score_row(engine: Engine, score_id: uuid.UUID) -> dict[str, object]:
    with engine.begin() as conn:
        return (
            conn.execute(select(scores_table).where(scores_table.c.id == score_id)).one()._asdict()
        )


# -- recording ---------------------------------------------------------------


def test_a_pushback_is_recorded_verbatim_and_applies_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    assert push_back(client, application_id).status_code == 303

    page = client.get(f"/applications/{application_id}").text
    assert "I ran the whole platform for two years" in page
    # Nothing has been applied and the page says which kind it thinks it is
    # only as a default on a radio the person has to confirm.
    assert "What kind of statement is this?" in page
    with engine.begin() as conn:
        row = conn.execute(
            select(pushbacks_table).where(pushbacks_table.c.user_id == user_id)
        ).one()
    assert row.user_text == WORDS
    assert row.status == "awaiting_classification"
    assert row.applied_delta is None


def test_recording_a_pushback_queues_exactly_one_classification(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    push_back(client, application_id)

    with engine.begin() as conn:
        kinds = list(
            conn.execute(
                select(tasks_table.c.kind).where(tasks_table.c.user_id == user_id)
            ).scalars()
        )
    assert kinds.count("classify_pushback") == 1


def test_a_pushback_needs_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    response = client.post(
        f"/applications/{application_id}/pushback",
        data={"axis": "want", "dimension": WANT_OVERALL, "direction": "up", "user_text": WORDS},
        follow_redirects=False,
    )
    assert response.status_code == 403


def test_a_pushback_cannot_name_something_protected(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    response = push_back(client, application_id, dimension="coverage:requirement-1")
    assert response.status_code == 400
    assert "not something a pushback can change" in response.text


# -- seeing and correcting the classification --------------------------------


def test_the_user_sees_the_classification_and_can_correct_it(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    """The misclassification guard, end to end.

    The model calls it a preference. The person says it is a claim about their
    own depth. What is applied is the person's answer, and the consequence is
    the asymmetric one: the number does not move.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)
    push_back(client, application_id, axis="get", dimension=COULD_GET_OVERALL)
    pushback_id = latest_pushback(engine, user_id)
    classify_as(engine, user_id, pushback_id, "preference")

    page = client.get(f"/applications/{application_id}").text
    assert "Reads as a statement about what you want." in page
    assert 'value="preference"' in page and "checked" in page

    assert apply_as(client, pushback_id, application_id, "capability").status_code == 303

    with engine.begin() as conn:
        row = conn.execute(select(pushbacks_table).where(pushbacks_table.c.id == pushback_id)).one()
    assert row.classification == "capability"
    assert float(row.applied_delta) == 0.0
    assert row.disposition == "pending_evidence"
    assert score_row(engine, score_id)["could_get_score"] == 4


def test_the_receipt_says_what_changed_what_did_not_and_what_would(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    push_back(client, application_id, axis="get", dimension=COULD_GET_OVERALL)
    pushback_id = latest_pushback(engine, user_id)
    apply_as(client, pushback_id, application_id, "capability")

    page = client.get(f"/applications/{application_id}").text
    assert "What changed now" in page
    assert "What did not change" in page
    assert "What would change it" in page
    assert "The number stayed at 4" in page


# -- the asymmetry, on the screen -------------------------------------------


def test_a_preference_moves_the_displayed_number_and_never_the_stored_one(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)
    before = score_row(engine, score_id)

    push_back(client, application_id, text="I would love a job like this")
    apply_as(client, latest_pushback(engine, user_id), application_id, "preference")

    page = client.get(f"/applications/{application_id}").text
    assert "moved by your corrections" in page
    assert "The tool said 5." in page
    assert score_row(engine, score_id) == before


def test_a_claim_that_you_are_underrated_opens_a_question_and_moves_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)
    push_back(client, application_id, axis="get", dimension=COULD_GET_OVERALL)
    pushback_id = latest_pushback(engine, user_id)
    apply_as(client, pushback_id, application_id, "capability")

    page = client.get(f"/applications/{application_id}").text
    assert "The sentence that would move this" in page
    assert "what decisions were yours" in page
    assert score_row(engine, score_id)["could_get_score"] == 4

    answer = "I owned the platform at Acme: nine engineers, the on-call rota, a 400k budget."
    response = client.post(
        f"/pushbacks/{pushback_id}/evidence",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "answer": answer,
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    with engine.begin() as conn:
        row = conn.execute(select(pushbacks_table).where(pushbacks_table.c.id == pushback_id)).one()
        span = conn.execute(
            select(spans_table.c.text).where(spans_table.c.id == row.resulting_span_id)
        ).scalar_one()
    # Verbatim, and the number STILL has not moved: it moves when the job is
    # scored again against the facts that now include this one.
    assert span == answer
    assert float(row.applied_delta) == 0.0
    assert score_row(engine, score_id)["could_get_score"] == 4


def test_saying_it_again_moves_the_number_less_not_more(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    push_back(client, application_id, text="I really want this kind of role")
    first = latest_pushback(engine, user_id)
    apply_as(client, first, application_id, "preference")

    push_back(client, application_id, text="I really want this kind of role")
    second = latest_pushback(engine, user_id)
    apply_as(client, second, application_id, "preference")

    with engine.begin() as conn:
        deltas = {
            row.id: float(row.applied_delta)
            for row in conn.execute(
                select(pushbacks_table).where(pushbacks_table.c.user_id == user_id)
            )
        }
    assert deltas[first] == pytest.approx(1.0)
    assert deltas[second] == 0.0
    assert "a restatement contributes nothing" in client.get(f"/applications/{application_id}").text


def test_the_third_correction_on_one_dimension_offers_a_comparison(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    for i in range(3):
        push_back(client, application_id, text=f"a genuinely different point number {i}")
        apply_as(client, latest_pushback(engine, user_id), application_id, "preference")

    page = client.get(f"/applications/{application_id}").text
    assert "Let us do this differently" in page
    assert "which of these two would you rather have?" in page


# -- the drift meter ---------------------------------------------------------


def test_the_drift_meter_is_on_the_screen_where_the_pushing_back_happens(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    push_back(client, application_id, text="I would love a job like this")
    apply_as(client, latest_pushback(engine, user_id), application_id, "preference")

    page = client.get(f"/applications/{application_id}").text
    assert "1 pushback, 1 upward, +1.0 net" in page
    assert "What your corrections have done" in page


def test_the_corrections_page_lists_every_one_in_the_users_own_words(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    push_back(client, application_id, text="I would love a job like this")
    apply_as(client, latest_pushback(engine, user_id), application_id, "preference")

    page = client.get("/pushbacks").text
    assert "I would love a job like this" in page
    assert "1 pushback, 1 upward, +1.0 net" in page


def test_every_page_links_to_the_corrections_log(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    assert '<a href="/pushbacks">Corrections</a>' in client.get("/applications").text


# -- the local override ------------------------------------------------------


def test_an_override_is_labelled_scoped_and_feeds_nothing(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)
    other_id = add_application(client)
    other_score = finish_score(engine, user_id, other_id)

    response = client.post(
        f"/applications/{application_id}/override",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "axis": "get",
            "value": "9",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    page = client.get(f"/applications/{application_id}").text
    assert "Your override. The tool said 4." in page
    # Scoped to this application, and not a correction: the other job is
    # untouched and the drift meter has not moved.
    assert "Your override" not in client.get(f"/applications/{other_id}").text
    assert score_row(engine, score_id)["could_get_score"] == 4
    assert score_row(engine, other_score)["could_get_score"] == 4
    assert "0.0 net" not in page  # no meter at all: nothing has been corrected
    assert "What your corrections have done" not in page


# -- what corrections never reach -------------------------------------------


def test_an_application_already_sent_keeps_the_score_it_was_sent_under(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    push_back(client, application_id, text="I would love a job like this")
    apply_as(client, latest_pushback(engine, user_id), application_id, "preference")

    client.post(
        f"/applications/{application_id}/status",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "to_status": "applied",
        },
        follow_redirects=False,
    )
    page = client.get(f"/applications/{application_id}").text
    assert "keeps the score it was sent under" in page
    assert "moved by your corrections" not in page


def test_the_page_says_what_pushing_back_cannot_change(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    page = client.get(f"/applications/{application_id}").text
    assert "What pushing back cannot change, ever" in page
    assert "what the claim gate says about any sentence" in page
    assert "the golden set, the eval labels or the measured over-claim rate" in page


def test_corrections_cannot_lift_a_number_over_a_broken_must_have(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    from jfl_core.models import HardGateBreach

    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(
        engine,
        user_id,
        application_id,
        want_it_score=2,
        hard_gate_breaches=[HardGateBreach(gate="location", breach="On site five days a week.")],
    )
    for i in range(2):
        push_back(client, application_id, text=f"a different point {i}")
        apply_as(client, latest_pushback(engine, user_id), application_id, "preference")

    page = client.get(f"/applications/{application_id}").text
    assert "holds this down whatever your corrections say" in page


# -- tenancy -----------------------------------------------------------------


def test_another_accounts_pushback_cannot_be_applied(
    client: TestClient,
    settings: WebSettings,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    push_back(client, application_id)
    theirs = latest_pushback(engine, user_id)

    other = TestClient(
        create_app(settings, identity_provider=google), base_url="https://testserver"
    )
    with other:
        sign_in(other, google, subs)
        mine = add_application(other)
        finish_score(engine, _signed_in_user_id(other), mine)
        response = other.post(
            f"/pushbacks/{theirs}/apply",
            data={
                "csrf_token": _csrf(other, f"/applications/{mine}"),
                "classification": "preference",
                "new_information": "yes",
            },
            follow_redirects=False,
        )
        assert response.status_code == 404
    with engine.begin() as conn:
        row = conn.execute(select(pushbacks_table).where(pushbacks_table.c.id == theirs)).one()
    assert row.status == "awaiting_classification"
