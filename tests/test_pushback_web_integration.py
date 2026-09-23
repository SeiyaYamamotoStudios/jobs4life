"""The one-box pushback flow through the real routes, against a live Postgres.

Marked `integration`; needs a migrated database. Google is stubbed, same
pattern as `test_scoring_web_integration.py`. No Anthropic call anywhere: the
worker's real handler is run, but with the model call replaced by a fixed
reading, so the path from "read" to "applied" is the one production takes.

What this pins is the flow as a person walks it: type into one box, see one
card that says what we took it to mean and what the number did (before ->
after, or "stays at"), and undo it with "Not what I meant". Plus the guarantees
that are the whole point: the stored score is never rewritten, a claim that the
tool has underrated you moves nothing until a fact is confirmed, repetition
moves nothing, the drift sentence escalates past its threshold, and no internal
word reaches the screen.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import uuid
from collections.abc import Iterator
from html import unescape

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
from jfl_core.models import ConstraintVerdict, ObjectiveVerdict, Task
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_generate.errors import GenerateError
from jfl_generate.pushback import PushbackClassification
from jfl_web.app import create_app
from jfl_web.oauth import GoogleIdentity, OAuthError
from jfl_web.pushbacks import DRIFT_LOUD
from jfl_web.settings import WebSettings
from jfl_worker.handlers import pushback as pushback_handler
from jfl_worker.registry import PermanentTaskError, TaskContext
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


def push_back(client: TestClient, application_id: uuid.UUID, text: str = WORDS) -> Response:
    """The box: words, and nothing else."""
    return client.post(
        f"/applications/{application_id}/pushback",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "user_text": text,
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


def run_reader(
    engine: Engine,
    user_id: uuid.UUID,
    pushback_id: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str = "preference",
    direction: str = "up",
    new_information: bool = True,
    fail: str | None = None,
) -> None:
    """The worker's real handler, with the model call replaced by a fixed
    reading (or a fixed failure). Everything after the call -- set the reading,
    classify, apply -- is the production path against this database.
    """
    monkeypatch.setattr(pushback_handler, "load_api_key", lambda *a, **k: "sk-ant-test")

    def _read(*args: object, **kwargs: object) -> PushbackClassification:
        if fail is not None:
            raise GenerateError(fail)
        return PushbackClassification(
            kind=kind,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            new_information=new_information,
            note="",
        )

    monkeypatch.setattr(pushback_handler, "call_classify_pushback", _read)
    now = dt.datetime.now(dt.UTC)
    task = Task(
        id=uuid.uuid4(),
        user_id=user_id,
        kind="classify_pushback",
        payload={"pushback_id": str(pushback_id)},
        status="running",
        attempts=1,
        max_attempts=5,
        scheduled_at=now,
        created_at=now,
        updated_at=now,
    )
    handler = pushback_handler.build_classify_pushback(master_key=MasterKey.generate())
    if fail is None:
        handler(TaskContext(task=task, engine=engine, now=now))
    else:
        with pytest.raises((PermanentTaskError, GenerateError)):
            handler(TaskContext(task=task, engine=engine, now=now))


def say(
    client: TestClient,
    engine: Engine,
    user_id: uuid.UUID,
    application_id: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
    text: str = WORDS,
    **reading: object,
) -> uuid.UUID:
    """Type into the box and let the worker read it."""
    assert push_back(client, application_id, text).status_code == 303
    pushback_id = latest_pushback(engine, user_id)
    run_reader(engine, user_id, pushback_id, monkeypatch, **reading)  # type: ignore[arg-type]
    return pushback_id


def pick(
    client: TestClient, pushback_id: uuid.UUID, application_id: uuid.UUID, reading: str
) -> Response:
    return client.post(
        f"/pushbacks/{pushback_id}/reading",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "reading": reading,
        },
        follow_redirects=False,
    )


def row(engine: Engine, pushback_id: uuid.UUID) -> object:
    with engine.begin() as conn:
        return conn.execute(
            select(pushbacks_table).where(pushbacks_table.c.id == pushback_id)
        ).one()


def score_row(engine: Engine, score_id: uuid.UUID) -> dict[str, object]:
    with engine.begin() as conn:
        return (
            conn.execute(select(scores_table).where(scores_table.c.id == score_id)).one()._asdict()
        )


def panel(client: TestClient, application_id: uuid.UUID) -> str:
    """The pushback panel as the polled fragment renders it."""
    response = client.get(f"/applications/{application_id}/pushbacks")
    assert response.status_code == 200
    return response.text


def text_of(html: str) -> str:
    """What a person reads: tags stripped, whitespace folded."""
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", html)).split())


# -- the box -------------------------------------------------------------------


def test_the_box_is_one_textarea_and_one_button_and_nothing_else(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    fragment = panel(client, application_id)
    form = re.search(r'<form[^>]*class="pushback-box".*?</form>', fragment, re.S)
    assert form is not None
    assert "Disagree with this? Tell us why." in form.group(0)
    assert form.group(0).count("<textarea") == 1
    assert form.group(0).count("<button") == 1
    for gone in ("<select", 'type="radio"', 'type="checkbox"', "<fieldset"):
        assert gone not in form.group(0)
    # Nothing else until it is used: no card, no history, no drift sentence,
    # no list of limits.
    assert "pushback-card" not in fragment
    assert "drift-line" not in fragment
    assert "pushback-limits" not in fragment


def test_submitting_records_the_words_verbatim_and_shows_it_being_read(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    assert push_back(client, application_id).status_code == 303
    pushback_id = latest_pushback(engine, user_id)
    stored = row(engine, pushback_id)
    assert stored.user_text == WORDS  # type: ignore[attr-defined]
    assert stored.status == "awaiting_classification"  # type: ignore[attr-defined]
    assert stored.applied_delta is None  # type: ignore[attr-defined]

    fragment = panel(client, application_id)
    assert "Reading what you wrote" in fragment
    assert 'hx-trigger="every 3s"' in fragment

    with engine.begin() as conn:
        kinds = list(
            conn.execute(
                select(tasks_table.c.kind).where(tasks_table.c.user_id == user_id)
            ).scalars()
        )
    assert kinds.count("classify_pushback") == 1


def test_the_box_needs_a_csrf_token(
    client: TestClient, google: StubGoogle, subs: list[str], engine: Engine
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    response = client.post(
        f"/applications/{application_id}/pushback",
        data={"user_text": WORDS},
        follow_redirects=False,
    )
    assert response.status_code == 403


# -- the card --------------------------------------------------------------------


def test_a_preference_shows_the_plain_reading_and_the_number_before_and_after(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)
    before = score_row(engine, score_id)

    say(client, engine, user_id, application_id, monkeypatch, "I would love a job like this")

    page = text_of(client.get(f"/applications/{application_id}").text)
    assert "You'd take roles like this more readily than we scored." in page
    assert "Do I want this: 5 → 6" in page
    assert "This counts for every job you score, not just this one." in page
    # The stored run is never rewritten; the big number says it was moved.
    assert "moved by your corrections" in page
    assert score_row(engine, score_id) == before


def test_a_second_correction_moves_it_less_and_says_why(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    say(client, engine, user_id, application_id, monkeypatch, "I would love a job like this")
    say(client, engine, user_id, application_id, monkeypatch, "And it is fully remote, too")

    card = text_of(panel(client, application_id))
    # 1 * 5/(5+1) = 0.83 of a point: 6 -> 6.8, which the big number rounds to 7.
    assert "Do I want this: 6 → 6.8 (shown as 7)" in card
    assert "One correction moves it a little; repeated ones move it less" in card


def test_saying_the_same_thing_again_moves_nothing_and_says_so(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    say(client, engine, user_id, application_id, monkeypatch, "I really want this kind of role")
    second = say(
        client, engine, user_id, application_id, monkeypatch, "I really want this kind of role"
    )

    assert float(row(engine, second).applied_delta) == 0.0  # type: ignore[attr-defined]
    card = text_of(panel(client, application_id))
    assert "Do I want this: stays at 6" in card
    assert "saying it again doesn't move it" in card


def test_at_the_limit_it_stops_and_offers_a_comparison(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    for i in range(4):
        say(client, engine, user_id, application_id, monkeypatch, f"a different point {i}")

    card = text_of(panel(client, application_id))
    assert "stays at 7" in card
    assert "already moved this as far as they can" in card
    assert "pick one to compare it with" in card


def test_a_stronger_fit_claim_moves_nothing_and_links_to_the_fact_that_would(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)

    pushback_id = say(
        client, engine, user_id, application_id, monkeypatch, kind="capability", direction="up"
    )

    fragment = panel(client, application_id)
    card = text_of(fragment)
    assert "You think you're a stronger fit than we scored." in card
    assert "Could I get this: stays at 4" in card
    assert "Your score won't move on your word alone" in card
    assert f'href="/pushbacks/{pushback_id}/evidence">answer this</a>' in fragment
    stored = row(engine, pushback_id)
    assert float(stored.applied_delta) == 0.0  # type: ignore[attr-defined]
    assert stored.axis == "get"  # type: ignore[attr-defined]
    assert score_row(engine, score_id)["could_get_score"] == 4

    form = client.get(f"/pushbacks/{pushback_id}/evidence")
    assert form.status_code == 200
    assert "What's the fact behind it?" in text_of(form.text)

    answer = "I owned the platform at Acme: nine engineers, the on-call rota, a 400k budget."
    response = client.post(
        f"/pushbacks/{pushback_id}/evidence",
        data={"csrf_token": _csrf(client, f"/pushbacks/{pushback_id}/evidence"), "answer": answer},
        follow_redirects=False,
    )
    assert response.status_code == 303
    stored = row(engine, pushback_id)
    with engine.begin() as conn:
        span = conn.execute(
            select(spans_table.c.text).where(
                spans_table.c.id == stored.resulting_span_id  # type: ignore[attr-defined]
            )
        ).scalar_one()
    # Verbatim, and the number STILL has not moved: it moves when the job is
    # scored again against the facts that now include this one.
    assert span == answer
    assert score_row(engine, score_id)["could_get_score"] == 4
    assert "You've confirmed the fact behind it." in text_of(panel(client, application_id))


def test_an_overrated_fit_moves_down_in_full(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    say(
        client,
        engine,
        user_id,
        application_id,
        monkeypatch,
        "Honestly I have never run anything that size",
        kind="capability",
        direction="down",
    )
    card = text_of(panel(client, application_id))
    assert "You think we've overrated your fit." in card
    assert "Could I get this: 4 → 3" in card
    assert "Taken as you said it, in full" in card


def test_an_objection_about_the_ad_changes_nothing_about_you_and_says_so_honestly(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    say(
        client,
        engine,
        user_id,
        application_id,
        monkeypatch,
        "The ad says remote in the second paragraph",
        kind="factual",
        direction="up",
    )
    fragment = panel(client, application_id)
    card = text_of(fragment)
    assert "You think we've misread the ad." in card
    assert "Nothing about you changed" in card
    # It does not claim to have re-read the ad with the correction: nothing
    # feeds a correction into scoring yet, so the card says that instead.
    assert "We can't yet re-read the ad with your correction in mind" in card
    assert f'action="/applications/{application_id}/score"' in fragment


# -- "Not what I meant" ------------------------------------------------------------


def test_not_what_i_meant_undoes_and_reapplies_under_the_chosen_reading(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The misreading guard, end to end.

    The words are read as a preference and move "do I want this" up. The
    person says they meant they are a stronger fit. The first correction stops
    counting, and the new reading moves nothing -- the asymmetric rule.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    score_id = finish_score(engine, user_id, application_id)
    first = say(client, engine, user_id, application_id, monkeypatch)
    assert "Do I want this: 5 → 6" in text_of(panel(client, application_id))
    assert "Not what I meant" in panel(client, application_id)

    assert pick(client, first, application_id, "fit_more").status_code == 303

    assert row(engine, first).withdrawn_at is not None  # type: ignore[attr-defined]
    second = latest_pushback(engine, user_id)
    assert second != first
    replaced = row(engine, second)
    assert replaced.user_text == WORDS  # type: ignore[attr-defined]
    assert replaced.classification == "capability"  # type: ignore[attr-defined]
    assert replaced.asserted_direction == "up"  # type: ignore[attr-defined]
    assert replaced.classification_source == "user"  # type: ignore[attr-defined]
    assert float(replaced.applied_delta) == 0.0  # type: ignore[attr-defined]

    page = text_of(client.get(f"/applications/{application_id}").text)
    assert "You think you're a stronger fit than we scored." in page
    assert "Could I get this: stays at 4" in page
    # The withdrawn preference no longer lifts the number.
    assert "moved by your corrections" not in page
    assert score_row(engine, score_id)["want_it_score"] == 5


def test_just_undo_it_takes_it_back_and_puts_nothing_in_its_place(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    first = say(client, engine, user_id, application_id, monkeypatch)

    assert pick(client, first, application_id, "undo").status_code == 303
    assert latest_pushback(engine, user_id) == first
    page = text_of(client.get(f"/applications/{application_id}").text)
    assert "Undone. That correction no longer counts for anything." in page
    assert "moved by your corrections" not in page
    # And it does not count on the drift sentence either.
    assert "You've pushed back" not in page


def test_when_the_reading_fails_the_person_picks_and_it_applies_to_the_same_row(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    pushback_id = say(
        client,
        engine,
        user_id,
        application_id,
        monkeypatch,
        fail="authentication_error: invalid x-api-key",
    )
    fragment = panel(client, application_id)
    assert "We couldn't work out what you meant" in text_of(fragment)
    assert "hx-trigger" not in fragment
    assert "I'd want this less than you scored" in text_of(fragment)

    assert pick(client, pushback_id, application_id, "want_less").status_code == 303
    assert latest_pushback(engine, user_id) == pushback_id
    applied = row(engine, pushback_id)
    assert applied.status == "applied"  # type: ignore[attr-defined]
    assert applied.asserted_direction == "down"  # type: ignore[attr-defined]
    assert "Do I want this: 5 → 4" in text_of(panel(client, application_id))


def test_an_unknown_reading_is_refused(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    first = say(client, engine, user_id, application_id, monkeypatch)
    assert pick(client, first, application_id, "capability_up").status_code == 400
    assert row(engine, first).withdrawn_at is None  # type: ignore[attr-defined]


def test_choosing_a_reading_needs_a_csrf_token(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    first = say(client, engine, user_id, application_id, monkeypatch)
    response = client.post(
        f"/pushbacks/{first}/reading", data={"reading": "undo"}, follow_redirects=False
    )
    assert response.status_code == 403
    assert row(engine, first).withdrawn_at is None  # type: ignore[attr-defined]
    response = client.post(
        f"/pushbacks/{first}/evidence", data={"answer": "words"}, follow_redirects=False
    )
    assert response.status_code == 403


# -- the drift sentence --------------------------------------------------------


def test_the_drift_sentence_is_quiet_at_first(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    say(client, engine, user_id, application_id, monkeypatch, kind="capability", direction="up")

    fragment = panel(client, application_id)
    assert "You've pushed back once on this profile, upward." in text_of(fragment)
    assert 'class="drift-line"' in fragment
    assert DRIFT_LOUD not in fragment
    assert '<a href="/pushbacks">See everything you&#39;ve said</a>' in fragment


def test_the_drift_sentence_escalates_past_the_threshold(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three upward pushes, all of them upward, and none of them moving a
    number (so it is the share that trips it, not the net movement).
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    for i in range(2):
        say(
            client,
            engine,
            user_id,
            application_id,
            monkeypatch,
            f"I have done more than this, point {i}",
            kind="capability",
            direction="up",
        )
    assert DRIFT_LOUD not in panel(client, application_id)

    say(
        client,
        engine,
        user_id,
        application_id,
        monkeypatch,
        "And more again",
        kind="capability",
        direction="up",
    )
    fragment = panel(client, application_id)
    assert "You've pushed back 3 times on this profile, all upward." in text_of(fragment)
    assert 'class="drift-line drift-meter"' in fragment
    assert DRIFT_LOUD in text_of(fragment)
    # The same sentence heads the log page.
    assert DRIFT_LOUD in text_of(client.get("/pushbacks").text)


def test_a_two_way_history_stays_quiet(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    for i, direction in enumerate(("up", "up", "down", "up", "down")):
        say(
            client,
            engine,
            user_id,
            application_id,
            monkeypatch,
            f"point {i}",
            kind="capability",
            direction=direction,
        )
    fragment = panel(client, application_id)
    assert "5 times on this profile, mostly upward" in text_of(fragment)
    assert DRIFT_LOUD not in fragment


# -- the log page ------------------------------------------------------------------


def test_the_log_is_a_plain_history_in_the_users_own_words(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    say(client, engine, user_id, application_id, monkeypatch, "I would love a job like this")
    say(
        client,
        engine,
        user_id,
        application_id,
        monkeypatch,
        "I ran that platform",
        kind="capability",
        direction="up",
    )

    page = text_of(client.get("/pushbacks").text)
    assert "I would love a job like this" in page
    assert "Moved “Do I want this” up by 1 point, for every job." in page
    assert "I ran that platform" in page
    assert "Changed nothing yet: waiting on the fact behind it." in page


def test_every_page_links_to_the_corrections_log(
    client: TestClient, google: StubGoogle, subs: list[str]
) -> None:
    sign_in(client, google, subs)
    assert '<a href="/pushbacks">Corrections</a>' in client.get("/applications").text


# -- no internal words on screen ----------------------------------------------------

# The words the owner named, plus the rest of the machinery's vocabulary. Checked
# against what a person reads -- tags stripped -- so class names and form field
# names may keep their own words.
INTERNAL = (
    "δ",
    "shrinkage",
    "shrunk",
    "dimension",
    "capability",
    "preference",
    "classif",
    "observation",
    "displacement",
    "claim gate",
    "golden set",
    "corpus",
    "want_overall",
    "could_get_overall",
    "pending_evidence",
)


def _assert_plain(html: str, where: str) -> None:
    visible = text_of(html).lower()
    found = [word for word in INTERNAL if word in visible]
    assert not found, f"{where} shows internal words {found}"


def _pushback_markup(html: str) -> str:
    match = re.search(r'<section id="pushbacks".*?</section>', html, re.S)
    assert match is not None
    return match.group(0)


def test_no_internal_word_appears_in_any_pushback_markup(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every card state, the history, the limits list, the log and the
    evidence page -- each rendered for real and swept.
    """
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)

    push_back(client, application_id, "one")
    _assert_plain(panel(client, application_id), "the reading card")

    readings = [
        {"kind": "preference", "direction": "up"},
        {"kind": "preference", "direction": "down"},
        {"kind": "preference", "direction": "down", "new_information": False},
        {"kind": "capability", "direction": "up"},
        {"kind": "capability", "direction": "down"},
        {"kind": "factual", "direction": "up"},
        {"fail": "authentication_error: nope"},
    ]
    last = None
    for i, reading in enumerate(readings):
        last = say(client, engine, user_id, application_id, monkeypatch, f"words {i}", **reading)
        _assert_plain(panel(client, application_id), f"the card for {reading}")
        _assert_plain(
            _pushback_markup(client.get(f"/applications/{application_id}").text),
            f"the page panel for {reading}",
        )

    assert last is not None
    pick(client, last, application_id, "want_more")
    pick(client, latest_pushback(engine, user_id), application_id, "undo")
    _assert_plain(panel(client, application_id), "the undone card")

    _assert_plain(client.get("/pushbacks").text, "the log page")
    with engine.begin() as conn:
        waiting = conn.execute(
            select(pushbacks_table.c.id).where(
                pushbacks_table.c.user_id == user_id,
                pushbacks_table.c.disposition == "pending_evidence",
            )
        ).scalar_one()
    _assert_plain(client.get(f"/pushbacks/{waiting}/evidence").text, "the evidence page")


# -- what corrections never reach ------------------------------------------------


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
    assert "Your override" not in client.get(f"/applications/{other_id}").text
    assert score_row(engine, score_id)["could_get_score"] == 4
    assert score_row(engine, other_score)["could_get_score"] == 4
    # Not a correction: no drift sentence at all.
    assert "You've pushed back" not in page


def test_an_application_already_sent_keeps_the_score_it_was_sent_under(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    client.post(
        f"/applications/{application_id}/status",
        data={
            "csrf_token": _csrf(client, f"/applications/{application_id}"),
            "to_status": "applied",
        },
        follow_redirects=False,
    )
    say(client, engine, user_id, application_id, monkeypatch, "I would love a job like this")

    page = text_of(client.get(f"/applications/{application_id}").text)
    assert "keeps the score it was sent under" in page
    assert "Do I want this: stays at 5" in page
    assert "moved by your corrections" not in page


def test_the_card_says_what_disagreeing_can_never_change(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    say(client, engine, user_id, application_id, monkeypatch)
    page = client.get(f"/applications/{application_id}").text
    assert "What disagreeing can never change" in page
    assert "what we say about any sentence in your CV or cover letter" in page


def test_corrections_cannot_lift_a_number_over_a_broken_must_have(
    client: TestClient,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
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
        say(client, engine, user_id, application_id, monkeypatch, f"a different point {i}")

    page = text_of(client.get(f"/applications/{application_id}").text)
    assert "holds this down whatever your corrections say" in page
    assert "A must-have this ad breaks holds it here" in page


# -- tenancy -------------------------------------------------------------------------


def test_another_accounts_correction_is_a_404_everywhere(
    client: TestClient,
    settings: WebSettings,
    google: StubGoogle,
    subs: list[str],
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = sign_in(client, google, subs)
    application_id = add_application(client)
    finish_score(engine, user_id, application_id)
    theirs = say(client, engine, user_id, application_id, monkeypatch, kind="capability")

    other = TestClient(
        create_app(settings, identity_provider=google), base_url="https://testserver"
    )
    with other:
        sign_in(other, google, subs)
        mine = add_application(other)
        finish_score(engine, _signed_in_user_id(other), mine)
        token = _csrf(other, f"/applications/{mine}")
        for response in (
            other.post(
                f"/pushbacks/{theirs}/reading",
                data={"csrf_token": token, "reading": "undo"},
                follow_redirects=False,
            ),
            other.post(
                f"/pushbacks/{theirs}/reading",
                data={"csrf_token": token, "reading": "want_more"},
                follow_redirects=False,
            ),
            other.get(f"/pushbacks/{theirs}/evidence"),
            other.post(
                f"/pushbacks/{theirs}/evidence",
                data={"csrf_token": token, "answer": "not mine to give"},
                follow_redirects=False,
            ),
            other.get(f"/applications/{application_id}/pushbacks"),
            other.post(
                f"/applications/{application_id}/pushback",
                data={"csrf_token": token, "user_text": "not mine"},
                follow_redirects=False,
            ),
        ):
            assert response.status_code == 404
        assert WORDS not in other.get("/pushbacks").text
    stored = row(engine, theirs)
    assert stored.withdrawn_at is None  # type: ignore[attr-defined]
    assert stored.resulting_span_id is None  # type: ignore[attr-defined]
