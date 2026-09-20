"""Application-score storage against real Postgres -- PLAN.md slice B4.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures, as in `test_profile_repo_integration.py`: every test leaves the
database as it found it. No network and no model call.

The properties worth pinning here are the ones that are invisible until they
go wrong: a re-score must keep the earlier run rather than overwrite it, both
axes must survive a round trip unchanged, the CHECK constraints must actually
refuse a score outside 1-10, and another user's row must be invisible even to
a caller holding its id.
"""

from __future__ import annotations

import decimal
import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import application_scores, applications, users
from jfl_core.models import HardGateBreach, NotStated, ObjectiveVerdict, ScoreLever
from jfl_core.storage.scores import PostgresScoreRepository
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine() -> Engine:
    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()


def _make_user(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


def _make_application(conn: Connection, user_id: uuid.UUID) -> uuid.UUID:
    application_id = uuid.uuid4()
    conn.execute(
        insert(applications).values(id=application_id, user_id=user_id, title="Engineering Manager")
    )
    return application_id


@pytest.fixture
def alice(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


@pytest.fixture
def bob(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


def _finish(repo: PostgresScoreRepository, score_id: uuid.UUID, **kw: object) -> None:
    defaults: dict[str, object] = {
        "could_get_score": 6,
        "could_get_assessment": "Three roles evidence the platform work.",
        "want_it_score": 3,
        "want_it_assessment": "The commute breaks what you said you would travel.",
        "objective_verdicts": [
            ObjectiveVerdict(ordinal=1, objective="Back to hands-on", verdict="Unlikely here.")
        ],
        "hard_gate_breaches": [HardGateBreach(gate="location", breach="On site five days a week.")],
        "levers": [
            ScoreLever(
                fact_text="Ran a team of 12",
                role_label="Northwind",
                would_move_to=8,
                note="Covers the headcount requirement.",
            )
        ],
        "not_stated": [NotStated(question_key="comp_floor", wording="Lowest total package?")],
        "model": "claude-opus-5",
        "cost_usd": decimal.Decimal("0.1234"),
        "trace_id": uuid.uuid4(),
    }
    defaults.update(kw)
    repo.mark_done(score_id, **defaults)  # type: ignore[arg-type]


# -- the round trip -----------------------------------------------------------


def test_both_axes_and_everything_around_them_survive_a_round_trip(
    conn: Connection, alice: uuid.UUID
) -> None:
    application_id = _make_application(conn, alice)
    repo = PostgresScoreRepository(conn, alice)

    pending = repo.create_pending(application_id)
    assert pending.status == "pending"
    assert pending.could_get_score is None and pending.want_it_score is None

    _finish(repo, pending.id)
    stored = repo.get(pending.id)

    assert stored is not None
    assert stored.status == "done"
    assert stored.error_code is None
    # Two axes, read back exactly as written, and not reconciled with each other.
    assert stored.could_get_score == 6
    assert stored.want_it_score == 3
    assert stored.could_get_assessment.startswith("Three roles")
    assert stored.want_it_assessment.startswith("The commute")
    assert [v.ordinal for v in stored.objective_verdicts] == [1]
    assert stored.hard_gate_breaches[0].gate == "location"
    assert stored.levers[0].fact_text == "Ran a team of 12"
    assert stored.levers[0].would_move_to == 8
    assert stored.not_stated[0].question_key == "comp_floor"
    assert stored.model == "claude-opus-5"
    assert stored.cost_usd == decimal.Decimal("0.123400")
    assert stored.trace_id is not None


def test_a_failure_stores_a_code_and_no_scores(conn: Connection, alice: uuid.UUID) -> None:
    application_id = _make_application(conn, alice)
    repo = PostgresScoreRepository(conn, alice)
    row = repo.create_pending(application_id)

    repo.mark_failed(row.id, "no_api_key")
    stored = repo.get(row.id)

    assert stored is not None
    assert stored.status == "failed"
    assert stored.error_code == "no_api_key"
    assert stored.could_get_score is None


# -- append-only --------------------------------------------------------------


def test_rescoring_keeps_the_earlier_run_and_shows_the_latest(
    conn: Connection, alice: uuid.UUID
) -> None:
    application_id = _make_application(conn, alice)
    repo = PostgresScoreRepository(conn, alice)

    first = repo.create_pending(application_id)
    _finish(repo, first.id, could_get_score=4, want_it_score=9)
    second = repo.create_pending(application_id)
    _finish(repo, second.id, could_get_score=7, want_it_score=2)

    latest = repo.latest(application_id)
    assert latest is not None and latest.id == second.id
    assert (latest.could_get_score, latest.want_it_score) == (7, 2)

    history = repo.history(application_id)
    assert [h.id for h in history] == [first.id, second.id]
    assert (history[0].could_get_score, history[0].want_it_score) == (4, 9)


def test_latest_is_the_pending_run_while_one_is_in_flight(
    conn: Connection, alice: uuid.UUID
) -> None:
    """Which is what makes the panel say "scoring" with no client-side state."""
    application_id = _make_application(conn, alice)
    repo = PostgresScoreRepository(conn, alice)
    done = repo.create_pending(application_id)
    _finish(repo, done.id)
    pending = repo.create_pending(application_id)

    latest = repo.latest(application_id)
    assert latest is not None and latest.id == pending.id and latest.status == "pending"


def test_no_score_yet_reads_as_none_rather_than_a_zero(conn: Connection, alice: uuid.UUID) -> None:
    application_id = _make_application(conn, alice)
    assert PostgresScoreRepository(conn, alice).latest(application_id) is None


# -- the constraints ----------------------------------------------------------


@pytest.mark.parametrize(("column", "value"), [("could_get_score", 0), ("want_it_score", 11)])
def test_postgres_refuses_a_score_outside_one_to_ten(
    conn: Connection, alice: uuid.UUID, column: str, value: int
) -> None:
    application_id = _make_application(conn, alice)
    row = PostgresScoreRepository(conn, alice).create_pending(application_id)
    with pytest.raises(IntegrityError):
        conn.execute(
            update(application_scores)
            .where(application_scores.c.id == row.id)
            .values(**{column: value})
        )


# -- tenancy ------------------------------------------------------------------


def test_another_users_score_is_invisible_even_holding_its_id(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    application_id = _make_application(conn, alice)
    alice_row = PostgresScoreRepository(conn, alice).create_pending(application_id)

    bobs = PostgresScoreRepository(conn, bob)
    assert bobs.get(alice_row.id) is None
    assert bobs.latest(application_id) is None
    assert bobs.history(application_id) == []


def test_another_user_cannot_finish_or_fail_someone_elses_run(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    application_id = _make_application(conn, alice)
    row = PostgresScoreRepository(conn, alice).create_pending(application_id)

    bobs = PostgresScoreRepository(conn, bob)
    _finish(bobs, row.id)
    bobs.mark_failed(row.id, "model_error")

    still = PostgresScoreRepository(conn, alice).get(row.id)
    assert still is not None and still.status == "pending"


def test_a_row_is_written_under_the_constructing_users_id(
    conn: Connection, alice: uuid.UUID
) -> None:
    """There is no per-call user argument to get wrong -- the id comes from
    `__init__` and nowhere else. This checks what actually landed.
    """
    application_id = _make_application(conn, alice)
    row = PostgresScoreRepository(conn, alice).create_pending(application_id)
    stored_user = conn.execute(
        select(application_scores.c.user_id).where(application_scores.c.id == row.id)
    ).scalar_one()
    assert stored_user == alice
