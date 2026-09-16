"""Profile setup storage against real Postgres -- PLAN.md slice B3a.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures, as in `test_job_filters_integration.py`: every test leaves the
database as it found it. No network and no model call.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import users
from jfl_core.storage.profile import PostgresProfileRepository, UnknownQuestionKeyError
from sqlalchemy import create_engine, insert
from sqlalchemy.engine import Connection, Engine

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


@pytest.fixture
def alice(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


@pytest.fixture
def bob(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


# -- all-blank save is valid -------------------------------------------------


def test_saving_nothing_leaves_everything_not_stated(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresProfileRepository(conn, alice)
    assert repo.get_current_answers() == {}
    assert repo.list_objectives() == []
    assert repo.list_ruled_out() == []


def test_saving_blank_answers_writes_nothing(conn: Connection, alice: uuid.UUID) -> None:
    """An all-blank section save must not manufacture rows for questions the
    user has never touched -- PLAN.md's "a skipped question is never
    defaulted".
    """
    repo = PostgresProfileRepository(conn, alice)
    repo.save_answers(
        {
            "location_commute": ("", None),
            "notice_period": ("", None),
        }
    )
    assert repo.get_current_answers() == {}


# -- verbatim round trip, including unicode and newlines ---------------------


def test_answers_round_trip_verbatim_including_unicode_and_newlines(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    text = "London — E14.\nWill travel ~1 day/week; café meetings fine. 你好"
    saved = repo.save_answer("location_commute", answer_text=text)
    assert saved is not None
    assert saved.answer_text == text

    current = repo.get_current_answers()
    assert current["location_commute"].answer_text == text


def test_structured_value_round_trips_alongside_free_text(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    saved = repo.save_answer(
        "levels", answer_text="EM or above only", structured={"selected": ["em", "above_em"]}
    )
    assert saved is not None
    assert saved.structured == {"selected": ["em", "above_em"]}

    current = repo.get_current_answers()["levels"]
    assert current.answer_text == "EM or above only"
    assert current.structured == {"selected": ["em", "above_em"]}


# -- structured inputs are optional -------------------------------------------


def test_structured_is_optional_free_text_alone_is_a_complete_answer(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    saved = repo.save_answer(
        "contract_types", answer_text="Permanent preferred, will consider outside IR35"
    )
    assert saved is not None
    assert saved.structured is None
    assert repo.get_current_answers()["contract_types"].structured is None


def test_unknown_question_key_is_rejected_defensively(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresProfileRepository(conn, alice)
    with pytest.raises(UnknownQuestionKeyError):
        repo.save_answer("not_a_real_question", answer_text="x")


# -- a second answer supersedes but history retains the first ----------------


def test_a_second_answer_supersedes_but_history_keeps_the_first(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    first = repo.save_answer("trajectory", answer_text="Staff engineer track")
    assert first is not None
    second = repo.save_answer("trajectory", answer_text="Back to EM, ideally director in 2 years")
    assert second is not None
    assert second.id != first.id

    current = repo.get_current_answers()["trajectory"]
    assert current.answer_text == "Back to EM, ideally director in 2 years"

    history = repo.history("trajectory")
    assert [h.answer_text for h in history] == [
        "Staff engineer track",
        "Back to EM, ideally director in 2 years",
    ]
    assert history[0].id == first.id
    assert history[1].id == second.id


def test_resaving_the_same_answer_is_a_no_op_and_writes_no_history(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    repo.save_answer("warning_signs", answer_text="Vague comp band, urgency language")
    result = repo.save_answer("warning_signs", answer_text="Vague comp band, urgency language")
    assert result is None
    assert len(repo.history("warning_signs")) == 1


def test_clearing_a_previous_answer_is_itself_a_recorded_change(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    repo.save_answer("categorical_no", answer_text="No pure IC roles")
    cleared = repo.save_answer("categorical_no", answer_text="")
    assert cleared is not None
    assert cleared.answer_text == ""
    assert repo.get_current_answers()["categorical_no"].answer_text == ""
    assert len(repo.history("categorical_no")) == 2


# -- tenancy: user A cannot read or write user B's profile --------------------


def test_a_user_cannot_read_another_users_answers(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    PostgresProfileRepository(conn, alice).save_answer(
        "employer_deal_breakers", answer_text="No recent layoffs"
    )
    bobs = PostgresProfileRepository(conn, bob)
    assert bobs.get_current_answers() == {}
    assert bobs.history("employer_deal_breakers") == []


def test_a_user_cannot_read_another_users_objectives_or_ruled_out(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    alices = PostgresProfileRepository(conn, alice)
    alices.save_objective(1, objective_text="More scope", evidence_text="Bigger org chart")
    alices.add_ruled_out("Not considering Acme again")

    bobs = PostgresProfileRepository(conn, bob)
    assert bobs.list_objectives() == []
    assert bobs.list_ruled_out() == []


def test_a_user_cannot_reopen_another_users_ruled_out_entry(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    alices = PostgresProfileRepository(conn, alice)
    entry = alices.add_ruled_out("Not considering Acme again")
    bobs = PostgresProfileRepository(conn, bob)
    assert bobs.mark_reopened(entry.id) is None
    # Untouched for its actual owner.
    assert alices.list_ruled_out()[0].reopened_at is None


# -- objectives: separate records, never combined -----------------------------


def test_objectives_are_separate_records_up_to_four(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresProfileRepository(conn, alice)
    repo.save_objective(1, objective_text="More comp", evidence_text="Offer at or above floor")
    repo.save_objective(2, objective_text="More scope", evidence_text="Org of 40+, budget owner")

    objectives = repo.list_objectives()
    assert [o.ordinal for o in objectives] == [1, 2]
    assert objectives[0].objective_text == "More comp"
    assert objectives[1].evidence_text == "Org of 40+, budget owner"


def test_saving_an_objective_twice_updates_it_in_place(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresProfileRepository(conn, alice)
    repo.save_objective(1, objective_text="More comp", evidence_text="v1")
    repo.save_objective(1, objective_text="More comp", evidence_text="v2")
    objectives = repo.list_objectives()
    assert len(objectives) == 1
    assert objectives[0].evidence_text == "v2"


def test_clearing_both_fields_removes_the_objective_slot(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    repo.save_objective(1, objective_text="More comp", evidence_text="v1")
    result = repo.save_objective(1, objective_text="  ", evidence_text="")
    assert result is None
    assert repo.list_objectives() == []


# -- ruled-out: dated, kept, reopen never deletes ------------------------------


def test_ruled_out_entries_keep_their_dates_oldest_first(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    first = repo.add_ruled_out("Not considering Acme again")
    second = repo.add_ruled_out("No more agencies from the 2024 batch")

    entries = repo.list_ruled_out()
    assert [e.id for e in entries] == [first.id, second.id]
    assert entries[0].recorded_at <= entries[1].recorded_at
    assert entries[0].reopened_at is None


def test_marking_reopened_sets_the_date_and_never_deletes(
    conn: Connection, alice: uuid.UUID
) -> None:
    repo = PostgresProfileRepository(conn, alice)
    entry = repo.add_ruled_out("Not considering Acme again")
    reopened = repo.mark_reopened(entry.id)
    assert reopened is not None
    assert reopened.id == entry.id
    assert reopened.reopened_at is not None
    assert reopened.decision_text == "Not considering Acme again"

    # Still on the record, not deleted.
    entries = repo.list_ruled_out()
    assert len(entries) == 1
    assert entries[0].reopened_at is not None


def test_marking_an_unknown_entry_reopened_returns_none(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresProfileRepository(conn, alice)
    assert repo.mark_reopened(uuid.uuid4()) is None
