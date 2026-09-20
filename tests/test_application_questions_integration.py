"""`PostgresApplicationQuestionRepository` against real Postgres -- NEXT.md's
task 4. Needs `docker compose up -d` and `alembic upgrade head`.

Transaction-rollback fixtures, as in `test_applications_repo_integration.py`.
No Anthropic API call anywhere -- there is nothing here that could make one.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import users
from jfl_core.storage.application_questions import PostgresApplicationQuestionRepository
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
        tx.rollback()  # every test leaves the database as it found it


def _make_user(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


def _make_application(conn: Connection, user_id: uuid.UUID, title: str = "Some role") -> uuid.UUID:
    application_id = uuid.uuid4()
    conn.execute(insert(applications_table).values(id=application_id, user_id=user_id, title=title))
    return application_id


@pytest.fixture
def alice(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


@pytest.fixture
def bob(conn: Connection) -> uuid.UUID:
    return _make_user(conn)


@pytest.fixture
def alice_application(conn: Connection, alice: uuid.UUID) -> uuid.UUID:
    return _make_application(conn, alice)


@pytest.fixture
def repo(conn: Connection, alice: uuid.UUID) -> PostgresApplicationQuestionRepository:
    return PostgresApplicationQuestionRepository(conn, alice)


@pytest.fixture
def bob_repo(conn: Connection, bob: uuid.UUID) -> PostgresApplicationQuestionRepository:
    return PostgresApplicationQuestionRepository(conn, bob)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def test_add_and_get_question(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    question = repo.add_question(alice_application, "Why do you want to work here?")
    assert question.application_id == alice_application
    assert question.question_text == "Why do you want to work here?"

    fetched = repo.get_question(question.id)
    assert fetched is not None and fetched.id == question.id


def test_list_questions_is_oldest_first(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    first = repo.add_question(alice_application, "First question")
    second = repo.add_question(alice_application, "Second question")

    listed = repo.list_questions(alice_application)
    assert [q.id for q in listed] == [first.id, second.id]


def test_a_question_belonging_to_another_user_is_invisible(
    repo: PostgresApplicationQuestionRepository,
    bob_repo: PostgresApplicationQuestionRepository,
    alice_application: uuid.UUID,
) -> None:
    question = repo.add_question(alice_application, "Why do you want to work here?")

    assert bob_repo.get_question(question.id) is None
    assert bob_repo.list_questions(alice_application) == []


# --------------------------------------------------------------------------
# Answer attempts -- append-only history
# --------------------------------------------------------------------------


def test_create_user_answer_is_pending_and_carries_the_typed_text(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    question = repo.add_question(alice_application, "Why this role?")
    answer = repo.create_user_answer(question.id, "Because I love the mission.")
    assert answer is not None
    assert answer.kind == "user"
    assert answer.status == "pending"
    assert answer.answer_text == "Because I love the mission."
    assert answer.gate_result is None
    assert answer.assessment is None


def test_create_draft_answer_starts_with_empty_text(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    question = repo.add_question(alice_application, "Why this role?")
    answer = repo.create_draft_answer(question.id)
    assert answer is not None
    assert answer.kind == "draft"
    assert answer.status == "pending"
    assert answer.answer_text == ""


def test_create_answer_for_an_unknown_question_writes_nothing(
    repo: PostgresApplicationQuestionRepository,
) -> None:
    assert repo.create_user_answer(uuid.uuid4(), "text") is None
    assert repo.create_draft_answer(uuid.uuid4()) is None


def test_pressing_check_twice_creates_two_versions_never_overwriting_the_first(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    """The same rule `profile_answers` follows, for the same reason: a tool
    whose whole claim is measuring distance from what someone actually said
    must never let that record be edited out from under them.
    """
    question = repo.add_question(alice_application, "Why this role?")
    first = repo.create_user_answer(question.id, "First attempt.")
    assert first is not None
    second = repo.create_user_answer(question.id, "Second, better attempt.")
    assert second is not None
    assert first.id != second.id

    history = repo.list_answers(question.id)
    assert [a.answer_text for a in history] == ["First attempt.", "Second, better attempt."]

    latest = repo.latest_answer(question.id)
    assert latest is not None and latest.id == second.id
    assert latest.answer_text == "Second, better attempt."

    # The first version is still exactly what was typed -- nothing rewrote it.
    first_again = repo.get_answer(first.id)
    assert first_again is not None
    assert first_again.answer_text == "First attempt."


def test_mark_done_records_gate_result_and_assessment_without_touching_answer_text(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    question = repo.add_question(alice_application, "Why this role?")
    answer = repo.create_user_answer(question.id, "Because I love the mission.")
    assert answer is not None

    trace_id = uuid.uuid4()
    repo.mark_done(
        answer.id,
        answer_text=None,
        gate_result={"sentences": []},
        assessment={"assessment": "Covers it well.", "gaps": ""},
        model="claude-opus-5",
        trace_id=trace_id,
    )

    done = repo.get_answer(answer.id)
    assert done is not None
    assert done.status == "done"
    assert done.error_code is None
    assert done.answer_text == "Because I love the mission."
    assert done.gate_result == {"sentences": []}
    assert done.assessment == {"assessment": "Covers it well.", "gaps": ""}
    assert done.model == "claude-opus-5"
    assert done.trace_id == trace_id


def test_mark_done_for_a_draft_fills_in_the_generated_text(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    question = repo.add_question(alice_application, "Why this role?")
    answer = repo.create_draft_answer(question.id)
    assert answer is not None

    repo.mark_done(
        answer.id,
        answer_text="A drafted answer, grounded in the corpus.",
        gate_result={"sentences": []},
        assessment=None,
        model="claude-opus-5",
        trace_id=uuid.uuid4(),
    )

    done = repo.get_answer(answer.id)
    assert done is not None
    assert done.answer_text == "A drafted answer, grounded in the corpus."
    assert done.assessment is None


def test_mark_failed_sets_status_and_code(
    repo: PostgresApplicationQuestionRepository, alice_application: uuid.UUID
) -> None:
    question = repo.add_question(alice_application, "Why this role?")
    answer = repo.create_user_answer(question.id, "An answer.")
    assert answer is not None

    repo.mark_failed(answer.id, "no_api_key")

    failed = repo.get_answer(answer.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "no_api_key"


def test_an_answer_belonging_to_another_user_is_invisible(
    repo: PostgresApplicationQuestionRepository,
    bob_repo: PostgresApplicationQuestionRepository,
    alice_application: uuid.UUID,
) -> None:
    question = repo.add_question(alice_application, "Why this role?")
    answer = repo.create_user_answer(question.id, "An answer.")
    assert answer is not None

    assert bob_repo.get_answer(answer.id) is None
    assert bob_repo.list_answers(question.id) == []
    assert bob_repo.latest_answer(question.id) is None

    # And Bob cannot mark it done or failed either -- the WHERE clause on
    # `user_id` makes both a silent no-op against a row that isn't his.
    bob_repo.mark_failed(answer.id, "no_api_key")
    still_alices = repo.get_answer(answer.id)
    assert still_alices is not None and still_alices.status == "pending"
