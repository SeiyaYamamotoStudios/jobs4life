"""`PostgresTitleSuggestionRepository` against real Postgres -- slice C7a.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures, as in `test_job_filters_integration.py`: every test leaves the
database as it found it. No network and no model call anywhere in this file.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import users
from jfl_core.models import SuggestedTitle
from jfl_core.storage.title_suggestions import PostgresTitleSuggestionRepository
from jfl_intake.normalise import normalise
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


def test_create_pending_then_get_by_phrase_key(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresTitleSuggestionRepository(conn, alice)
    phrase = "Engineering Manager"
    key = normalise(phrase)

    created = repo.create_pending(phrase=phrase, phrase_key=key)
    assert created is not None
    assert created.phrase == phrase
    assert created.phrase_key == key
    assert created.status == "pending"
    assert created.suggestions == []
    assert created.dismissed_at is None

    fetched = repo.get_by_phrase_key(key)
    assert fetched is not None and fetched.id == created.id


def test_create_pending_is_a_no_op_for_an_existing_phrase_key(
    conn: Connection, alice: uuid.UUID
) -> None:
    """The unique constraint, exercised: the second call for the same
    normalised phrase must not mint a second row and must not enqueue a
    second call -- `ON CONFLICT DO NOTHING` returning None is what
    `jobs.save_filter` reads as "already handled".
    """
    repo = PostgresTitleSuggestionRepository(conn, alice)
    key = normalise("engineering manager")

    first = repo.create_pending(phrase="engineering manager", phrase_key=key)
    assert first is not None

    second = repo.create_pending(phrase="Engineering Manager", phrase_key=key)
    assert second is None

    # Still exactly one row, holding the first phrase's own casing.
    fetched = repo.get_by_phrase_key(key)
    assert fetched is not None
    assert fetched.id == first.id
    assert fetched.phrase == "engineering manager"


def test_mark_done_stores_suggestions_and_clears_error(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresTitleSuggestionRepository(conn, alice)
    row = repo.create_pending(
        phrase="engineering manager", phrase_key=normalise("engineering manager")
    )
    assert row is not None

    repo.mark_failed(row.id, "model_error")
    assert repo.get(row.id).status == "failed"  # type: ignore[union-attr]

    suggestions = [
        SuggestedTitle(title="Senior Engineering Manager", gloss="a step up"),
        SuggestedTitle(title="Engineering Lead", gloss=""),
    ]
    repo.mark_done(row.id, suggestions)

    fetched = repo.get(row.id)
    assert fetched is not None
    assert fetched.status == "done"
    assert fetched.error_code is None
    assert [s.title for s in fetched.suggestions] == [
        "Senior Engineering Manager",
        "Engineering Lead",
    ]
    assert fetched.suggestions[0].gloss == "a step up"


def test_mark_failed_sets_status_and_code(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresTitleSuggestionRepository(conn, alice)
    row = repo.create_pending(
        phrase="engineering manager", phrase_key=normalise("engineering manager")
    )
    assert row is not None

    repo.mark_failed(row.id, "no_api_key")

    fetched = repo.get(row.id)
    assert fetched is not None
    assert fetched.status == "failed"
    assert fetched.error_code == "no_api_key"


def test_dismiss_sets_dismissed_at_and_reports_success(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresTitleSuggestionRepository(conn, alice)
    row = repo.create_pending(
        phrase="engineering manager", phrase_key=normalise("engineering manager")
    )
    assert row is not None
    assert row.dismissed_at is None

    assert repo.dismiss(row.id) is True

    fetched = repo.get(row.id)
    assert fetched is not None and fetched.dismissed_at is not None


def test_dismiss_of_an_unknown_id_reports_failure(conn: Connection, alice: uuid.UUID) -> None:
    repo = PostgresTitleSuggestionRepository(conn, alice)
    assert repo.dismiss(uuid.uuid4()) is False


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_a_users_row_is_invisible_to_another_user(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    mine = PostgresTitleSuggestionRepository(conn, alice)
    row = mine.create_pending(
        phrase="engineering manager", phrase_key=normalise("engineering manager")
    )
    assert row is not None

    theirs = PostgresTitleSuggestionRepository(conn, bob)
    assert theirs.get(row.id) is None
    assert theirs.get_by_phrase_key(row.phrase_key) is None


def test_a_second_user_can_use_the_same_phrase_key(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    """The unique constraint is `(user_id, phrase_key)`, not `phrase_key` alone
    -- two users adding the same title phrase must not collide.
    """
    key = normalise("engineering manager")
    alice_repo = PostgresTitleSuggestionRepository(conn, alice)
    bob_repo = PostgresTitleSuggestionRepository(conn, bob)

    alice_row = alice_repo.create_pending(phrase="engineering manager", phrase_key=key)
    bob_row = bob_repo.create_pending(phrase="engineering manager", phrase_key=key)

    assert alice_row is not None
    assert bob_row is not None
    assert alice_row.id != bob_row.id


def test_a_user_cannot_dismiss_another_users_row(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    mine = PostgresTitleSuggestionRepository(conn, alice)
    row = mine.create_pending(
        phrase="engineering manager", phrase_key=normalise("engineering manager")
    )
    assert row is not None

    theirs = PostgresTitleSuggestionRepository(conn, bob)
    assert theirs.dismiss(row.id) is False

    fetched = mine.get(row.id)
    assert fetched is not None and fetched.dismissed_at is None


def test_a_user_cannot_mark_another_users_row_done(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    mine = PostgresTitleSuggestionRepository(conn, alice)
    row = mine.create_pending(
        phrase="engineering manager", phrase_key=normalise("engineering manager")
    )
    assert row is not None

    theirs = PostgresTitleSuggestionRepository(conn, bob)
    theirs.mark_done(row.id, [SuggestedTitle(title="Should Not Land", gloss="")])

    fetched = mine.get(row.id)
    assert fetched is not None
    assert fetched.status == "pending"
    assert fetched.suggestions == []
