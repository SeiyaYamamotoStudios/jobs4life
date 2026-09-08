"""Integration tests for slice A5's repository: `PostgresApplicationRepository`.
Needs `docker compose up -d` and `alembic upgrade head`.

Follows the transaction-rollback fixture pattern from test_jobs_integration.py.
No Anthropic API calls anywhere -- there is nothing here that could make one.
"""

from __future__ import annotations

import os
import uuid

import pytest
from jfl_core.db.tables import application_events as application_events_table
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import jobs as jobs_table
from jfl_core.db.tables import users
from jfl_core.storage.applications import ApplicationNotFoundError, PostgresApplicationRepository
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine):
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()  # every test leaves the database as it found it


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


@pytest.fixture
def repo(conn: Connection, alice: uuid.UUID) -> PostgresApplicationRepository:
    return PostgresApplicationRepository(conn, alice)


# --- basic round trip -------------------------------------------------------


def test_create_round_trips_and_defaults_to_interested(
    repo: PostgresApplicationRepository,
) -> None:
    application = repo.create_application(title="Platform Engineer", employer="Acme")
    assert application.status == "interested"
    assert application.employer == "Acme"
    assert application.job_id is None
    assert application.created_at is not None
    assert application.updated_at is not None

    fetched = repo.get_application(application.id)
    assert fetched is not None
    assert fetched.application == application


def test_get_application_returns_none_for_an_unknown_id(
    repo: PostgresApplicationRepository,
) -> None:
    assert repo.get_application(uuid.uuid4()) is None


# --- the timeline: create writes the first event, status changes append ----


def test_creating_an_application_writes_the_opening_event(
    repo: PostgresApplicationRepository,
) -> None:
    """The event log is the timeline from the start -- "added" is itself an
    entry, not an implicit gap before the first real transition.
    """
    application = repo.create_application(title="Backend Engineer")
    detail = repo.get_application(application.id)
    assert detail is not None
    assert len(detail.events) == 1
    assert detail.events[0].from_status is None
    assert detail.events[0].to_status == "interested"


def test_a_status_change_appends_an_event_rather_than_replacing_history(
    conn: Connection, repo: PostgresApplicationRepository
) -> None:
    """The acceptance criterion from the brief, at the repository layer: two
    transitions must leave TWO events behind, not one row overwritten twice.
    """
    application = repo.create_application(title="SRE")

    repo.change_status(application.id, to_status="applied")
    repo.change_status(application.id, to_status="screening", note="Recruiter call booked")

    detail = repo.get_application(application.id)
    assert detail is not None
    assert detail.application.status == "screening"

    # Three events total: the opening "added" plus the two transitions.
    assert len(detail.events) == 3
    assert [e.to_status for e in detail.events] == ["interested", "applied", "screening"]
    assert detail.events[0].from_status is None
    assert detail.events[1].from_status == "interested"
    assert detail.events[2].from_status == "applied"
    assert detail.events[2].note == "Recruiter call booked"

    # And the raw table really does hold three rows -- nothing was updated in place.
    rows = conn.execute(
        select(application_events_table.c.id).where(
            application_events_table.c.application_id == application.id
        )
    ).all()
    assert len(rows) == 3


def test_change_status_updates_updated_at(repo: PostgresApplicationRepository) -> None:
    application = repo.create_application(title="Data Engineer")
    before = application.updated_at
    updated = repo.change_status(application.id, to_status="applied")
    assert updated.updated_at >= before


def test_change_status_on_an_unknown_application_raises(
    repo: PostgresApplicationRepository,
) -> None:
    with pytest.raises(ApplicationNotFoundError):
        repo.change_status(uuid.uuid4(), to_status="applied")


def test_update_notes_on_an_unknown_application_raises(
    repo: PostgresApplicationRepository,
) -> None:
    with pytest.raises(ApplicationNotFoundError):
        repo.update_notes(uuid.uuid4(), "hello")


def test_update_notes_round_trips_and_does_not_touch_the_timeline(
    repo: PostgresApplicationRepository,
) -> None:
    application = repo.create_application(title="Engineering Manager", notes="first draft")
    updated = repo.update_notes(application.id, "revised notes")
    assert updated.notes == "revised notes"

    detail = repo.get_application(application.id)
    assert detail is not None
    assert len(detail.events) == 1  # unchanged: notes are not a status transition


# --- listing: ordering and filtering ----------------------------------------


def test_list_applications_orders_most_recently_updated_first(
    repo: PostgresApplicationRepository,
) -> None:
    first = repo.create_application(title="First")
    second = repo.create_application(title="Second")
    repo.change_status(first.id, to_status="applied")  # bump first back to the top

    ids_in_order = [a.id for a in repo.list_applications()]
    assert ids_in_order.index(first.id) < ids_in_order.index(second.id)


def test_list_applications_filters_by_status(repo: PostgresApplicationRepository) -> None:
    interested = repo.create_application(title="Still interested")
    applied = repo.create_application(title="Already applied")
    repo.change_status(applied.id, to_status="applied")

    only_applied = repo.list_applications(status="applied")
    assert [a.id for a in only_applied] == [applied.id]

    only_interested = repo.list_applications(status="interested")
    assert [a.id for a in only_interested] == [interested.id]


# --- a pasted job ad is stored verbatim, never parsed -----------------------


def test_a_pasted_job_ad_is_stored_verbatim_and_linked(
    conn: Connection, repo: PostgresApplicationRepository
) -> None:
    raw_text = "Senior Engineer at Acme. Remote. Own the platform."
    application = repo.create_application(title="Senior Engineer", raw_job_text=raw_text)

    assert application.job_id is not None
    row = conn.execute(
        select(jobs_table.c.raw_text, jobs_table.c.source, jobs_table.c.employer).where(
            jobs_table.c.id == application.job_id
        )
    ).one()
    assert row.raw_text == raw_text
    assert row.source == "paste"
    # Nothing extracted -- see the module docstring: extraction is a model
    # call and belongs to slice B.
    assert row.employer is None


def test_re_pasting_the_same_ad_on_a_second_application_reuses_the_job_row(
    repo: PostgresApplicationRepository,
) -> None:
    raw_text = "Staff Engineer at Acme. Hybrid."
    first = repo.create_application(title="Staff Engineer (round 1)", raw_job_text=raw_text)
    second = repo.create_application(title="Staff Engineer (round 2)", raw_job_text=raw_text)
    assert first.job_id == second.job_id


def test_no_job_ad_means_no_job_row(repo: PostgresApplicationRepository) -> None:
    application = repo.create_application(title="No ad pasted")
    assert application.job_id is None


# --- tenancy: the acceptance criterion from PLAN.md -------------------------


def test_two_users_cannot_see_each_others_applications(
    conn: Connection, alice: uuid.UUID, bob: uuid.UUID
) -> None:
    """Both repositories run on the SAME connection, so nothing but the id
    fixed at construction separates them -- no per-request filter, no
    session, no middleware. If scoping were a WHERE clause someone had to
    remember, this is where it would show.
    """
    alice_repo = PostgresApplicationRepository(conn, alice)
    bob_repo = PostgresApplicationRepository(conn, bob)

    alice_app = alice_repo.create_application(title="Alice's application")
    bob_app = bob_repo.create_application(title="Bob's application")

    assert [a.id for a in alice_repo.list_applications()] == [alice_app.id]
    assert [a.id for a in bob_repo.list_applications()] == [bob_app.id]

    # Bob's repository cannot reach Alice's row by id, guessed or not.
    assert bob_repo.get_application(alice_app.id) is None
    with pytest.raises(ApplicationNotFoundError):
        bob_repo.change_status(alice_app.id, to_status="applied")
    with pytest.raises(ApplicationNotFoundError):
        bob_repo.update_notes(alice_app.id, "not yours")

    # And Alice's row is untouched by any of Bob's attempts.
    fetched = alice_repo.get_application(alice_app.id)
    assert fetched is not None
    assert fetched.application.status == "interested"
    assert fetched.application.notes is None


def test_a_deleted_user_takes_their_applications_and_events_with_them(
    conn: Connection, alice: uuid.UUID
) -> None:
    """ON DELETE CASCADE, checked rather than assumed."""
    repo = PostgresApplicationRepository(conn, alice)
    application = repo.create_application(title="Will be cascaded away")

    conn.execute(delete(users).where(users.c.id == alice))

    assert (
        conn.execute(
            select(applications_table.c.id).where(applications_table.c.id == application.id)
        ).first()
        is None
    )
    assert (
        conn.execute(
            select(application_events_table.c.id).where(
                application_events_table.c.application_id == application.id
            )
        ).first()
        is None
    )


def test_deleting_the_linked_job_does_not_delete_the_application(
    conn: Connection, repo: PostgresApplicationRepository
) -> None:
    """`job_id` is ON DELETE SET NULL, not CASCADE -- losing the ad text must
    never take the tracked application down with it.
    """
    application = repo.create_application(
        title="Ad text may disappear", raw_job_text="Some job ad text."
    )
    assert application.job_id is not None

    conn.execute(delete(jobs_table).where(jobs_table.c.id == application.job_id))

    fetched = repo.get_application(application.id)
    assert fetched is not None
    assert fetched.application.job_id is None


# --- the status CHECK constraint --------------------------------------------


def test_an_invalid_status_is_rejected_at_the_database(conn: Connection, alice: uuid.UUID) -> None:
    with pytest.raises(IntegrityError):
        conn.execute(
            insert(applications_table).values(
                id=uuid.uuid4(),
                user_id=alice,
                title="Bad status",
                status="not-a-real-status",
            )
        )
    # No further use of `conn` in this test -- the fixture's teardown
    # `tx.rollback()` clears the aborted transaction, same as
    # test_schema_integration.py's constraint tests.


def test_updated_at_is_bumped_by_a_notes_edit_too(repo: PostgresApplicationRepository) -> None:
    application = repo.create_application(title="Notes bump timestamp")
    before = application.updated_at
    updated = repo.update_notes(application.id, "edited")
    assert updated.updated_at >= before
