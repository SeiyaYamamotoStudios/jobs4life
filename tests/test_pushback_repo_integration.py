"""The pushback log against real Postgres. Needs a migrated database.

Transaction-rollback fixtures, as in `test_scores_repo_integration.py`: every
test leaves the database as it found it. No network, no model call.

The properties worth pinning here are the ones invisible until they go wrong:
that the words come back byte for byte, that displacement is derived from the
log rather than stored anywhere, that an exact restatement is caught whatever
anyone ticks, that the database itself refuses a capability claim upward
carrying a movement, and that one user's corrections are invisible to another
even to a caller holding the id.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import application_scores, applications, score_pushbacks, users
from jfl_core.pushback import COULD_GET_OVERALL, WANT_OVERALL
from jfl_core.storage.pushbacks import (
    PostgresPushbackRepository,
    PostgresScoreOverrideRepository,
)
from sqlalchemy import create_engine, insert, select, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration

WORDS = "That's wrong, I ran the whole platform for two years."


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
        insert(applications).values(id=application_id, user_id=user_id, title="Staff Platform")
    )
    return application_id


def _make_score(conn: Connection, user_id: uuid.UUID, application_id: uuid.UUID) -> uuid.UUID:
    score_id = uuid.uuid4()
    conn.execute(
        insert(application_scores).values(
            id=score_id,
            user_id=user_id,
            application_id=application_id,
            status="done",
            could_get_score=4,
            could_get_assessment="The ad asks for org-wide platform ownership.",
            want_it_score=5,
            want_it_assessment="Two of five things you said matter are evidenced.",
        )
    )
    return score_id


class Fixture:
    def __init__(self, conn: Connection) -> None:
        self.user_id = _make_user(conn)
        self.application_id = _make_application(conn, self.user_id)
        self.score_id = _make_score(conn, self.user_id, self.application_id)
        self.repo = PostgresPushbackRepository(conn, self.user_id)

    def record(
        self,
        *,
        dimension: str = WANT_OVERALL,
        axis: str = "want",
        direction: str = "up",
        text: str = WORDS,
        points: float = 1.0,
    ):  # type: ignore[no-untyped-def]
        return self.repo.record(
            application_id=self.application_id,
            score_id=self.score_id,
            axis=axis,  # type: ignore[arg-type]
            dimension=dimension,
            shown_score=5,
            shown_explanation="Two of five things you said matter are evidenced.",
            user_text=text,
            asserted_direction=direction,  # type: ignore[arg-type]
            asserted_points=points,
        )


@pytest.fixture
def alice(conn: Connection) -> Fixture:
    return Fixture(conn)


@pytest.fixture
def bob(conn: Connection) -> Fixture:
    return Fixture(conn)


# -- recording ---------------------------------------------------------------


def test_a_pushback_is_recorded_before_anything_is_classified(alice: Fixture) -> None:
    row = alice.record()
    assert row.status == "awaiting_classification"
    assert row.applied_delta is None
    assert row.disposition is None
    assert row.classification is None


def test_the_users_words_come_back_byte_for_byte(alice: Fixture) -> None:
    odd = "  I ran  the WHOLE platform -- two years, 9 engineers.\n"
    row = alice.record(text=odd)
    assert alice.repo.get(row.id).user_text == odd  # type: ignore[union-attr]


def test_the_stimulus_is_part_of_the_record(alice: Fixture) -> None:
    row = alice.record()
    assert row.shown_score == 5
    assert row.shown_explanation.startswith("Two of five")


def test_two_identical_pushbacks_are_two_rows(alice: Fixture) -> None:
    first = alice.record()
    second = alice.record()
    assert first.id != second.id
    assert len(alice.repo.for_application(alice.application_id)) == 2


# -- applying ----------------------------------------------------------------


def test_a_preference_is_accepted_shrunk_and_recorded(alice: Fixture) -> None:
    row = alice.record()
    applied = alice.repo.apply(row.id, classification="preference", new_information=True)
    assert applied is not None
    assert applied.status == "applied"
    assert applied.disposition == "accepted"
    assert applied.applied_delta == pytest.approx(1.0)
    assert applied.target_dimension == WANT_OVERALL


def test_displacement_is_derived_from_the_log_not_stored(alice: Fixture) -> None:
    for _ in range(2):
        row = alice.record(text=f"new point {uuid.uuid4()}")
        alice.repo.apply(row.id, classification="preference", new_information=True)
    # 1.0, then 1.0 * 5/6 shrunk by the first. Stored as NUMERIC(6,3), which
    # is three decimal places of a number whose unit is a point of a 1-10
    # score -- far finer than anything displayed, and the tolerance says so.
    assert alice.repo.displacement(WANT_OVERALL) == pytest.approx(1.0 + 5 / 6, abs=1e-3)
    assert alice.repo.displacements() == pytest.approx({WANT_OVERALL: 1.0 + 5 / 6}, abs=1e-3)


def test_an_exact_restatement_moves_nothing_whatever_the_user_ticks(alice: Fixture) -> None:
    """The half of "repetition is not evidence" that no prompt and no checkbox
    can be argued out of.
    """
    first = alice.record()
    alice.repo.apply(first.id, classification="preference", new_information=True)
    again = alice.record(text=WORDS.upper())
    applied = alice.repo.apply(again.id, classification="preference", new_information=True)
    assert applied is not None
    assert applied.applied_delta == 0.0
    assert applied.new_information is False
    assert applied.effect["repetition"] is True


def test_the_third_correction_on_one_dimension_offers_the_comparison(alice: Fixture) -> None:
    effects = []
    for i in range(3):
        row = alice.record(text=f"a different point number {i}")
        applied = alice.repo.apply(row.id, classification="preference", new_information=True)
        assert applied is not None
        effects.append(applied.effect)
    assert [e["comparison_offered"] for e in effects] == [False, False, True]
    assert alice.repo.displacement(WANT_OVERALL) == pytest.approx(2.0)


def test_a_capability_claim_upward_opens_a_question_and_moves_nothing(alice: Fixture) -> None:
    row = alice.record(axis="get", dimension=COULD_GET_OVERALL)
    applied = alice.repo.apply(
        row.id,
        classification="capability",
        new_information=True,
        evidence_question="What did running the platform involve?",
    )
    assert applied is not None
    assert applied.applied_delta == 0.0
    assert applied.disposition == "pending_evidence"
    assert applied.evidence_question
    assert alice.repo.displacement(COULD_GET_OVERALL) == 0.0


def test_a_capability_claim_downward_lands_in_full(alice: Fixture) -> None:
    row = alice.record(axis="get", dimension=COULD_GET_OVERALL, direction="down", points=3.0)
    applied = alice.repo.apply(row.id, classification="capability", new_information=True)
    assert applied is not None
    assert applied.applied_delta == pytest.approx(-3.0)


def test_applying_twice_is_not_two_effects(alice: Fixture) -> None:
    row = alice.record()
    first = alice.repo.apply(row.id, classification="preference", new_information=True)
    second = alice.repo.apply(row.id, classification="preference", new_information=True)
    assert first is not None and second is not None
    assert first.applied_delta == second.applied_delta
    assert alice.repo.displacement(WANT_OVERALL) == pytest.approx(first.applied_delta)


def test_an_applied_pushback_cannot_be_reclassified(alice: Fixture) -> None:
    row = alice.record()
    alice.repo.apply(row.id, classification="preference", new_information=True)
    alice.repo.set_classification(row.id, classification="factual", new_information=False)
    after = alice.repo.get(row.id)
    assert after is not None
    assert after.classification == "preference"


# -- what a pushback never touches ------------------------------------------


def test_no_pushback_changes_the_stored_score(conn: Connection, alice: Fixture) -> None:
    before = (
        conn.execute(select(application_scores).where(application_scores.c.id == alice.score_id))
        .one()
        ._asdict()
    )
    for kind, axis, dimension, direction in (
        ("preference", "want", WANT_OVERALL, "up"),
        ("capability", "get", COULD_GET_OVERALL, "up"),
        ("capability", "get", COULD_GET_OVERALL, "down"),
        ("factual", "want", WANT_OVERALL, "down"),
    ):
        row = alice.record(
            axis=axis, dimension=dimension, direction=direction, text=f"{kind} {direction}"
        )
        alice.repo.apply(row.id, classification=kind, new_information=True)  # type: ignore[arg-type]
    after = (
        conn.execute(select(application_scores).where(application_scores.c.id == alice.score_id))
        .one()
        ._asdict()
    )
    assert before == after


def test_the_database_refuses_a_capability_claim_upward_that_moved(
    conn: Connection, alice: Fixture
) -> None:
    """The rule written where no future caller can get past it.

    `jfl_core.pushback` returns before any arithmetic; this is the same
    guarantee in the schema, so a hand-written UPDATE cannot install the
    ratchet either.
    """
    row = alice.record(axis="get", dimension=COULD_GET_OVERALL)
    alice.repo.apply(row.id, classification="capability", new_information=True)
    with pytest.raises(IntegrityError):
        conn.execute(
            update(score_pushbacks).where(score_pushbacks.c.id == row.id).values(applied_delta=2)
        )


def test_the_database_refuses_a_movement_on_an_unapplied_pushback(
    conn: Connection, alice: Fixture
) -> None:
    row = alice.record()
    with pytest.raises(IntegrityError):
        conn.execute(
            update(score_pushbacks).where(score_pushbacks.c.id == row.id).values(applied_delta=1)
        )


# -- the drift meter ---------------------------------------------------------


def test_the_drift_meter_counts_what_was_asked_for_not_what_was_applied(
    alice: Fixture,
) -> None:
    up = alice.record(axis="get", dimension=COULD_GET_OVERALL, text="you have underrated me")
    alice.repo.apply(up.id, classification="capability", new_information=True)
    down = alice.record(text="I would not take this", direction="down")
    alice.repo.apply(down.id, classification="preference", new_information=True)

    meter = alice.repo.drift_meter()
    assert meter.total == 2
    assert meter.upward == 1
    assert meter.downward == 1
    assert meter.pending_evidence == 1
    assert meter.net == pytest.approx(-1.0)


def test_an_unapplied_pushback_is_not_counted_as_though_it_had_done_something(
    alice: Fixture,
) -> None:
    alice.record()
    assert alice.repo.drift_meter().total == 0


def test_awaiting_evidence_lists_what_the_loop_is_waiting_on(alice: Fixture) -> None:
    row = alice.record(axis="get", dimension=COULD_GET_OVERALL)
    alice.repo.apply(
        row.id,
        classification="capability",
        new_information=True,
        evidence_question="What did it involve?",
    )
    assert [p.id for p in alice.repo.awaiting_evidence()] == [row.id]

    alice.repo.record_evidence(row.id, _some_span(alice))
    assert alice.repo.awaiting_evidence() == []


def _some_span(fixture: Fixture) -> uuid.UUID:
    """A span id to point at. `record_evidence` only records which span
    answered the question -- it does not create one, and could not.
    """
    from jfl_core.corpus_source import append_confirmed_fact

    return append_confirmed_fact(
        fixture.repo._conn,  # noqa: SLF001 -- the test owns this connection
        fixture.user_id,
        "I owned the platform at Acme: nine engineers and the on-call rota.",
        section="Answered questions",
    )


# -- overrides ---------------------------------------------------------------


def test_an_override_is_local_and_feeds_nothing(conn: Connection, alice: Fixture) -> None:
    overrides = PostgresScoreOverrideRepository(conn, alice.user_id)
    overrides.set_override(alice.application_id, axis="get", value=9, note="trust me")
    assert overrides.current(alice.application_id)["get"].value == 9
    assert alice.repo.displacements() == {}
    assert alice.repo.drift_meter().total == 0


def test_clearing_an_override_keeps_the_row_that_set_it(conn: Connection, alice: Fixture) -> None:
    overrides = PostgresScoreOverrideRepository(conn, alice.user_id)
    overrides.set_override(alice.application_id, axis="want", value=8)
    overrides.set_override(alice.application_id, axis="want", value=None)
    assert "want" not in overrides.current(alice.application_id)
    rows = conn.execute(select(score_pushbacks.c.id)).all()
    assert rows == []  # and nothing leaked into the pushback log


# -- tenancy -----------------------------------------------------------------


def test_another_users_pushback_is_invisible_even_with_its_id(
    conn: Connection, alice: Fixture, bob: Fixture
) -> None:
    row = bob.record()
    assert alice.repo.get(row.id) is None
    assert alice.repo.apply(row.id, classification="preference", new_information=True) is None
    assert alice.repo.for_application(bob.application_id) == []


def test_one_users_corrections_do_not_move_anothers_numbers(alice: Fixture, bob: Fixture) -> None:
    row = bob.record()
    bob.repo.apply(row.id, classification="preference", new_information=True)
    assert bob.repo.displacement(WANT_OVERALL) == pytest.approx(1.0)
    assert alice.repo.displacement(WANT_OVERALL) == 0.0
    assert alice.repo.drift_meter().total == 0


def test_another_users_override_is_invisible(
    conn: Connection, alice: Fixture, bob: Fixture
) -> None:
    PostgresScoreOverrideRepository(conn, bob.user_id).set_override(
        bob.application_id, axis="get", value=9
    )
    assert PostgresScoreOverrideRepository(conn, alice.user_id).current(bob.application_id) == {}
