"""CV intake storage against real Postgres -- PLAN.md slice B6.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures as in `test_profile_repo_integration.py`: every test leaves the
database as it found it. No network and no model call.

The grounding half -- a confirmed fact becoming a corpus span, and a CV never
becoming one -- is in `test_confirmed_facts_grounding_integration.py`.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import users
from jfl_core.ids import fact_fingerprint, role_key
from jfl_core.models import ProposedFact
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository, cv_path
from sqlalchemy import create_engine, insert
from sqlalchemy.engine import Connection, Engine

pytestmark = pytest.mark.integration

ACME = "Acme Ltd -- Engineering Manager, 2021-2024"
NORTHWIND = "Northwind -- Staff Engineer, 2018-2021"

CV_ONE = "# Jane Doe\n\n## Acme Ltd\n\n- Led a team of 8.\n- Owned the payments platform.\n"
CV_TWO = "# Jane Doe\n\n## Acme Ltd\n\n- Led a team of eight.\n"


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


def _fact(role: str, text: str, *, probe: str | None = None, ordinal: int = 0) -> ProposedFact:
    return ProposedFact(
        role_label=role,
        role_key=role_key(role),
        source_line=f"- {text}",
        fact_text=text,
        probe=probe,
        fingerprint=fact_fingerprint(role, text),
        ordinal=ordinal,
    )


# -- the CV store --------------------------------------------------------------


class TestSentDocuments:
    def test_a_cv_is_stored_verbatim(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresSentDocumentRepository(conn, alice)
        stored = repo.add_cv(filename="cv.md", text=CV_ONE)
        assert repo.cv_text(stored.id) == CV_ONE
        assert stored.extraction_status == "pending"

    def test_re_uploading_the_same_bytes_does_not_duplicate(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """A second row would mean a second model call charged to the user for
        a document already read.
        """
        repo = PostgresSentDocumentRepository(conn, alice)
        first = repo.add_cv(filename="cv.md", text=CV_ONE)
        again = repo.add_cv(filename="cv.md", text=CV_ONE)
        assert again.id == first.id
        assert len(repo.list_cvs()) == 1

    def test_the_same_cv_under_another_name_is_still_the_same_cv(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresSentDocumentRepository(conn, alice)
        first = repo.add_cv(filename="cv.md", text=CV_ONE)
        renamed = repo.add_cv(filename="cv-final-FINAL.md", text=CV_ONE)
        assert renamed.id == first.id

    def test_the_same_filename_with_different_text_is_a_different_cv(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresSentDocumentRepository(conn, alice)
        first = repo.add_cv(filename="cv.md", text=CV_ONE)
        second = repo.add_cv(filename="cv.md", text=CV_TWO)
        assert second.id != first.id
        assert cv_path("cv.md", CV_ONE) != cv_path("cv.md", CV_TWO)

    def test_another_users_cv_is_invisible(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        hers = PostgresSentDocumentRepository(conn, alice).add_cv(filename="cv.md", text=CV_ONE)
        his = PostgresSentDocumentRepository(conn, bob)
        assert his.get_cv(hers.id) is None
        assert his.cv_text(hers.id) is None
        assert his.claim_extraction(hers.id) is None
        assert his.list_cvs() == []

    def test_two_users_may_upload_byte_identical_cvs(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        """Content identity is per user: dedupe must never reach across
        tenants, or one person's upload would silently answer another's.
        """
        hers = PostgresSentDocumentRepository(conn, alice).add_cv(filename="cv.md", text=CV_ONE)
        his = PostgresSentDocumentRepository(conn, bob).add_cv(filename="cv.md", text=CV_ONE)
        assert hers.id != his.id


class TestClaimExtraction:
    def test_a_pending_cv_yields_its_text(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresSentDocumentRepository(conn, alice)
        stored = repo.add_cv(filename="cv.md", text=CV_ONE)
        assert repo.claim_extraction(stored.id) == CV_ONE

    def test_a_finished_cv_yields_nothing(self, conn: Connection, alice: uuid.UUID) -> None:
        """At-least-once delivery means this runs again on work already done.
        A second claim would be a second charge."""
        repo = PostgresSentDocumentRepository(conn, alice)
        stored = repo.add_cv(filename="cv.md", text=CV_ONE)
        repo.finish_extraction(stored.id, facts_proposed=4)
        assert repo.claim_extraction(stored.id) is None
        assert repo.get_cv(stored.id).facts_proposed == 4  # type: ignore[union-attr]

    def test_a_failed_cv_can_be_claimed_again(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresSentDocumentRepository(conn, alice)
        stored = repo.add_cv(filename="cv.md", text=CV_ONE)
        repo.fail_extraction(stored.id, "model_error")
        assert repo.claim_extraction(stored.id) == CV_ONE
        assert repo.get_cv(stored.id).extraction_error_code == "model_error"  # type: ignore[union-attr]

    def test_an_unknown_id_yields_nothing(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresSentDocumentRepository(conn, alice)
        assert repo.claim_extraction(uuid.uuid4()) is None


# -- candidate facts -----------------------------------------------------------


class TestAddProposed:
    def test_facts_are_inserted_and_counted(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        added = repo.add_proposed(
            [_fact(ACME, "Led a team of 8.", ordinal=0), _fact(ACME, "Ran the rota.", ordinal=1)]
        )
        assert added == 2
        assert repo.counts().proposed == 2

    def test_the_same_fact_from_a_second_cv_is_not_a_second_row(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """Thirty-three near-identical CVs collapse into one list to confirm,
        or onboarding is unusable.
        """
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Led a team of 8.")])
        assert repo.add_proposed([_fact(ACME, "led  a  team of 8")]) == 0
        assert repo.counts().proposed == 1

    def test_a_repeat_never_resets_a_decided_fact(self, conn: Connection, alice: uuid.UUID) -> None:
        """A fact the user rejected must not come back as `proposed` because
        another CV mentioned it again.
        """
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Led a team of 8.")])
        fact = repo.list_facts()[0]
        repo.reject(fact.id)
        repo.add_proposed([_fact(ACME, "Led a team of 8.")])
        assert repo.get_fact(fact.id).state == "rejected"  # type: ignore[union-attr]

    def test_two_users_may_hold_the_same_fingerprint(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        assert PostgresCandidateFactRepository(conn, alice).add_proposed([_fact(ACME, "x")]) == 1
        assert PostgresCandidateFactRepository(conn, bob).add_proposed([_fact(ACME, "x")]) == 1

    def test_an_empty_batch_is_a_no_op(self, conn: Connection, alice: uuid.UUID) -> None:
        assert PostgresCandidateFactRepository(conn, alice).add_proposed([]) == 0


class TestRoles:
    def test_roles_group_by_key_and_keep_cv_order(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed(
            [
                _fact(ACME, "Acme one.", ordinal=0),
                _fact(ACME, "Acme two.", ordinal=1),
                _fact(NORTHWIND, "Northwind one.", ordinal=2),
            ]
        )
        roles = repo.roles()
        assert [r.role_label for r in roles] == [ACME, NORTHWIND]
        assert [r.proposed for r in roles] == [2, 1]

    def test_the_same_role_written_differently_is_one_group(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "One."), _fact(ACME.lower(), "Two.")])
        assert len(repo.roles()) == 1

    def test_counts_move_between_states(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "One."), _fact(ACME, "Two."), _fact(ACME, "Three.")])
        facts = repo.list_facts()
        repo.confirm(facts[0].id)
        repo.reject(facts[1].id)
        role = repo.roles()[0]
        assert (role.confirmed, role.rejected, role.proposed) == (1, 1, 1)
        assert repo.counts().model_dump() == {"proposed": 1, "confirmed": 1, "rejected": 1}

    def test_another_users_roles_are_invisible(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        PostgresCandidateFactRepository(conn, alice).add_proposed([_fact(ACME, "Hers.")])
        his = PostgresCandidateFactRepository(conn, bob)
        assert his.roles() == []
        assert his.list_facts() == []
        assert his.counts().total == 0


class TestConfirmAndReject:
    def test_confirming_stores_the_users_edit_verbatim(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Led a team of 8.")])
        fact = repo.list_facts()[0]

        confirmed = repo.confirm(fact.id, text="Led a team of six, and two contractors.")

        assert confirmed is not None
        assert confirmed.state == "confirmed"
        assert confirmed.confirmed_text == "Led a team of six, and two contractors."
        assert confirmed.fact_text == "Led a team of 8."  # the proposal is kept as it was
        assert confirmed.corpus_text == "Led a team of six, and two contractors."
        assert confirmed.span_id is not None

    def test_confirming_the_proposal_as_written_needs_no_edit(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Ran the on-call rota.")])
        fact = repo.list_facts()[0]
        confirmed = repo.confirm(fact.id)
        assert confirmed is not None
        assert confirmed.confirmed_text is None
        assert confirmed.corpus_text == "Ran the on-call rota."

    def test_a_probe_answer_is_the_users_own_words(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Led a team of 8.", probe="How many reported to you?")])
        fact = repo.list_facts()[0]
        confirmed = repo.confirm(fact.id, probe_answer="Six directly, two dotted line.")
        assert confirmed is not None
        assert confirmed.probe_answer == "Six directly, two dotted line."

    def test_confirming_twice_is_idempotent(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Ran the rota.")])
        fact = repo.list_facts()[0]
        first = repo.confirm(fact.id)
        second = repo.confirm(fact.id)
        assert first is not None and second is not None
        assert first.span_id == second.span_id

    def test_rejecting_clears_the_span(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        repo.add_proposed([_fact(ACME, "Ran the rota.")])
        fact = repo.list_facts()[0]
        repo.confirm(fact.id)
        rejected = repo.reject(fact.id)
        assert rejected is not None
        assert rejected.state == "rejected"
        assert rejected.span_id is None

    def test_another_users_fact_cannot_be_confirmed_or_rejected(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        hers = PostgresCandidateFactRepository(conn, alice)
        hers.add_proposed([_fact(ACME, "Hers.")])
        fact = hers.list_facts()[0]

        his = PostgresCandidateFactRepository(conn, bob)
        assert his.get_fact(fact.id) is None
        assert his.confirm(fact.id) is None
        assert his.reject(fact.id) is None
        assert hers.get_fact(fact.id).state == "proposed"  # type: ignore[union-attr]

    def test_an_unknown_fact_is_none_rather_than_an_error(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresCandidateFactRepository(conn, alice)
        assert repo.confirm(uuid.uuid4()) is None
        assert repo.reject(uuid.uuid4()) is None
