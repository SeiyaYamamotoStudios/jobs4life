"""Profile storage against real Postgres -- `docs/profile-schema.md`.

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures: every test leaves the database as it found it. No network and no model
call anywhere in this file.

What is worth pinning here is what a defect would be silent about: a re-submitted
form manufacturing a version the user never chose, history coming back in the
wrong order so "the current profile" is the wrong one, a profile that fails to
parse back out of JSONB, and one user reading another's.

The self-assessment's corpus half -- profile questions 15 and 16 becoming spans
through the one write path -- is at the end, because it is the only part of a
save that leaves this table.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.db.tables import users
from jfl_core.ids import fact_fingerprint, role_key
from jfl_core.models import ProposedFact
from jfl_core.profile import (
    Capability,
    Constraint,
    Disciplines,
    Objective,
    Profile,
    SelfAssessment,
)
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.postgres import PostgresGroundingRepository
from jfl_core.storage.profile import (
    PostgresProfileRepository,
    propose_capabilities_from_facts,
    save_profile,
)
from jfl_core.storage.user_corpus import PostgresUserCorpusRepository
from sqlalchemy import create_engine, insert
from sqlalchemy.engine import Connection, Engine

pytestmark = pytest.mark.integration

ACME = "Acme Ltd -- Engineering Manager, 2021-2024"


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


def _profile(**kw: object) -> Profile:
    defaults: dict[str, object] = {
        "constraints": [
            Constraint(
                kind="comp_floor",
                stance="must",
                value={"guaranteed": 120000, "headline": 145000, "ccy": "GBP"},
                note="base + pension, ignoring equity",
            )
        ],
        "disciplines": Disciplines(practises=["engineering management"], **{"not": ["frontend"]}),
        "objectives": [Objective(rank=1, text="Back to hands-on work")],
    }
    defaults.update(kw)
    return Profile(**defaults)  # type: ignore[arg-type]


# -- reading ------------------------------------------------------------------


class TestCurrent:
    def test_a_user_who_has_saved_nothing_gets_an_empty_profile_not_none(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """ "No profile" and "an empty profile" are the same statement to every
        caller. Returning None would put a check in six places and leave it out
        of one.
        """
        current = PostgresProfileRepository(conn, alice).current()
        assert current == Profile()
        assert current.is_empty

    def test_the_current_profile_is_the_latest_save(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresProfileRepository(conn, alice)
        repo.save(_profile())
        repo.save(_profile(objectives=[Objective(rank=1, text="Stop commuting")]))
        assert repo.current().objectives[0].text == "Stop commuting"

    def test_everything_saved_reads_back_exactly(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresProfileRepository(conn, alice)
        span = uuid.uuid4()
        saved = _profile(
            capabilities=[
                Capability(
                    label="FX pricing platforms",
                    tier="production_depth",
                    interest="want_more",
                    last_used=2024,
                    evidence=[span],
                    source="cv_fact",
                )
            ],
            self_assessment=SelfAssessment(depth_genuine="Deep on payments."),
        )
        repo.save(saved)
        current = repo.current()
        assert current == saved
        assert current.capabilities[0].evidence == [span]
        assert current.disciplines.not_practised == ["frontend"]


# -- writing ------------------------------------------------------------------


class TestSave:
    def test_a_save_appends_rather_than_overwriting(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """What you believed about yourself in March stays readable, and undo
        is free.
        """
        repo = PostgresProfileRepository(conn, alice)
        repo.save(_profile())
        repo.save(_profile(objectives=[Objective(rank=1, text="Stop commuting")]))
        assert len(repo.history()) == 2

    def test_an_identical_save_is_a_no_op(self, conn: Connection, alice: uuid.UUID) -> None:
        """Re-submitting an untouched form must not appear in history as a
        decision the user made -- the rule the retired `save_answer` followed.
        """
        repo = PostgresProfileRepository(conn, alice)
        repo.save(_profile())
        returned = repo.save(_profile())
        assert len(repo.history()) == 1
        assert returned == repo.current()

    def test_saving_an_empty_profile_over_nothing_is_still_a_no_op(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresProfileRepository(conn, alice)
        repo.save(Profile())
        assert repo.history() == []

    def test_clearing_a_section_is_a_change_worth_recording(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """Emptying a section is a decision, not an absence of one."""
        repo = PostgresProfileRepository(conn, alice)
        repo.save(_profile())
        repo.save(_profile(objectives=[]))
        assert len(repo.history()) == 2
        assert repo.current().objectives == []

    def test_reordering_a_ranked_list_is_a_change(self, conn: Connection, alice: uuid.UUID) -> None:
        """Disciplines are ranked, so a re-ranking is exactly the kind of
        change history exists to keep.
        """
        repo = PostgresProfileRepository(conn, alice)
        repo.save(_profile(disciplines=Disciplines(practises=["a", "b"])))
        repo.save(_profile(disciplines=Disciplines(practises=["b", "a"])))
        assert len(repo.history()) == 2

    def test_save_returns_what_was_stored(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresProfileRepository(conn, alice)
        assert repo.save(_profile()) == repo.current()


# -- history ------------------------------------------------------------------


class TestHistory:
    def test_versions_come_back_newest_first(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresProfileRepository(conn, alice)
        for text in ("first", "second", "third"):
            repo.save(_profile(objectives=[Objective(rank=1, text=text)]))
        assert [v.data.objectives[0].text for v in repo.history()] == [
            "third",
            "second",
            "first",
        ]

    def test_two_saves_in_one_transaction_do_not_tie(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """`created_at` defaults to `clock_timestamp()`, not `now()`: the
        current profile is *the latest row*, and `now()` is transaction-start
        time, so two saves in one request would make "latest" ambiguous exactly
        where it decides what the user sees.
        """
        repo = PostgresProfileRepository(conn, alice)
        repo.save(_profile(objectives=[Objective(rank=1, text="first")]))
        repo.save(_profile(objectives=[Objective(rank=1, text="second")]))
        stamps = [v.created_at for v in repo.history()]
        assert stamps[0] > stamps[1]
        assert repo.current().objectives[0].text == "second"

    def test_the_limit_is_honoured_and_keeps_the_newest(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresProfileRepository(conn, alice)
        for text in ("first", "second", "third"):
            repo.save(_profile(objectives=[Objective(rank=1, text=text)]))
        assert [v.data.objectives[0].text for v in repo.history(limit=2)] == [
            "third",
            "second",
        ]

    def test_no_history_for_a_user_who_has_saved_nothing(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        assert PostgresProfileRepository(conn, alice).history() == []


class TestAt:
    def test_a_version_reads_back_whole(self, conn: Connection, alice: uuid.UUID) -> None:
        repo = PostgresProfileRepository(conn, alice)
        first = _profile(objectives=[Objective(rank=1, text="first")])
        repo.save(first)
        repo.save(_profile(objectives=[Objective(rank=1, text="second")]))
        oldest = repo.history()[-1]
        assert repo.at(oldest.id) == first

    def test_an_unknown_version_is_none(self, conn: Connection, alice: uuid.UUID) -> None:
        assert PostgresProfileRepository(conn, alice).at(uuid.uuid4()) is None


# -- tenancy ------------------------------------------------------------------


class TestTenancy:
    def test_one_user_s_profile_is_invisible_to_another(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        PostgresProfileRepository(conn, alice).save(_profile())
        bob_repo = PostgresProfileRepository(conn, bob)
        assert bob_repo.current() == Profile()
        assert bob_repo.history() == []

    def test_another_user_s_version_id_reads_as_absent(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        alice_repo = PostgresProfileRepository(conn, alice)
        alice_repo.save(_profile())
        version_id = alice_repo.history()[0].id
        assert PostgresProfileRepository(conn, bob).at(version_id) is None

    def test_a_save_by_one_user_does_not_touch_another_s_history(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        PostgresProfileRepository(conn, alice).save(_profile())
        PostgresProfileRepository(conn, bob).save(_profile(objectives=[]))
        assert len(PostgresProfileRepository(conn, alice).history()) == 1
        assert len(PostgresProfileRepository(conn, bob).history()) == 1


# -- the self-assessment's corpus half ----------------------------------------


def _corpus_texts(conn: Connection, user_id: uuid.UUID) -> list[str]:
    """Every live (non-retired) span's text for this user -- what the claim gate
    would actually be handed.
    """
    return [s.text for s in PostgresGroundingRepository(conn).all_spans(user_id)]


class TestSelfAssessmentReachesTheCorpus:
    def test_saving_a_profile_records_questions_15_and_16_as_spans(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """The two answers that are claims about the person rather than
        preferences still become corpus text, through the one write path.
        """
        save_profile(
            PostgresProfileRepository(conn, alice),
            PostgresUserCorpusRepository(conn, alice),
            _profile(
                self_assessment=SelfAssessment(
                    depth_genuine="Deep on payments, exposure only on ML.",
                    recurring_gaps="Kubernetes keeps coming up.",
                )
            ),
        )
        texts = _corpus_texts(conn, alice)
        assert "Deep on payments, exposure only on ML." in texts
        assert "Kubernetes keeps coming up." in texts

    def test_the_words_are_stored_verbatim(self, conn: Connection, alice: uuid.UUID) -> None:
        """No model is anywhere on this path. A tidied sentence would hold the
        user to wording they did not choose.
        """
        words = "Deep on payments; NOT on ML -- I've only *reviewed* models."
        save_profile(
            PostgresProfileRepository(conn, alice),
            PostgresUserCorpusRepository(conn, alice),
            Profile(self_assessment=SelfAssessment(depth_genuine=words)),
        )
        assert words in _corpus_texts(conn, alice)

    def test_a_re_answer_supersedes_rather_than_joining(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        repo = PostgresProfileRepository(conn, alice)
        corpus = PostgresUserCorpusRepository(conn, alice)
        save_profile(repo, corpus, Profile(self_assessment=SelfAssessment(depth_genuine="Old.")))
        save_profile(repo, corpus, Profile(self_assessment=SelfAssessment(depth_genuine="New.")))
        texts = _corpus_texts(conn, alice)
        assert "New." in texts
        assert "Old." not in texts

    def test_a_cleared_answer_stops_grounding(self, conn: Connection, alice: uuid.UUID) -> None:
        """An answer the user deleted must not go on being cited at them."""
        repo = PostgresProfileRepository(conn, alice)
        corpus = PostgresUserCorpusRepository(conn, alice)
        save_profile(repo, corpus, Profile(self_assessment=SelfAssessment(recurring_gaps="K8s.")))
        save_profile(repo, corpus, Profile())
        assert "K8s." not in _corpus_texts(conn, alice)

    def test_the_profile_row_keeps_the_text_too(self, conn: Connection, alice: uuid.UUID) -> None:
        """The corpus holds the citable fact; the profile holds what the screen
        shows back. Both, because neither serves the other's purpose.
        """
        repo = PostgresProfileRepository(conn, alice)
        save_profile(
            repo,
            PostgresUserCorpusRepository(conn, alice),
            Profile(self_assessment=SelfAssessment(depth_genuine="Deep on payments.")),
        )
        assert repo.current().self_assessment.depth_genuine == "Deep on payments."

    def test_nothing_reaches_the_corpus_from_any_other_section(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        """Constraints, capabilities, disciplines and objectives are
        preferences, not claims about the person. A preference in the corpus
        would be groundable evidence for something nobody asserted.
        """
        save_profile(
            PostgresProfileRepository(conn, alice),
            PostgresUserCorpusRepository(conn, alice),
            _profile(capabilities=[Capability(label="FX pricing platforms", tier="working")]),
        )
        texts = " ".join(_corpus_texts(conn, alice))
        assert "FX pricing platforms" not in texts
        assert "Back to hands-on work" not in texts


# -- seeding from confirmed CV facts ------------------------------------------


class TestSeedingFromConfirmedFacts:
    def _proposed(self, text: str, ordinal: int = 0) -> ProposedFact:
        return ProposedFact(
            role_label=ACME,
            role_key=role_key(ACME),
            source_line=f"- {text}",
            fact_text=text,
            fingerprint=fact_fingerprint(ACME, text),
            ordinal=ordinal,
        )

    def test_a_confirmed_fact_seeds_a_capability_carrying_its_span(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        facts = PostgresCandidateFactRepository(conn, alice)
        facts.add_proposed([self._proposed("Led a team of eight")])
        fact = facts.list_facts()[0]
        confirmed = facts.confirm(fact.id)
        assert confirmed is not None and confirmed.span_id is not None

        proposed = propose_capabilities_from_facts(facts, Profile())
        assert [c.label for c in proposed] == [ACME]
        assert proposed[0].evidence == [confirmed.span_id]
        assert proposed[0].tier is None
        assert proposed[0].source == "cv_fact"

    def test_an_unconfirmed_fact_never_seeds_one(self, conn: Connection, alice: uuid.UUID) -> None:
        """A CV's claim is not the user's. This is the 2026-09-18 decision
        reaching the profile: grounding on unconfirmed CV text is what switches
        the over-claim measurement off silently.
        """
        facts = PostgresCandidateFactRepository(conn, alice)
        facts.add_proposed([self._proposed("Led a team of eight")])
        assert propose_capabilities_from_facts(facts, Profile()) == []

    def test_a_rejected_fact_never_seeds_one(self, conn: Connection, alice: uuid.UUID) -> None:
        facts = PostgresCandidateFactRepository(conn, alice)
        facts.add_proposed([self._proposed("Led a team of eight")])
        fact = facts.list_facts()[0]
        facts.confirm(fact.id)
        facts.reject(fact.id)
        assert propose_capabilities_from_facts(facts, Profile()) == []

    def test_a_role_the_profile_already_covers_is_not_proposed_again(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        facts = PostgresCandidateFactRepository(conn, alice)
        facts.add_proposed([self._proposed("Led a team of eight")])
        facts.confirm(facts.list_facts()[0].id)
        profile = Profile(capabilities=[Capability(label=ACME, tier="production_depth")])
        assert propose_capabilities_from_facts(facts, profile) == []

    def test_a_seeded_capability_can_be_saved_as_it_stands(
        self, conn: Connection, alice: uuid.UUID
    ) -> None:
        facts = PostgresCandidateFactRepository(conn, alice)
        facts.add_proposed([self._proposed("Led a team of eight")])
        facts.confirm(facts.list_facts()[0].id)
        repo = PostgresProfileRepository(conn, alice)
        proposed = propose_capabilities_from_facts(facts, repo.current())
        repo.save(Profile(capabilities=proposed))
        assert repo.current().capabilities == proposed

    def test_another_user_s_confirmed_facts_seed_nothing(
        self, conn: Connection, alice: uuid.UUID, bob: uuid.UUID
    ) -> None:
        alice_facts = PostgresCandidateFactRepository(conn, alice)
        alice_facts.add_proposed([self._proposed("Led a team of eight")])
        alice_facts.confirm(alice_facts.list_facts()[0].id)
        bob_facts = PostgresCandidateFactRepository(conn, bob)
        assert propose_capabilities_from_facts(bob_facts, Profile()) == []
