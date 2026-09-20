"""The write-back path, end to end against real Postgres -- PLAN.md slice B6.

This is the test the slice exists for. Two claims, and the second matters more
than the first:

  1. a **confirmed** fact reaches the corpus, as an ordinary
     `provenance='document'` span that `PostgresGroundingRepository.all_spans`
     returns -- through markdown, never a direct span insert;
  2. the **CV** never does. Not its text, not its lines, not a proposed fact.
     Grounding on CVs makes every later CV "supported" and switches the
     over-claim measurement off silently, which is the failure this whole
     design is shaped around (CLAUDE.md, 2026-09-18).

Needs `docker compose up -d` and `alembic upgrade head`. Transaction-rollback
fixtures. No network and no model call.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.corpus_source import SOURCE_URI, document_markdown
from jfl_core.db.tables import documents as documents_table
from jfl_core.db.tables import users
from jfl_core.ids import fact_fingerprint, role_key
from jfl_core.models import ProposedFact
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.postgres import PostgresGroundingRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import Connection, Engine

pytestmark = pytest.mark.integration

ACME = "Acme Ltd -- Engineering Manager, 2021-2024"
NORTHWIND = "Northwind -- Staff Engineer, 2018-2021"

CV_TEXT = (
    "# Jane Doe\n\n"
    "## Acme Ltd -- Engineering Manager, 2021-2024\n\n"
    "- Single-handedly rescued the payments platform from certain doom.\n"
    "- Led a team of 8.\n"
)


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


@pytest.fixture
def alice(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


def _fact(role: str, text: str, *, document: uuid.UUID | None = None) -> ProposedFact:
    return ProposedFact(
        sent_document_id=document,
        role_label=role,
        role_key=role_key(role),
        source_line=f"- {text}",
        fact_text=text,
        fingerprint=fact_fingerprint(role, text),
    )


def _corpus_facts(conn: Connection, user_id: uuid.UUID) -> list[str]:
    """The confirmed facts in the corpus, in document order.

    Bullets only: a corpus document also has heading spans (the document title
    and one per role), exactly as `corpus/*.md` always has, and those are
    structure rather than facts.
    """
    return [
        s.text for s in PostgresGroundingRepository(conn).all_spans(user_id) if s.kind == "bullet"
    ]


def _all_corpus_text(conn: Connection, user_id: uuid.UUID) -> list[str]:
    return [s.text for s in PostgresGroundingRepository(conn).all_spans(user_id)]


# -- 1. a confirmed fact becomes a grounding span ------------------------------


def test_a_confirmed_fact_is_an_ordinary_document_span(conn: Connection, alice: uuid.UUID) -> None:
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed([_fact(ACME, "Ran the on-call rota for the payments team.")])
    fact = facts.list_facts()[0]

    confirmed = facts.confirm(fact.id)

    assert confirmed is not None and confirmed.span_id is not None
    span = PostgresGroundingRepository(conn).get_span(alice, confirmed.span_id)
    assert span is not None
    assert span.text == "Ran the on-call rota for the payments team."
    # A document span, not an adjudicated one: it came out of markdown, like
    # every other corpus fact, so nothing downstream can tell it apart.
    assert span.provenance == "document"
    assert span.kind == "bullet"
    # The role is the breadcrumb, which is what the gate sees beside the claim.
    assert span.section_path == ACME
    assert span.id in {s.id for s in PostgresGroundingRepository(conn).all_spans(alice)}


def test_the_users_own_words_are_what_lands(conn: Connection, alice: uuid.UUID) -> None:
    """No model on this path: the edit is stored byte for byte, not tidied."""
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed([_fact(ACME, "Single-handedly rescued the payments platform.")])
    fact = facts.list_facts()[0]

    edit = "Was one of four people on the payments platform recovery; I ran the rota."
    confirmed = facts.confirm(fact.id, text=edit)

    assert confirmed is not None
    assert edit in _corpus_facts(conn, alice)
    assert "Single-handedly" not in " ".join(_corpus_facts(conn, alice))


def test_the_markdown_is_stored_and_readable_back(conn: Connection, alice: uuid.UUID) -> None:
    """ "Markdown is the source of truth, the database is a rebuildable index"
    has to survive a hosted user with no file on disk.
    """
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed([_fact(ACME, "Ran the rota."), _fact(NORTHWIND, "Wrote the scheduler.")])
    for fact in facts.list_facts():
        facts.confirm(fact.id)

    markdown = document_markdown(conn, alice)
    assert markdown is not None
    assert f"## {ACME}" in markdown
    assert "- Ran the rota." in markdown
    assert f"## {NORTHWIND}" in markdown
    assert "- Wrote the scheduler." in markdown

    row = conn.execute(
        select(documents_table.c.storage_kind, documents_table.c.source_uri).where(
            documents_table.c.user_id == alice
        )
    ).one()
    assert (row.storage_kind, row.source_uri) == ("hosted", SOURCE_URI)


def test_two_confirmations_share_one_corpus_document(conn: Connection, alice: uuid.UUID) -> None:
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed([_fact(ACME, "One."), _fact(ACME, "Two.")])
    for fact in facts.list_facts():
        facts.confirm(fact.id)

    documents = conn.execute(
        select(documents_table.c.id).where(documents_table.c.user_id == alice)
    ).all()
    assert len(documents) == 1


def test_rejecting_a_confirmed_fact_takes_it_out_of_grounding(
    conn: Connection, alice: uuid.UUID
) -> None:
    """ "I can delete the fact you recorded about me" has to be true, not
    aspirational -- and deleting it has to stop the gate grounding on it.
    """
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed([_fact(ACME, "Kept this one."), _fact(ACME, "Took this one back.")])
    for fact in facts.list_facts():
        facts.confirm(fact.id)
    assert len(_corpus_facts(conn, alice)) == 2

    taken_back = next(f for f in facts.list_facts() if f.fact_text == "Took this one back.")
    facts.reject(taken_back.id)

    assert _corpus_facts(conn, alice) == ["Kept this one."]
    markdown = document_markdown(conn, alice)
    assert markdown is not None and "Took this one back." not in markdown


def test_editing_a_confirmed_fact_replaces_rather_than_adds(
    conn: Connection, alice: uuid.UUID
) -> None:
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed([_fact(ACME, "Led a team of 8.")])
    fact = facts.list_facts()[0]
    facts.confirm(fact.id, text="Led a team of eight.")

    facts.confirm(fact.id, text="Led a team of six.")

    assert _corpus_facts(conn, alice) == ["Led a team of six."]


# -- 2. the CV itself never reaches grounding ----------------------------------


def test_an_uploaded_cv_never_appears_in_the_corpus(conn: Connection, alice: uuid.UUID) -> None:
    """The sent-document store is separate by construction, not by a WHERE
    clause. Storing a CV, and proposing facts from it, must leave the grounding
    store completely empty.
    """
    cvs = PostgresSentDocumentRepository(conn, alice)
    stored = cvs.add_cv(filename="cv.md", text=CV_TEXT)

    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed(
        [
            _fact(ACME, "Single-handedly rescued the payments platform.", document=stored.id),
            _fact(ACME, "Led a team of 8.", document=stored.id),
        ]
    )

    assert PostgresGroundingRepository(conn).all_spans(alice) == []
    assert (
        conn.execute(select(documents_table.c.id).where(documents_table.c.user_id == alice)).all()
        == []
    )


def test_only_the_confirmed_fact_grounds_not_its_cv_line(
    conn: Connection, alice: uuid.UUID
) -> None:
    cvs = PostgresSentDocumentRepository(conn, alice)
    stored = cvs.add_cv(filename="cv.md", text=CV_TEXT)
    facts = PostgresCandidateFactRepository(conn, alice)
    facts.add_proposed(
        [
            _fact(ACME, "Led a team of 8.", document=stored.id),
            _fact(ACME, "Single-handedly rescued the payments platform.", document=stored.id),
        ]
    )
    confirmable = next(f for f in facts.list_facts() if f.fact_text == "Led a team of 8.")

    facts.confirm(confirmable.id)

    corpus = _all_corpus_text(conn, alice)
    assert "Led a team of 8." in corpus
    # The unconfirmed one is kept in `candidate_facts` and grounds nothing.
    assert not any("Single-handedly" in text for text in corpus)
    # And nothing verbatim from the CV -- its own lines, its headings -- is in
    # the corpus either.
    assert not any("Jane Doe" in text for text in corpus)
    assert facts.counts().proposed == 1
