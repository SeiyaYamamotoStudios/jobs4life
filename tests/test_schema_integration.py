"""Integration tests: the constraints that encode design rules must actually bite.

Marked `integration`; excluded from the default run. Needs `docker compose up -d`
and `alembic upgrade head`.
"""

from __future__ import annotations

import os
import uuid

import pytest
from jfl_core.db.tables import documents, spans, users
from sqlalchemy import create_engine, insert, text
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


@pytest.fixture
def user(conn) -> uuid.UUID:
    """A throwaway user per test; the enclosing transaction rolls it back."""
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


def _doc(conn, user: uuid.UUID) -> uuid.UUID:
    doc_id = uuid.uuid4()
    conn.execute(
        insert(documents).values(
            id=doc_id,
            user_id=user,
            source_uri=f"file:corpus/{doc_id}.md",
            storage_kind="local_file",
            content_hash="0" * 64,
        )
    )
    return doc_id


def test_document_span_requires_a_document(conn, user) -> None:
    with pytest.raises(IntegrityError):
        conn.execute(
            insert(spans).values(
                id=uuid.uuid4(),
                user_id=user,
                document_id=None,
                provenance="document",
                kind="bullet",
                text="orphaned",
                content_hash="a" * 64,
            )
        )


def test_adjudicated_span_must_not_have_a_document(conn, user) -> None:
    doc_id = _doc(conn, user)
    with pytest.raises(IntegrityError):
        conn.execute(
            insert(spans).values(
                id=uuid.uuid4(),
                user_id=user,
                document_id=doc_id,
                provenance="adjudicated",
                kind="bullet",
                text="wrong",
                content_hash="b" * 64,
            )
        )


def test_valid_spans_of_both_provenances_insert(conn, user) -> None:
    doc_id = _doc(conn, user)
    conn.execute(
        insert(spans).values(
            id=uuid.uuid4(),
            user_id=user,
            document_id=doc_id,
            provenance="document",
            kind="bullet",
            text="from a file",
            content_hash="c" * 64,
        )
    )
    conn.execute(
        insert(spans).values(
            id=uuid.uuid4(),
            user_id=user,
            document_id=None,
            provenance="adjudicated",
            kind="bullet",
            text="confirmed in review",
            content_hash="d" * 64,
        )
    )


def test_unknown_provenance_is_rejected(conn, user) -> None:
    with pytest.raises(IntegrityError):
        conn.execute(
            insert(spans).values(
                id=uuid.uuid4(),
                user_id=user,
                document_id=None,
                provenance="invented",
                kind="bullet",
                text="x",
                content_hash="e" * 64,
            )
        )


def test_sent_store_has_no_foreign_key_path_into_the_corpus(conn) -> None:
    """The isolation of the sent store is structural, not a WHERE clause.

    No FK connects sent_* to a corpus table, so a grounding query cannot reach
    sent material by joining. Both sides legitimately reference `users` -- a
    shared parent is not a path between them, since joining through it would
    return the whole corpus rather than anything sent-specific.
    """
    corpus_tables = {"documents", "spans", "span_sentences", "span_embeddings"}
    rows = conn.execute(
        text(
            """
            select tc.table_name, ccu.table_name as references_table
            from information_schema.table_constraints tc
            join information_schema.constraint_column_usage ccu
              on tc.constraint_name = ccu.constraint_name
            where tc.constraint_type = 'FOREIGN KEY'
            """
        )
    ).all()
    crossings = [
        (r.table_name, r.references_table)
        for r in rows
        if (r.table_name.startswith("sent_") and r.references_table in corpus_tables)
        or (r.references_table.startswith("sent_") and r.table_name in corpus_tables)
    ]
    assert crossings == [], f"sent store is joinable to the corpus: {crossings}"


def test_grounding_repository_exposes_no_sent_material() -> None:
    """The interface, not just the schema, must have no route to sent documents."""
    import inspect as _inspect

    from jfl_core.repositories import GroundingRepository

    source = _inspect.getsource(GroundingRepository)
    assert "sent" not in source.lower()


def test_hnsw_index_exists_on_embeddings(conn) -> None:
    idx = conn.execute(
        text("select indexdef from pg_indexes where indexname = 'ix_span_embeddings_embedding'")
    ).scalar_one()
    assert "hnsw" in idx and "vector_cosine_ops" in idx
