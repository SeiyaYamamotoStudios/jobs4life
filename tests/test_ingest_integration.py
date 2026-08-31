"""Integration tests for corpus ingestion. Needs `docker compose up -d` and
`alembic upgrade head`.

Follows the transaction-rollback fixture pattern from test_schema_integration.py:
PostgresIngestRepository takes the test's own `conn`, inside its transaction, so
tearing that down at the end of each test undoes every write.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from jfl_core.context import RequestContext
from jfl_core.db.tables import documents, spans, users
from jfl_core.ids import content_hash, span_id
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.storage.postgres import PostgresIngestRepository
from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import Connection

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    import os

    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine):
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()


@pytest.fixture
def user(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


def _ctx(user_id: uuid.UUID) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        anthropic_api_key=None,
        database_url="unused-in-tests",
    )


def _write_corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for name, content in files.items():
        path = corpus_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return corpus_dir


CV = "## Northwind\n\n- Led the platform team\n- Shipped v2\n"


def test_ingesting_a_fixture_corpus_twice_is_idempotent(
    tmp_path: Path, conn: Connection, user: uuid.UUID
) -> None:
    corpus_dir = _write_corpus(tmp_path, {"cv.md": CV})
    repo = PostgresIngestRepository(conn)
    ctx = _ctx(user)

    first = run_ingestion(ctx, repo, corpus_dir)
    assert first.documents_seen == 1
    assert first.spans_created == 3  # heading + 2 bullets
    assert first.spans_updated == 0

    second = run_ingestion(ctx, repo, corpus_dir)
    assert second.documents_seen == 1
    assert second.spans_created == 0  # nothing new
    assert second.spans_updated == 3  # same rows, re-seen
    assert second.spans_retired == 0
    assert second.documents_retired == 0


def test_removing_a_file_retires_its_spans_rather_than_deleting_them(
    tmp_path: Path, conn: Connection, user: uuid.UUID
) -> None:
    corpus_dir = _write_corpus(tmp_path, {"cv.md": CV, "other.md": "- A separate claim\n"})
    repo = PostgresIngestRepository(conn)
    ctx = _ctx(user)
    run_ingestion(ctx, repo, corpus_dir)

    removed_span_id = span_id(user, "file:corpus/other.md", "", "- A separate claim")

    (corpus_dir / "other.md").unlink()
    summary = run_ingestion(ctx, repo, corpus_dir)

    assert summary.documents_retired == 1
    assert summary.spans_retired == 1

    row = conn.execute(select(spans).where(spans.c.id == removed_span_id)).one()
    assert row.retired_at is not None  # still resolvable, not deleted

    doc_row = conn.execute(
        select(documents).where(
            documents.c.user_id == user, documents.c.source_uri.contains("other")
        )
    ).one()
    assert doc_row.retired_at is not None


def test_editing_a_bullet_retires_the_old_span_and_creates_a_new_one(
    tmp_path: Path, conn: Connection, user: uuid.UUID
) -> None:
    corpus_dir = _write_corpus(tmp_path, {"cv.md": "- Led a team of 6\n"})
    repo = PostgresIngestRepository(conn)
    ctx = _ctx(user)
    run_ingestion(ctx, repo, corpus_dir)

    old_id = span_id(user, "file:corpus/cv.md", "", "- Led a team of 6")

    (corpus_dir / "cv.md").write_text("- Led a team of 12\n", encoding="utf-8")
    summary = run_ingestion(ctx, repo, corpus_dir)

    assert summary.spans_created == 1
    assert summary.spans_retired == 1

    old_row = conn.execute(select(spans).where(spans.c.id == old_id)).one()
    assert old_row.retired_at is not None  # old span still resolvable

    new_id = span_id(user, "file:corpus/cv.md", "", "- Led a team of 12")
    new_row = conn.execute(select(spans).where(spans.c.id == new_id)).one()
    assert new_row.retired_at is None
    assert new_row.text == "Led a team of 12"


def test_adjudicated_spans_are_never_retired_by_ingestion(
    tmp_path: Path, conn: Connection, user: uuid.UUID
) -> None:
    corpus_dir = _write_corpus(tmp_path, {"cv.md": CV})
    repo = PostgresIngestRepository(conn)
    ctx = _ctx(user)
    run_ingestion(ctx, repo, corpus_dir)

    adjudicated_id = uuid.uuid4()
    conn.execute(
        insert(spans).values(
            id=adjudicated_id,
            user_id=user,
            document_id=None,
            provenance="adjudicated",
            kind="bullet",
            text="confirmed in review",
            content_hash=content_hash("confirmed in review"),
        )
    )

    # Empty the corpus dir entirely; only the document-provenance spans should be retired.
    (corpus_dir / "cv.md").unlink()
    run_ingestion(ctx, repo, corpus_dir)

    row = conn.execute(select(spans).where(spans.c.id == adjudicated_id)).one()
    assert row.retired_at is None


def test_cli_end_to_end_via_run_ingestion_lands_rows_in_postgres(
    tmp_path: Path, conn: Connection, user: uuid.UUID
) -> None:
    """Exercises the same path the `jfl ingest` CLI command drives, without a
    subprocess, so it can share the rollback fixture.
    """
    corpus_dir = _write_corpus(
        tmp_path,
        {"cv.md": "# Alex\n\n## Northwind\n\n- Led the platform team\n- Shipped v2\n"},
    )
    repo = PostgresIngestRepository(conn)
    ctx = _ctx(user)
    summary = run_ingestion(ctx, repo, corpus_dir)

    assert summary.documents_seen == 1
    doc_rows = conn.execute(select(documents).where(documents.c.user_id == user)).all()
    assert len(doc_rows) == 1
    assert doc_rows[0].title == "Alex"

    span_rows = conn.execute(select(spans).where(spans.c.user_id == user)).all()
    assert len(span_rows) == 4  # two headings ("Alex", "Northwind") + 2 bullets
    assert {r.kind for r in span_rows} == {"heading", "bullet"}
