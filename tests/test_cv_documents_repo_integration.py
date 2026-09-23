"""`cv_documents` against real Postgres: append-only versions, the latest one
shown, and one user never seeing or writing another's. Needs `docker compose up
-d` and `alembic upgrade head`. Fictional fixtures only.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvRole
from jfl_core.db.tables import users as users_table
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.cv_documents import PostgresCvDocumentRepository
from sqlalchemy import create_engine, delete, insert
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    created = create_engine(DATABASE_URL)
    yield created
    created.dispose()


def _user(engine: Engine, display_name: str | None = None) -> uuid.UUID:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(
            insert(users_table).values(
                id=uid, email=f"{uid}@test.invalid", display_name=display_name
            )
        )
    return uid


@pytest.fixture
def users(engine: Engine) -> Iterator[tuple[uuid.UUID, uuid.UUID]]:
    first, second = _user(engine, "Morgan Fictional"), _user(engine)
    try:
        yield first, second
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id.in_([first, second])))


def _application(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        return (
            PostgresApplicationRepository(conn, user_id)
            .create_application(title="Platform Lead")
            .id
        )


def _document(summary: str, template: str = "modern") -> CvDocument:
    return CvDocument(
        template=template,  # type: ignore[arg-type]
        header=CvHeader(name="Morgan Fictional"),
        summary=[CvLine(text=summary, verdict="review", note="Not stated.")],
        roles=[
            CvRole(
                title="Head of Engineering",
                employer="Northwind Traders",
                dates="Nov 2021 – Present",
                bullets=[CvLine(text="Led a team.", verdict="supported")],
            )
        ],
        education=[CvLine(text="BSc, 2008", origin="fact")],
    )


def test_versions_are_appended_and_the_latest_is_shown(
    engine: Engine, users: tuple[uuid.UUID, uuid.UUID]
) -> None:
    user, _ = users
    application_id = _application(engine, user)
    trace = uuid.uuid4()
    with engine.begin() as conn:
        repo = PostgresCvDocumentRepository(conn, user)
        first = repo.add_version(
            application_id,
            _document("First."),
            status="generated",
            gate_result={"sentences": []},
            trace_id=trace,
        )
        edited = repo.add_version(
            application_id, _document("Edited.", template="classic"), status="edited"
        )
    assert first is not None and edited is not None

    with engine.begin() as conn:
        repo = PostgresCvDocumentRepository(conn, user)
        latest = repo.latest(application_id)
        versions = repo.list_versions(application_id)
        by_trace = repo.version_for_trace(trace)
        fetched = repo.get_version(first.id)

    assert latest is not None and latest.id == edited.id
    assert latest.template == "classic" and latest.document.template == "classic"
    assert latest.status == "edited" and latest.gate_result is None
    assert [v.id for v in versions] == [edited.id, first.id]
    assert by_trace is not None and by_trace.id == first.id
    # The document round-trips whole, verdicts and origins included.
    assert fetched is not None and fetched.document == _document("First.")


def test_another_user_can_neither_read_nor_write_them(
    engine: Engine, users: tuple[uuid.UUID, uuid.UUID]
) -> None:
    owner, other = users
    application_id = _application(engine, owner)
    trace = uuid.uuid4()
    with engine.begin() as conn:
        version = PostgresCvDocumentRepository(conn, owner).add_version(
            application_id, _document("Mine."), status="generated", trace_id=trace
        )
    assert version is not None

    with engine.begin() as conn:
        repo = PostgresCvDocumentRepository(conn, other)
        assert repo.latest(application_id) is None
        assert repo.list_versions(application_id) == []
        assert repo.get_version(version.id) is None
        assert repo.version_for_trace(trace) is None
        # Writing against someone else's application writes nothing.
        assert repo.add_version(application_id, _document("Theirs."), status="edited") is None

    with engine.begin() as conn:
        assert len(PostgresCvDocumentRepository(conn, owner).list_versions(application_id)) == 1


def test_the_account_display_name_is_the_users_own(
    engine: Engine, users: tuple[uuid.UUID, uuid.UUID]
) -> None:
    named, unnamed = users
    with engine.begin() as conn:
        assert PostgresCvDocumentRepository(conn, named).account_display_name() == (
            "Morgan Fictional"
        )
        assert PostgresCvDocumentRepository(conn, unnamed).account_display_name() == ""
