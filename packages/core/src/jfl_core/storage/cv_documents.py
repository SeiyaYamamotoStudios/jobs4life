"""PLACEHOLDER (cvedit branch) -- the append-only CV document store.

The generation branch owns the real repository; at merge its version wins and
the editing/export screens are re-pointed if a name differs. The three methods
below are the agreed surface:

    latest(application_id)
    list_versions(application_id)
    add_version(application_id, doc, *, status, trace_id)

Append-only: an edit, a template switch and a check each write a new version,
so every version a user has seen stays readable and downloadable.

Tenancy is structural: the repository is constructed with its user, every read
is filtered by it, and `add_version` refuses an application that is not theirs.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, insert, select

from jfl_core.cv_document import CvDocument
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import cv_documents as table
from jfl_core.storage.tenancy import TenantScopedRepository


class CvDocumentVersion(BaseModel):
    """One stored version. Immutable -- there is no update path."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    application_id: uuid.UUID
    version: int
    doc: CvDocument
    status: str
    trace_id: uuid.UUID | None
    created_at: dt.datetime


class ApplicationNotFoundError(LookupError):
    """`add_version` for an application this user does not own."""


_COLUMNS = (
    table.c.id,
    table.c.application_id,
    table.c.version,
    table.c.doc,
    table.c.status,
    table.c.trace_id,
    table.c.created_at,
)


def _row(row: object) -> CvDocumentVersion:
    return CvDocumentVersion(
        id=row.id,  # type: ignore[attr-defined]
        application_id=row.application_id,  # type: ignore[attr-defined]
        version=row.version,  # type: ignore[attr-defined]
        doc=CvDocument.model_validate(row.doc),  # type: ignore[attr-defined]
        status=row.status,  # type: ignore[attr-defined]
        trace_id=row.trace_id,  # type: ignore[attr-defined]
        created_at=row.created_at,  # type: ignore[attr-defined]
    )


class PostgresCvDocumentRepository(TenantScopedRepository):
    """One user's CV documents, and no one else's."""

    def latest(self, application_id: uuid.UUID) -> CvDocumentVersion | None:
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id, table.c.application_id == application_id)
            .order_by(table.c.version.desc())
            .limit(1)
        ).first()
        return None if row is None else _row(row)

    def list_versions(self, application_id: uuid.UUID) -> list[CvDocumentVersion]:
        """Every version, newest first."""
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id, table.c.application_id == application_id)
            .order_by(table.c.version.desc())
        ).all()
        return [_row(row) for row in rows]

    def add_version(
        self,
        application_id: uuid.UUID,
        doc: CvDocument,
        *,
        status: str,
        trace_id: uuid.UUID | None,
    ) -> CvDocumentVersion:
        owned = self._conn.execute(
            select(applications_table.c.id).where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
        ).first()
        if owned is None:
            raise ApplicationNotFoundError(str(application_id))
        current = self._conn.execute(
            select(func.coalesce(func.max(table.c.version), 0)).where(
                table.c.application_id == application_id
            )
        ).scalar_one()
        row = self._conn.execute(
            insert(table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                version=int(current) + 1,
                doc=doc.model_dump(mode="json"),
                status=status,
                trace_id=trace_id,
            )
            .returning(*_COLUMNS)
        ).one()
        return _row(row)


__all__ = ["ApplicationNotFoundError", "CvDocumentVersion", "PostgresCvDocumentRepository"]
