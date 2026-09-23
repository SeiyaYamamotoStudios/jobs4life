"""Complete CVs, one row per version, tenancy-scoped.

Append-only: `add_version` is the only write. A regenerate writes a new
`generated` row; an edit writes a new `edited` row holding the edited
document. Nothing here UPDATEs an earlier version, so what the user was shown
last week is still there to compare against -- the same rule `profiles` and
`application_question_answers` follow, for the same reason.

No model call anywhere in this module. The worker writes the `generated` rows
(`jfl_worker.handlers.draft_generation`); the edit screen writes the rest.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import insert, select

from jfl_core.cv_document import CvDocument, CvTemplate
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import cv_documents as table
from jfl_core.db.tables import users as users_table
from jfl_core.models import CvDocumentStatus
from jfl_core.storage.tenancy import TenantScopedRepository


class CvDocumentVersion(BaseModel):
    """One stored version of an application's CV."""

    id: uuid.UUID
    user_id: uuid.UUID
    application_id: uuid.UUID
    document: CvDocument
    template: CvTemplate
    status: CvDocumentStatus
    gate_result: dict[str, Any] | None
    trace_id: uuid.UUID | None
    created_at: dt.datetime
    updated_at: dt.datetime


_COLUMNS = (
    table.c.id,
    table.c.user_id,
    table.c.application_id,
    table.c.document,
    table.c.template,
    table.c.status,
    table.c.gate_result,
    table.c.trace_id,
    table.c.created_at,
    table.c.updated_at,
)


def _from_row(row: Any) -> CvDocumentVersion:
    return CvDocumentVersion(
        id=row.id,
        user_id=row.user_id,
        application_id=row.application_id,
        document=CvDocument.model_validate(row.document),
        template=row.template,
        status=row.status,
        gate_result=row.gate_result,
        trace_id=row.trace_id,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresCvDocumentRepository(TenantScopedRepository):
    """One user's CV versions, and no one else's."""

    def add_version(
        self,
        application_id: uuid.UUID,
        document: CvDocument,
        *,
        status: CvDocumentStatus,
        gate_result: dict[str, Any] | None = None,
        trace_id: uuid.UUID | None = None,
    ) -> CvDocumentVersion | None:
        """Store `document` as the newest version for this application. None --
        and nothing written -- if the application is not this user's, so an id
        from another account cannot be written against.

        `template` is taken from the document itself, never passed separately,
        so the column and the JSON cannot disagree.
        """
        owned = self._conn.execute(
            select(applications_table.c.id).where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
        ).first()
        if owned is None:
            return None
        row = self._conn.execute(
            insert(table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                document=document.model_dump(mode="json"),
                template=document.template,
                status=status,
                gate_result=gate_result,
                trace_id=trace_id,
            )
            .returning(*_COLUMNS)
        ).one()
        return _from_row(row)

    def latest(self, application_id: uuid.UUID) -> CvDocumentVersion | None:
        """The version shown: the newest row for this application."""
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.application_id == application_id, table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc(), table.c.id.desc())
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def list_versions(self, application_id: uuid.UUID) -> list[CvDocumentVersion]:
        """Every version for this application, newest first."""
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.application_id == application_id, table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc(), table.c.id.desc())
        ).all()
        return [_from_row(row) for row in rows]

    def get_version(self, version_id: uuid.UUID) -> CvDocumentVersion | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == version_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def version_for_trace(self, trace_id: uuid.UUID) -> CvDocumentVersion | None:
        """The version a task wrote, by the task's id -- how a redelivered task
        finds the work it already did instead of paying for it twice."""
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.trace_id == trace_id, table.c.user_id == self._user_id)
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def account_display_name(self) -> str:
        """The signed-in account's display name, or "" -- the CV header's
        fallback when the profile names no one. Read here because it is what
        this repository's header needs, and scoped like everything else."""
        value = self._conn.execute(
            select(users_table.c.display_name).where(users_table.c.id == self._user_id)
        ).scalar_one_or_none()
        return (value or "").strip()
