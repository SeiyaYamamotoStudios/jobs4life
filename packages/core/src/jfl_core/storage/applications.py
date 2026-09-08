"""The application tracker: slice A5, tenancy-scoped.

This is the wedge -- see CLAUDE.md's 2026-09-07 decision log entry and
`PLAN.md`'s slice A. The owner's own words for the gap conversations cannot
close are "a clear list of all the applications I have going"; this repository
is that list.

No model call anywhere in this module, deliberately: slice A is model-free so
it costs nothing to run and keeps auth bugs separate from engine bugs. A
pasted job ad's raw text is stored verbatim in `jobs.raw_text` -- reusing the
existing domain-2a table rather than inventing a second place for job text --
and NOTHING extracts `employer`/`title`/`location` from it here. That is a
model call (`jfl_generate.extract.extract_requirements`) and belongs to slice
B; when it lands, `jobs.upsert_job`-style logic will happily fill those columns
in on the same deterministic row id this module already wrote.

**Transitions are recorded, never overwritten.** `change_status` updates
`applications.status` AND inserts an `application_events` row in the same
statement group -- the event log is the timeline, and later slices inject it
into model context, so it has to be complete. `create_application` does the
same on the way in, with `from_status=NULL`, so "added" is itself the first
timeline entry rather than an implicit gap before the first real transition.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import insert, select, update

from jfl_core.db.tables import application_events as application_events_table
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import jobs as jobs_table
from jfl_core.ids import content_hash
from jfl_core.ids import job_id as derive_job_id
from jfl_core.models import Application, ApplicationDetail, ApplicationEvent, ApplicationStatus
from jfl_core.storage.tenancy import TenantScopedRepository

DEFAULT_STATUS: ApplicationStatus = "interested"

_APPLICATION_COLUMNS = (
    applications_table.c.id,
    applications_table.c.user_id,
    applications_table.c.job_id,
    applications_table.c.title,
    applications_table.c.employer,
    applications_table.c.url,
    applications_table.c.status,
    applications_table.c.source,
    applications_table.c.notes,
    applications_table.c.created_at,
    applications_table.c.updated_at,
)

_EVENT_COLUMNS = (
    application_events_table.c.id,
    application_events_table.c.user_id,
    application_events_table.c.application_id,
    application_events_table.c.from_status,
    application_events_table.c.to_status,
    application_events_table.c.note,
    application_events_table.c.occurred_at,
    application_events_table.c.created_at,
)


class ApplicationNotFoundError(RuntimeError):
    """No application with this id belongs to this user.

    Deliberately the same message shape whether the id does not exist at all
    or belongs to someone else -- distinguishing the two would tell a caller
    which ids are real, which is the tenancy leak this whole scheme exists to
    prevent.
    """

    def __init__(self, application_id: uuid.UUID) -> None:
        super().__init__(f"no application {application_id} for this user")


def _application_from_row(row: Any) -> Application:
    return Application(
        id=row.id,
        user_id=row.user_id,
        job_id=row.job_id,
        title=row.title,
        employer=row.employer,
        url=row.url,
        status=row.status,
        source=row.source,
        notes=row.notes,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _event_from_row(row: Any) -> ApplicationEvent:
    return ApplicationEvent(
        id=row.id,
        user_id=row.user_id,
        application_id=row.application_id,
        from_status=row.from_status,
        to_status=row.to_status,
        note=row.note,
        occurred_at=row.occurred_at,
        created_at=row.created_at,
    )


class PostgresApplicationRepository(TenantScopedRepository):
    """Applications and their status-change timeline, for exactly one user."""

    def create_application(
        self,
        *,
        title: str,
        employer: str | None = None,
        url: str | None = None,
        source: str | None = None,
        notes: str | None = None,
        raw_job_text: str | None = None,
        status: ApplicationStatus = DEFAULT_STATUS,
    ) -> Application:
        """Add an application, and write the "added" event that opens its
        timeline. `raw_job_text`, if given, is stored verbatim as a `jobs` row
        (see the module docstring) and linked via `job_id`; it is never parsed.
        """
        job_id = self._store_raw_job(raw_job_text) if raw_job_text else None

        application_id = uuid.uuid4()
        row = self._conn.execute(
            insert(applications_table)
            .values(
                id=application_id,
                user_id=self._user_id,
                job_id=job_id,
                title=title,
                employer=employer,
                url=url,
                status=status,
                source=source,
                notes=notes,
            )
            .returning(*_APPLICATION_COLUMNS)
        ).one()

        self._conn.execute(
            insert(application_events_table).values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                from_status=None,
                to_status=status,
                note=None,
            )
        )
        return _application_from_row(row)

    def list_applications(self, *, status: ApplicationStatus | None = None) -> list[Application]:
        """Most recently updated first -- the primary screen's ordering."""
        query = (
            select(*_APPLICATION_COLUMNS)
            .where(applications_table.c.user_id == self._user_id)
            .order_by(applications_table.c.updated_at.desc())
        )
        if status is not None:
            query = query.where(applications_table.c.status == status)
        rows = self._conn.execute(query).all()
        return [_application_from_row(row) for row in rows]

    def get_application(self, application_id: uuid.UUID) -> ApplicationDetail | None:
        """The application plus its full timeline, oldest event first."""
        row = self._conn.execute(
            select(*_APPLICATION_COLUMNS).where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
        ).first()
        if row is None:
            return None

        event_rows = self._conn.execute(
            select(*_EVENT_COLUMNS)
            .where(
                application_events_table.c.application_id == application_id,
                application_events_table.c.user_id == self._user_id,
            )
            .order_by(
                application_events_table.c.occurred_at.asc(),
                application_events_table.c.created_at.asc(),
            )
        ).all()
        return ApplicationDetail(
            application=_application_from_row(row),
            events=[_event_from_row(r) for r in event_rows],
        )

    def change_status(
        self, application_id: uuid.UUID, *, to_status: ApplicationStatus, note: str | None = None
    ) -> Application:
        """Update `status` and append an event in one call. Raises
        `ApplicationNotFoundError` rather than silently doing nothing, so a
        caller cannot mistake "no such application" for "no change needed".
        """
        current = self._conn.execute(
            select(applications_table.c.status).where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
        ).first()
        if current is None:
            raise ApplicationNotFoundError(application_id)
        from_status = current.status

        # `updated_at` is not in `.values()` -- the column's `onupdate=func.now()`
        # (see tables.py) fills it in on every UPDATE built from this table.
        row = self._conn.execute(
            update(applications_table)
            .where(applications_table.c.id == application_id)
            .values(status=to_status)
            .returning(*_APPLICATION_COLUMNS)
        ).one()

        self._conn.execute(
            insert(application_events_table).values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                application_id=application_id,
                from_status=from_status,
                to_status=to_status,
                note=note,
            )
        )
        return _application_from_row(row)

    def update_notes(self, application_id: uuid.UUID, notes: str | None) -> Application:
        row = self._conn.execute(
            update(applications_table)
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
            .values(notes=notes)
            .returning(*_APPLICATION_COLUMNS)
        ).first()
        if row is None:
            raise ApplicationNotFoundError(application_id)
        return _application_from_row(row)

    def _store_raw_job(self, raw_text: str) -> uuid.UUID:
        """Store a pasted job ad verbatim as a `jobs` row and return its id.

        Deterministic on (user, text) via `jfl_core.ids.job_id`, same as the
        CLI path -- re-pasting the same ad from a second application resolves
        to the same row rather than minting a duplicate. Nothing here reads
        `employer`/`title`/`location` out of the text: that is extraction, a
        model call, and belongs to slice B.
        """
        jid = derive_job_id(self._user_id, raw_text)
        exists = self._conn.execute(select(jobs_table.c.id).where(jobs_table.c.id == jid)).first()
        if exists is None:
            self._conn.execute(
                insert(jobs_table).values(
                    id=jid,
                    user_id=self._user_id,
                    source="paste",
                    raw_text=raw_text,
                    content_hash=content_hash(raw_text),
                )
            )
        return jid
