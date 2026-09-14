"""The application tracker: slice A5, tenancy-scoped.

This is the wedge -- see CLAUDE.md's 2026-09-07 decision log entry and
`PLAN.md`'s slice A. The owner's own words for the gap conversations cannot
close are "a clear list of all the applications I have going"; this repository
is that list.

No model call anywhere in this module, deliberately -- and that stays true in
slice B3, which is what the extraction methods at the bottom are for. A pasted
job ad's raw text is stored verbatim in `jobs.raw_text` (reusing the existing
domain-2a table rather than inventing a second place for job text) and nothing
here reads `employer`/`title`/`location` out of it. That is a model call
(`jfl_generate.extract.extract_requirements`), it takes ~30 seconds, and it
happens in the worker: this module only records that it was asked for, hands
the worker its input, and records what came back.

**Extraction never overwrites something the user typed.** `title` is replaced
only while `title_is_provisional` is true -- the placeholder this app derived
from the ad's first line -- and `employer` only while it is NULL. That is a
condition in the UPDATE's WHERE clause, not a check a caller has to remember,
because the caller that forgets silently rewrites someone's own words.

**Extraction costs the user money**, since users bring their own API key. So
`claim_extraction` refuses to hand out work for an application that is already
`done`: at-least-once delivery means a handler can be run twice, and twice here
means paying twice. A genuine re-read is `request_extraction`, which a person
has to press.

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

from sqlalchemy import func, insert, select, update

from jfl_core.db.tables import application_events as application_events_table
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import job_requirements as job_requirements_table
from jfl_core.db.tables import jobs as jobs_table
from jfl_core.ids import content_hash
from jfl_core.ids import job_id as derive_job_id
from jfl_core.models import (
    Application,
    ApplicationDetail,
    ApplicationEvent,
    ApplicationExtraction,
    ApplicationStatus,
    ExtractionErrorCode,
    ExtractionInput,
    ExtractionStatus,
    JobRequirement,
)
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
    applications_table.c.extraction_status,
    applications_table.c.extraction_error_code,
    applications_table.c.extracted_at,
    applications_table.c.title_is_provisional,
    applications_table.c.archived_at,
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
        extraction_status=row.extraction_status,
        extraction_error_code=row.extraction_error_code,
        extracted_at=row.extracted_at,
        title_is_provisional=row.title_is_provisional,
        archived_at=row.archived_at,
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
        title_is_provisional: bool = False,
        extraction_status: ExtractionStatus = "none",
    ) -> Application:
        """Add an application, and write the "added" event that opens its
        timeline. `raw_job_text`, if given, is stored verbatim as a `jobs` row
        (see the module docstring) and linked via `job_id`; it is never parsed
        here.

        `title_is_provisional` says the title is a placeholder this app derived
        rather than the user's own words, and `extraction_status="pending"` says
        a task has been (or is about to be) enqueued to read the ad properly.
        Both default off, so the CLI-era callers keep their old behaviour.
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
                title_is_provisional=title_is_provisional,
                extraction_status=extraction_status,
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

    def list_applications(
        self, *, status: ApplicationStatus | None = None, archived: bool = False
    ) -> list[Application]:
        """Most recently updated first -- the primary screen's ordering.

        Live applications by default; `archived=True` lists only archived ones.
        Anything that later feeds an owner's application history to a model must
        read through this default, so an archived test entry is never presented as
        part of their real record.
        """
        query = (
            select(*_APPLICATION_COLUMNS)
            .where(applications_table.c.user_id == self._user_id)
            .where(
                applications_table.c.archived_at.is_not(None)
                if archived
                else applications_table.c.archived_at.is_(None)
            )
            # `created_at` and `id` break ties, and they are not decoration.
            # Postgres `now()` is transaction-start time, so two rows written in
            # one transaction share a timestamp exactly; ordering by `updated_at`
            # alone then returns whatever the executor prefers, and the list
            # reorders itself between page loads for no visible reason.
            .order_by(
                applications_table.c.updated_at.desc(),
                applications_table.c.created_at.desc(),
                applications_table.c.id.desc(),
            )
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

    def archive(self, application_id: uuid.UUID) -> Application:
        """Take an application off the owner's lists. Status and timeline are untouched."""
        return self._set_archived(application_id, archived=True)

    def unarchive(self, application_id: uuid.UUID) -> Application:
        """Restore an archived application to the owner's lists."""
        return self._set_archived(application_id, archived=False)

    def _set_archived(self, application_id: uuid.UUID, *, archived: bool) -> Application:
        row = self._conn.execute(
            update(applications_table)
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
            .values(
                archived_at=func.now() if archived else None,
                # Explicitly keep `updated_at`, overriding the column's onupdate.
                # Archiving is housekeeping, not progress on the application, so it
                # must not reorder the list or show as the application's latest
                # activity.
                updated_at=applications_table.c.updated_at,
            )
            .returning(*_APPLICATION_COLUMNS)
        ).first()
        if row is None:
            raise ApplicationNotFoundError(application_id)
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

    # -- extraction (slice B3) ---------------------------------------------
    #
    # Four methods, one per moment: asked for, picked up, finished, failed.
    # Plus one read for the panel that shows the state.

    def request_extraction(self, application_id: uuid.UUID) -> bool:
        """Mark this application's ad as waiting to be read. Returns False if
        there is no ad to read, in which case nothing is written and the caller
        must not enqueue a task.

        Separate from enqueueing on purpose: the row and the task row are two
        writes and the second can fail, so the state the user sees is set by the
        same transaction that decides whether there is anything to do.
        """
        row = self._conn.execute(
            update(applications_table)
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
                applications_table.c.job_id.is_not(None),
            )
            .values(extraction_status="pending", extraction_error_code=None)
            .returning(applications_table.c.id)
        ).first()
        return row is not None

    def claim_extraction(self, application_id: uuid.UUID) -> ExtractionInput | None:
        """The worker's read: the ad text to extract from, or None if there is
        nothing to do.

        None on any of three cases, all of which mean "do not call the model":
        no such application for this user, no ad stored against it, or an
        extraction that has already succeeded. The last is the one that matters
        -- a redelivered task must not spend the user's money a second time on
        an answer that is already in the database.
        """
        row = self._conn.execute(
            select(applications_table.c.job_id, jobs_table.c.raw_text)
            .select_from(applications_table.join(jobs_table))
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
                applications_table.c.extraction_status != "done",
            )
        ).first()
        if row is None:
            return None
        return ExtractionInput(
            application_id=application_id, job_id=row.job_id, raw_text=row.raw_text
        )

    def finish_extraction(
        self, application_id: uuid.UUID, *, title: str | None, employer: str | None
    ) -> None:
        """Record a successful read, and fold what it found into the row --
        but only into fields the user has not filled in themselves.

        Two UPDATEs rather than one because they have different WHERE clauses,
        and the WHERE clause is where the guarantee lives: a title is replaced
        only while it is provisional, an employer only while it is NULL.
        """
        if title and title.strip():
            self._conn.execute(
                update(applications_table)
                .where(
                    applications_table.c.id == application_id,
                    applications_table.c.user_id == self._user_id,
                    applications_table.c.title_is_provisional.is_(True),
                )
                .values(title=title.strip(), title_is_provisional=False)
            )
        if employer and employer.strip():
            self._conn.execute(
                update(applications_table)
                .where(
                    applications_table.c.id == application_id,
                    applications_table.c.user_id == self._user_id,
                    applications_table.c.employer.is_(None),
                )
                .values(employer=employer.strip())
            )
        self._conn.execute(
            update(applications_table)
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
            .values(extraction_status="done", extraction_error_code=None, extracted_at=func.now())
        )

    def fail_extraction(self, application_id: uuid.UUID, code: ExtractionErrorCode) -> None:
        """Record that the read failed, as a code and never as a message.

        The caller is holding the user's decrypted API key while it calls this.
        A code from a closed set cannot carry one; a formatted exception can.
        """
        self._conn.execute(
            update(applications_table)
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
            .values(extraction_status="failed", extraction_error_code=code)
        )

    def get_extraction(self, application_id: uuid.UUID) -> ApplicationExtraction | None:
        """Everything the extraction panel renders, in one call. None if the
        application is not this user's -- same answer as "does not exist".
        """
        row = self._conn.execute(
            select(
                applications_table.c.job_id,
                applications_table.c.extraction_status,
                applications_table.c.extraction_error_code,
                applications_table.c.extracted_at,
                jobs_table.c.employer,
                jobs_table.c.title,
                jobs_table.c.location,
            )
            .select_from(applications_table.outerjoin(jobs_table))
            .where(
                applications_table.c.id == application_id,
                applications_table.c.user_id == self._user_id,
            )
        ).first()
        if row is None:
            return None

        requirements: list[JobRequirement] = []
        if row.job_id is not None:
            requirement_rows = self._conn.execute(
                select(
                    job_requirements_table.c.id,
                    job_requirements_table.c.user_id,
                    job_requirements_table.c.job_id,
                    job_requirements_table.c.ordinal,
                    job_requirements_table.c.text,
                    job_requirements_table.c.necessity,
                )
                .where(
                    job_requirements_table.c.job_id == row.job_id,
                    job_requirements_table.c.user_id == self._user_id,
                )
                .order_by(job_requirements_table.c.ordinal.asc())
            ).all()
            requirements = [
                JobRequirement(
                    id=r.id,
                    user_id=r.user_id,
                    job_id=r.job_id,
                    ordinal=r.ordinal,
                    text=r.text,
                    necessity=r.necessity,
                )
                for r in requirement_rows
            ]

        return ApplicationExtraction(
            application_id=application_id,
            status=row.extraction_status,
            error_code=row.extraction_error_code,
            extracted_at=row.extracted_at,
            has_job_ad=row.job_id is not None,
            employer=row.employer,
            title=row.title,
            location=row.location,
            requirements=requirements,
        )

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
