"""The sent-document store, and what CV intake does with it -- slice B6.

**This store is never reachable from a grounding query, and that is structural
rather than careful.** The tables here (`sent_documents`, `sent_spans`,
`cv_extractions`) are separate from `spans`, this repository is separate from
`GroundingRepository`, and there is no method here that returns a corpus span
and none there that returns a sent document. A flag on `spans` would have been
a flag someone's WHERE clause could forget; the isolation is in the interface
instead. See CLAUDE.md's architectural constraints, and "generated documents
influence form, never truth": a previous CV is legitimate for structure, voice
and consistency checking, and is never evidence.

A CV arrives verbatim -- `add_cv` stores exactly the bytes the user uploaded or
pasted, with no cleanup, because the whole slice downstream quotes lines from
it back to them and a line we tidied is a line they did not write.

**Re-uploading the same CV does not duplicate it.** Identity is the content
hash: the same file under a different name is the same CV, and the same name
with different content is a different one (so the `path` a row is stored under
carries a hash prefix -- see `cv_path`). This matters more than it looks: the
owner has thirty-three generated CVs and re-uploads are the normal case, and a
duplicate row would mean a second model call, charged to the user, for a
document already read.

`cv_extractions` is the state of the one model call per CV. One row per CV,
unique on `sent_document_id`, which is what lets a redelivered task ask "is
this already done?" with a single read instead of extracting -- and charging --
twice.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from jfl_core.db.tables import cv_extractions as extractions_table
from jfl_core.db.tables import sent_documents as documents_table
from jfl_core.db.tables import sent_spans as spans_table
from jfl_core.ids import content_hash
from jfl_core.models import CvExtractionErrorCode, StoredCv
from jfl_core.storage.tenancy import TenantScopedRepository

CV_KIND = "cv"


def cv_path(filename: str, text: str) -> str:
    """The `sent_documents.path` a CV is stored under.

    `(user_id, path)` is unique, so the path has to distinguish two different
    CVs that happen to share a filename -- "cv.md" is not a rare name. Prefixing
    the content hash does that, and makes the same bytes under the same name
    resolve to the same row rather than colliding.
    """
    safe = " ".join(filename.split()).replace("/", "-").strip() or "pasted"
    return f"cv/{content_hash(text)[:12]}-{safe}"


def _split_units(text: str) -> list[str]:
    """`sent_spans` rows: one per non-blank line, in order.

    Deliberately not `jfl_core.ingest.parser.parse_document`: that produces
    `Span`s carrying a corpus `document_id` and a grounding-shaped identity,
    and building those for a document that must never ground anything would be
    inviting exactly the mistake this store exists to prevent. A line is the
    unit a later consistency comparison wants anyway.
    """
    return [line.strip() for line in text.splitlines() if line.strip()]


def _to_cv(row: Any) -> StoredCv:
    return StoredCv(
        id=row.id,
        path=row.path,
        title=row.role,
        content_hash=row.content_hash,
        created_at=row.created_at,
        extraction_status=row.status or "pending",
        extraction_error_code=row.error_code,
        facts_proposed=row.facts_proposed or 0,
        length_chars=row.length_chars or 0,
    )


_LISTING = (
    documents_table.c.id,
    documents_table.c.path,
    documents_table.c.role,
    documents_table.c.content_hash,
    documents_table.c.created_at,
    func.length(documents_table.c.text).label("length_chars"),
    extractions_table.c.status,
    extractions_table.c.error_code,
    extractions_table.c.facts_proposed,
)

_LISTING_FROM = documents_table.outerjoin(
    extractions_table, extractions_table.c.sent_document_id == documents_table.c.id
)


class PostgresSentDocumentRepository(TenantScopedRepository):
    """This user's sent documents, and no one else's. Reads and writes the
    sent-document store only -- it touches no table the claim gate can see.
    """

    def add_cv(self, *, filename: str, text: str, title: str | None = None) -> StoredCv:
        """Store a CV verbatim, or return the one already stored for this text.

        Idempotent on content: re-uploading a CV this user already has stores
        nothing new. A caller deciding whether to queue a read should ask
        `find_by_content` first rather than infer it from what comes back here
        -- a CV uploaded twice before the first read finished would otherwise
        get two tasks and, at-least-once delivery aside, two chances to be
        charged for.
        """
        existing = self.find_by_content(text)
        if existing is not None:
            return existing

        digest = content_hash(text)
        document_id = uuid.uuid4()
        self._conn.execute(
            insert(documents_table).values(
                id=document_id,
                user_id=self._user_id,
                kind=CV_KIND,
                role=title,
                path=cv_path(filename, text),
                content_hash=digest,
                text=text,
            )
        )
        units = _split_units(text)
        if units:
            self._conn.execute(
                insert(spans_table),
                [
                    {
                        "id": uuid.uuid4(),
                        "user_id": self._user_id,
                        "sent_document_id": document_id,
                        "ordinal": ordinal,
                        "text": unit,
                        "content_hash": content_hash(unit),
                    }
                    for ordinal, unit in enumerate(units)
                ],
            )
        self._conn.execute(
            pg_insert(extractions_table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                sent_document_id=document_id,
                status="pending",
            )
            .on_conflict_do_nothing(index_elements=["sent_document_id"])
        )
        stored = self._conn.execute(
            select(*_LISTING).select_from(_LISTING_FROM).where(documents_table.c.id == document_id)
        ).one()
        return _to_cv(stored)

    def find_by_content(self, text: str) -> StoredCv | None:
        """The CV this user already has for exactly these bytes, or None.

        Identity is the content hash, so the same file under a different name
        is the same CV and the same name with different content is a different
        one. Scoped to this user, like everything here: dedupe reaching across
        tenants would let one person's upload silently answer another's.
        """
        row = self._conn.execute(
            select(*_LISTING)
            .select_from(_LISTING_FROM)
            .where(
                documents_table.c.user_id == self._user_id,
                documents_table.c.kind == CV_KIND,
                documents_table.c.content_hash == content_hash(text),
            )
            .limit(1)
        ).first()
        return None if row is None else _to_cv(row)

    def list_cvs(self) -> list[StoredCv]:
        """Newest first. Never returns the CV text -- see `StoredCv`."""
        rows = self._conn.execute(
            select(*_LISTING)
            .select_from(_LISTING_FROM)
            .where(
                documents_table.c.user_id == self._user_id,
                documents_table.c.kind == CV_KIND,
            )
            .order_by(documents_table.c.created_at.desc())
        ).all()
        return [_to_cv(row) for row in rows]

    def get_cv(self, sent_document_id: uuid.UUID) -> StoredCv | None:
        row = self._conn.execute(
            select(*_LISTING)
            .select_from(_LISTING_FROM)
            .where(
                documents_table.c.id == sent_document_id,
                documents_table.c.user_id == self._user_id,
            )
        ).first()
        return None if row is None else _to_cv(row)

    def cv_text(self, sent_document_id: uuid.UUID) -> str | None:
        """The stored CV, verbatim. Only the extraction handler needs this."""
        row = self._conn.execute(
            select(documents_table.c.text).where(
                documents_table.c.id == sent_document_id,
                documents_table.c.user_id == self._user_id,
            )
        ).first()
        return None if row is None else row.text

    def claim_extraction(self, sent_document_id: uuid.UUID) -> str | None:
        """The CV's text if it still needs reading, None if it does not.

        None covers all three of "no such CV for this user", "no extraction row"
        and "already done" -- every one of which means *do not call the model*,
        and none of which is a failure worth retrying. A `failed` extraction is
        claimable again, because re-reading a CV is a button the user presses
        and it is their money.
        """
        row = self._conn.execute(
            select(documents_table.c.text, extractions_table.c.status)
            .select_from(_LISTING_FROM)
            .where(
                documents_table.c.id == sent_document_id,
                documents_table.c.user_id == self._user_id,
            )
        ).first()
        if row is None or row.status is None or row.status == "done":
            return None
        return str(row.text)

    def finish_extraction(self, sent_document_id: uuid.UUID, *, facts_proposed: int) -> None:
        self._conn.execute(
            update(extractions_table)
            .where(
                extractions_table.c.sent_document_id == sent_document_id,
                extractions_table.c.user_id == self._user_id,
            )
            .values(status="done", error_code=None, facts_proposed=facts_proposed)
        )

    def fail_extraction(self, sent_document_id: uuid.UUID, code: CvExtractionErrorCode) -> None:
        self._conn.execute(
            update(extractions_table)
            .where(
                extractions_table.c.sent_document_id == sent_document_id,
                extractions_table.c.user_id == self._user_id,
            )
            .values(status="failed", error_code=code)
        )
