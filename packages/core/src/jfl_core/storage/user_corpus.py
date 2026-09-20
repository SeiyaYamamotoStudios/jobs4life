"""The one way a user's own words become corpus, on the server.

Two flows need this and they must not each grow their own: confirming a fact
read out of a CV, and answering profile questions 15 and 16 (where your depth is
genuine, and the gaps that keep coming up). Both are the user stating something
true about themselves, in their words; both must land in exactly one place, in
exactly one shape. Two write paths for one kind of fact is how the same
statement ends up with two span ids, which the claim gate then reads as two
independent pieces of evidence for one thing.

**No model call anywhere in this module, ever.** The text goes in byte for byte,
aside from collapsing embedded newlines so one statement is one span -- the same
trimming `jfl_core.ingest.gap_answers` does and for the same reason. A model
tidying a user's sentence into a neater corpus fact is the ratchet in miniature:
the user is then held to wording they did not choose, by a tool whose whole
claim is that it measures distance from what they actually said.

**Spans are retired, never deleted.** A statement the user supersedes stops
grounding anything (retired spans are excluded from `all_spans`) but its row
stays, so a citation recorded against it still resolves.

**Where this sits against "corpus markdown is the source of truth".** That rule
is about the CLI, whose corpus is a directory of markdown files it re-ingests.
The hosted app has no per-user directory and the question of whether it should
is open (PLAN.md B6). So this writes the span directly -- but it writes the same
*shape* the parser would produce from a markdown bullet (`provenance='document'`,
a real `documents` row, a section path, sentence offsets) and derives the span id
with `jfl_core.ids.span_id` exactly as the parser does. If a per-user markdown
file is materialised later, re-ingesting it reproduces these ids rather than
minting a second set.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, insert, select, update

from jfl_core.db.tables import documents as documents_table
from jfl_core.db.tables import span_sentences as span_sentences_table
from jfl_core.db.tables import spans as spans_table
from jfl_core.ids import content_hash, document_id, sentence_id, span_id
from jfl_core.ingest.parser import split_sentences
from jfl_core.storage.tenancy import TenantScopedRepository

# One document per user, holding everything they have confirmed by hand. Named
# like a corpus file (and prefixed `upload:`, one of `documents.storage_kind`'s
# three values) so that if the hosted corpus ever becomes real markdown, this is
# the file it becomes.
SOURCE_URI = "upload:corpus/confirmed.md"
TITLE = "Confirmed"


def line_text(text: str) -> str:
    """Trim outer whitespace and collapse embedded newlines, so one statement is
    always exactly one span. Nothing else is touched: internal spacing,
    punctuation and any markdown characters the user typed survive as written.
    """
    return " ".join(text.strip().splitlines())


class PostgresUserCorpusRepository(TenantScopedRepository):
    """This user's hand-confirmed corpus text. Bound to one user at
    construction, like every other repository here -- see
    `jfl_core.storage.tenancy`.
    """

    # -- the document these spans belong to -----------------------------------

    def _document_id(self) -> uuid.UUID:
        return document_id(self._user_id, SOURCE_URI)

    def _ensure_document(self) -> uuid.UUID:
        doc_id = self._document_id()
        exists = self._conn.execute(
            select(documents_table.c.id).where(documents_table.c.id == doc_id)
        ).first()
        if exists is None:
            self._conn.execute(
                insert(documents_table).values(
                    id=doc_id,
                    user_id=self._user_id,
                    source_uri=SOURCE_URI,
                    storage_kind="upload",
                    title=TITLE,
                    # The document is a container, not a file with a fixed
                    # content -- its hash is of its identity, and the spans
                    # carry their own hashes.
                    content_hash=content_hash(SOURCE_URI),
                )
            )
        else:
            self._conn.execute(
                update(documents_table)
                .where(documents_table.c.id == doc_id)
                .values(last_seen_at=func.now(), retired_at=None)
            )
        return doc_id

    def _next_ordinal(self, doc_id: uuid.UUID) -> int:
        highest = self._conn.execute(
            select(func.max(spans_table.c.ordinal)).where(
                spans_table.c.user_id == self._user_id,
                spans_table.c.document_id == doc_id,
            )
        ).scalar()
        return 0 if highest is None else int(highest) + 1

    # -- writing --------------------------------------------------------------

    def record(self, text: str, *, section: str) -> uuid.UUID:
        """Put `text` in the corpus under `section`, verbatim, and return its
        span id.

        Idempotent by content: recording a statement that is already live in
        that section returns the existing span rather than a second copy of it,
        because the id is derived from (user, source, section, normalised text)
        -- the parser's own scheme. Recording something that was previously
        retired revives it, which is the right answer for a user who rejects a
        fact and then brings it back.

        Raises `ValueError` on blank text: silence is not a corpus fact.
        """
        line = line_text(text)
        if not line:
            raise ValueError("refusing to record an empty corpus statement")

        doc_id = self._ensure_document()
        new_id = span_id(self._user_id, SOURCE_URI, section, line, occurrence=0)
        existing = self._conn.execute(
            select(spans_table.c.id).where(spans_table.c.id == new_id)
        ).first()
        if existing is None:
            self._conn.execute(
                insert(spans_table).values(
                    id=new_id,
                    user_id=self._user_id,
                    document_id=doc_id,
                    provenance="document",
                    kind="bullet",
                    section_path=section,
                    ordinal=self._next_ordinal(doc_id),
                    text=line,
                    content_hash=content_hash(line),
                )
            )
        else:
            self._conn.execute(
                update(spans_table)
                .where(spans_table.c.id == new_id)
                .values(last_seen_at=func.now(), retired_at=None)
            )
        self._write_sentences(new_id, line)
        return new_id

    def _write_sentences(self, span: uuid.UUID, line: str) -> None:
        """Sentence offsets, re-derived rather than diffed -- they are fully
        determined by the text, exactly as in `jfl_core.storage.postgres`.
        """
        self._conn.execute(
            delete(span_sentences_table).where(span_sentences_table.c.span_id == span)
        )
        offsets = split_sentences(line)
        if not offsets:
            return
        self._conn.execute(
            insert(span_sentences_table),
            [
                {
                    "id": sentence_id(span, idx),
                    "user_id": self._user_id,
                    "span_id": span,
                    "idx": idx,
                    "start_offset": start,
                    "end_offset": end,
                }
                for idx, (start, end) in enumerate(offsets)
            ],
        )

    def replace_section(self, section: str, texts: Sequence[str]) -> list[uuid.UUID]:
        """Make `texts` exactly what this section holds, retiring whatever else
        was live there. Returns the surviving span ids, in the order given.

        This is what a re-answered profile question needs: the user's newer
        words replace their older ones and the older ones stop grounding
        anything. Passing an empty sequence clears the section -- an answer the
        user deleted must not go on being cited at them.
        """
        kept: list[uuid.UUID] = []
        for text in texts:
            if line_text(text):
                kept.append(self.record(text, section=section))

        stmt = update(spans_table).where(
            spans_table.c.user_id == self._user_id,
            spans_table.c.document_id == self._document_id(),
            spans_table.c.section_path == section,
            spans_table.c.retired_at.is_(None),
        )
        if kept:
            stmt = stmt.where(spans_table.c.id.notin_(kept))
        self._conn.execute(stmt.values(retired_at=func.now()))
        return kept

    def retire(self, span: uuid.UUID) -> bool:
        """Stop one span grounding anything. The row stays; only `retired_at` is
        set. Returns False if this user has no such live span.
        """
        result = self._conn.execute(
            update(spans_table)
            .where(
                spans_table.c.id == span,
                spans_table.c.user_id == self._user_id,
                spans_table.c.retired_at.is_(None),
            )
            .values(retired_at=func.now())
        )
        return result.rowcount > 0

    # -- reading --------------------------------------------------------------

    def texts_in(self, section: str) -> list[str]:
        """The live statements in one section, in the order they were recorded.
        Reading, for a page or a test -- grounding reads the corpus through
        `GroundingRepository`, not through here.
        """
        rows = self._conn.execute(
            select(spans_table.c.text)
            .where(
                spans_table.c.user_id == self._user_id,
                spans_table.c.document_id == self._document_id(),
                spans_table.c.section_path == section,
                spans_table.c.retired_at.is_(None),
            )
            .order_by(spans_table.c.ordinal)
        ).all()
        return [row.text for row in rows]

    def live_count(self) -> int:
        """How many statements this user has confirmed by hand."""
        return int(
            self._conn.execute(
                select(func.count())
                .select_from(spans_table)
                .where(
                    spans_table.c.user_id == self._user_id,
                    spans_table.c.document_id == self._document_id(),
                    spans_table.c.retired_at.is_(None),
                )
            ).scalar_one()
        )


__all__ = ["SOURCE_URI", "TITLE", "PostgresUserCorpusRepository", "line_text"]
