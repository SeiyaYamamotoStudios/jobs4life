"""Postgres implementation of the repository interfaces. SQLAlchemy Core, sync.

This is the only module allowed to hold SQL/SQLAlchemy-Core query building --
`jfl_core.repositories` defines what callers may ask for, this defines how.

The repository takes a `Connection`, not an `Engine`, and never opens or
commits a transaction itself: the caller owns the transaction boundary (a
`with engine.begin():` block in the CLI, an already-open transaction in a
test). That is what lets integration tests use the same rollback-per-test
fixture as test_schema_integration.py -- ingestion writes on the test's own
connection, inside its transaction, so tearing that down undoes everything.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Connection

from jfl_core.db.tables import documents as documents_table
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import span_sentences
from jfl_core.db.tables import spans as spans_table
from jfl_core.ids import sentence_id
from jfl_core.models import RunRecord, Span, SpanCandidate


class PostgresIngestRepository:
    """`IngestRepository` against Postgres. Ids are precomputed and deterministic
    (see ids.py), so "upsert" here is a plain existence check plus insert-or-update
    rather than an ON CONFLICT dance.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def upsert_document(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        source_uri: str,
        title: str | None,
        content_hash: str,
    ) -> bool:
        exists = self._conn.execute(
            select(documents_table.c.id).where(documents_table.c.id == document_id)
        ).first()
        if exists is None:
            self._conn.execute(
                insert(documents_table).values(
                    id=document_id,
                    user_id=user_id,
                    source_uri=source_uri,
                    storage_kind="local_file",
                    title=title,
                    content_hash=content_hash,
                )
            )
            return True
        self._conn.execute(
            update(documents_table)
            .where(documents_table.c.id == document_id)
            .values(
                title=title,
                content_hash=content_hash,
                last_seen_at=func.now(),
                retired_at=None,  # a file that reappears is live again
            )
        )
        return False

    def upsert_span(self, span: Span) -> bool:
        exists = self._conn.execute(
            select(spans_table.c.id).where(spans_table.c.id == span.id)
        ).first()
        if exists is None:
            self._conn.execute(
                insert(spans_table).values(
                    id=span.id,
                    user_id=span.user_id,
                    document_id=span.document_id,
                    provenance=span.provenance,
                    kind=span.kind,
                    section_path=span.section_path,
                    ordinal=span.ordinal,
                    text=span.text,
                    content_hash=span.content_hash,
                    char_start=span.char_start,
                    char_end=span.char_end,
                )
            )
            created = True
        else:
            self._conn.execute(
                update(spans_table)
                .where(spans_table.c.id == span.id)
                .values(
                    section_path=span.section_path,
                    ordinal=span.ordinal,
                    text=span.text,
                    content_hash=span.content_hash,
                    char_start=span.char_start,
                    char_end=span.char_end,
                    last_seen_at=func.now(),
                    retired_at=None,  # a span that reappears is live again
                )
            )
            created = False

        # Sentences are fully determined by span text, so re-derive rather than
        # diff: delete what was there and reinsert what parsing produced now.
        self._conn.execute(delete(span_sentences).where(span_sentences.c.span_id == span.id))
        if span.sentences:
            self._conn.execute(
                insert(span_sentences),
                [
                    {
                        "id": sentence_id(span.id, sentence.idx),
                        "user_id": span.user_id,
                        "span_id": span.id,
                        "idx": sentence.idx,
                        "start_offset": sentence.start_offset,
                        "end_offset": sentence.end_offset,
                    }
                    for sentence in span.sentences
                ],
            )
        return created

    def retire_missing_documents(self, user_id: uuid.UUID, seen: set[uuid.UUID]) -> int:
        stmt = update(documents_table).where(
            documents_table.c.user_id == user_id,
            documents_table.c.retired_at.is_(None),
        )
        if seen:
            stmt = stmt.where(documents_table.c.id.notin_(seen))
        result = self._conn.execute(stmt.values(retired_at=func.now()))
        return result.rowcount

    def retire_missing_spans(self, user_id: uuid.UUID, seen: set[uuid.UUID]) -> int:
        stmt = update(spans_table).where(
            spans_table.c.user_id == user_id,
            spans_table.c.provenance == "document",
            spans_table.c.retired_at.is_(None),
        )
        if seen:
            stmt = stmt.where(spans_table.c.id.notin_(seen))
        result = self._conn.execute(stmt.values(retired_at=func.now()))
        return result.rowcount


def _row_to_span(row: Any) -> Span:
    # `Any`: a SQLAlchemy Core `Row`'s attributes are dynamic, not statically typed.
    # `sentences` is left empty: no current reader needs sentence offsets, and
    # populating it would mean an extra join per span for a baseline gate that
    # only formats span text into a prompt. Revisit if a caller needs it.
    return Span(
        id=row.id,
        user_id=row.user_id,
        document_id=row.document_id,
        provenance=row.provenance,
        kind=row.kind,
        section_path=row.section_path,
        ordinal=row.ordinal,
        text=row.text,
        content_hash=row.content_hash,
        char_start=row.char_start,
        char_end=row.char_end,
    )


class PostgresGroundingRepository:
    """`GroundingRepository` against Postgres.

    `search` is unimplemented: retrieval is unused in v1 (see CLAUDE.md's decisions
    log, "Retrieval unused in v1") -- the gate stuffs the whole corpus into context,
    so nothing calls it yet. `span_embeddings` stays empty until that changes.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None:
        row = self._conn.execute(
            select(spans_table).where(spans_table.c.id == span_id, spans_table.c.user_id == user_id)
        ).first()
        return _row_to_span(row) if row is not None else None

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]:
        raise NotImplementedError("retrieval is unused in v1 -- see CLAUDE.md decisions log")

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        """All spans for the baseline gate: both provenances, retired ones excluded
        by default. Ordered by document then ordinal so the prompt reads in document
        order rather than insertion order.
        """
        stmt = select(spans_table).where(spans_table.c.user_id == user_id)
        if not include_retired:
            stmt = stmt.where(spans_table.c.retired_at.is_(None))
        stmt = stmt.order_by(spans_table.c.document_id, spans_table.c.ordinal)
        return [_row_to_span(row) for row in self._conn.execute(stmt).all()]

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID:
        self._conn.execute(
            insert(spans_table).values(
                id=span.id,
                user_id=user_id,
                document_id=None,
                provenance="adjudicated",
                kind=span.kind,
                section_path=span.section_path,
                ordinal=span.ordinal,
                text=span.text,
                content_hash=span.content_hash,
                char_start=None,
                char_end=None,
            )
        )
        return span.id


class PostgresRunRepository:
    """`RunRepository` against Postgres. One insert per call; the caller owns the
    transaction, same convention as the other repositories in this module.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def record(self, run: RunRecord) -> None:
        self._conn.execute(
            insert(runs_table).values(
                id=run.id,
                user_id=run.user_id,
                trace_id=run.trace_id,
                parent_run_id=run.parent_run_id,
                component=run.component,
                stage=run.stage,
                model=run.model,
                tokens_in=run.tokens_in,
                tokens_out=run.tokens_out,
                cache_read_tokens=run.cache_read_tokens,
                cache_write_tokens=run.cache_write_tokens,
                cost_usd=run.cost_usd,
                latency_ms=run.latency_ms,
                outcome=run.outcome,
                error=run.error,
                attributes=run.attributes,
                started_at=run.started_at,
            )
        )
