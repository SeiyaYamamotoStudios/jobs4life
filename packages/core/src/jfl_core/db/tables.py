"""Schema definition. SQLAlchemy Core tables; no ORM session machinery.

Conventions used throughout:
  * every table carries `user_id` (constant locally, real tenant key later)
  * status/kind columns are TEXT + CHECK, not PG ENUM -- CHECKs are alterable in
    one migration, enums are not
  * timestamps are timestamptz, defaulted server-side
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID

# Single embedding space for both the GPU and CPU implementations: same model,
# different device. Keeps the pgvector column a fixed width.
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_N_name)s",
        "uq": "uq_%(table_name)s_%(column_0_N_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s",
        "pk": "pk_%(table_name)s",
    }
)


def _ts(name: str, **kw: object) -> Column:
    return Column(name, TIMESTAMP(timezone=True), **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Corpus: the grounding store. Rebuildable from corpus/*.md at any time.
# --------------------------------------------------------------------------

documents = Table(
    "documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("user_id", Text, nullable=False),
    Column("path", Text, nullable=False),  # relative to corpus/
    Column("title", Text),
    Column("content_hash", String(64), nullable=False),
    _ts("first_seen_at", nullable=False, server_default=func.now()),
    _ts("last_seen_at", nullable=False, server_default=func.now()),
    _ts("retired_at"),  # set when the file disappears; rows are never deleted
    UniqueConstraint("user_id", "path"),
)

spans = Table(
    "spans",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # deterministic, see ids.py
    Column("user_id", Text, nullable=False),
    # NULL for adjudicated spans: they have no source file.
    Column("document_id", UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True),
    Column("provenance", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("section_path", Text),  # heading breadcrumb, e.g. "Kaluza > Platform"
    Column("ordinal", Integer),  # position in doc; ordering only, NOT part of the id
    Column("text", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("char_start", Integer),  # offsets into the source document
    Column("char_end", Integer),
    _ts("first_seen_at", nullable=False, server_default=func.now()),
    _ts("last_seen_at", nullable=False, server_default=func.now()),
    _ts("retired_at"),  # edited-away spans are retired, not deleted: golden-set
    # items and past adjudications keep pointing at a real row
    CheckConstraint("provenance in ('document','adjudicated')", name="provenance"),
    CheckConstraint("kind in ('bullet','paragraph','heading')", name="kind"),
    CheckConstraint(
        "(provenance = 'document') = (document_id is not null)", name="document_id_iff_document"
    ),
    Index("ix_spans_user_id_document_id_ordinal", "user_id", "document_id", "ordinal"),
    Index("ix_spans_user_id_provenance", "user_id", "provenance"),
)

# Sentence offsets within a span, so the citation unit can be tightened from
# span to sentence later without re-labelling the golden set.
span_sentences = Table(
    "span_sentences",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # deterministic from span id + idx
    Column("user_id", Text, nullable=False),
    Column(
        "span_id", UUID(as_uuid=True), ForeignKey("spans.id", ondelete="CASCADE"), nullable=False
    ),
    Column("idx", Integer, nullable=False),
    Column("start_offset", Integer, nullable=False),  # relative to spans.text
    Column("end_offset", Integer, nullable=False),
    UniqueConstraint("span_id", "idx"),
)

span_embeddings = Table(
    "span_embeddings",
    metadata,
    Column(
        "span_id", UUID(as_uuid=True), ForeignKey("spans.id", ondelete="CASCADE"), primary_key=True
    ),
    Column("model", Text, primary_key=True),
    Column("user_id", Text, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
    Column("source_content_hash", String(64), nullable=False),  # detects staleness
    _ts("created_at", nullable=False, server_default=func.now()),
    Index(
        "ix_span_embeddings_embedding",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
)

# --------------------------------------------------------------------------
# Previously-sent documents. A SEPARATE STORE, deliberately.
#
# These must never be reachable from a grounding query. They are not a flag on
# `spans` because a flag is something a future WHERE clause can forget; a
# separate table means the grounding repository has no join that reaches them.
# --------------------------------------------------------------------------

sent_documents = Table(
    "sent_documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("user_id", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("employer", Text),
    Column("role", Text),
    Column("sent_on", Date),
    Column("path", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("text", Text, nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("kind in ('cv','cover_letter','application_answer')", name="kind"),
    UniqueConstraint("user_id", "path"),
)

sent_spans = Table(
    "sent_spans",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("user_id", Text, nullable=False),
    Column(
        "sent_document_id",
        UUID(as_uuid=True),
        ForeignKey("sent_documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
)

# --------------------------------------------------------------------------
# Review queue. Ambiguous cases only; clear passes and clear failures never land here.
# --------------------------------------------------------------------------

review_items = Table(
    "review_items",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("user_id", Text, nullable=False),
    Column("trace_id", UUID(as_uuid=True), nullable=False),
    Column("claim_text", Text, nullable=False),
    Column("source_text", Text, nullable=False),  # the full input the claim came from
    Column("sentence_idx", Integer, nullable=False),
    Column("candidate_span_ids", ARRAY(UUID(as_uuid=True)), nullable=False),
    Column("drift_label", Text),  # taxonomy TBD -- deliberately unconstrained for now
    Column("status", Text, nullable=False, server_default="pending"),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("resolved_at"),
    CheckConstraint("status in ('pending','adjudicated','dismissed')", name="status"),
    Index("ix_review_items_user_id_status_created_at", "user_id", "status", "created_at"),
)

adjudications = Table(
    "adjudications",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("user_id", Text, nullable=False),
    Column(
        "review_item_id",
        UUID(as_uuid=True),
        ForeignKey("review_items.id"),
        nullable=False,
        unique=True,
    ),
    Column("decision", Text, nullable=False),
    Column("note", Text),
    # The adjudicated span written back to the corpus, so the same compression is
    # not re-litigated on the next draft.
    Column("resulting_span_id", UUID(as_uuid=True), ForeignKey("spans.id")),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("decision in ('grounded','not_grounded','rewrite')", name="decision"),
)

# --------------------------------------------------------------------------
# Instrumentation. One row per model call (and per non-model stage worth timing).
# Flat and wide on purpose: this is queried with GROUP BY for the writeup, not
# rendered on a dashboard.
# --------------------------------------------------------------------------

runs = Table(
    "runs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("user_id", Text, nullable=False),
    Column("trace_id", UUID(as_uuid=True), nullable=False),  # one gate invocation
    Column("parent_run_id", UUID(as_uuid=True), ForeignKey("runs.id")),
    Column("component", Text, nullable=False),  # gate | evals | ingest
    Column("stage", Text, nullable=False),  # split | classify | retrieve | extrapolate | baseline
    Column("model", Text),  # NULL for non-model stages
    Column("tokens_in", Integer),
    Column("tokens_out", Integer),
    Column("cache_read_tokens", Integer),
    Column("cache_write_tokens", Integer),
    Column("cost_usd", Numeric(12, 6)),
    Column("latency_ms", Integer),
    Column("outcome", Text, nullable=False),
    Column("error", Text),
    Column("attributes", JSONB),  # stage-specific extras; never queried structurally
    _ts("started_at", nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("outcome in ('ok','error','refused','skipped')", name="outcome"),
    Index("ix_runs_trace_id", "trace_id"),
    Index("ix_runs_user_id_stage_created_at", "user_id", "stage", "created_at"),
)
