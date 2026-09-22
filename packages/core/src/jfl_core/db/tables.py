"""Schema definition. SQLAlchemy Core tables; no ORM session machinery.

Conventions used throughout:
  * every table carries `user_id` (constant locally, real tenant key later)
  * status/kind columns are TEXT + CHECK, not PG ENUM -- CHECKs are alterable in
    one migration, enums are not
  * timestamps are timestamptz, defaulted server-side
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
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


def _ts(name: str, **kw: object) -> Column[datetime]:
    return Column(name, TIMESTAMP(timezone=True), **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Accounts. The CLI still runs single-user against the seeded local row; the web
# app (packages/web, slice A) signs users in with Google and scopes every read
# and write to the session's user. Tenancy is enforced by construction, not by a
# WHERE clause -- see jfl_core.storage.tenancy.
# --------------------------------------------------------------------------

# Deterministic, so the local user is the same row on every machine and in
# every test database. uuid5(NS_ROOT, "user:local").
LOCAL_USER_ID = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")

users = Table(
    "users",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    # Google's `sub` claim: the ONLY identity key. Email addresses change hands
    # between people and get reassigned inside a workspace; `sub` is stable and
    # unique forever. Nullable because the seeded local user (and any future
    # non-Google account) has none -- absence must not collide, so lookups filter
    # on a non-null value rather than matching NULL.
    Column("google_sub", Text, unique=True),
    # Presentation only, refreshed from the id token on every login. Never used
    # to find or match a user. Neither NOT NULL nor UNIQUE (migration
    # 6b3ce06d7b4e): Google reassigns an address to a different
    # person, and the old unique constraint turned that reassignment into a
    # hard login failure for the address's new owner -- their upsert clashed
    # with the row still displaying it under the previous owner's `sub`.
    Column("email", Text),
    Column("display_name", Text),
    Column("is_active", Boolean, nullable=False, server_default=text("true")),
    _ts("created_at", nullable=False, server_default=func.now()),
)

# Opaque server-side sessions. Deliberately NOT a JWT: this app holds other
# people's API keys, so revoking a session has to take effect on the next
# request, which a self-contained token cannot promise.
#
# `token_hash` is sha256 of the cookie value, never the value itself -- reading
# this table gives an attacker no usable session. `csrf_token` IS stored in the
# clear because it has to be rendered into a form; on its own it authenticates
# nothing, since a request needs the session cookie too.
sessions = Table(
    "sessions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("token_hash", String(64), nullable=False, unique=True),
    Column("csrf_token", Text, nullable=False),
    Column("user_agent", Text),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("last_seen_at", nullable=False, server_default=func.now()),
    _ts("expires_at", nullable=False),  # rolling: extended by `touch` on use
    Index("ix_sessions_user_id", "user_id"),
    Index("ix_sessions_expires_at", "expires_at"),
)

# Envelope encryption: a per-record data key (`wrapped_dek`) encrypted under a
# master key held outside the database, and the secret encrypted under the DEK.
# Rotating the master key rewraps DEKs without touching ciphertext.
#
# Plaintext must never reach this table, the runs table, a log line, or a
# traceback. See `jfl_core.crypto.envelope` for the seal/unseal pair; a
# repository here only ever moves ciphertext.
user_credentials = Table(
    "user_credentials",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("kind", Text, nullable=False),
    Column("label", Text, nullable=False, server_default=""),
    Column("ciphertext", LargeBinary, nullable=False),  # secret, under the DEK
    Column("nonce", LargeBinary, nullable=False),  # fresh per encryption, never reused
    Column("wrapped_dek", LargeBinary, nullable=False),  # DEK, under the KEK
    Column("dek_nonce", LargeBinary, nullable=False),  # fresh per wrap, never reused
    Column("master_key_id", Text, nullable=False),  # which KEK wrapped the DEK
    # Last four characters of the secret, so the UI can say which key is stored
    # beside a field that can be written but never read back.
    Column("key_hint", Text, nullable=False, server_default=""),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("rotated_at"),
    _ts("last_used_at"),
    CheckConstraint(
        "kind in ('anthropic_api_key','openai_api_key','ats_token','smtp_password')",
        name="kind",
    ),
    UniqueConstraint("user_id", "kind", "label"),
)

# Pluggable intake. The table lands now because ATS JSON, RSS, a forwarded-email
# inbox and manual paste differ enough in shape that retrofitting hurts; the
# postings themselves belong to the intake domain and are not designed yet.
job_sources = Table(
    "job_sources",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("kind", Text, nullable=False),
    Column("name", Text, nullable=False),
    Column("config", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("enabled", Boolean, nullable=False, server_default=text("true")),
    Column("cursor", Text),  # opaque per-kind resume token: etag, last id, date
    _ts("last_polled_at"),
    _ts("last_success_at"),
    Column("consecutive_failures", Integer, nullable=False, server_default=text("0")),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint(
        "kind in ('greenhouse','lever','ashby','workable','rss','forwarded_email','manual')",
        name="kind",
    ),
    UniqueConstraint("user_id", "kind", "name"),
)


# --------------------------------------------------------------------------
# Corpus: the grounding store. Rebuildable from corpus/*.md at any time.
# --------------------------------------------------------------------------

# Where the markdown behind a document actually lives. `hosted` was added with
# CV intake (slice B6): a hosted user has no `corpus/` directory, so "markdown
# is the source of truth, the database is a rebuildable index over it" can only
# stay true if the markdown itself is stored -- see `documents.text` below and
# `jfl_core.corpus_source`.
_DOCUMENT_STORAGE_KINDS = ("local_file", "upload", "paste", "hosted")

documents = Table(
    "documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    # Generalised from a bare path so local files, uploads and pastes share one
    # identity scheme: "file:corpus/cv.md", "upload:<uuid>", "paste:<uuid>".
    Column("source_uri", Text, nullable=False),
    Column("storage_kind", Text, nullable=False),
    Column("title", Text),
    Column("content_hash", String(64), nullable=False),
    # The markdown itself, when this deployment is the only place it exists
    # (storage_kind='hosted'). NULL for a `local_file` document, whose source
    # of truth is a file on the owner's machine and which this table only
    # indexes. Never read by the gate -- spans are what grounding sees; this is
    # here so a hosted corpus document can be re-parsed, shown back to its
    # author, and edited by them.
    Column("text", Text),
    _ts("first_seen_at", nullable=False, server_default=func.now()),
    _ts("last_seen_at", nullable=False, server_default=func.now()),
    _ts("retired_at"),  # set when the source disappears; rows are never deleted
    CheckConstraint(
        "storage_kind in ('" + "','".join(_DOCUMENT_STORAGE_KINDS) + "')",
        name="storage_kind",
    ),
    UniqueConstraint("user_id", "source_uri"),
)

spans = Table(
    "spans",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # deterministic, see ids.py
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    # NULL for adjudicated spans: they have no source file.
    Column("document_id", UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True),
    Column("provenance", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("section_path", Text),  # heading breadcrumb, e.g. "Northwind > Platform"
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
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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
# Generation (domain 2a): jobs, extracted requirements, and per-requirement
# coverage against the corpus. No drafting lives here yet -- see CLAUDE.md's
# build order.
# --------------------------------------------------------------------------

jobs = Table(
    "jobs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # deterministic, see ids.py
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("source", Text, nullable=False),
    Column("employer", Text),
    Column("title", Text),
    Column("location", Text),
    Column("url", Text),
    Column("raw_text", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("source in ('paste','file')", name="source"),
    # Re-pasting the same ad resolves to the same row instead of minting a duplicate.
    UniqueConstraint("user_id", "content_hash"),
)

job_requirements = Table(
    "job_requirements",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # deterministic, see ids.py
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("job_id", UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    # order in the ad; ordering only, NOT part of the id
    Column("ordinal", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("necessity", Text, nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("necessity in ('essential','desirable','unstated')", name="necessity"),
)

# APPEND-ONLY: one row per requirement, per coverage run. Showing coverage
# change before and after a gap answer is the point of this slice, so a re-run
# adds a row rather than overwriting the last one. Query "latest per
# requirement" with `DISTINCT ON (requirement_id) ... ORDER BY requirement_id,
# created_at DESC`.
requirement_coverage = Table(
    "requirement_coverage",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # random: one id per run, not deterministic
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "requirement_id",
        UUID(as_uuid=True),
        ForeignKey("job_requirements.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("trace_id", UUID(as_uuid=True), nullable=False),  # groups one coverage run
    Column("status", Text, nullable=False),
    Column("cited_span_ids", ARRAY(UUID(as_uuid=True)), nullable=False),
    Column("evidence_note", Text, nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("status in ('evidenced','partial','absent','contradicted')", name="status"),
    Index(
        "ix_requirement_coverage_user_id_requirement_id_created_at",
        "user_id",
        "requirement_id",
        "created_at",
    ),
)

gap_questions = Table(
    "gap_questions",
    metadata,
    # Deterministic from requirement_id ALONE (see ids.py): re-running coverage
    # refreshes one stable question per requirement rather than accumulating
    # near-duplicates of it.
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "requirement_id",
        UUID(as_uuid=True),
        ForeignKey("job_requirements.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("question", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="open"),
    Column("answer_text", Text),
    _ts("answered_at"),
    # No FK cascade: an adjudicated span outlives the question that produced it,
    # same as `adjudications.resulting_span_id` above.
    Column("resulting_span_id", UUID(as_uuid=True), ForeignKey("spans.id")),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("status in ('open','answered','dismissed')", name="status"),
    Index("ix_gap_questions_user_id_status_created_at", "user_id", "status", "created_at"),
)


# --------------------------------------------------------------------------
# Generation (domain 2b-core): drafts, anchored on a job, generated from the
# corpus and gated automatically -- see CLAUDE.md's decisions log, "The claim
# gate runs automatically on generated text." Autonomous mode only; the
# interactive gaps-first mode is 2b-full and not built.
# --------------------------------------------------------------------------

drafts = Table(
    "drafts",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # random: one id per draft
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("job_id", UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
    Column("kind", Text, nullable=False),
    Column("text", Text, nullable=False),  # the generated draft, verbatim
    Column("gate_result", JSONB, nullable=False),  # the parsed claim-gate output for this draft
    # Shared with both this draft's `runs` rows (the draft call, stage='draft', and
    # the automatic claim-gate pass, stage='baseline'), so a draft's total cost is
    # one query: SELECT sum(cost_usd) FROM runs WHERE trace_id = drafts.trace_id.
    Column("trace_id", UUID(as_uuid=True), nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint("kind in ('cv_bullets','cover_letter')", name="kind"),
    Index("ix_drafts_user_id_job_id_created_at", "user_id", "job_id", "created_at"),
)


# --------------------------------------------------------------------------
# Review queue. Ambiguous cases only; clear passes and clear failures never land here.
# --------------------------------------------------------------------------

review_items = Table(
    "review_items",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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
# Application tracker (slice A5). This is the wedge: what conversations cannot
# do is remember a running list of applications with real timestamps -- see
# CLAUDE.md's 2026-09-07 decision. Nothing here calls a model.
#
# `applications` holds current state; `application_events` is the append-only
# timeline that state is derived from. A status change writes BOTH -- the row
# is updated and an event is inserted -- because the event log is what later
# slices inject into model context, so it must stay complete, never
# overwritten. See `jfl_core.storage.applications.PostgresApplicationRepository`.
# --------------------------------------------------------------------------

_APPLICATION_STATUSES = (
    "interested",
    "applied",
    "screening",
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
)

# Slice B3: where the background read of the pasted ad has got to. Separate from
# `status` on purpose -- `status` is where the *application* is in the world, and
# an extraction failing has nothing to do with whether the user has applied.
#
# `none` is the honest fourth value: rows added before B3, and rows added with no
# ad text at all, have never had an extraction and are not "pending" one.
_EXTRACTION_STATUSES = ("none", "pending", "done", "failed")

# Why a code and not a message. The failure is written by the worker, which is
# holding the user's decrypted API key three frames up the stack; a free-text
# error column is exactly where a careless `str(exc)` from the SDK ends up. A
# closed set of codes cannot carry a secret, and the wording belongs to the web
# layer anyway, where it can be changed without a migration.
_EXTRACTION_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "no_job_ad",
    "ad_too_long",
    "model_refused",
    "model_error",
    "credential_unreadable",
    # Slice C7: "Track as application" could not read the description off the
    # watched board -- see `jfl_core.models.ExtractionErrorCode`.
    "description_unavailable",
)

applications = Table(
    "applications",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    # Nullable: an application can exist with no linked job (added before a
    # pasted ad, or one that was never pasted at all). SET NULL rather than
    # CASCADE -- losing the ad text should never take the tracked application
    # with it.
    Column("job_id", UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL")),
    Column("title", Text, nullable=False),
    Column("employer", Text),
    Column("url", Text),
    Column("status", Text, nullable=False, server_default="interested"),
    # Free text ("referral", "LinkedIn", "company site"), not a controlled
    # vocabulary yet -- unlike `job_sources.kind`, which is. Domain 3 (intake)
    # is where a real taxonomy belongs; this is a user-typed label.
    Column("source", Text),
    Column("notes", Text),
    # -- slice B3: the background read of the pasted ad -----------------------
    Column("extraction_status", Text, nullable=False, server_default="none"),
    Column("extraction_error_code", Text),
    _ts("extracted_at"),
    # Whether `title` is a placeholder this app derived from the ad's first line
    # rather than something the user typed. It is what makes "never overwrite
    # something the user typed themselves" a fact about the row instead of a
    # guess: extraction may replace a provisional title and may not replace any
    # other kind, and once it has, the title stops being provisional.
    Column("title_is_provisional", Boolean, nullable=False, server_default=text("false")),
    # Soft delete. Archiving takes an application off the owner's lists without
    # touching its status or its event timeline, so a test entry or a duplicate can
    # disappear without being recorded as something the owner did -- "withdrawn" is
    # a real outcome of a real process and must stay reserved for one. NULL means
    # live. Nothing is ever deleted; restoring clears it.
    _ts("archived_at"),
    # Slice C7: the watched-board job "Track as application" was pressed on, or
    # NULL for an application added by paste. SET NULL rather than CASCADE --
    # losing the board, or the job falling off it, must never take the tracked
    # application down with it. Provenance only: nothing here reads it to
    # decide what the application is allowed to do.
    Column(
        "board_job_id",
        UUID(as_uuid=True),
        ForeignKey("board_jobs.id", ondelete="SET NULL"),
    ),
    _ts("created_at", nullable=False, server_default=func.now()),
    # `onupdate` is a Core-level default: SQLAlchemy adds `updated_at = now()`
    # to any UPDATE built from this table that does not itself set the column
    # -- which is every status change and every notes edit, so the repository
    # never has to remember to touch it by hand.
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_APPLICATION_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "extraction_status in ('" + "','".join(_EXTRACTION_STATUSES) + "')",
        name="extraction_status",
    ),
    CheckConstraint(
        "extraction_error_code is null or extraction_error_code in ('"
        + "','".join(_EXTRACTION_ERROR_CODES)
        + "')",
        name="extraction_error_code",
    ),
    Index("ix_applications_user_id_updated_at", "user_id", "updated_at"),
    Index("ix_applications_user_id_status", "user_id", "status"),
    # "Is this board job already tracked?", for rendering the button on /jobs
    # and a board's page across every open job in one query.
    Index("ix_applications_user_id_board_job_id", "user_id", "board_job_id"),
)

# APPEND-ONLY: never updated or deleted. `from_status` is NULL on the row
# created alongside the application itself, so the timeline includes "added"
# as well as every transition after it.
application_events = Table(
    "application_events",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "application_id",
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("from_status", Text),
    Column("to_status", Text, nullable=False),
    Column("note", Text),
    _ts("occurred_at", nullable=False, server_default=func.now()),
    _ts("created_at", nullable=False, server_default=func.now()),
    CheckConstraint(
        "to_status in ('" + "','".join(_APPLICATION_STATUSES) + "')",
        name="to_status",
    ),
    Index(
        "ix_application_events_user_id_application_id_occurred_at",
        "user_id",
        "application_id",
        "occurred_at",
    ),
)

# --------------------------------------------------------------------------
# Application questions -- two equal paths, "check my answer" and "draft one
# for me". See CLAUDE.md's 2026-09-18 decision ("check my answer" / "draft one
# for me" side by side, advising, never prescribing) and NEXT.md's task 4.
#
# `application_questions` is the question itself, written once. Every attempt
# to answer it -- typed by the user and checked, or generated and gated -- is a
# fresh row in `application_question_answers`, never an UPDATE to a previous
# one: the same append-only rule `profiles` follows, and for the same
# reason -- a tool whose whole claim is measuring distance from what someone
# actually said must never let that record be silently edited out from under
# them. `kind` says which path produced the row; `answer_text` is the user's
# own words for `kind='user'` and starts `''` for `kind='draft'`, filled in
# once the model has written something.
#
# Cost is not stored on the row. `trace_id` groups the model call(s) one
# attempt made -- one for a check (the assessment call; the claim gate's own
# call already writes its own `runs` row automatically), two for a draft (the
# draft call, then the automatic gate pass) -- the same pattern
# `jfl_core.models.Draft.trace_id` uses, so the per-attempt cost is one query:
# `SELECT sum(cost_usd) FROM runs WHERE trace_id = ...`.
# --------------------------------------------------------------------------

application_questions = Table(
    "application_questions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "application_id",
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("question_text", Text, nullable=False),
    _ts("created_at", nullable=False, server_default=func.now()),
    Index(
        "ix_application_questions_user_id_application_id",
        "user_id",
        "application_id",
    ),
)

_APPLICATION_QUESTION_ANSWER_KINDS = ("user", "draft")

_APPLICATION_QUESTION_ANSWER_STATUSES = ("pending", "done", "failed")

# A subset of `_EXTRACTION_ERROR_CODES`'s shape -- the ones this pair of calls
# can actually produce. No `no_job_ad` / `ad_too_long` (there is no ad here).
# `no_requirements` is `draft_application_answer`'s own precondition failure:
# there is nothing job-specific to draft from until the ad has been read.
_APPLICATION_QUESTION_ANSWER_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
    "no_requirements",
)

application_question_answers = Table(
    "application_question_answers",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "question_id",
        UUID(as_uuid=True),
        ForeignKey("application_questions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", Text, nullable=False),
    Column("answer_text", Text, nullable=False, server_default=""),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("error_code", Text),
    # jfl_gate.schema.GateOutput.model_dump(), same convention as
    # jfl_core.models.Draft.gate_result -- a plain JSONB dict here rather than
    # typed against that model, since jfl_core has no dependency on jfl_gate
    # (see CLAUDE.md's architectural constraints). NULL until `status='done'`.
    Column("gate_result", JSONB),
    # {"assessment": ..., "gaps": ...} -- jfl_generate.schema.AssessAnswerOutput
    # dumped. Only `kind='user'` ever populates this: a draft is judged by the
    # gate the same as any generated text, and asking the model to assess its
    # own draft against the question it was just given would be circular.
    Column("assessment", JSONB),
    Column("model", Text),
    Column("trace_id", UUID(as_uuid=True)),
    # `clock_timestamp()`, not `now()` -- same reasoning as
    # `profiles.created_at`: two versions written in the same transaction must
    # still order correctly, since "the latest row" is what "the current
    # answer" means.
    _ts("created_at", nullable=False, server_default=text("clock_timestamp()")),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "kind in ('" + "','".join(_APPLICATION_QUESTION_ANSWER_KINDS) + "')",
        name="kind",
    ),
    CheckConstraint(
        "status in ('" + "','".join(_APPLICATION_QUESTION_ANSWER_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('"
        + "','".join(_APPLICATION_QUESTION_ANSWER_ERROR_CODES)
        + "')",
        name="error_code",
    ),
    Index(
        "ix_application_question_answers_question_id_created_at",
        "question_id",
        "created_at",
    ),
)


# --------------------------------------------------------------------------
# Background work (slice B1). A claim-gate call takes ~2 minutes and an
# extraction is not much quicker, so nothing that slow may run inside a request:
# the form returns immediately and a worker container picks the work up here.
#
# Claiming is `SELECT ... FOR UPDATE SKIP LOCKED` (see
# `jfl_core.storage.tasks.PostgresTaskQueue`), which is why the two partial
# indexes below exist -- they are the claim query and the stale-reclaim query,
# nothing else.
#
# **Delivery is at-least-once, not exactly-once, and no schema can change
# that.** A worker killed between claiming a row and finishing it leaves the row
# in `running` forever; the reclaim query finds those by `started_at` and puts
# them back. A handler must therefore tolerate being run twice.
#
# `kind` carries NO check constraint, unlike every other kind/status column
# here. The set of kinds is a code-level registry in `jfl_worker`, grows with
# every slice, and a task whose kind no worker recognises is simply never
# claimed -- so a CHECK would buy nothing and cost a migration per handler.
# `status` does carry one: those four values are the state machine itself.
# --------------------------------------------------------------------------

_TASK_STATUSES = ("pending", "running", "succeeded", "failed")

tasks = Table(
    "tasks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    # Every task belongs to someone, including maintenance work: the worker
    # attributes system tasks (session purging) to the seeded local user, so
    # there is no second, unowned code path into this table.
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("kind", Text, nullable=False),
    # Arguments only -- ids, flags, a job id. NEVER a secret: an API key lives
    # encrypted in `user_credentials` and is fetched by the handler from there,
    # because a payload is read back by the worker, shown in admin queries and
    # quoted into error messages.
    Column("payload", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("status", Text, nullable=False, server_default="pending"),
    # Incremented when the row is CLAIMED, not when it fails -- a task that kills
    # the worker outright still burns an attempt, so a poison pill cannot loop
    # forever.
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("max_attempts", Integer, nullable=False, server_default=text("3")),
    # Why the last attempt failed. Kept on success too: it is the record of what
    # went wrong before the retry that worked.
    Column("last_error", Text),
    # Not before this. Backoff between attempts is written here, so a failing
    # task waits instead of spinning.
    _ts("scheduled_at", nullable=False, server_default=func.now()),
    _ts("started_at"),  # when the current/last attempt was claimed
    _ts("finished_at"),  # set on 'succeeded' and on terminal 'failed' only
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_TASK_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint("attempts >= 0 and max_attempts >= 1", name="attempt_counts"),
    # The claim query, exactly: status = 'pending' and scheduled_at <= now() and
    # kind = any(...), ordered by scheduled_at. Partial, because pending rows are
    # a small and shrinking minority of a table that keeps its history.
    Index(
        "ix_tasks_pending_scheduled_at",
        "scheduled_at",
        "kind",
        postgresql_where=text("status = 'pending'"),
    ),
    # The reclaim query: rows stuck in 'running' past the visibility timeout.
    Index(
        "ix_tasks_running_started_at",
        "started_at",
        postgresql_where=text("status = 'running'"),
    ),
    # "What is happening to my application?" -- the user-facing list.
    Index("ix_tasks_user_id_created_at", "user_id", "created_at"),
)

# --------------------------------------------------------------------------
# Watched job boards (domain 3, intake). The owner pastes an employer's
# careers-board URL; a daily check records which jobs were listed. What matters
# is the history -- which jobs appeared, vanished, came back, were reposted --
# so this is our own record of presence, not a cache of the ATS.
#
# **Only a complete, successful check may close a presence interval.** That
# rule is enforced in `jfl_intake.engine.plan_check` and applied by
# `jfl_core.storage.boards`; the schema's part is to make the history
# impossible to double-write (the partial unique index on open intervals) and
# to record every check, including the ones that changed nothing.
#
# Presence is stored as INTERVALS, not per-check sightings. One 2,600-job board
# checked daily would be ~950k sighting rows a year; intervals store exactly the
# events that matter (opened, closed) and nothing in between.
#
# Watches are per user in v1: two users watching the same board each get their
# own rows and their own history. Nothing here is shared across tenants.
# --------------------------------------------------------------------------

_BOARD_PLATFORMS = (
    "greenhouse",
    "ashby",
    "lever",
    "workday",
    "smartrecruiters",
    "rippling",
    "breezy",
    "teamtailor",
    "personio",
    "recruitee",
    "pinpoint",
    "workable",
)

# `held` is a complete fetch the drop guard refused to apply; see
# `jfl_intake.engine`. Every status but `complete` changes no job's state.
_BOARD_CHECK_STATUSES = ("complete", "incomplete", "truncated", "unreachable", "failed", "held")

# A closed set, never free text -- same reasoning as `_EXTRACTION_ERROR_CODES`:
# wording belongs to the web layer, and an error column that takes a formatted
# exception is one that eventually takes something it should not.
_BOARD_CHECK_ERROR_CODES = (
    "not_found",  # HTTP 404: wrong board name, or the board was taken down
    "http_client_error",  # any other 4xx -- including Workday's 400 for limit > 20
    "rate_limited",  # HTTP 429
    "server_error",  # HTTP 5xx
    "timeout",
    "connection_error",
    "malformed_response",  # 200, but not the shape the platform documents
    "unidentifiable_job",  # a listed job with no usable id, so it cannot be tracked
    "count_mismatch",  # saw a different number of jobs than the board said it had
    "page_cap_reached",  # the pagination safety net fired
    "duplicate_posting",  # a token-paged listing repeated an id without finishing
    "request_budget_exhausted",
    "deadline_exceeded",
    "listing_ceiling",  # Workday board over its listing ceiling with no usable facet
    "unsupported_board",  # the stored board key is not one the adapter can use
    "drop_guard",  # complete, but a sudden collapse -- held, not applied
)

# How a posting says the work is done (`jfl_core.models.Workplace`). `unknown`
# is a first-class value: most platforms state nothing, and "not stated" must
# never be stored as on-site. See `jfl_intake.workplace`.
_WORKPLACES = ("remote", "hybrid", "onsite", "unknown")
# The saved filter's workplace preset (`jfl_core.models.WorkplaceMode`).
_WORKPLACE_MODES = ("remote_only", "remote_friendly", "custom")

watched_boards = Table(
    "watched_boards",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("platform", Text, nullable=False),
    Column("board_url", Text, nullable=False),  # verbatim, as pasted
    # The platform-specific identifier parsed out of the URL -- a Greenhouse
    # token, a Workday tenant/wd/site triple. The adapter reads this, never the
    # pasted URL, so a URL with tracking parameters and one without are one board.
    Column("board_key", JSONB, nullable=False),
    Column("label", Text),
    _ts("created_at", nullable=False, server_default=func.now()),
    # When the scheduler should next enqueue a check. A new board is due at once,
    # so its baseline is taken without anyone pressing anything; after that each
    # board keeps a fixed daily slot derived from its id (`jfl_intake.scheduling`).
    _ts("next_check_at", nullable=False, server_default=func.now()),
    # The three pointers below are `use_alter` foreign keys: `board_checks` also
    # references this table, and a cycle cannot be created in one CREATE TABLE.
    Column(
        "last_check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="SET NULL", use_alter=True),
    ),
    # Checks in a row that did not complete. Reset by the next complete check
    # that is applied; a held check leaves it alone, because the fetch worked.
    Column("consecutive_failures", Integer, nullable=False, server_default=text("0")),
    # The first complete check: the jobs it saw were "open when you started
    # watching", never "new". Set once.
    Column(
        "baseline_check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="SET NULL", use_alter=True),
    ),
    # The drop guard's flag. Non-null means the board is held: the check it
    # names returned a sudden collapse and was not applied, and every later
    # collapse is held too until a person accepts it or the count recovers.
    Column(
        "held_check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="SET NULL", use_alter=True),
    ),
    # A person has said the collapse is real ("they did close those roles"), so
    # the next complete check is applied whatever its count. Cleared by any
    # applied check.
    Column("drop_accepted", Boolean, nullable=False, server_default=text("false")),
    # Whether the saved job filter lets this board's jobs with an unstated
    # workplace through a workplace constraint. NULL means the platform default
    # (`jfl_intake.workplace.include_unstated_by_default`: on for platforms with
    # no structured workplace field, off for those that state it), so a board
    # nobody has touched follows that default; true/false is the owner's choice.
    Column("include_unstated_workplace", Boolean),
    # The owner says this employer's hybrid is more than about a day a week, so
    # its hybrid and remote-friendly jobs are left out of the remote-friendly
    # preset. Off by default: hybrid is shown, badged "days not stated".
    Column("hybrid_too_heavy", Boolean, nullable=False, server_default=text("false")),
    CheckConstraint(
        "platform in ('" + "','".join(_BOARD_PLATFORMS) + "')",
        name="platform",
    ),
    CheckConstraint("consecutive_failures >= 0", name="consecutive_failures"),
    UniqueConstraint("user_id", "platform", "board_key"),
    Index("ix_watched_boards_user_id_created_at", "user_id", "created_at"),
    # The scheduler's query, across tenants: which boards are due.
    Index("ix_watched_boards_next_check_at", "next_check_at"),
)

board_checks = Table(
    "board_checks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "board_id",
        UUID(as_uuid=True),
        ForeignKey("watched_boards.id", ondelete="CASCADE"),
        nullable=False,
    ),
    _ts("started_at", nullable=False),
    _ts("finished_at", nullable=False),
    Column("status", Text, nullable=False),
    # Distinct jobs the adapter returned -- recorded for every status, so a
    # held or incomplete check still says what it saw.
    Column("jobs_seen", Integer, nullable=False, server_default=text("0")),
    # What the board said it had, when it said anything. NULL where the platform
    # gives no total (Lever, Ashby) or none could be trusted (a truncated board).
    Column("expected_total", Integer),
    Column("error_code", Text),
    Column("is_baseline", Boolean, nullable=False, server_default=text("false")),
    CheckConstraint(
        "status in ('" + "','".join(_BOARD_CHECK_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('" + "','".join(_BOARD_CHECK_ERROR_CODES) + "')",
        name="error_code",
    ),
    # A complete check has nothing to explain, and every other status must say
    # why it is not complete.
    CheckConstraint(
        "(status = 'complete') = (error_code is null)", name="error_code_iff_not_complete"
    ),
    CheckConstraint("not is_baseline or status = 'complete'", name="baseline_is_complete"),
    CheckConstraint("jobs_seen >= 0", name="jobs_seen"),
    Index("ix_board_checks_board_id_started_at", "board_id", "started_at"),
    # One baseline per board, structurally: a redelivered first check cannot
    # mint a second one.
    Index(
        "ix_board_checks_board_id_baseline",
        "board_id",
        unique=True,
        postgresql_where=text("is_baseline"),
    ),
)

board_jobs = Table(
    "board_jobs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "board_id",
        UUID(as_uuid=True),
        ForeignKey("watched_boards.id", ondelete="CASCADE"),
        nullable=False,
    ),
    # The platform's own id for the POSTING -- Greenhouse/Ashby/Lever job id, the
    # Workday `externalPath` suffix (`R171808-1`). Identity for "the same job came
    # back". Deliberately the finest grain the platform exposes: history recorded
    # at a coarser grain (the requisition) could never be re-keyed afterwards.
    Column("external_id", Text, nullable=False),
    # The employer's requisition where the platform exposes one (Workday
    # `bulletFields[0]`, Greenhouse `requisition_id`), else NULL. STORED ONLY: no
    # rule reads it. A requisition re-listed under a new posting id has not been
    # observed, and rules are built from observed patterns -- this column is what
    # lets that pattern be learned from real history later. Refreshed on every
    # applied sighting, like the display fields below.
    Column("requisition_id", Text),
    # Latest seen values, refreshed on every applied sighting.
    Column("title", Text, nullable=False),
    Column("location", Text),
    Column("url", Text),
    # Normalised title + location (`jfl_intake.normalise.fingerprint`). Identity
    # for "a different posting of the same role" -- the repost signal.
    Column("fingerprint", Text, nullable=False),
    # Descriptive, refreshed on every applied sighting like `requisition_id`, and
    # never part of identity or the fingerprint. `workplace` comes from the
    # platform's structured field where one exists (`jfl_intake.workplace`);
    # `locations` is every location the posting lists, in a stable order, while
    # `location` above stays the single string the fingerprint was built from.
    # Rows written before these columns existed read `unknown` and `{}` until
    # their next sighting.
    Column("workplace", Text, nullable=False, server_default="unknown"),
    # The employer's own words for the workplace, where they wrote some (a
    # Greenhouse custom field's value, e.g. "On-Site"), kept verbatim so the
    # page shows what the employer published rather than our mapping of it.
    Column("workplace_label", Text),
    Column("locations", ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")),
    Column(
        "first_seen_check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="CASCADE"),
        nullable=False,
    ),
    _ts("first_seen_at", nullable=False),
    _ts("last_seen_at", nullable=False),
    # Set when this job was new and matched a recently closed job on the same
    # board by fingerprint. Recorded at check time rather than derived at read
    # time, so changing the repost window later does not rewrite history.
    Column(
        "reposted_from_job_id",
        UUID(as_uuid=True),
        ForeignKey("board_jobs.id", ondelete="SET NULL"),
    ),
    CheckConstraint(
        "workplace in ('" + "','".join(_WORKPLACES) + "')",
        name="workplace",
    ),
    UniqueConstraint("board_id", "external_id"),
    Index("ix_board_jobs_board_id_fingerprint", "board_id", "fingerprint"),
    Index("ix_board_jobs_first_seen_check_id", "first_seen_check_id"),
    # "What changed since" -- new and reposted jobs.
    Index("ix_board_jobs_user_id_first_seen_at", "user_id", "first_seen_at"),
    Index("ix_board_jobs_reposted_from_job_id", "reposted_from_job_id"),
)

board_job_presence = Table(
    "board_job_presence",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "job_id",
        UUID(as_uuid=True),
        ForeignKey("board_jobs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "opened_check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="CASCADE"),
        nullable=False,
    ),
    _ts("opened_at", nullable=False),
    Column(
        "closed_check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="CASCADE"),
    ),
    _ts("closed_at"),
    CheckConstraint("(closed_check_id is null) = (closed_at is null)", name="closed_together"),
    # At most ONE open interval per job. This is the idempotency guarantee made
    # structural: a check applied twice, or two checks racing, cannot open a
    # second interval for a job that is already open.
    Index(
        "ix_board_job_presence_job_id_open",
        "job_id",
        unique=True,
        postgresql_where=text("closed_check_id is null"),
    ),
    Index("ix_board_job_presence_job_id_closed_at", "job_id", "closed_at"),
    Index("ix_board_job_presence_opened_check_id", "opened_check_id"),
    Index("ix_board_job_presence_closed_check_id", "closed_check_id"),
    # "What changed since" -- returned and gone.
    Index("ix_board_job_presence_user_id_opened_at", "user_id", "opened_at"),
    Index("ix_board_job_presence_user_id_closed_at", "user_id", "closed_at"),
)

# One saved job filter per user: a lens over the open jobs of every board they
# watch. **Never applied at the source** -- boards are fetched whole and this is
# matched locally (`jfl_intake.filtering`), so editing it cannot make a job look
# as though it vanished. Text fields are stored exactly as typed (comma-separated
# alternatives); normalisation happens at match time, so the rule can change
# without rewriting what the user wrote. Empty means "no constraint".
job_filters = Table(
    "job_filters",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    # One of `_WORKPLACE_MODES`. `custom` -- the default, and what every filter
    # saved before the presets existed reads as -- consults `workplaces` below.
    Column("workplace_mode", Text, nullable=False, server_default="custom"),
    # A subset of `_WORKPLACES`. Empty = any workplace. Only consulted under `custom`.
    Column("workplaces", ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")),
    Column("title_includes", Text, nullable=False, server_default=""),
    Column("title_excludes", Text, nullable=False, server_default=""),
    Column("location", Text, nullable=False, server_default=""),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "workplaces <@ array['" + "','".join(_WORKPLACES) + "']::text[]",
        name="workplaces",
    ),
    CheckConstraint(
        "workplace_mode in ('" + "','".join(_WORKPLACE_MODES) + "')",
        name="workplace_mode",
    ),
)

# Per-board "also include" rules, OR'd with the saved filter: the owner knows
# something about an employer the board does not say ("accepts ~25% in office").
# An exception widens WORKPLACE and LOCATION only; the saved filter's title
# includes/excludes still apply to anything it lets through, so it can never
# widen the role. `note` is the owner's own words, stored verbatim -- no model
# touches it, and the page shows it beside the employer's unaltered label.
board_filter_exceptions = Table(
    "board_filter_exceptions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "board_id",
        UUID(as_uuid=True),
        ForeignKey("watched_boards.id", ondelete="CASCADE"),
        nullable=False,
    ),
    # A subset of `_WORKPLACES`. Empty = any workplace.
    Column("workplaces", ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")),
    Column("location", Text, nullable=False, server_default=""),
    Column("note", Text, nullable=False, server_default=""),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "workplaces <@ array['" + "','".join(_WORKPLACES) + "']::text[]",
        name="workplaces",
    ),
    Index("ix_board_filter_exceptions_user_id_board_id", "user_id", "board_id"),
    Index("ix_board_filter_exceptions_board_id", "board_id"),
)

# The "what changed" feed (PLAN.md C7). There is still no events table -- events
# are derived from `board_jobs` and `board_job_presence` -- so these two tables
# record only the READER's side: when this user last looked, and which derived
# events they have seen or dismissed. An event's identity is (job, kind, check):
# every event is produced by exactly one check, and one check can give a job at
# most one event of a kind. Deleting a board cascades through its jobs and checks
# and takes these marks with it.
_BOARD_JOB_EVENT_KINDS = ("new", "reposted", "returned", "gone")

job_feed_state = Table(
    "job_feed_state",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    ),
    # When the user last viewed the feed. NULL until the first view. Only ever
    # moves forward, so an event that was once "before you last looked" can never
    # become unseen again.
    _ts("last_looked_at"),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
)

job_feed_marks = Table(
    "job_feed_marks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "job_id",
        UUID(as_uuid=True),
        ForeignKey("board_jobs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "check_id",
        UUID(as_uuid=True),
        ForeignKey("board_checks.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", Text, nullable=False),
    # When the event happened (the check's finish), copied so the feed can find
    # how far back to derive events without joining back through the history.
    _ts("event_at", nullable=False),
    # When this user was first shown it. The 24-hour visibility runs from here.
    _ts("first_seen_at", nullable=False),
    _ts("dismissed_at"),
    CheckConstraint(
        "kind in ('" + "','".join(_BOARD_JOB_EVENT_KINDS) + "')",
        name="kind",
    ),
    # One mark per event per user, so two tabs viewing at once cannot mint two.
    UniqueConstraint("user_id", "job_id", "kind", "check_id"),
    Index("ix_job_feed_marks_user_id_first_seen_at", "user_id", "first_seen_at"),
)

# Suggested title expansions -- slice C7a. One row per (user, phrase_key): the
# model call runs once per phrase, ever, cached here. `phrase` is the text as
# typed; `phrase_key` is `jfl_intake.normalise.normalise(phrase)`, which is what
# the unique constraint is on -- re-saving the same phrase in different
# capitalisation or spacing must not enqueue a second call. `suggestions` is a
# JSONB list of `{title, gloss}`, empty until `status = 'done'`. Suggestions are
# never written into `job_filters.title_includes` by anything in this table --
# see `jfl_web.routes.title_suggestions`, where a tickbox is the only path.
_TITLE_SUGGESTION_STATUSES = ("pending", "done", "failed")

# A subset of `_EXTRACTION_ERROR_CODES` -- the ones this call can actually
# produce. No `no_job_ad` / `ad_too_long`: there is no ad here.
_TITLE_SUGGESTION_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
)

title_suggestions = Table(
    "title_suggestions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("phrase", Text, nullable=False),
    Column("phrase_key", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("suggestions", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("error_code", Text),
    _ts("dismissed_at"),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_TITLE_SUGGESTION_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('" + "','".join(_TITLE_SUGGESTION_ERROR_CODES) + "')",
        name="error_code",
    ),
    UniqueConstraint("user_id", "phrase_key"),
)

# --------------------------------------------------------------------------
# Capability clustering: one cheap model call that groups a user's CONFIRMED
# candidate facts into capability labels.
#
# One row per run, kept rather than replaced, because a run records a decision
# the user was asked to make and the accepted/rejected answers live in it.
# `proposals` is a JSONB list of `jfl_core.models.ProposedCapability`; nothing
# in it is a profile row until the user accepts it, and their rename wins
# permanently -- see `jfl_web.routes.profile`.
#
# `unclustered_fact_ids` (sent, placed in nothing) and `omitted_fact_ids` (more
# confirmed facts than one bounded call takes) exist so that no confirmed fact
# is ever silently dropped: both lists are shown on the profile screen and both
# are picked up by the next run.
#
# `trace_id` prices the run through `runs` (`cost_for_trace`) rather than
# storing a cost here -- one place holds spend, and it is the one built for
# querying it.
# --------------------------------------------------------------------------
_CAPABILITY_CLUSTER_STATUSES = ("pending", "done", "failed")

# The same subset `_TITLE_SUGGESTION_ERROR_CODES` takes: no document here, so
# nothing can be missing or too long.
_CAPABILITY_CLUSTER_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
)

capability_clusters = Table(
    "capability_clusters",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("status", Text, nullable=False, server_default="pending"),
    # Minted when the run is created, not when it finishes, so a failed run can
    # still be priced.
    Column("trace_id", UUID(as_uuid=True), nullable=False),
    Column("proposals", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("fact_count", Integer, nullable=False, server_default=text("0")),
    Column("unclustered_fact_ids", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("omitted_fact_ids", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("error_code", Text),
    _ts("dismissed_at"),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_CAPABILITY_CLUSTER_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('"
        + "','".join(_CAPABILITY_CLUSTER_ERROR_CODES)
        + "')",
        name="error_code",
    ),
    Index("ix_capability_clusters_user_id_created_at", "user_id", text("created_at DESC")),
)

# --------------------------------------------------------------------------
# Profile suggestions: one row per CV-reading call that proposes plain profile
# *settings* -- disciplines, where the person has worked, the level the CV
# describes. Distinct from `candidate_facts`, which holds the claims a CV makes
# about the world and which are confirmed one at a time into the corpus.
#
# `proposals` is a JSONB list of `jfl_core.models.ProposedSetting`. Nothing in
# it reaches `profiles.data` until the user accepts it, and an answered
# proposal stays on the row so that rejecting one keeps it from being offered
# again.
#
# `trace_id` prices the run through `runs` (`cost_for_trace`) rather than
# storing a cost here.
# --------------------------------------------------------------------------
_PROFILE_SUGGESTION_STATUSES = ("pending", "done", "failed")

# The same subset `_CAPABILITY_CLUSTER_ERROR_CODES` takes: the CVs are already
# stored, so nothing here can be missing or too long.
_PROFILE_SUGGESTION_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
)

profile_suggestions = Table(
    "profile_suggestions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("status", Text, nullable=False, server_default="pending"),
    # Minted when the run is created, not when it finishes, so a failed run can
    # still be priced.
    Column("trace_id", UUID(as_uuid=True), nullable=False),
    Column("proposals", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("cv_count", Integer, nullable=False, server_default=text("0")),
    Column("error_code", Text),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_PROFILE_SUGGESTION_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('"
        + "','".join(_PROFILE_SUGGESTION_ERROR_CODES)
        + "')",
        name="error_code",
    ),
    Index("ix_profile_suggestions_user_id_created_at", "user_id", text("created_at DESC")),
)


# --------------------------------------------------------------------------
# The profile (docs/profile-schema.md, 2026-09-21). One denormalised row per
# save, append-only, latest wins -- replacing `profile_answers`,
# `profile_objectives` and `profile_ruled_out`, which held the eighteen
# free-text questions of PLAN.md B3a and zero rows in production.
#
# `data` is JSONB and therefore carries no CHECK constraint: every closed set
# in it is guaranteed by `jfl_core.profile.Profile`, the only write path, and
# by the test that pairs its Literals with what the screens offer. That is
# weaker than a CHECK and was accepted deliberately (owner, 2026-09-21) as the
# price of a shape we expect to change. Nothing here is logged -- comp floors
# and deal-breakers are sensitive.
# --------------------------------------------------------------------------

profiles = Table(
    "profiles",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    # What shape `data` was written in. Written on every row so a later reader
    # can tell rather than guess; no CHECK, because the set grows by one every
    # time the model changes and a constraint would have to be migrated in
    # lockstep with a value that is already only meaningful to Python.
    Column("schema_version", Integer, nullable=False),
    Column("data", JSONB, nullable=False),
    # `clock_timestamp()`, not `now()`, for the reason the retired
    # `profile_answers.created_at` gave and this table inherits: the current
    # profile is *the latest row*, and `now()` is transaction-start time, so
    # two saves in one transaction would tie and make "latest" ambiguous
    # exactly where it decides what the user sees.
    _ts("created_at", nullable=False, server_default=text("clock_timestamp()")),
    Index("ix_profiles_user_id_created_at", "user_id", text("created_at DESC")),
)

# --------------------------------------------------------------------------
# CV onboarding (PLAN.md B6, redesigned 2026-09-18): candidate facts a model
# proposed from an uploaded CV, each awaiting the user's confirmation.
#
# This table is NOT corpus, and that is its whole reason for existing. The CV
# itself lives in `sent_documents` (form, never truth); a fact extracted from it
# becomes a `spans` row only when the user confirms it, at which point
# `span_id` points at the span their words produced. Grounding on the CV
# directly would make every later CV "supported" and silently switch the
# over-claim measurement off.
# --------------------------------------------------------------------------
# Instrumentation. One row per model call (and per non-model stage worth timing).
# Flat and wide on purpose: this is queried with GROUP BY for the writeup, not
# rendered on a dashboard.
# --------------------------------------------------------------------------

runs = Table(
    "runs",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
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


# --------------------------------------------------------------------------
# Two scores for one application -- PLAN.md slice B4.
#
# **Two axes, 1-10 each, and there is deliberately no third column.** CLAUDE.md
# and PLAN.md B4: "do I want this" and "could I get this" are reported
# separately and never averaged, so a composite has nowhere to live in this
# schema. The CHECKs pin each to 1-10.
#
# Append-only across runs, like `requirement_coverage`: a re-score inserts a
# new row and the page reads the latest, so the history of what the tool said
# and what it cost stays readable. A row's own `status` moves `pending` ->
# `done`/`failed` once (the run's state), which is not a rewrite of an earlier
# score.
# --------------------------------------------------------------------------

_SCORE_STATUSES = ("pending", "done", "failed")

# A subset of `_EXTRACTION_ERROR_CODES` plus `no_requirements`, which is this
# call's own: the ad has not been read, so there is nothing to score against
# and nothing was called. No `ad_too_long` -- the ad is not in this prompt.
_SCORE_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
    "no_requirements",
)

application_scores = Table(
    "application_scores",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # random: one id per run
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "application_id",
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("error_code", Text),
    # Both NULL until the run finishes. Two columns, two paragraphs, no third
    # number -- see the section comment above.
    Column("could_get_score", Integer),
    Column("could_get_assessment", Text, nullable=False, server_default=""),
    Column("want_it_score", Integer),
    Column("want_it_assessment", Text, nullable=False, server_default=""),
    # JSONB lists of the shapes in jfl_core.models: ConstraintVerdict,
    # ObjectiveVerdict, HardGateBreach, ScoreLever, NotStated. Read back whole
    # and rendered; never queried structurally, same convention as
    # `runs.attributes`.
    #
    # The two verdict lists are what `want_it_score` was derived from rather
    # than decoration: the model is never asked for that number, it gives a
    # four-word verdict per item and `jfl_core.fit` does the arithmetic. A row
    # from before the derivation reads `'[]'`, which says "this run predates
    # it" and not "nothing matched".
    Column("constraint_verdicts", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("objective_verdicts", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("hard_gate_breaches", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("levers", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("not_stated", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("model", Text),
    # The whole run's cost: the scoring call, plus the coverage call when this
    # run had to make one. Same precision as `runs.cost_usd`.
    Column("cost_usd", Numeric(12, 6)),
    # Ties this row to its `runs` rows -- cost attribution, not a foreign key.
    Column("trace_id", UUID(as_uuid=True)),
    # `clock_timestamp()`, not `now()`, for the reason `profiles` gives:
    # this is an append-only history table, `now()` is transaction-start time,
    # and two runs written in one transaction would share a timestamp, making
    # "the latest run" ambiguous exactly where the page reads it.
    _ts("created_at", nullable=False, server_default=text("clock_timestamp()")),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_SCORE_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('" + "','".join(_SCORE_ERROR_CODES) + "')",
        name="error_code",
    ),
    CheckConstraint(
        "could_get_score is null or (could_get_score between 1 and 10)",
        name="could_get_score",
    ),
    CheckConstraint(
        "want_it_score is null or (want_it_score between 1 and 10)",
        name="want_it_score",
    ),
    Index(
        "ix_application_scores_user_id_application_id_created_at",
        "user_id",
        "application_id",
        "created_at",
    ),
)


# --------------------------------------------------------------------------
# CV intake (PLAN.md slice B6, redesigned 2026-09-18).
#
# The uploaded CVs themselves live in `sent_documents` above -- the separate
# store that grounding cannot reach. These two tables are what happens to them:
# `cv_extractions` records the one model call per CV, and `candidate_facts`
# holds what that call proposed, one row per fact, until the user confirms,
# edits or rejects it.
#
# **A confirmed fact does not become a span from here.** It goes through
# markdown (`jfl_core.corpus_source`) and comes back as an ordinary
# `provenance='document'` span, so there is exactly one write path into the
# corpus and the user can read, edit and delete the text that was recorded
# about them. `candidate_facts.span_id` records which span that produced.
# --------------------------------------------------------------------------

_CV_EXTRACTION_STATUSES = ("pending", "done", "failed")

# A closed set, never free text -- same reasoning as `_EXTRACTION_ERROR_CODES`.
_CV_EXTRACTION_ERROR_CODES = (
    "no_cv_text",
    "no_api_key",
    "api_key_rejected",
    "credential_unreadable",
    "cv_too_long",
    "model_refused",
    "model_error",
)

_CANDIDATE_FACT_STATES = ("proposed", "confirmed", "rejected")

cv_extractions = Table(
    "cv_extractions",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "sent_document_id",
        UUID(as_uuid=True),
        ForeignKey("sent_documents.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("status", Text, nullable=False, server_default="pending"),
    Column("error_code", Text),
    Column("facts_proposed", Integer, nullable=False, server_default=text("0")),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "status in ('" + "','".join(_CV_EXTRACTION_STATUSES) + "')",
        name="status",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('" + "','".join(_CV_EXTRACTION_ERROR_CODES) + "')",
        name="error_code",
    ),
    # One extraction row per CV: re-reading a CV re-uses it rather than
    # accumulating a history of attempts, which is what makes a redelivered
    # task able to ask "is this already done?" with one read.
    UniqueConstraint("sent_document_id"),
    Index("ix_cv_extractions_user_id_status", "user_id", "status"),
)

candidate_facts = Table(
    "candidate_facts",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    # SET NULL rather than CASCADE: deleting the CV a fact came from must not
    # delete a fact the user has already confirmed. The citation of which CV
    # said it is lost; the fact, and its corpus span, are not.
    Column(
        "sent_document_id",
        UUID(as_uuid=True),
        ForeignKey("sent_documents.id", ondelete="SET NULL"),
    ),
    Column("role_label", Text, nullable=False),
    Column("role_key", Text, nullable=False),  # jfl_core.ids.role_key(role_label)
    Column("source_line", Text, nullable=False),  # the CV's own words, verbatim
    Column("fact_text", Text, nullable=False),  # the model's proposed statement
    Column("probe", Text),  # one-line question, where the shape needs one
    Column("probe_answer", Text),  # the user's own words, never a model's
    Column("state", Text, nullable=False, server_default="proposed"),
    Column("confirmed_text", Text),  # the user's words, if they edited
    # The corpus span this fact became. No FK ON DELETE: spans are retired,
    # never deleted (see `spans.retired_at`), so a dangling id cannot arise.
    Column("span_id", UUID(as_uuid=True), ForeignKey("spans.id")),
    Column("fingerprint", String(64), nullable=False),  # jfl_core.ids.fact_fingerprint
    Column("ordinal", Integer, nullable=False, server_default=text("0")),
    _ts("created_at", nullable=False, server_default=func.now()),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    CheckConstraint(
        "state in ('" + "','".join(_CANDIDATE_FACT_STATES) + "')",
        name="state",
    ),
    # Only a confirmed fact may carry a span. Structural, because "we grounded
    # on something the user rejected" is the failure this whole slice exists to
    # prevent, and a WHERE clause is not enforcement.
    CheckConstraint("span_id is null or state = 'confirmed'", name="span_iff_confirmed"),
    # The dedupe across 33 near-identical CVs: same role, same fact, one row.
    UniqueConstraint("user_id", "fingerprint"),
    Index("ix_candidate_facts_user_id_role_key_ordinal", "user_id", "role_key", "ordinal"),
    Index("ix_candidate_facts_user_id_state", "user_id", "state"),
)


# --------------------------------------------------------------------------
# Pushback: what happens when the user disagrees with a score.
#
# Design: `~/jobs4life-profile-research/feedback-loops.md`, "The loop we should
# build"; the arithmetic is `jfl_core.pushback`, pure and testable without a
# database.
#
# **Append-only, and the log is the store.** There is no stored preference
# weight anywhere, on purpose: a dimension's displacement is the sum of the
# applied deltas in this table, so "the profile drifted" is not a state that can
# happen quietly -- it is a query, and the drift meter is that query shown to
# the user. Nothing here writes to `profiles`, `spans`, `requirement_coverage`
# or `application_scores`, and nothing in this table could.
#
# **The user's words are stored verbatim**, with the exact number and sentence
# they were shown beside them: a pushback typed straight after reading our
# explanation is partly a response to our explanation, so the stimulus is part
# of the record rather than context that has to be reconstructed later.
#
# A row is inserted the moment the user submits, before anything is classified
# and before anything moves -- "recorded whether or not it changes anything" is
# the rule, and `status` says which of those happened.
# --------------------------------------------------------------------------

# `awaiting_classification` -- recorded, nothing applied, a cheap model call in
# flight (or failed, which is not an error the user has to care about: they can
# classify it themselves). `classified` -- the classification is on the row and
# the user has not confirmed it yet. `applied` -- the user confirmed or
# corrected it and the effect is recorded. Terminal, because the log is
# append-only: changing your mind is a new pushback, not an edit.
_PUSHBACK_STATUSES = ("awaiting_classification", "classified", "applied")

# A closed set, never a message: this column is written by the worker while it
# holds the user's decrypted API key.
_PUSHBACK_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "model_refused",
    "model_error",
    "credential_unreadable",
)

# The three kinds, three consequences: `jfl_core.pushback` holds the Literal and
# the rule, this holds what the table will accept, and the migration holds what
# Postgres will accept. All three are compared by
# `packages/core/tests/test_value_lists_agree.py` and its integration sibling.
_PUSHBACK_CLASSIFICATIONS = ("preference", "capability", "factual")
_PUSHBACK_DISPOSITIONS = ("accepted", "recorded_only", "pending_evidence")
_CLASSIFICATION_SOURCES = ("none", "model", "user")
_SCORE_AXES = ("want", "get")

score_pushbacks = Table(
    "score_pushbacks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),  # random: one id per pushback
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "application_id",
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    ),
    # The exact run they were arguing with. CASCADE: a pushback against a score
    # that no longer exists has lost the stimulus that gives it meaning.
    Column(
        "score_id",
        UUID(as_uuid=True),
        ForeignKey("application_scores.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("axis", Text, nullable=False),
    # What they were disagreeing about, from `jfl_core.pushback`'s allowlist.
    # Free TEXT with no CHECK because the vocabulary is per-user -- a
    # constraint kind, an objective rank, a capability key -- and the allowlist
    # that matters is the one `decide()` enforces before anything is applied.
    Column("dimension", Text, nullable=False),
    # Where the displacement actually landed, once the classification was known
    # (`jfl_core.pushback.target_dimension`). Empty until applied.
    Column("target_dimension", Text, nullable=False, server_default=""),
    # The stimulus, per the design: the exact number and sentence shown.
    Column("shown_score", Integer),
    Column("shown_explanation", Text, nullable=False, server_default=""),
    # VERBATIM. A model may classify this; it may never rewrite it. Same rule
    # as a gap answer (CLAUDE.md, 2026-09-01) and for the same reason: a user
    # held to wording they did not choose, by a tool whose claim is that it
    # measures distance from what they actually said.
    Column("user_text", Text, nullable=False),
    Column("asserted_direction", Text, nullable=False),
    # How far out they say it is, 1-3. Not "what should the number be": people
    # are far more reliable at relative judgements than absolute ones, and the
    # cap means the exact figure barely matters anyway.
    Column("asserted_points", Numeric(4, 2), nullable=False, server_default=text("1")),
    Column("status", Text, nullable=False, server_default="awaiting_classification"),
    Column("classification", Text),
    # Who decided the classification. `user` means they corrected the model, or
    # classified it themselves when the call failed -- worth being able to
    # count, because a model that is corrected often is a model to replace.
    Column("classification_source", Text, nullable=False, server_default="none"),
    Column("classification_note", Text, nullable=False, server_default=""),
    # Whether this carried a fact the record did not already have. Nullable
    # until classified. False makes it a restatement, which contributes no
    # delta -- and an exact textual restatement is forced False in code
    # whatever this says.
    Column("new_information", Boolean),
    Column("error_code", Text),
    Column("trace_id", UUID(as_uuid=True)),
    # What it did. NULL until applied; 0 is a real, common and honest answer.
    Column("applied_delta", Numeric(6, 3)),
    Column("prior_observations", Integer),
    Column("disposition", Text),
    # The receipt, whole: `jfl_core.pushback.PushbackEffect` as JSONB. Read back
    # and rendered, never queried structurally -- same convention as
    # `runs.attributes`.
    Column("effect", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    # For a capability claim the tool will not take on trust: the exact fact
    # that would move the number. The user's answer goes into the corpus
    # verbatim by the one existing write path, and `resulting_span_id` records
    # which span it became.
    Column("evidence_question", Text, nullable=False, server_default=""),
    # No FK cascade: spans are retired, never deleted, same as
    # `gap_questions.resulting_span_id`.
    Column("resulting_span_id", UUID(as_uuid=True), ForeignKey("spans.id")),
    # `clock_timestamp()`, not `now()`: this is an append-only log read in
    # order, and two rows written in one transaction would otherwise tie.
    _ts("created_at", nullable=False, server_default=text("clock_timestamp()")),
    _ts("updated_at", nullable=False, server_default=func.now(), onupdate=func.now()),
    _ts("applied_at"),
    CheckConstraint("axis in ('" + "','".join(_SCORE_AXES) + "')", name="axis"),
    CheckConstraint("asserted_direction in ('up','down')", name="asserted_direction"),
    CheckConstraint("status in ('" + "','".join(_PUSHBACK_STATUSES) + "')", name="status"),
    CheckConstraint(
        "classification is null or classification in ('"
        + "','".join(_PUSHBACK_CLASSIFICATIONS)
        + "')",
        name="classification",
    ),
    CheckConstraint(
        "classification_source in ('" + "','".join(_CLASSIFICATION_SOURCES) + "')",
        name="classification_source",
    ),
    CheckConstraint(
        "disposition is null or disposition in ('" + "','".join(_PUSHBACK_DISPOSITIONS) + "')",
        name="disposition",
    ),
    CheckConstraint(
        "error_code is null or error_code in ('" + "','".join(_PUSHBACK_ERROR_CODES) + "')",
        name="error_code",
    ),
    CheckConstraint("asserted_points > 0 and asserted_points <= 3", name="asserted_points"),
    # **The asymmetric bar, in the database.** A capability claim that the
    # number should go UP may never carry a non-zero delta. The rule is in
    # `jfl_core.pushback._capability`, which returns before any arithmetic; this
    # is the same rule written where no future caller can get past it, because
    # "a claim that increases what you assert needs grounding" is the one
    # guarantee this whole feature exists to provide.
    CheckConstraint(
        "not (classification = 'capability' and asserted_direction = 'up' "
        "and applied_delta is not null and applied_delta <> 0)",
        name="capability_up_never_moves",
    ),
    # A pushback that has not been applied has moved nothing, and one that has
    # been applied says what it did. Neither state is expressible halfway.
    CheckConstraint(
        "(status = 'applied') = (applied_delta is not null)",
        name="applied_iff_delta",
    ),
    Index(
        "ix_score_pushbacks_user_id_created_at",
        "user_id",
        text("created_at DESC"),
    ),
    Index("ix_score_pushbacks_user_id_target_dimension", "user_id", "target_dimension"),
    Index("ix_score_pushbacks_user_id_application_id", "user_id", "application_id"),
)


# A local override of a displayed number. Deliberately NOT a pushback: it is the
# escape hatch offered when the tool has said its piece and the user still
# disagrees, and it is honest about being one. It is scoped to one application,
# labelled as an override everywhere it appears, feeds no dimension's
# displacement, reaches no other job's score, and changes nothing about what the
# claim gate will say about a CV bullet. People will use an imperfect tool if
# they are allowed to modify it, even slightly; an unbounded lever destroys the
# product, and a lever that is real, bounded and labelled is the resolution.
score_overrides = Table(
    "score_overrides",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column(
        "application_id",
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("axis", Text, nullable=False),
    # NULL clears the override. Append-only, latest row per (application, axis)
    # wins, so "I took the override off" stays in the record.
    Column("value", Integer),
    Column("note", Text, nullable=False, server_default=""),
    _ts("created_at", nullable=False, server_default=text("clock_timestamp()")),
    CheckConstraint("axis in ('" + "','".join(_SCORE_AXES) + "')", name="axis"),
    CheckConstraint("value is null or (value between 1 and 10)", name="value"),
    Index(
        "ix_score_overrides_user_id_application_id_created_at",
        "user_id",
        "application_id",
        text("created_at DESC"),
    ),
)


# --------------------------------------------------------------------------
# Which sections of a screen this user leaves open (docs/ui-sections.md).
#
# Two things at once, and the second is why the table exists rather than a
# cookie. `is_open` is the preference: a choice made here wins over the default
# on the next visit, on every device. `default_open`, `toggles` and
# `against_default` are the *record of disagreement* -- the owner's own reason
# for asking ("we will have to track if people go against this") -- so a default
# that everybody immediately undoes is visible in a query instead of being
# something someone eventually notices.
#
# Deliberately NOT part of `profiles`. That row is append-only and is read back
# as "what you believed about yourself in March"; a save per collapsed panel
# would bury real decisions under UI noise. This is the other shape: one row per
# (user, section), upserted, never versioned.
#
# `section_key` carries no CHECK. The set is open by construction -- a per-draft
# section is keyed by the draft's own id -- so a closed list would have to be
# migrated every time a screen grows a panel. The route validates the shape
# instead (`jfl_web.routes.ui_sections`).
# --------------------------------------------------------------------------

ui_section_states = Table(
    "ui_section_states",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column(
        "user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    ),
    Column("section_key", Text, nullable=False),
    # The user's own choice, as of the last toggle.
    Column("is_open", Boolean, nullable=False),
    # What the screen would have shown had this row not existed, as reported by
    # the page that was on screen at the time. Kept so `against_default` means
    # something a year from now, when the default itself may have changed.
    Column("default_open", Boolean, nullable=False),
    Column("toggles", Integer, nullable=False, server_default=text("0")),
    Column("against_default", Integer, nullable=False, server_default=text("0")),
    # When this section was last open in front of the user. It is what "new
    # since you last looked" is measured from, and it is NULL until the first
    # toggle -- a section nobody has ever opened or closed has no watermark, and
    # inventing one would announce old items as new. Same rule as a newly
    # watched board's first check: a baseline, not news.
    _ts("last_opened_at"),
    _ts("created_at", nullable=False, server_default=text("clock_timestamp()")),
    _ts("updated_at", nullable=False, server_default=text("now()")),
    UniqueConstraint("user_id", "section_key"),
)
