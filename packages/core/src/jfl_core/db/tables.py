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
    # to find or match a user.
    Column("email", Text, nullable=False, unique=True),
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
    _ts("first_seen_at", nullable=False, server_default=func.now()),
    _ts("last_seen_at", nullable=False, server_default=func.now()),
    _ts("retired_at"),  # set when the source disappears; rows are never deleted
    CheckConstraint("storage_kind in ('local_file','upload','paste')", name="storage_kind"),
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
