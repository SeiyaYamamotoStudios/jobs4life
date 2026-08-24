"""initial schema

Revision ID: 16a1a380bfb1
Revises: 
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

revision = '16a1a380bfb1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table('documents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('path', sa.Text(), nullable=False),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('first_seen_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_seen_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('retired_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_documents')),
    sa.UniqueConstraint('user_id', 'path', name=op.f('uq_documents_user_id_path'))
    )
    op.create_table('review_items',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.Column('claim_text', sa.Text(), nullable=False),
    sa.Column('source_text', sa.Text(), nullable=False),
    sa.Column('sentence_idx', sa.Integer(), nullable=False),
    sa.Column('candidate_span_ids', postgresql.ARRAY(sa.UUID()), nullable=False),
    sa.Column('drift_label', sa.Text(), nullable=True),
    sa.Column('status', sa.Text(), server_default='pending', nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('resolved_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("status in ('pending','adjudicated','dismissed')", name=op.f('ck_review_items_status')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_review_items'))
    )
    op.create_index('ix_review_items_user_id_status_created_at', 'review_items', ['user_id', 'status', 'created_at'], unique=False)
    op.create_table('runs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.Column('parent_run_id', sa.UUID(), nullable=True),
    sa.Column('component', sa.Text(), nullable=False),
    sa.Column('stage', sa.Text(), nullable=False),
    sa.Column('model', sa.Text(), nullable=True),
    sa.Column('tokens_in', sa.Integer(), nullable=True),
    sa.Column('tokens_out', sa.Integer(), nullable=True),
    sa.Column('cache_read_tokens', sa.Integer(), nullable=True),
    sa.Column('cache_write_tokens', sa.Integer(), nullable=True),
    sa.Column('cost_usd', sa.Numeric(precision=12, scale=6), nullable=True),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('outcome', sa.Text(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('started_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("outcome in ('ok','error','refused','skipped')", name=op.f('ck_runs_outcome')),
    sa.ForeignKeyConstraint(['parent_run_id'], ['runs.id'], name=op.f('fk_runs_parent_run_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_runs'))
    )
    op.create_index('ix_runs_trace_id', 'runs', ['trace_id'], unique=False)
    op.create_index('ix_runs_user_id_stage_created_at', 'runs', ['user_id', 'stage', 'created_at'], unique=False)
    op.create_table('sent_documents',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('employer', sa.Text(), nullable=True),
    sa.Column('role', sa.Text(), nullable=True),
    sa.Column('sent_on', sa.Date(), nullable=True),
    sa.Column('path', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind in ('cv','cover_letter','application_answer')", name=op.f('ck_sent_documents_kind')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_sent_documents')),
    sa.UniqueConstraint('user_id', 'path', name=op.f('uq_sent_documents_user_id_path'))
    )
    op.create_table('sent_spans',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('sent_document_id', sa.UUID(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.ForeignKeyConstraint(['sent_document_id'], ['sent_documents.id'], name=op.f('fk_sent_spans_sent_document_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_sent_spans'))
    )
    op.create_table('spans',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('document_id', sa.UUID(), nullable=True),
    sa.Column('provenance', sa.Text(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('section_path', sa.Text(), nullable=True),
    sa.Column('ordinal', sa.Integer(), nullable=True),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('char_start', sa.Integer(), nullable=True),
    sa.Column('char_end', sa.Integer(), nullable=True),
    sa.Column('first_seen_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_seen_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('retired_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("(provenance = 'document') = (document_id is not null)", name=op.f('ck_spans_document_id_iff_document')),
    sa.CheckConstraint("kind in ('bullet','paragraph','heading')", name=op.f('ck_spans_kind')),
    sa.CheckConstraint("provenance in ('document','adjudicated')", name=op.f('ck_spans_provenance')),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_spans_document_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_spans'))
    )
    op.create_index('ix_spans_user_id_document_id_ordinal', 'spans', ['user_id', 'document_id', 'ordinal'], unique=False)
    op.create_index('ix_spans_user_id_provenance', 'spans', ['user_id', 'provenance'], unique=False)
    op.create_table('adjudications',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('review_item_id', sa.UUID(), nullable=False),
    sa.Column('decision', sa.Text(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('resulting_span_id', sa.UUID(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("decision in ('grounded','not_grounded','rewrite')", name=op.f('ck_adjudications_decision')),
    sa.ForeignKeyConstraint(['resulting_span_id'], ['spans.id'], name=op.f('fk_adjudications_resulting_span_id')),
    sa.ForeignKeyConstraint(['review_item_id'], ['review_items.id'], name=op.f('fk_adjudications_review_item_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_adjudications')),
    sa.UniqueConstraint('review_item_id', name=op.f('uq_adjudications_review_item_id'))
    )
    op.create_table('span_embeddings',
    sa.Column('span_id', sa.UUID(), nullable=False),
    sa.Column('model', sa.Text(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=1024), nullable=False),
    sa.Column('source_content_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['span_id'], ['spans.id'], name=op.f('fk_span_embeddings_span_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('span_id', 'model', name=op.f('pk_span_embeddings'))
    )
    op.create_index('ix_span_embeddings_embedding', 'span_embeddings', ['embedding'], unique=False, postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.create_table('span_sentences',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('span_id', sa.UUID(), nullable=False),
    sa.Column('idx', sa.Integer(), nullable=False),
    sa.Column('start_offset', sa.Integer(), nullable=False),
    sa.Column('end_offset', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['span_id'], ['spans.id'], name=op.f('fk_span_sentences_span_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_span_sentences')),
    sa.UniqueConstraint('span_id', 'idx', name=op.f('uq_span_sentences_span_id_idx'))
    )


def downgrade() -> None:
    op.drop_table('span_sentences')
    op.drop_index('ix_span_embeddings_embedding', table_name='span_embeddings', postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'})
    op.drop_table('span_embeddings')
    op.drop_table('adjudications')
    op.drop_index('ix_spans_user_id_provenance', table_name='spans')
    op.drop_index('ix_spans_user_id_document_id_ordinal', table_name='spans')
    op.drop_table('spans')
    op.drop_table('sent_spans')
    op.drop_table('sent_documents')
    op.drop_index('ix_runs_user_id_stage_created_at', table_name='runs')
    op.drop_index('ix_runs_trace_id', table_name='runs')
    op.drop_table('runs')
    op.drop_index('ix_review_items_user_id_status_created_at', table_name='review_items')
    op.drop_table('review_items')
    op.drop_table('documents')
