"""accounts, credentials, job sources, generalised document source

Revision ID: 309cf277970b
Revises: 16a1a380bfb1
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '309cf277970b'
down_revision = '16a1a380bfb1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('email', sa.Text(), nullable=False),
    sa.Column('display_name', sa.Text(), nullable=True),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name=op.f('uq_users_email'))
    )
    op.execute(
        """
        insert into users (id, email, display_name)
        values ('0425d123-ed29-5a6a-a06d-d00267574046', 'local@localhost', 'Local')
        on conflict (id) do nothing
        """
    )
    op.create_table('job_sources',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('enabled', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('cursor', sa.Text(), nullable=True),
    sa.Column('last_polled_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('last_success_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('consecutive_failures', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind in ('greenhouse','lever','ashby','workable','rss','forwarded_email','manual')", name=op.f('ck_job_sources_kind')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_job_sources_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_job_sources')),
    sa.UniqueConstraint('user_id', 'kind', 'name', name=op.f('uq_job_sources_user_id_kind_name'))
    )
    op.create_table('user_credentials',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('label', sa.Text(), server_default='', nullable=False),
    sa.Column('ciphertext', sa.LargeBinary(), nullable=False),
    sa.Column('wrapped_dek', sa.LargeBinary(), nullable=False),
    sa.Column('nonce', sa.LargeBinary(), nullable=False),
    sa.Column('master_key_id', sa.Text(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('rotated_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('last_used_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("kind in ('anthropic_api_key','openai_api_key','ats_token','smtp_password')", name=op.f('ck_user_credentials_kind')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_credentials_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_credentials')),
    sa.UniqueConstraint('user_id', 'kind', 'label', name=op.f('uq_user_credentials_user_id_kind_label'))
    )
    op.alter_column('adjudications', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_adjudications_user_id'), 'adjudications', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.add_column('documents', sa.Column('source_uri', sa.Text(), nullable=False))
    op.add_column('documents', sa.Column('storage_kind', sa.Text(), nullable=False))
    op.alter_column('documents', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.drop_constraint(op.f('uq_documents_user_id_path'), 'documents', type_='unique')
    op.create_unique_constraint(op.f('uq_documents_user_id_source_uri'), 'documents', ['user_id', 'source_uri'])
    op.create_foreign_key(op.f('fk_documents_user_id'), 'documents', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.create_check_constraint(op.f('ck_documents_storage_kind'), 'documents', "storage_kind in ('local_file','upload','paste')")
    op.drop_column('documents', 'path')
    op.alter_column('review_items', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_review_items_user_id'), 'review_items', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.alter_column('runs', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_runs_user_id'), 'runs', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.alter_column('sent_documents', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_sent_documents_user_id'), 'sent_documents', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.alter_column('sent_spans', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_sent_spans_user_id'), 'sent_spans', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.alter_column('span_embeddings', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_span_embeddings_user_id'), 'span_embeddings', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.alter_column('span_sentences', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_span_sentences_user_id'), 'span_sentences', 'users', ['user_id'], ['id'], ondelete='CASCADE')
    op.alter_column('spans', 'user_id',
               existing_type=sa.TEXT(),
               type_=sa.UUID(),
               existing_nullable=False,
               postgresql_using='user_id::uuid')
    op.create_foreign_key(op.f('fk_spans_user_id'), 'spans', 'users', ['user_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    op.drop_constraint(op.f('fk_spans_user_id'), 'spans', type_='foreignkey')
    op.alter_column('spans', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_constraint(op.f('fk_span_sentences_user_id'), 'span_sentences', type_='foreignkey')
    op.alter_column('span_sentences', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_constraint(op.f('fk_span_embeddings_user_id'), 'span_embeddings', type_='foreignkey')
    op.alter_column('span_embeddings', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_constraint(op.f('fk_sent_spans_user_id'), 'sent_spans', type_='foreignkey')
    op.alter_column('sent_spans', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_constraint(op.f('fk_sent_documents_user_id'), 'sent_documents', type_='foreignkey')
    op.alter_column('sent_documents', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_constraint(op.f('fk_runs_user_id'), 'runs', type_='foreignkey')
    op.alter_column('runs', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_constraint(op.f('fk_review_items_user_id'), 'review_items', type_='foreignkey')
    op.alter_column('review_items', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.add_column('documents', sa.Column('path', sa.TEXT(), autoincrement=False, nullable=False))
    op.drop_constraint(op.f('ck_documents_storage_kind'), 'documents', type_='check')
    op.drop_constraint(op.f('fk_documents_user_id'), 'documents', type_='foreignkey')
    op.drop_constraint(op.f('uq_documents_user_id_source_uri'), 'documents', type_='unique')
    op.create_unique_constraint(op.f('uq_documents_user_id_path'), 'documents', ['user_id', 'path'], postgresql_nulls_not_distinct=False)
    op.alter_column('documents', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_column('documents', 'storage_kind')
    op.drop_column('documents', 'source_uri')
    op.drop_constraint(op.f('fk_adjudications_user_id'), 'adjudications', type_='foreignkey')
    op.alter_column('adjudications', 'user_id',
               existing_type=sa.UUID(),
               type_=sa.TEXT(),
               existing_nullable=False,
               postgresql_using='user_id::text')
    op.drop_table('user_credentials')
    op.drop_table('job_sources')
    op.drop_table('users')
