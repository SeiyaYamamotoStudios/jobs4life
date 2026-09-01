"""drafts

Revision ID: 81d65d3075c8
Revises: ad22c244c67b
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '81d65d3075c8'
down_revision = 'ad22c244c67b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('drafts',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('job_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('gate_result', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("kind in ('cv_bullets','cover_letter')", name=op.f('ck_drafts_kind')),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name=op.f('fk_drafts_job_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_drafts_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_drafts'))
    )
    op.create_index('ix_drafts_user_id_job_id_created_at', 'drafts', ['user_id', 'job_id', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_drafts_user_id_job_id_created_at', table_name='drafts')
    op.drop_table('drafts')
