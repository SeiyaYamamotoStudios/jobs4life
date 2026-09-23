"""PLACEHOLDER (cvedit branch): cv_documents

Revision ID: c9e0d1a2b3f4
Revises: c4e8a2f61d57

The generation branch owns the real `cv_documents` migration. This one exists
only so the editing/export screens can be built and tested against a real
table; drop it at merge in favour of that branch's.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'c9e0d1a2b3f4'
down_revision = 'c4e8a2f61d57'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'cv_documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('application_id', sa.UUID(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('doc', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('trace_id', sa.UUID(), nullable=True),
        sa.Column(
            'created_at',
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['application_id'], ['applications.id'],
            name=op.f('fk_cv_documents_application_id'), ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_cv_documents_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cv_documents')),
        sa.UniqueConstraint(
            'application_id', 'version', name=op.f('uq_cv_documents_application_id_version')
        ),
    )


def downgrade() -> None:
    op.drop_table('cv_documents')
