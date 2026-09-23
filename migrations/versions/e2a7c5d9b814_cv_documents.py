"""cv documents: the complete CV, one row per version

Revision ID: e2a7c5d9b814
Revises: c4e8a2f61d57

"Write the CV" now produces a whole CV -- header, summary, skills, every role
with its bullets, education -- as one `CvDocument` (`jfl_core.cv_document`),
stored as JSONB. Append-only: a regenerate is a new row, and so is an edit;
the latest row per application is the one shown. `status` is how the version
came to be (`generated`, `edited`, `approved`). `gate_result` is the claim
gate's raw output for a generated version, NULL otherwise. `trace_id` is shared
with the version's `runs` rows.

Hand-written off c4e8a2f61d57 (no `Vector` column, so no pgvector import). The
value lists are literals, never imported, because a migration is a snapshot.
Reverses cleanly: downgrading drops every stored CV.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'e2a7c5d9b814'
down_revision = 'c4e8a2f61d57'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'cv_documents',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('application_id', sa.UUID(), nullable=False),
        sa.Column('document', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('template', sa.Text(), server_default='modern', nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('gate_result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('trace_id', sa.UUID(), nullable=True),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('generated','edited','approved')", name=op.f('ck_cv_documents_status')
        ),
        sa.CheckConstraint(
            "template in ('classic','modern')", name=op.f('ck_cv_documents_template')
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_cv_documents_user_id'), ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['application_id'],
            ['applications.id'],
            name=op.f('fk_cv_documents_application_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cv_documents')),
    )
    op.create_index(
        op.f('ix_cv_documents_user_id_application_id_created_at'),
        'cv_documents',
        ['user_id', 'application_id', 'created_at'],
    )


def downgrade() -> None:
    op.drop_table('cv_documents')
