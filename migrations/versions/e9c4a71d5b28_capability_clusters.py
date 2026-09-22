"""capability clusters

Revision ID: e9c4a71d5b28
Revises: d1f4b7c20a55

One row per clustering run: the cheap Haiku call that groups a user's
**confirmed** candidate facts into capability labels, because a role is not a
capability and a single fact is not one either.

`proposals` is a JSONB list of `jfl_core.models.ProposedCapability`
(`{label, fact_ids, span_ids, state}`); nothing in it reaches `profiles.data`
until the user accepts it. `unclustered_fact_ids` and `omitted_fact_ids` are
what this run did not place and what did not fit in one bounded call -- kept
so that a confirmed fact is never silently dropped. `trace_id` prices the run
through `runs`, so no cost is stored here.

Hand-written (no `Vector` column, so the usual pgvector import is moot). The
value lists are written out as literals, never imported, because a migration is
a snapshot. Reverses cleanly: the table holds proposals the user has not acted
on, and the accepted ones already live on the profile.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'e9c4a71d5b28'
down_revision = 'd1f4b7c20a55'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'capability_clusters',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('status', sa.Text(), server_default='pending', nullable=False),
        sa.Column('trace_id', sa.UUID(), nullable=False),
        sa.Column(
            'proposals',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column('fact_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column(
            'unclustered_fact_ids',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'omitted_fact_ids',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('dismissed_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('pending','done','failed')",
            name=op.f('ck_capability_clusters_status'),
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in "
            "('no_api_key','api_key_rejected','model_refused','model_error','credential_unreadable')",
            name=op.f('ck_capability_clusters_error_code'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_capability_clusters_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_capability_clusters')),
    )
    op.create_index(
        'ix_capability_clusters_user_id_created_at',
        'capability_clusters',
        ['user_id', sa.text('created_at DESC')],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_capability_clusters_user_id_created_at', table_name='capability_clusters')
    op.drop_table('capability_clusters')
