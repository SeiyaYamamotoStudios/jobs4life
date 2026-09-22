"""profile suggestions

Revision ID: b7d3e9f10a42
Revises: b2c7e41d9f30

One row per run of the cheap Haiku call that reads a user's uploaded CVs and
proposes plain profile *settings* -- which disciplines they practise, where
they have worked, what level the CV describes. Distinct from `candidate_facts`,
which holds the claims a CV makes about the world.

`proposals` is a JSONB list of `jfl_core.models.ProposedSetting`
(`{kind, key, values, source_lines, state}`); nothing in it reaches
`profiles.data` until the user accepts it, and an answered proposal stays on
the row so that rejecting one keeps it from being offered again. `trace_id`
prices the run through `runs`, so no cost is stored here.

Hand-written (no `Vector` column, so the usual pgvector import is moot). The
value lists are written out as literals, never imported, because a migration is
a snapshot. Reverses cleanly: the table holds proposals the user has not acted
on, and the accepted ones already live on the profile.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'b7d3e9f10a42'
down_revision = 'b2c7e41d9f30'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'profile_suggestions',
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
        sa.Column('cv_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('error_code', sa.Text(), nullable=True),
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
            name=op.f('ck_profile_suggestions_status'),
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in "
            "('no_api_key','api_key_rejected','model_refused','model_error','credential_unreadable')",
            name=op.f('ck_profile_suggestions_error_code'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_profile_suggestions_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_profile_suggestions')),
    )
    op.create_index(
        'ix_profile_suggestions_user_id_created_at',
        'profile_suggestions',
        ['user_id', sa.text('created_at DESC')],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_profile_suggestions_user_id_created_at', table_name='profile_suggestions')
    op.drop_table('profile_suggestions')
