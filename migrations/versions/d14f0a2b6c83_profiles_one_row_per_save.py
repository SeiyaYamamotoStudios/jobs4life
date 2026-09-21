"""Profiles: one append-only row per save

Revision ID: d14f0a2b6c83
Revises: c4d9a1f6e207

`docs/profile-schema.md`, agreed 2026-09-21. One row per save, the current
profile being the latest; `data` is JSONB and `jfl_core.profile.Profile` is its
only write path.

The three tables this replaces -- `profile_answers`, `profile_objectives`,
`profile_ruled_out` -- are deliberately **not** dropped here. Production holds
zero rows of them so there is nothing to migrate, but the scoring path still
reads them and is being moved separately; dropping them in the same change would
break it for the sake of tidiness.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd14f0a2b6c83'
down_revision = 'c4d9a1f6e207'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('schema_version', sa.Integer(), server_default=sa.text('1'), nullable=False),
        sa.Column(
            'data',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        # clock_timestamp(), not now(): the latest row *is* the profile, and two
        # section saves in one transaction must not tie.
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_profiles_user_id_users'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_profiles')),
    )
    op.create_index('ix_profiles_user_id_created_at', 'profiles', ['user_id', 'created_at'])


def downgrade() -> None:
    op.drop_index('ix_profiles_user_id_created_at', table_name='profiles')
    op.drop_table('profiles')
