"""title suggestions

Revision ID: deceb9ea736a
Revises: e5396ef31c67

Slice C7a. `title_suggestions`: one row per `(user, phrase_key)`, the cached
result of the one cheap model call a newly added title-filter phrase gets --
see CLAUDE.md's 2026-09-15 decision and PLAN.md's C7a. `phrase` is the text as
typed; `phrase_key` is `jfl_intake.normalise.normalise(phrase)`, and the unique
constraint on it is what makes "once per phrase, ever" true under a race, not
just in the common case. `suggestions` is a JSONB list of `{title, gloss}`,
empty until `status = 'done'`.

Hand-written (no `Vector` column, so the usual pgvector import is moot). The
value lists are written out as literals, never imported, because a migration is
a snapshot. Reverses cleanly.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'deceb9ea736a'
down_revision = 'e5396ef31c67'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'title_suggestions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('phrase', sa.Text(), nullable=False),
        sa.Column('phrase_key', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), server_default='pending', nullable=False),
        sa.Column(
            'suggestions',
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
            "status in ('pending','done','failed')", name=op.f('ck_title_suggestions_status')
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in "
            "('no_api_key','api_key_rejected','model_refused','model_error','credential_unreadable')",
            name=op.f('ck_title_suggestions_error_code'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_title_suggestions_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_title_suggestions')),
        sa.UniqueConstraint(
            'user_id', 'phrase_key', name=op.f('uq_title_suggestions_user_id_phrase_key')
        ),
    )


def downgrade() -> None:
    op.drop_table('title_suggestions')
