"""The profile as one row, and the constraint verdicts a score is derived from

Revision ID: d1f4b7c20a55
Revises: c4d9a1f6e207

`docs/profile-schema.md`, agreed 2026-09-21. Two changes.

**`profiles`** -- one append-only row per save, four sections in one JSONB
document, latest row wins. It replaces `profile_answers`, `profile_objectives`
and `profile_ruled_out`, which production held zero rows of, so nothing is
migrated. Those three tables are deliberately left standing: the screens that
write them are replaced separately, and dropping a table before its last writer
is gone buys nothing.

There is **no CHECK constraint on what is inside `data`**. A CHECK cannot see
into a document, so the stances, tiers and kinds in there are guarded only by
`jfl_core.models.Profile` being the single write path. That is weaker than this
schema's usual three-way agreement between a Literal, a tuple and a CHECK; it
was accepted deliberately (owner, 2026-09-21) as the price of a shape we expect
to change while we learn what belongs in it.

**`application_scores.constraint_verdicts`** -- what the ad evidences about
each constraint the user recorded, as `evidenced` / `partial` / `silent` /
`contradicted`. The "do I want this" number is derived from these rather than
asked for, so without the column there is nothing behind the number; existing
rows default to `'[]'`, which reads correctly as "this run predates the
derivation" rather than as "no constraints matched".

Downgrade drops both. It is lossy in the ordinary way -- the derived number
stays in `want_it_score` with nothing left saying what it was derived from.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd1f4b7c20a55'
down_revision = 'c4d9a1f6e207'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'profiles',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('schema_version', sa.Integer(), server_default=sa.text('1'), nullable=False),
        sa.Column(
            'data',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_profiles_user_id_created_at', 'profiles', ['user_id', 'created_at'], unique=False
    )

    op.add_column(
        'application_scores',
        sa.Column(
            'constraint_verdicts',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column('application_scores', 'constraint_verdicts')
    op.drop_index('ix_profiles_user_id_created_at', table_name='profiles')
    op.drop_table('profiles')
