"""api key health

Revision ID: d157716081c0
Revises: e2a7c5d9b814

The owner's feedback: when the Anthropic credits ran out nothing said so, and
the queued work burned its three attempts in a few minutes and died.

`api_key_health` is one row per user: `ok`, or the account-level reason the
model API is refusing calls (`credits_exhausted`, `invalid_key`,
`permission_denied`), when that started and when it was last seen. A category,
never the SDK's error text. It drives the banner on every signed-in page, read
by primary key. Parked tasks need no schema: they are `pending` rows whose
`last_error` starts `parked: ` (see `jfl_core.model_api`), counted through the
existing `ix_tasks_user_id_created_at`.

Hand-edited from autogenerate (no `Vector` column, so no pgvector import). The
value list is a literal, never imported, because a migration is a snapshot.
Reverses cleanly: downgrading drops the health rows; parked tasks stay pending
and run at their scheduled time as before.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd157716081c0'
down_revision = 'e2a7c5d9b814'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'api_key_health',
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column(
            'since',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'checked_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('ok','credits_exhausted','invalid_key','permission_denied')",
            name=op.f('ck_api_key_health_status'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_api_key_health_user_id'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('user_id', name=op.f('pk_api_key_health')),
    )


def downgrade() -> None:
    op.drop_table('api_key_health')
