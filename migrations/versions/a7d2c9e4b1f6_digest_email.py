"""digest email: per-user opt-in and cursor, and a record of what was sent

Revision ID: a7d2c9e4b1f6
Revises: f3a91c7e5b26

The owner is putting his search on ice and wants the daily board checks to come
to him. Two tables (PLAN.md C9):

`digest_settings` -- one row per user who has ever opted in. `enabled` is the
opt-in (default off: nothing is mailed to anyone who did not ask). Two
timestamps: `notified_up_to`, the cursor events are read from, and
`last_evaluated_at`, when a digest run last completed for the user whether or not
it sent anything -- a quiet day must still count as done, or the scheduling pass
would re-enqueue the user every quarter of an hour. Deliberately NOT columns on
`job_feed_state`: the email must never move the feed's last-looked time.

`email_sends` -- what was sent and when, counts only (no address, no title, no
body). It is what the weekly-heartbeat rule reads, and what the settings page
shows as "last sent".

Hand-written (no `Vector` column, so the usual pgvector import is moot).
Reverses cleanly: dropping both loses the opt-ins and the send log, and every
user reverts to receiving nothing.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a7d2c9e4b1f6'
down_revision: str | None = "f3a91c7e5b26"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'digest_settings',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('enabled', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('opted_in_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('notified_up_to', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_evaluated_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_digest_settings_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_digest_settings')),
        sa.UniqueConstraint('user_id', name=op.f('uq_digest_settings_user_id')),
    )
    op.create_table(
        'email_sends',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('sent_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('jobs_listed', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('events_found', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('boards_failed', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.CheckConstraint(
            "kind in ('digest','heartbeat','test')", name=op.f('ck_email_sends_kind')
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_email_sends_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_email_sends')),
    )
    op.create_index(
        'ix_email_sends_user_id_sent_at', 'email_sends', ['user_id', 'sent_at'], unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_email_sends_user_id_sent_at', table_name='email_sends')
    op.drop_table('email_sends')
    op.drop_table('digest_settings')
