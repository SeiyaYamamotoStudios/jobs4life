"""job feed state and marks

Revision ID: b7046bc3b4e6
Revises: 60e3dea4bda5

The reader's side of the "what changed" feed (PLAN.md C7). Events stay derived
from `board_jobs` and `board_job_presence`; nothing here duplicates them.

  * `job_feed_state` -- one row per user: when they last looked at the feed;
  * `job_feed_marks` -- per user, per event (job, kind, check): when they were
    first shown it and whether they dismissed it. `kind` is CHECKed against the
    BoardJobEventKind values, written out as literals because a migration is a
    snapshot.

Hand-written. No `Vector` column, so no pgvector import. Reverses cleanly;
downgrade discards last-looked times and dismissals, which belong to the feature
it removes.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = 'b7046bc3b4e6'
down_revision = '60e3dea4bda5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('job_feed_state',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('last_looked_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_job_feed_state_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_job_feed_state')),
    sa.UniqueConstraint('user_id', name=op.f('uq_job_feed_state_user_id'))
    )
    op.create_table('job_feed_marks',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('job_id', sa.UUID(), nullable=False),
    sa.Column('check_id', sa.UUID(), nullable=False),
    sa.Column('kind', sa.Text(), nullable=False),
    sa.Column('event_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('first_seen_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('dismissed_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint("kind in ('new','reposted','returned','gone')", name=op.f('ck_job_feed_marks_kind')),
    sa.ForeignKeyConstraint(['check_id'], ['board_checks.id'], name=op.f('fk_job_feed_marks_check_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['job_id'], ['board_jobs.id'], name=op.f('fk_job_feed_marks_job_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_job_feed_marks_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_job_feed_marks')),
    sa.UniqueConstraint('user_id', 'job_id', 'kind', 'check_id', name=op.f('uq_job_feed_marks_user_id_job_id_kind_check_id'))
    )
    op.create_index('ix_job_feed_marks_user_id_first_seen_at', 'job_feed_marks', ['user_id', 'first_seen_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_job_feed_marks_user_id_first_seen_at', table_name='job_feed_marks')
    op.drop_table('job_feed_marks')
    op.drop_table('job_feed_state')
