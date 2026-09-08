"""application tracker: applications and application_events

Slice A5. Two new tables, nothing else touched.

`applications` holds current state; `application_events` is the append-only
timeline it is derived from -- see `jfl_core.storage.applications` for why a
status change writes both. `updated_at` on `applications` carries
`onupdate=func.now()` in `tables.py`, which is a client-side SQLAlchemy Core
default applied when an UPDATE statement omits the column, not a database
trigger -- there is nothing for autogenerate to detect for it, so no DDL for
it appears below, and none is needed.

No hand edits needed this time: neither table touches a `Vector` column (so
the usual missing `import pgvector.sqlalchemy` is moot), and every new column
is nullable or carries a server default, so there is no NOT NULL-on-an-
existing-table hazard to work around.

Reverses cleanly: downgrade drops only what upgrade added.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '6546d61a3172'
down_revision = 'cb19e34095ce'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('applications',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('job_id', sa.UUID(), nullable=True),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('employer', sa.Text(), nullable=True),
    sa.Column('url', sa.Text(), nullable=True),
    sa.Column('status', sa.Text(), server_default='interested', nullable=False),
    sa.Column('source', sa.Text(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status in ('interested','applied','screening','interviewing','offer','rejected','withdrawn')", name=op.f('ck_applications_status')),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name=op.f('fk_applications_job_id'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_applications_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_applications'))
    )
    op.create_index('ix_applications_user_id_status', 'applications', ['user_id', 'status'], unique=False)
    op.create_index('ix_applications_user_id_updated_at', 'applications', ['user_id', 'updated_at'], unique=False)
    op.create_table('application_events',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('application_id', sa.UUID(), nullable=False),
    sa.Column('from_status', sa.Text(), nullable=True),
    sa.Column('to_status', sa.Text(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('occurred_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("to_status in ('interested','applied','screening','interviewing','offer','rejected','withdrawn')", name=op.f('ck_application_events_to_status')),
    sa.ForeignKeyConstraint(['application_id'], ['applications.id'], name=op.f('fk_application_events_application_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_application_events_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_application_events'))
    )
    op.create_index('ix_application_events_user_id_application_id_occurred_at', 'application_events', ['user_id', 'application_id', 'occurred_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_application_events_user_id_application_id_occurred_at', table_name='application_events')
    op.drop_table('application_events')
    op.drop_index('ix_applications_user_id_updated_at', table_name='applications')
    op.drop_index('ix_applications_user_id_status', table_name='applications')
    op.drop_table('applications')
