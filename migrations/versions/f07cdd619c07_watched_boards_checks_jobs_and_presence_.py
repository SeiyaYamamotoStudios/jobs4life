"""watched boards: checks, jobs and presence intervals

Revision ID: f07cdd619c07
Revises: 2f6b0c1d94a7

Domain 3 (intake), the watched-job-boards engine. Four new tables, nothing
existing touched.

  * `watched_boards` -- one row per board a user watches, per user;
  * `board_checks` -- every check, including the ones that changed nothing,
    with a status from a closed set and an error code from a closed set;
  * `board_jobs` -- one row per distinct external job on a board;
  * `board_job_presence` -- INTERVALS of presence, not per-check sightings.

The rule the history rests on -- only a complete check may close an interval --
lives in code (`jfl_intake.engine`). The schema's contribution is structural
idempotency: `ix_board_job_presence_job_id_open` is a partial UNIQUE index
allowing at most one open interval per job, and `ix_board_checks_board_id_baseline`
allows one baseline per board, so a redelivered check cannot double-write either.

**One hand edit.** `watched_boards` points at `board_checks` (`last_check_id`,
`baseline_check_id`, `held_check_id`) and `board_checks` points back at
`watched_boards`. Autogenerate renders the `use_alter` foreign keys inline in
`create_table`, which fails because `board_checks` does not exist yet; they are
moved to `op.create_foreign_key` after both tables exist, and dropped first on
downgrade. No `Vector` column, so the usual `pgvector` import is moot.

Reverses cleanly: downgrade drops only what upgrade added. It discards every
board's history, which is the correct behaviour for a rollback of the feature
that owns it.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'f07cdd619c07'
down_revision = '2f6b0c1d94a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('watched_boards',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('platform', sa.Text(), nullable=False),
    sa.Column('board_url', sa.Text(), nullable=False),
    sa.Column('board_key', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('label', sa.Text(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('next_check_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_check_id', sa.UUID(), nullable=True),
    sa.Column('consecutive_failures', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('baseline_check_id', sa.UUID(), nullable=True),
    sa.Column('held_check_id', sa.UUID(), nullable=True),
    sa.Column('drop_accepted', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.CheckConstraint("platform in ('greenhouse','ashby','lever','workday')", name=op.f('ck_watched_boards_platform')),
    sa.CheckConstraint('consecutive_failures >= 0', name=op.f('ck_watched_boards_consecutive_failures')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_watched_boards_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_watched_boards')),
    sa.UniqueConstraint('user_id', 'platform', 'board_key', name=op.f('uq_watched_boards_user_id_platform_board_key'))
    )
    op.create_index('ix_watched_boards_next_check_at', 'watched_boards', ['next_check_at'], unique=False)
    op.create_index('ix_watched_boards_user_id_created_at', 'watched_boards', ['user_id', 'created_at'], unique=False)
    op.create_table('board_checks',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('board_id', sa.UUID(), nullable=False),
    sa.Column('started_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('finished_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('jobs_seen', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('expected_total', sa.Integer(), nullable=True),
    sa.Column('error_code', sa.Text(), nullable=True),
    sa.Column('is_baseline', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.CheckConstraint("(status = 'complete') = (error_code is null)", name=op.f('ck_board_checks_error_code_iff_not_complete')),
    sa.CheckConstraint("error_code is null or error_code in ('not_found','http_client_error','rate_limited','server_error','timeout','connection_error','malformed_response','unidentifiable_job','count_mismatch','page_cap_reached','request_budget_exhausted','deadline_exceeded','listing_ceiling','unsupported_board','drop_guard')", name=op.f('ck_board_checks_error_code')),
    sa.CheckConstraint("not is_baseline or status = 'complete'", name=op.f('ck_board_checks_baseline_is_complete')),
    sa.CheckConstraint("status in ('complete','incomplete','truncated','unreachable','failed','held')", name=op.f('ck_board_checks_status')),
    sa.CheckConstraint('jobs_seen >= 0', name=op.f('ck_board_checks_jobs_seen')),
    sa.ForeignKeyConstraint(['board_id'], ['watched_boards.id'], name=op.f('fk_board_checks_board_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_board_checks_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_board_checks'))
    )
    op.create_index('ix_board_checks_board_id_baseline', 'board_checks', ['board_id'], unique=True, postgresql_where=sa.text('is_baseline'))
    op.create_index('ix_board_checks_board_id_started_at', 'board_checks', ['board_id', 'started_at'], unique=False)

    # Hand edit: the cycle's back-references, now that `board_checks` exists.
    for column in ('last_check_id', 'baseline_check_id', 'held_check_id'):
        op.create_foreign_key(
            op.f(f'fk_watched_boards_{column}'),
            'watched_boards', 'board_checks',
            [column], ['id'],
            ondelete='SET NULL',
        )

    op.create_table('board_jobs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('board_id', sa.UUID(), nullable=False),
    sa.Column('external_id', sa.Text(), nullable=False),
    sa.Column('requisition_id', sa.Text(), nullable=True),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('location', sa.Text(), nullable=True),
    sa.Column('url', sa.Text(), nullable=True),
    sa.Column('fingerprint', sa.Text(), nullable=False),
    sa.Column('first_seen_check_id', sa.UUID(), nullable=False),
    sa.Column('first_seen_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('last_seen_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('reposted_from_job_id', sa.UUID(), nullable=True),
    sa.ForeignKeyConstraint(['board_id'], ['watched_boards.id'], name=op.f('fk_board_jobs_board_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['first_seen_check_id'], ['board_checks.id'], name=op.f('fk_board_jobs_first_seen_check_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['reposted_from_job_id'], ['board_jobs.id'], name=op.f('fk_board_jobs_reposted_from_job_id'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_board_jobs_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_board_jobs')),
    sa.UniqueConstraint('board_id', 'external_id', name=op.f('uq_board_jobs_board_id_external_id'))
    )
    op.create_index('ix_board_jobs_board_id_fingerprint', 'board_jobs', ['board_id', 'fingerprint'], unique=False)
    op.create_index('ix_board_jobs_first_seen_check_id', 'board_jobs', ['first_seen_check_id'], unique=False)
    op.create_index('ix_board_jobs_reposted_from_job_id', 'board_jobs', ['reposted_from_job_id'], unique=False)
    op.create_index('ix_board_jobs_user_id_first_seen_at', 'board_jobs', ['user_id', 'first_seen_at'], unique=False)
    op.create_table('board_job_presence',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('job_id', sa.UUID(), nullable=False),
    sa.Column('opened_check_id', sa.UUID(), nullable=False),
    sa.Column('opened_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.Column('closed_check_id', sa.UUID(), nullable=True),
    sa.Column('closed_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.CheckConstraint('(closed_check_id is null) = (closed_at is null)', name=op.f('ck_board_job_presence_closed_together')),
    sa.ForeignKeyConstraint(['closed_check_id'], ['board_checks.id'], name=op.f('fk_board_job_presence_closed_check_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['job_id'], ['board_jobs.id'], name=op.f('fk_board_job_presence_job_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['opened_check_id'], ['board_checks.id'], name=op.f('fk_board_job_presence_opened_check_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_board_job_presence_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_board_job_presence'))
    )
    op.create_index('ix_board_job_presence_closed_check_id', 'board_job_presence', ['closed_check_id'], unique=False)
    op.create_index('ix_board_job_presence_job_id_closed_at', 'board_job_presence', ['job_id', 'closed_at'], unique=False)
    op.create_index('ix_board_job_presence_job_id_open', 'board_job_presence', ['job_id'], unique=True, postgresql_where=sa.text('closed_check_id is null'))
    op.create_index('ix_board_job_presence_opened_check_id', 'board_job_presence', ['opened_check_id'], unique=False)
    op.create_index('ix_board_job_presence_user_id_closed_at', 'board_job_presence', ['user_id', 'closed_at'], unique=False)
    op.create_index('ix_board_job_presence_user_id_opened_at', 'board_job_presence', ['user_id', 'opened_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_board_job_presence_user_id_opened_at', table_name='board_job_presence')
    op.drop_index('ix_board_job_presence_user_id_closed_at', table_name='board_job_presence')
    op.drop_index('ix_board_job_presence_opened_check_id', table_name='board_job_presence')
    op.drop_index('ix_board_job_presence_job_id_open', table_name='board_job_presence', postgresql_where=sa.text('closed_check_id is null'))
    op.drop_index('ix_board_job_presence_job_id_closed_at', table_name='board_job_presence')
    op.drop_index('ix_board_job_presence_closed_check_id', table_name='board_job_presence')
    op.drop_table('board_job_presence')
    op.drop_index('ix_board_jobs_user_id_first_seen_at', table_name='board_jobs')
    op.drop_index('ix_board_jobs_reposted_from_job_id', table_name='board_jobs')
    op.drop_index('ix_board_jobs_first_seen_check_id', table_name='board_jobs')
    op.drop_index('ix_board_jobs_board_id_fingerprint', table_name='board_jobs')
    op.drop_table('board_jobs')

    # Hand edit: break the cycle before either side of it is dropped.
    for column in ('held_check_id', 'baseline_check_id', 'last_check_id'):
        op.drop_constraint(op.f(f'fk_watched_boards_{column}'), 'watched_boards', type_='foreignkey')

    op.drop_index('ix_board_checks_board_id_started_at', table_name='board_checks')
    op.drop_index('ix_board_checks_board_id_baseline', table_name='board_checks', postgresql_where=sa.text('is_baseline'))
    op.drop_table('board_checks')
    op.drop_index('ix_watched_boards_user_id_created_at', table_name='watched_boards')
    op.drop_index('ix_watched_boards_next_check_at', table_name='watched_boards')
    op.drop_table('watched_boards')
