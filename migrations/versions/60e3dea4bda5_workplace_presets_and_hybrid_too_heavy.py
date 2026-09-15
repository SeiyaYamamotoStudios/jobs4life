"""workplace presets and hybrid_too_heavy

Revision ID: 60e3dea4bda5
Revises: e5396ef31c67

The saved job filter's two named workplace presets (owner, 2026-09-15):

  * `job_filters.workplace_mode` -- `remote_only`, `remote_friendly` or `custom`
    (closed set, CHECK). Defaults to `custom`, which is exactly the checkbox
    behaviour every existing filter was saved under, so no saved filter changes
    what it matches;
  * `watched_boards.hybrid_too_heavy` -- the owner's judgement that an employer's
    hybrid is more than about a day a week, taking that board's hybrid out of
    `remote_friendly`. Defaults to false: hybrid is shown, badged "days not stated".

Hand-written. The value list is a literal, never imported, because a migration is
a snapshot. Adding NOT NULL columns with server defaults rewrites no data by hand.
Reverses cleanly; downgrade discards the chosen presets and board settings, which
belong to the feature it removes.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '60e3dea4bda5'
down_revision = 'e5396ef31c67'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'job_filters',
        sa.Column('workplace_mode', sa.Text(), server_default='custom', nullable=False),
    )
    op.create_check_constraint(
        op.f('ck_job_filters_workplace_mode'),
        'job_filters',
        "workplace_mode in ('remote_only','remote_friendly','custom')",
    )
    op.add_column(
        'watched_boards',
        sa.Column('hybrid_too_heavy', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    )


def downgrade() -> None:
    op.drop_column('watched_boards', 'hybrid_too_heavy')
    op.drop_constraint(op.f('ck_job_filters_workplace_mode'), 'job_filters', type_='check')
    op.drop_column('job_filters', 'workplace_mode')
