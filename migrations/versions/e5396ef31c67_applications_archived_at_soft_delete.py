"""applications archived_at soft delete

Revision ID: e5396ef31c67
Revises: 9eb127bee328

Adds `applications.archived_at`, a nullable timestamp: NULL means live, set means
archived. Archiving takes an application off the owner's lists without touching its
status or its event timeline -- so a test entry or a duplicate can disappear without
being recorded as something the owner did. "Withdrawn" is a real outcome of a real
process and stays reserved for one.

Earlier migrations are deployed to production and are not edited. Adding a nullable
column rewrites nothing: every existing application reads as live.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = 'e5396ef31c67'
down_revision = '9eb127bee328'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('applications', sa.Column('archived_at', sa.TIMESTAMP(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('applications', 'archived_at')
