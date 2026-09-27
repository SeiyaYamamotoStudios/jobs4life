"""ui table sorts: which column and direction a user last sorted each table by

Revision ID: f3a91c7e5b26
Revises: d157716081c0

The owner's complaint: "The sorting of the applications isn't persistent, in
fact it needs to be persistent on any tables, etc." A header click chose an
order that lasted only until the next page load. This table gives every
sortable table (applications, jobs, boards, changes, cvs) one saved (sort_key,
direction) per user, read once per render and upserted on a header click.

Same shape as `ui_section_states` (b2c7e41d9f30): a small preference table,
never on the path of anything that costs money, structurally tenant-scoped
like everything else. The one difference is `table_key` DOES carry a CHECK --
a table's columns are fixed by its template, not opened per row the way a
per-draft section key is, so there is a real closed list to check. `sort_key`
carries no CHECK: it is validated against each table's own closed set of
columns in Python (`jfl_web.sorting.SortSpec`), and a stale or unknown value
(a column a screen has since dropped) is meant to fall back to that table's
default silently rather than fail a constraint.

Hand-written (no `Vector` column, so the usual pgvector import is moot).
Reverses cleanly: dropping the table loses only a record of preference, and
every table reverts to its agreed default order.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'f3a91c7e5b26'
down_revision: str | None = "d157716081c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'ui_table_sorts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('table_key', sa.Text(), nullable=False),
        sa.Column('sort_key', sa.Text(), nullable=False),
        sa.Column('direction', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "table_key in ('applications','jobs','boards','changes','cvs')",
            name=op.f('ck_ui_table_sorts_table_key'),
        ),
        sa.CheckConstraint("direction in ('asc','desc')", name=op.f('ck_ui_table_sorts_direction')),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_ui_table_sorts_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_ui_table_sorts')),
        sa.UniqueConstraint(
            'user_id', 'table_key', name=op.f('uq_ui_table_sorts_user_id_table_key')
        ),
    )


def downgrade() -> None:
    op.drop_table('ui_table_sorts')
