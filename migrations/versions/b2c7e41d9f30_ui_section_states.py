"""ui section states: which panels a user leaves open, and where that disagrees

Revision ID: b2c7e41d9f30
Revises: a3f81c5d2e94

Sections became collapsible (`docs/ui-sections.md`) and a choice has to survive
a reload, so it is stored per user rather than in the browser. The table does a
second job at no extra cost: every toggle also records what the screen *would*
have done without the stored choice, which turns "do people go against these
defaults" into a query instead of a thing someone eventually notices.

Deliberately not a section of `profiles`. That row is append-only and reads back
as what the user believed about themselves on a given day; a new version per
folded panel would bury real decisions under UI noise.

`section_key` carries no CHECK constraint. The set of keys is open by
construction -- a per-draft section is keyed by the draft's own id -- so a
closed list would need migrating every time a screen grows a panel. The route
validates the key's shape instead.

Hand-written (no `Vector` column, so the usual pgvector import is moot).
Reverses cleanly: dropping the table restores the agreed defaults for everyone
and loses only a record of preference, never content.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'b2c7e41d9f30'
down_revision = 'a3f81c5d2e94'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'ui_section_states',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('section_key', sa.Text(), nullable=False),
        sa.Column('is_open', sa.Boolean(), nullable=False),
        sa.Column('default_open', sa.Boolean(), nullable=False),
        sa.Column('toggles', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('against_default', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('last_opened_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_ui_section_states_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_ui_section_states')),
        sa.UniqueConstraint(
            'user_id', 'section_key', name=op.f('uq_ui_section_states_user_id_section_key')
        ),
    )


def downgrade() -> None:
    op.drop_table('ui_section_states')
