"""score pushbacks: withdrawn_at, for "Not what I meant"

Revision ID: c4e8a2f61d57
Revises: b7d3e9f10a42

The pushback box now applies what it read straight away and offers an undo,
instead of asking the user to confirm a classification first. Undo is a mark on
the log, not an edit of it: the row keeps its delta and its receipt, and every
sum that makes up the profile -- displacement, observation count, the drift
meter -- skips a withdrawn row.

`withdrawn_only_if_applied` -- only something that was applied can be undone.
The two design CHECKs from a3f81c5d2e94 (`capability_up_never_moves`,
`applied_iff_delta`) are untouched.

Hand-written. Reverses cleanly: dropping the column makes every undone
correction count again, which is why a downgrade is a decision, not a chore.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = 'c4e8a2f61d57'
down_revision = 'b7d3e9f10a42'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'score_pushbacks',
        sa.Column('withdrawn_at', sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        op.f('ck_score_pushbacks_withdrawn_only_if_applied'),
        'score_pushbacks',
        "withdrawn_at is null or status = 'applied'",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f('ck_score_pushbacks_withdrawn_only_if_applied'), 'score_pushbacks', type_='check'
    )
    op.drop_column('score_pushbacks', 'withdrawn_at')
