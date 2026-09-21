"""application_scores.constraint_verdicts -- what the want-it number is derived from

Revision ID: d1f4b7c20a55
Revises: d7f2b8c1a394

`docs/profile-schema.md`, agreed 2026-09-21. **"Do I want this" is not a
predicted-satisfaction score.** The model is never asked for the number: it
gives a verdict -- `evidenced` / `partial` / `silent` / `contradicted` -- on
every constraint and every objective the user recorded, and `jfl_core.fit`
derives the number from those in code, where it can be read and argued with.

This column is where the constraint half of that lands. Without it the number
has nothing behind it, and a panel showing a 7 with no list under it is exactly
the unarguable score this project exists to avoid. The objective half already
has `objective_verdicts`, added with the table.

Existing rows default to `'[]'`, which reads correctly as "this run predates
the derivation" rather than as "no constraints matched".

This revision originally also created `profiles`. It no longer does: `profiles`
is created by `d7f2b8c1a394`, which this now follows, and two migrations
creating one table is how a chain stops upgrading on a fresh database.

Downgrade drops the column. Lossy in the ordinary way -- the derived number
stays in `want_it_score` with nothing left saying what it was derived from.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd1f4b7c20a55'
down_revision = 'd7f2b8c1a394'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'application_scores',
        sa.Column(
            'constraint_verdicts',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column('application_scores', 'constraint_verdicts')
