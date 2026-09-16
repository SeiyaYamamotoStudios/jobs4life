"""Version profile_objectives like profile_answers

Revision ID: a6b1efb733d7
Revises: 170e706e57bc

`profile_objectives` was the odd one out among the profile setup tables: a
mutable upsert per (user_id, ordinal), where clearing both fields deleted the
row -- so what a user once said an objective was could be lost. This makes it
append-only, the same shape as `profile_answers`: a save to an ordinal is a
new row, never an UPDATE; the current value of a slot is its latest row,
read back with `DISTINCT ON` in the repository; and `created_at` uses
`clock_timestamp()`, not `now()`, so two saves to the same ordinal inside one
transaction do not tie (the same reasoning as `profile_answers.created_at`).

Upgrade preserves every existing row as its ordinal's *first* version: the
unique (user_id, ordinal) constraint is dropped (a slot may now have many
rows), an index is added for the DISTINCT ON lookup, `updated_at` is dropped
(a version is immutable once written -- there is nothing left for it to
mean), and `created_at`'s default moves to `clock_timestamp()`. No data is
touched; every row that existed keeps its `id` and `created_at`.

Downgrade cannot invent the mutable-row history back. It keeps only the
latest version per (user_id, ordinal) -- the rest is deleted -- and then
deletes any latest version that is blank in both fields, matching the old
schema's rule that an unused slot is an absent row, not an empty one. The
restored `updated_at` is backfilled from that surviving row's `created_at`,
which is the closest available approximation, not the true original
last-edit time (that information does not survive a round trip once several
versions have been saved for one ordinal).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a6b1efb733d7'
down_revision = '170e706e57bc'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f('uq_profile_objectives_user_id_ordinal'), 'profile_objectives', type_='unique'
    )
    op.create_index(
        'ix_profile_objectives_user_id_ordinal_created_at',
        'profile_objectives',
        ['user_id', 'ordinal', 'created_at'],
        unique=False,
    )
    op.drop_column('profile_objectives', 'updated_at')
    # Alembic's autogenerate does not compare server defaults (compare_server_default
    # is off in migrations/env.py), so this does not show up in a diff -- adjusted by
    # hand. `clock_timestamp()`, not `now()`: see `profile_answers.created_at` for why.
    op.alter_column(
        'profile_objectives',
        'created_at',
        server_default=sa.text('clock_timestamp()'),
    )


def downgrade() -> None:
    # Collapse to one row per (user_id, ordinal): the latest version by
    # created_at, ties broken by id. Everything older is genuinely lost --
    # the mutable schema had nowhere to put it.
    op.execute(
        """
        DELETE FROM profile_objectives
        WHERE id NOT IN (
            SELECT DISTINCT ON (user_id, ordinal) id
            FROM profile_objectives
            ORDER BY user_id, ordinal, created_at DESC, id DESC
        )
        """
    )
    # A blank surviving version means the slot was cleared (or never
    # meaningfully set) -- the old schema represented that as no row at all.
    op.execute(
        """
        DELETE FROM profile_objectives
        WHERE btrim(objective_text) = '' AND btrim(evidence_text) = ''
        """
    )
    op.alter_column(
        'profile_objectives',
        'created_at',
        server_default=sa.text('now()'),
    )
    op.add_column(
        'profile_objectives',
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
    )
    # Backfilled from the surviving row's created_at -- see module docstring:
    # this is an approximation, not the true original last-edit time.
    op.execute('UPDATE profile_objectives SET updated_at = created_at')
    op.alter_column(
        'profile_objectives',
        'updated_at',
        nullable=False,
        server_default=sa.text('now()'),
    )
    op.drop_index('ix_profile_objectives_user_id_ordinal_created_at', table_name='profile_objectives')
    op.create_unique_constraint(
        op.f('uq_profile_objectives_user_id_ordinal'),
        'profile_objectives',
        ['user_id', 'ordinal'],
    )
