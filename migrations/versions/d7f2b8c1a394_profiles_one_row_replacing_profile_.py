"""The profile as one row, replacing profile_answers/objectives/ruled_out

Revision ID: d7f2b8c1a394
Revises: c4d9a1f6e207

`docs/profile-schema.md`, agreed 2026-09-21. The eighteen free-text questions of
PLAN.md B3a are replaced by one denormalised JSONB row per save: append-only,
latest wins, a few KB each. One read now serves scoring, drafting and the
filter, where three differently-versioned tables served none of them well.

**This replaces rather than migrates.** Production holds zero rows in all three
old tables, so there is nothing to carry across and no transform to get wrong;
the upgrade drops them outright. Had there been rows, the honest move would have
been a transform, because a free-text answer cannot be read into a structured
constraint without a model, and a model on that path would hold the user to
wording they did not choose.

`data` is JSONB and therefore has no CHECK constraint. `jfl_core.profile.Profile`
is the only write path and is where every closed set (`stance`, `tier`,
`interest`, constraint `kind`) is enforced, with a test pairing its Literals
against what the screens offer. Weaker than a CHECK, accepted deliberately
(owner, 2026-09-21), and named in the design as the price of a shape we expect
to change while we learn what belongs in it.

`created_at` defaults to `clock_timestamp()`, not `now()`, for the reason the
retired `profile_answers.created_at` gave and this table inherits: the current
profile is *the latest row*, `now()` is transaction-start time, and two saves in
one transaction would tie -- making "latest" ambiguous exactly where it decides
what the user is shown.

Downgrade recreates the three tables as they stood at `c4d9a1f6e207` -- empty.
Profiles saved under the new schema are deleted with the table and are **not**
fanned back out into answers: there is no honest mapping from a ranked objective
or a tiered capability onto a free-text question, and inventing one would put
words in the user's mouth. The downgrade says so rather than pretending.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd7f2b8c1a394'
down_revision = 'c4d9a1f6e207'
branch_labels = None
depends_on = None

# The key set as `c41a9f3b27d5` left it: the twelve preference questions plus
# the two that also became corpus text. Only the downgrade needs it now.
_QUESTION_KEYS = (
    'location_commute',
    'workplace_arrangements',
    'levels',
    'comp_floor',
    'contract_types',
    'notice_period',
    'right_to_work',
    'categorical_no',
    'disciplines',
    'trajectory',
    'employer_deal_breakers',
    'warning_signs',
    'depth_genuine',
    'recurring_gaps',
)


def upgrade() -> None:
    op.create_table(
        'profiles',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('schema_version', sa.Integer(), nullable=False),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        # clock_timestamp(), not now() -- see the module docstring.
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_profiles_user_id'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_profiles')),
    )
    # Descending on created_at: every read is "this user's latest", and the
    # history page walks the same order.
    op.create_index(
        'ix_profiles_user_id_created_at',
        'profiles',
        ['user_id', sa.text('created_at DESC')],
        unique=False,
    )

    op.drop_index(
        'ix_profile_ruled_out_user_id_recorded_at', table_name='profile_ruled_out'
    )
    op.drop_table('profile_ruled_out')
    op.drop_index(
        'ix_profile_objectives_user_id_ordinal_created_at', table_name='profile_objectives'
    )
    op.drop_table('profile_objectives')
    op.drop_index(
        'ix_profile_answers_user_id_question_key_created_at', table_name='profile_answers'
    )
    op.drop_table('profile_answers')


def downgrade() -> None:
    # The three tables exactly as c4d9a1f6e207 left them, and empty -- see the
    # module docstring for why nothing is fanned back out into them.
    op.create_table(
        'profile_answers',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('question_key', sa.Text(), nullable=False),
        sa.Column('answer_text', sa.Text(), server_default='', nullable=False),
        sa.Column('structured', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "question_key in ('" + "','".join(_QUESTION_KEYS) + "')",
            name=op.f('ck_profile_answers_question_key'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_profile_answers_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_profile_answers')),
    )
    op.create_index(
        'ix_profile_answers_user_id_question_key_created_at',
        'profile_answers',
        ['user_id', 'question_key', 'created_at'],
        unique=False,
    )

    op.create_table(
        'profile_objectives',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('objective_text', sa.Text(), server_default='', nullable=False),
        sa.Column('evidence_text', sa.Text(), server_default='', nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            'ordinal between 1 and 4', name=op.f('ck_profile_objectives_ordinal_range')
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_profile_objectives_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_profile_objectives')),
    )
    op.create_index(
        'ix_profile_objectives_user_id_ordinal_created_at',
        'profile_objectives',
        ['user_id', 'ordinal', 'created_at'],
        unique=False,
    )

    op.create_table(
        'profile_ruled_out',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('decision_text', sa.Text(), nullable=False),
        sa.Column(
            'recorded_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.Column('reopened_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_profile_ruled_out_user_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_profile_ruled_out')),
    )
    op.create_index(
        'ix_profile_ruled_out_user_id_recorded_at',
        'profile_ruled_out',
        ['user_id', 'recorded_at'],
        unique=False,
    )

    op.drop_index('ix_profiles_user_id_created_at', table_name='profiles')
    op.drop_table('profiles')
