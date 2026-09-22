"""score pushbacks and local overrides

Revision ID: a3f81c5d2e94
Revises: e9c4a71d5b28

What happens when the user disagrees with a score, recorded whether or not it
changes anything -- `~/jobs4life-profile-research/feedback-loops.md`, "The loop
we should build".

`score_pushbacks` is append-only and is **the store**: there is no preference
weight column anywhere, because a dimension's displacement is
`sum(applied_delta)` over this user's applied rows. That is what makes the
drift meter a query rather than a promise, and what makes "the profile drifted
towards whatever was comfortable" a thing you can see instead of a thing you
have to trust did not happen.

Two CHECK constraints carry design decisions rather than data hygiene:

  * `capability_up_never_moves` -- a claim that the tool has UNDERRATED you may
    never carry a non-zero delta. The rule lives in `jfl_core.pushback`, which
    returns before any arithmetic runs; this is the same rule written where no
    future caller can get past it.
  * `applied_iff_delta` -- a pushback has either not been applied and moved
    nothing, or been applied and says what it did. There is no halfway state in
    which something moved and the receipt does not know it.

`score_overrides` is the escape hatch, deliberately a separate table: the user
setting a displayed number by hand for one application, labelled as an
override, feeding no dimension and reaching no other job.

Hand-written (no `Vector` column, so the usual pgvector import is moot). The
value lists are literals, never imported, because a migration is a snapshot.
Reverses cleanly: dropping these two tables restores every score to exactly the
number the pipeline produced, which is the point -- nothing else was ever
edited.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a3f81c5d2e94'
down_revision = 'e9c4a71d5b28'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'score_pushbacks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('application_id', sa.UUID(), nullable=False),
        sa.Column('score_id', sa.UUID(), nullable=False),
        sa.Column('axis', sa.Text(), nullable=False),
        sa.Column('dimension', sa.Text(), nullable=False),
        sa.Column('target_dimension', sa.Text(), server_default='', nullable=False),
        sa.Column('shown_score', sa.Integer(), nullable=True),
        sa.Column('shown_explanation', sa.Text(), server_default='', nullable=False),
        sa.Column('user_text', sa.Text(), nullable=False),
        sa.Column('asserted_direction', sa.Text(), nullable=False),
        sa.Column(
            'asserted_points', sa.Numeric(4, 2), server_default=sa.text('1'), nullable=False
        ),
        sa.Column(
            'status', sa.Text(), server_default='awaiting_classification', nullable=False
        ),
        sa.Column('classification', sa.Text(), nullable=True),
        sa.Column('classification_source', sa.Text(), server_default='none', nullable=False),
        sa.Column('classification_note', sa.Text(), server_default='', nullable=False),
        sa.Column('new_information', sa.Boolean(), nullable=True),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('trace_id', sa.UUID(), nullable=True),
        sa.Column('applied_delta', sa.Numeric(6, 3), nullable=True),
        sa.Column('prior_observations', sa.Integer(), nullable=True),
        sa.Column('disposition', sa.Text(), nullable=True),
        sa.Column(
            'effect',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column('evidence_question', sa.Text(), server_default='', nullable=False),
        sa.Column('resulting_span_id', sa.UUID(), nullable=True),
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
        sa.Column('applied_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.CheckConstraint("axis in ('want','get')", name=op.f('ck_score_pushbacks_axis')),
        sa.CheckConstraint(
            "asserted_direction in ('up','down')",
            name=op.f('ck_score_pushbacks_asserted_direction'),
        ),
        sa.CheckConstraint(
            "status in ('awaiting_classification','classified','applied')",
            name=op.f('ck_score_pushbacks_status'),
        ),
        sa.CheckConstraint(
            "classification is null or classification in "
            "('preference','capability','factual')",
            name=op.f('ck_score_pushbacks_classification'),
        ),
        sa.CheckConstraint(
            "classification_source in ('none','model','user')",
            name=op.f('ck_score_pushbacks_classification_source'),
        ),
        sa.CheckConstraint(
            "disposition is null or disposition in "
            "('accepted','recorded_only','pending_evidence')",
            name=op.f('ck_score_pushbacks_disposition'),
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in "
            "('no_api_key','api_key_rejected','model_refused','model_error',"
            "'credential_unreadable')",
            name=op.f('ck_score_pushbacks_error_code'),
        ),
        sa.CheckConstraint(
            'asserted_points > 0 and asserted_points <= 3',
            name=op.f('ck_score_pushbacks_asserted_points'),
        ),
        sa.CheckConstraint(
            "not (classification = 'capability' and asserted_direction = 'up' "
            "and applied_delta is not null and applied_delta <> 0)",
            name=op.f('ck_score_pushbacks_capability_up_never_moves'),
        ),
        sa.CheckConstraint(
            "(status = 'applied') = (applied_delta is not null)",
            name=op.f('ck_score_pushbacks_applied_iff_delta'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_score_pushbacks_user_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['application_id'],
            ['applications.id'],
            name=op.f('fk_score_pushbacks_application_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['score_id'],
            ['application_scores.id'],
            name=op.f('fk_score_pushbacks_score_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['resulting_span_id'],
            ['spans.id'],
            name=op.f('fk_score_pushbacks_resulting_span_id'),
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_score_pushbacks')),
    )
    op.create_index(
        'ix_score_pushbacks_user_id_created_at',
        'score_pushbacks',
        ['user_id', sa.text('created_at DESC')],
        unique=False,
    )
    op.create_index(
        'ix_score_pushbacks_user_id_target_dimension',
        'score_pushbacks',
        ['user_id', 'target_dimension'],
        unique=False,
    )
    op.create_index(
        'ix_score_pushbacks_user_id_application_id',
        'score_pushbacks',
        ['user_id', 'application_id'],
        unique=False,
    )

    op.create_table(
        'score_overrides',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('application_id', sa.UUID(), nullable=False),
        sa.Column('axis', sa.Text(), nullable=False),
        sa.Column('value', sa.Integer(), nullable=True),
        sa.Column('note', sa.Text(), server_default='', nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.CheckConstraint("axis in ('want','get')", name=op.f('ck_score_overrides_axis')),
        sa.CheckConstraint(
            'value is null or (value between 1 and 10)',
            name=op.f('ck_score_overrides_value'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_score_overrides_user_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['application_id'],
            ['applications.id'],
            name=op.f('fk_score_overrides_application_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_score_overrides')),
    )
    op.create_index(
        'ix_score_overrides_user_id_application_id_created_at',
        'score_overrides',
        ['user_id', 'application_id', sa.text('created_at DESC')],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        'ix_score_overrides_user_id_application_id_created_at', table_name='score_overrides'
    )
    op.drop_table('score_overrides')
    op.drop_index('ix_score_pushbacks_user_id_application_id', table_name='score_pushbacks')
    op.drop_index('ix_score_pushbacks_user_id_target_dimension', table_name='score_pushbacks')
    op.drop_index('ix_score_pushbacks_user_id_created_at', table_name='score_pushbacks')
    op.drop_table('score_pushbacks')
