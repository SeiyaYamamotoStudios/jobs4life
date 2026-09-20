"""application questions

Revision ID: a8a7cf5c3bc4
Revises: a6b1efb733d7

Application questions -- both ways, advised not prescribed (NEXT.md's task 4,
CLAUDE.md's 2026-09-18 decision). Two tables:

`application_questions`: the question text, written once.

`application_question_answers`: every attempt to answer it, append-only --
never an UPDATE to a previous row, the same rule `profile_answers` follows for
the same reason. `kind` is `'user'` (the user's own words, checked by the claim
gate) or `'draft'` (generated from the corpus, then gated automatically).
`gate_result` and `assessment` are JSONB, NULL until `status='done'`.
`trace_id` groups the model call(s) one attempt made, so the per-attempt cost
is `SELECT sum(cost_usd) FROM runs WHERE trace_id = ...`, the same convention
`drafts.trace_id` uses.

Hand-written (no `Vector` column, so the usual pgvector import is moot). The
value lists are written out as literals, never imported, because a migration is
a snapshot. Reverses cleanly.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a8a7cf5c3bc4'
down_revision = 'a6b1efb733d7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'application_questions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('application_id', sa.UUID(), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_application_questions_user_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['application_id'],
            ['applications.id'],
            name=op.f('fk_application_questions_application_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_application_questions')),
    )
    op.create_index(
        op.f('ix_application_questions_user_id_application_id'),
        'application_questions',
        ['user_id', 'application_id'],
    )

    op.create_table(
        'application_question_answers',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('question_id', sa.UUID(), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('answer_text', sa.Text(), server_default='', nullable=False),
        sa.Column('status', sa.Text(), server_default='pending', nullable=False),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('gate_result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('assessment', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('model', sa.Text(), nullable=True),
        sa.Column('trace_id', sa.UUID(), nullable=True),
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
            "kind in ('user','draft')", name=op.f('ck_application_question_answers_kind')
        ),
        sa.CheckConstraint(
            "status in ('pending','done','failed')",
            name=op.f('ck_application_question_answers_status'),
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in "
            "('no_api_key','api_key_rejected','model_refused','model_error',"
            "'credential_unreadable','no_requirements')",
            name=op.f('ck_application_question_answers_error_code'),
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            name=op.f('fk_application_question_answers_user_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['question_id'],
            ['application_questions.id'],
            name=op.f('fk_application_question_answers_question_id'),
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_application_question_answers')),
    )
    op.create_index(
        op.f('ix_application_question_answers_question_id_created_at'),
        'application_question_answers',
        ['question_id', 'created_at'],
    )


def downgrade() -> None:
    op.drop_table('application_question_answers')
    op.drop_table('application_questions')
