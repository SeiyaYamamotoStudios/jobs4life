"""Candidate facts from CVs, and profile questions 15/16

Revision ID: c41a9f3b27d5
Revises: a6b1efb733d7

Two changes, both from CLAUDE.md's 2026-09-18 decision "the corpus starts from
CVs, but only what the user confirms is evidence".

`candidate_facts` holds what a model proposed from an uploaded CV, awaiting the
user's confirmation. It is deliberately not `spans`: a proposal is not evidence,
and grounding on a CV would make every later CV "supported" and switch the
over-claim measurement off silently. `span_id` is null until the user confirms,
and points at the span *their* words produced -- `confirmed_text`, never
`fact_text`. `created_at` uses `clock_timestamp()` rather than `now()` because
roles and facts are listed in CV order, which is insertion order, and one upload
writes every row in a single transaction.

The `profile_answers.question_key` CHECK gains `depth_genuine` and
`recurring_gaps` -- questions 15 and 16, the two answers that are claims about
the person rather than preferences and so also become corpus text.

Downgrade drops the table and narrows the CHECK back, deleting any answer to
either question first (the constraint cannot be restored while a row violates
it). Confirmed facts' spans are NOT removed by the downgrade: they are the
user's own words and outlive the table that proposed them.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'c41a9f3b27d5'
down_revision = 'a6b1efb733d7'
branch_labels = None
depends_on = None

_OLD_KEYS = (
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
)
_NEW_KEYS = (*_OLD_KEYS, 'depth_genuine', 'recurring_gaps')

_CANDIDATE_FACT_STATES = ('proposed', 'confirmed', 'rejected')


def _key_check(keys: tuple[str, ...]) -> str:
    return "question_key in ('" + "','".join(keys) + "')"


def upgrade() -> None:
    op.create_table(
        'candidate_facts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sent_document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role_label', sa.Text(), nullable=False),
        sa.Column('role_key', sa.Text(), nullable=False),
        sa.Column('source_line', sa.Text(), nullable=False),
        sa.Column('fact_text', sa.Text(), nullable=False),
        sa.Column('probe', sa.Text(), nullable=True),
        sa.Column('probe_answer', sa.Text(), nullable=True),
        sa.Column('state', sa.Text(), server_default='proposed', nullable=False),
        sa.Column('confirmed_text', sa.Text(), nullable=True),
        sa.Column('span_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('fingerprint', sa.String(length=64), nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('clock_timestamp()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state in ('" + "','".join(_CANDIDATE_FACT_STATES) + "')",
            name=op.f('ck_candidate_facts_state'),
        ),
        sa.CheckConstraint(
            "state <> 'confirmed' or confirmed_text is not null",
            name=op.f('ck_candidate_facts_confirmed_text_when_confirmed'),
        ),
        sa.ForeignKeyConstraint(
            ['sent_document_id'],
            ['sent_documents.id'],
            name=op.f('fk_candidate_facts_sent_document_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['span_id'], ['spans.id'], name=op.f('fk_candidate_facts_span_id')
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_candidate_facts_user_id'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_candidate_facts')),
        sa.UniqueConstraint(
            'user_id', 'fingerprint', name=op.f('uq_candidate_facts_user_id_fingerprint')
        ),
    )
    op.create_index(
        'ix_candidate_facts_user_id_state', 'candidate_facts', ['user_id', 'state'], unique=False
    )
    op.create_index(
        'ix_candidate_facts_user_id_created_at',
        'candidate_facts',
        ['user_id', 'created_at'],
        unique=False,
    )

    op.drop_constraint(op.f('ck_profile_answers_question_key'), 'profile_answers', type_='check')
    op.create_check_constraint(
        op.f('ck_profile_answers_question_key'), 'profile_answers', _key_check(_NEW_KEYS)
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM profile_answers WHERE question_key in ('depth_genuine','recurring_gaps')"
    )
    op.drop_constraint(op.f('ck_profile_answers_question_key'), 'profile_answers', type_='check')
    op.create_check_constraint(
        op.f('ck_profile_answers_question_key'), 'profile_answers', _key_check(_OLD_KEYS)
    )

    op.drop_index('ix_candidate_facts_user_id_created_at', table_name='candidate_facts')
    op.drop_index('ix_candidate_facts_user_id_state', table_name='candidate_facts')
    op.drop_table('candidate_facts')
