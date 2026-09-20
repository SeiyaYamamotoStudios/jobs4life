"""Profile questions 15/16

Revision ID: c41a9f3b27d5
Revises: a6b1efb733d7

The `profile_answers.question_key` CHECK gains `depth_genuine` and
`recurring_gaps` -- questions 15 and 16, the two profile answers that are claims
about the person rather than preferences, and so also become corpus text
(CLAUDE.md, 2026-09-18). `candidate_facts` itself is created by the CV-intake
migration further along the chain; this file carries only the CHECK.
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

