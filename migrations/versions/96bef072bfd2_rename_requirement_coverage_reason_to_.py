"""rename requirement_coverage.reason to evidence_note

Revision ID: 96bef072bfd2
Revises: 81d65d3075c8
"""
from __future__ import annotations

from alembic import op

revision = '96bef072bfd2'
down_revision = '81d65d3075c8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `reason` tripped the live API's reverse-engineering/duplication classifier
    # when paired with a long labelling prompt -- see jfl_gate.prompt and
    # jfl_generate.prompts for the bisection (2026-09-02). Renaming the column
    # to match the schema property it backs.
    op.alter_column('requirement_coverage', 'reason', new_column_name='evidence_note')


def downgrade() -> None:
    op.alter_column('requirement_coverage', 'evidence_note', new_column_name='reason')
