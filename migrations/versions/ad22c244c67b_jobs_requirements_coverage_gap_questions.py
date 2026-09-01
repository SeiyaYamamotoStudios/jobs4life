"""jobs, requirements, coverage, gap questions

Revision ID: ad22c244c67b
Revises: 309cf277970b
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'ad22c244c67b'
down_revision = '309cf277970b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('jobs',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('source', sa.Text(), nullable=False),
    sa.Column('employer', sa.Text(), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('location', sa.Text(), nullable=True),
    sa.Column('url', sa.Text(), nullable=True),
    sa.Column('raw_text', sa.Text(), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source in ('paste','file')", name=op.f('ck_jobs_source')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_jobs_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_jobs')),
    sa.UniqueConstraint('user_id', 'content_hash', name=op.f('uq_jobs_user_id_content_hash'))
    )
    op.create_table('job_requirements',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('job_id', sa.UUID(), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('necessity', sa.Text(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("necessity in ('essential','desirable','unstated')", name=op.f('ck_job_requirements_necessity')),
    sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], name=op.f('fk_job_requirements_job_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_job_requirements_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_job_requirements'))
    )
    op.create_table('gap_questions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('requirement_id', sa.UUID(), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('status', sa.Text(), server_default='open', nullable=False),
    sa.Column('answer_text', sa.Text(), nullable=True),
    sa.Column('answered_at', postgresql.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('resulting_span_id', sa.UUID(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status in ('open','answered','dismissed')", name=op.f('ck_gap_questions_status')),
    sa.ForeignKeyConstraint(['requirement_id'], ['job_requirements.id'], name=op.f('fk_gap_questions_requirement_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['resulting_span_id'], ['spans.id'], name=op.f('fk_gap_questions_resulting_span_id')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_gap_questions_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_gap_questions'))
    )
    op.create_index('ix_gap_questions_user_id_status_created_at', 'gap_questions', ['user_id', 'status', 'created_at'], unique=False)
    op.create_table('requirement_coverage',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('requirement_id', sa.UUID(), nullable=False),
    sa.Column('trace_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('cited_span_ids', postgresql.ARRAY(sa.UUID()), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("status in ('evidenced','partial','absent','contradicted')", name=op.f('ck_requirement_coverage_status')),
    sa.ForeignKeyConstraint(['requirement_id'], ['job_requirements.id'], name=op.f('fk_requirement_coverage_requirement_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_requirement_coverage_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_requirement_coverage'))
    )
    op.create_index('ix_requirement_coverage_user_id_requirement_id_created_at', 'requirement_coverage', ['user_id', 'requirement_id', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_requirement_coverage_user_id_requirement_id_created_at', table_name='requirement_coverage')
    op.drop_table('requirement_coverage')
    op.drop_index('ix_gap_questions_user_id_status_created_at', table_name='gap_questions')
    op.drop_table('gap_questions')
    op.drop_table('job_requirements')
    op.drop_table('jobs')
