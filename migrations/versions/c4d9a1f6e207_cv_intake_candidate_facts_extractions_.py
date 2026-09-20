"""CV intake: candidate facts, cv extractions, hosted corpus markdown

Revision ID: c4d9a1f6e207
Revises: a6b1efb733d7

PLAN.md slice B6, redesigned by CLAUDE.md's 2026-09-18 decision: a new user's
corpus starts from their CVs, but only what they confirm is evidence.

Three changes, and the first is the one worth reading.

**`documents.text`, and a fourth `storage_kind`.** "Corpus markdown is the
source of truth; the database is a rebuildable index over it" was written for a
CLI with a `corpus/` directory. A hosted user has no such directory, so the
only way that statement stays true is for the markdown itself to be stored --
`documents.text`, NULL for a `local_file` document whose truth is still a file
on the owner's machine, and set for a `hosted` one that exists nowhere else.
`jfl_core.corpus_source` is the only writer.

**`cv_extractions`** -- one row per uploaded CV, recording the single model
call that reads it. Unique on `sent_document_id`, which is what lets a
redelivered task ask "is this already done?" with one read rather than
extracting (and charging) twice.

**`candidate_facts`** -- one row per fact a CV claims. Unique on
`(user_id, fingerprint)`, which is how thirty-three near-identical generated
CVs collapse into one list to confirm. Two CHECKs carry design decisions:
`state` is the three-state rule (confirmed / claimed-unconfirmed / rejected),
and `span_id is null or state = 'confirmed'` makes "we grounded on something
the user rejected" unrepresentable rather than merely unlikely.

Downgrade is lossy and says so: `hosted` documents have nowhere to go once
`documents.text` is dropped -- the markdown would survive only as spans, with
no source to rebuild from -- so they and their spans are deleted, which is the
honest reading of "this deployment can no longer hold corpus markdown".
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'c4d9a1f6e207'
down_revision = 'a6b1efb733d7'
branch_labels = None
depends_on = None

_OLD_STORAGE_KINDS = "storage_kind in ('local_file','upload','paste')"
_NEW_STORAGE_KINDS = "storage_kind in ('local_file','upload','paste','hosted')"


def upgrade() -> None:
    op.add_column('documents', sa.Column('text', sa.Text(), nullable=True))
    op.drop_constraint('storage_kind', 'documents', type_='check')
    op.create_check_constraint('storage_kind', 'documents', _NEW_STORAGE_KINDS)

    op.create_table(
        'cv_extractions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('sent_document_id', sa.UUID(), nullable=False),
        sa.Column('status', sa.Text(), server_default='pending', nullable=False),
        sa.Column('error_code', sa.Text(), nullable=True),
        sa.Column('facts_proposed', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('pending','done','failed')", name=op.f('ck_cv_extractions_status')
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in ('no_cv_text','no_api_key',"
            "'api_key_rejected','credential_unreadable','cv_too_long','model_refused',"
            "'model_error')",
            name=op.f('ck_cv_extractions_error_code'),
        ),
        sa.ForeignKeyConstraint(
            ['sent_document_id'],
            ['sent_documents.id'],
            name=op.f('fk_cv_extractions_sent_document_id'),
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['user_id'], ['users.id'], name=op.f('fk_cv_extractions_user_id'), ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_cv_extractions')),
        sa.UniqueConstraint('sent_document_id', name=op.f('uq_cv_extractions_sent_document_id')),
    )
    op.create_index(
        'ix_cv_extractions_user_id_status', 'cv_extractions', ['user_id', 'status'], unique=False
    )

    op.create_table(
        'candidate_facts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('sent_document_id', sa.UUID(), nullable=True),
        sa.Column('role_label', sa.Text(), nullable=False),
        sa.Column('role_key', sa.Text(), nullable=False),
        sa.Column('source_line', sa.Text(), nullable=False),
        sa.Column('fact_text', sa.Text(), nullable=False),
        sa.Column('probe', sa.Text(), nullable=True),
        sa.Column('probe_answer', sa.Text(), nullable=True),
        sa.Column('state', sa.Text(), server_default='proposed', nullable=False),
        sa.Column('confirmed_text', sa.Text(), nullable=True),
        sa.Column('span_id', sa.UUID(), nullable=True),
        sa.Column('fingerprint', sa.String(length=64), nullable=False),
        sa.Column('ordinal', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column(
            'created_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state in ('proposed','confirmed','rejected')",
            name=op.f('ck_candidate_facts_state'),
        ),
        sa.CheckConstraint(
            "span_id is null or state = 'confirmed'",
            name=op.f('ck_candidate_facts_span_iff_confirmed'),
        ),
        sa.ForeignKeyConstraint(
            ['sent_document_id'],
            ['sent_documents.id'],
            name=op.f('fk_candidate_facts_sent_document_id'),
            ondelete='SET NULL',
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
        'ix_candidate_facts_user_id_role_key_ordinal',
        'candidate_facts',
        ['user_id', 'role_key', 'ordinal'],
        unique=False,
    )
    op.create_index(
        'ix_candidate_facts_user_id_state', 'candidate_facts', ['user_id', 'state'], unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_candidate_facts_user_id_state', table_name='candidate_facts')
    op.drop_index('ix_candidate_facts_user_id_role_key_ordinal', table_name='candidate_facts')
    op.drop_table('candidate_facts')
    op.drop_index('ix_cv_extractions_user_id_status', table_name='cv_extractions')
    op.drop_table('cv_extractions')

    # Hosted documents exist only here -- see the module docstring. Their spans
    # go first (spans.document_id has no ON DELETE), and span_sentences and
    # span_embeddings follow by their own CASCADE.
    op.execute(
        """
        DELETE FROM spans
        WHERE document_id IN (SELECT id FROM documents WHERE storage_kind = 'hosted')
        """
    )
    op.execute("DELETE FROM documents WHERE storage_kind = 'hosted'")
    op.drop_constraint('storage_kind', 'documents', type_='check')
    op.create_check_constraint('storage_kind', 'documents', _OLD_STORAGE_KINDS)
    op.drop_column('documents', 'text')
