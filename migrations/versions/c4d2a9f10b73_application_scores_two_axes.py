"""application scores: two axes, never composited

Revision ID: c4d2a9f10b73
Revises: a6b1efb733d7

PLAN.md slice B4. `application_scores`: one row per scoring run against one
application, append-only -- a re-score inserts a new row and the page reads the
latest, so what the tool said, when, and what it cost stays readable.

**There is deliberately no composite column.** CLAUDE.md's standing decision
and PLAN.md B4: "could I get this" and "do I want this" are reported separately
and never averaged, so a third number has nowhere to live here either. Each is
1-10, pinned by its own CHECK, and each carries its own paragraph.

`objective_verdicts`, `hard_gate_breaches`, `levers` and `not_stated` are JSONB
lists of the shapes in `jfl_core.models`; they are read back whole and
rendered, never queried structurally.

Hand-written (no `Vector` column, so the usual pgvector import is moot). The
value lists are written out as literals, never imported, because a migration is
a snapshot. Reverses cleanly.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c4d2a9f10b73"
down_revision = "a6b1efb733d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "application_scores",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("application_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("could_get_score", sa.Integer(), nullable=True),
        sa.Column("could_get_assessment", sa.Text(), server_default="", nullable=False),
        sa.Column("want_it_score", sa.Integer(), nullable=True),
        sa.Column("want_it_assessment", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "objective_verdicts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "hard_gate_breaches",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "levers",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "not_stated",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("trace_id", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            # clock_timestamp(), not now(): append-only history, and two runs in
            # one transaction must not share "the latest" -- see tables.py.
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status in ('pending','done','failed')",
            name=op.f("ck_application_scores_status"),
        ),
        sa.CheckConstraint(
            "error_code is null or error_code in "
            "('no_api_key','api_key_rejected','model_refused','model_error',"
            "'credential_unreadable','no_requirements')",
            name=op.f("ck_application_scores_error_code"),
        ),
        sa.CheckConstraint(
            "could_get_score is null or (could_get_score between 1 and 10)",
            name=op.f("ck_application_scores_could_get_score"),
        ),
        sa.CheckConstraint(
            "want_it_score is null or (want_it_score between 1 and 10)",
            name=op.f("ck_application_scores_want_it_score"),
        ),
        sa.ForeignKeyConstraint(
            ["application_id"],
            ["applications.id"],
            name=op.f("fk_application_scores_application_id"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_application_scores_user_id"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_application_scores")),
    )
    op.create_index(
        "ix_application_scores_user_id_application_id_created_at",
        "application_scores",
        ["user_id", "application_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_application_scores_user_id_application_id_created_at",
        table_name="application_scores",
    )
    op.drop_table("application_scores")
