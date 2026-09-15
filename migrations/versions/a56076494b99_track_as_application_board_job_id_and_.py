"""track as application: board_job_id and description_unavailable

Revision ID: a56076494b99
Revises: e5396ef31c67

Slice C7's "Track as application" button. Two changes, both on `applications`.

`board_job_id` links a tracked application back to the watched-board job it was
created from -- nullable, and `ON DELETE SET NULL`: losing the board, or the job
falling off it, must never take the tracked application down with it. This is
provenance, not a dependency the application's own life runs on.

`description_unavailable` joins the closed set `extraction_error_code` may hold,
for when the board fetch behind that button could not read a description at
all (unsupported platform, 404, exhausted retries). The CHECK constraint is
dropped and recreated with the wider set, per the standing rule that a CHECK
built from a Python tuple is edited by replacing it whole, never patched in
SQL.

Earlier migrations are deployed to production and are not edited. Both changes
here are additive to a table with rows in it: the new column is nullable and
every existing application reads `board_job_id IS NULL` (added by paste, which
is exactly what they are), and the wider CHECK accepts everything the narrower
one did.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a56076494b99"
down_revision = "e5396ef31c67"
branch_labels = None
depends_on = None

_EXTRACTION_ERROR_CODES = (
    "no_api_key",
    "api_key_rejected",
    "no_job_ad",
    "ad_too_long",
    "model_refused",
    "model_error",
    "credential_unreadable",
    "description_unavailable",
)


def upgrade() -> None:
    op.add_column(
        "applications",
        sa.Column(
            "board_job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("board_jobs.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_applications_user_id_board_job_id", "applications", ["user_id", "board_job_id"]
    )
    op.drop_constraint(op.f("ck_applications_extraction_error_code"), "applications", type_="check")
    op.create_check_constraint(
        op.f("ck_applications_extraction_error_code"),
        "applications",
        "extraction_error_code is null or extraction_error_code in ('"
        + "','".join(_EXTRACTION_ERROR_CODES)
        + "')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_applications_extraction_error_code"), "applications", type_="check")
    op.create_check_constraint(
        op.f("ck_applications_extraction_error_code"),
        "applications",
        "extraction_error_code is null or extraction_error_code in "
        "('no_api_key','api_key_rejected','no_job_ad','ad_too_long','model_refused',"
        "'model_error','credential_unreadable')",
    )
    op.drop_index("ix_applications_user_id_board_job_id", table_name="applications")
    op.drop_column("applications", "board_job_id")
