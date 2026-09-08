"""application extraction state: slice B3

Four columns on `applications`, nothing else touched.

The add form is now a paste box: the ad goes in, the application is created
with a provisional title, and a `extract_job_ad` task reads the ad in the
background. These columns are where that read's progress and result live.

  extraction_status      none | pending | done | failed. A separate axis from
                         `status`, which is where the application is in the
                         world -- an extraction failing says nothing about
                         whether the user has applied.
  extraction_error_code  a code from a closed set, never a message. The worker
                         writes this while holding the user's decrypted API
                         key, and a free-text column is exactly where a
                         careless `str(exc)` from the SDK ends up.
  extracted_at           when the read last succeeded.
  title_is_provisional   whether `title` is a placeholder taken from the ad's
                         first line rather than the user's own words. It is
                         what makes "extraction never overwrites something the
                         user typed" a condition in a WHERE clause instead of a
                         convention someone has to remember.

Every column is nullable or carries a server default, so this applies to a
table with rows in it: existing applications become `extraction_status='none'`
with a non-provisional title, which is exactly what they are -- their titles
were typed by hand on the old form.

Reverses cleanly: downgrade drops only what upgrade added. No hand edits were
needed (no `Vector` column is involved).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "2f6b0c1d94a7"
down_revision = "150bd8aa1e1a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "applications",
        sa.Column("extraction_status", sa.Text(), server_default="none", nullable=False),
    )
    op.add_column("applications", sa.Column("extraction_error_code", sa.Text(), nullable=True))
    op.add_column(
        "applications",
        sa.Column("extracted_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "applications",
        sa.Column(
            "title_is_provisional",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        op.f("ck_applications_extraction_status"),
        "applications",
        "extraction_status in ('none','pending','done','failed')",
    )
    op.create_check_constraint(
        op.f("ck_applications_extraction_error_code"),
        "applications",
        "extraction_error_code is null or extraction_error_code in "
        "('no_api_key','api_key_rejected','no_job_ad','ad_too_long','model_refused',"
        "'model_error','credential_unreadable')",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_applications_extraction_error_code"), "applications", type_="check"
    )
    op.drop_constraint(op.f("ck_applications_extraction_status"), "applications", type_="check")
    op.drop_column("applications", "title_is_provisional")
    op.drop_column("applications", "extracted_at")
    op.drop_column("applications", "extraction_error_code")
    op.drop_column("applications", "extraction_status")
