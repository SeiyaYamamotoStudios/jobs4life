"""drop users email unique not null

Revision ID: 6b3ce06d7b4e
Revises: a56076494b99

Identity is Google's `sub` (`users.google_sub`, already the lookup key --
`PostgresUserRepository.upsert_google_user` has never matched on email).
`email` is presentation-only, refreshed from the id token on every login, so
the UNIQUE NOT NULL constraint bought nothing but a failure mode: Google
reassigns an address to a different person, and the account still displaying
it under the old owner's `sub` made the new owner's first login a hard
`IntegrityError` (previously surfaced as `DuplicateEmailError` / the
`email_taken` login error, both removed in the same change as this
migration -- see `packages/core/src/jfl_core/storage/accounts.py` and
`packages/web/src/jfl_web/routes/auth.py`).

No `import pgvector.sqlalchemy` -- nothing here touches a Vector column.

Downgrade recreates NOT NULL and UNIQUE and **will fail** if, by the time it
runs, any row has a null email or two rows share one -- exactly the state
this migration exists to make reachable. Clean up or delete the offending
rows by hand before downgrading past this point.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = '6b3ce06d7b4e'
down_revision = 'a56076494b99'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(op.f('uq_users_email'), 'users', type_='unique')
    op.alter_column('users', 'email', existing_type=sa.TEXT(), nullable=True)


def downgrade() -> None:
    op.alter_column('users', 'email', existing_type=sa.TEXT(), nullable=False)
    op.create_unique_constraint(op.f('uq_users_email'), 'users', ['email'])
