"""web auth: google identity, sessions, credential custody

Slice A1-A4. Additive throughout -- no existing column changes type, loses a
constraint, or is dropped.

Two hand edits on top of what alembic generated:

  * `user_credentials.dek_nonce` is NOT NULL on a table that already exists. The
    table has never been written to (the crypto it was designed for did not
    exist until this slice), but "should be empty" is not a guarantee a migration
    may rely on, so the column arrives with a server default and the default is
    then dropped. On an empty table the result is identical to a plain NOT NULL
    add; on a non-empty one it succeeds instead of aborting, and the placeholder
    fails to authenticate at unseal time, which is the loud failure you want.
  * no `import pgvector.sqlalchemy` -- alembic always omits it, but nothing here
    touches a Vector column, so there is nothing to add.

`users.google_sub` is nullable: the seeded local user (migration 0002) has no
Google identity and must keep working. UNIQUE over a nullable column is right
here -- Postgres treats NULLs as distinct, so any number of non-Google accounts
coexist while every Google `sub` maps to at most one row.

Reverses cleanly: downgrade drops only what upgrade added.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'cb19e34095ce'
down_revision = '96bef072bfd2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('sessions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('csrf_token', sa.Text(), nullable=False),
    sa.Column('user_agent', sa.Text(), nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_seen_at', postgresql.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('expires_at', postgresql.TIMESTAMP(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_sessions_user_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_sessions')),
    sa.UniqueConstraint('token_hash', name=op.f('uq_sessions_token_hash'))
    )
    op.create_index('ix_sessions_expires_at', 'sessions', ['expires_at'], unique=False)
    op.create_index('ix_sessions_user_id', 'sessions', ['user_id'], unique=False)

    # Hand edit: default, then drop the default. See the module docstring.
    op.add_column(
        'user_credentials',
        sa.Column('dek_nonce', sa.LargeBinary(), nullable=False, server_default=sa.text("'\\x'::bytea")),
    )
    op.alter_column('user_credentials', 'dek_nonce', server_default=None)

    op.add_column('user_credentials', sa.Column('key_hint', sa.Text(), server_default='', nullable=False))
    op.add_column('users', sa.Column('google_sub', sa.Text(), nullable=True))
    op.create_unique_constraint(op.f('uq_users_google_sub'), 'users', ['google_sub'])


def downgrade() -> None:
    op.drop_constraint(op.f('uq_users_google_sub'), 'users', type_='unique')
    op.drop_column('users', 'google_sub')
    op.drop_column('user_credentials', 'key_hint')
    op.drop_column('user_credentials', 'dek_nonce')
    op.drop_index('ix_sessions_user_id', table_name='sessions')
    op.drop_index('ix_sessions_expires_at', table_name='sessions')
    op.drop_table('sessions')
