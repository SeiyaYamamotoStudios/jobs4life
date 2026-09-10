"""watched boards platform check accepts all twelve

Revision ID: 3c540957b0d2
Revises: f07cdd619c07

`f07cdd619c07` is deployed to production and is not edited here. Slice C's
adapter work (see `docs/ats-platforms.md`) added eight more board platforms --
SmartRecruiters, Rippling, Breezy, Teamtailor, Personio, Recruitee, Pinpoint,
Workable -- to `jfl_core.models.BoardPlatform` and the adapter registry, but
`ck_watched_boards_platform` was left listing only the original four, so a
real board on any of the eight failed at INSERT with no adapter-side signal
that anything was wrong. This drops and recreates that one constraint.

Production has zero boards today, so both directions are data-safe; the
downgrade is still written to actually restore the four, not just be present.
"""
from __future__ import annotations

from alembic import op

revision = '3c540957b0d2'
down_revision = 'f07cdd619c07'
branch_labels = None
depends_on = None

_TWELVE = (
    'greenhouse', 'ashby', 'lever', 'workday',
    'smartrecruiters', 'rippling', 'breezy', 'teamtailor',
    'personio', 'recruitee', 'pinpoint', 'workable',
)
_ORIGINAL_FOUR = ('greenhouse', 'ashby', 'lever', 'workday')


def upgrade() -> None:
    op.drop_constraint(op.f('ck_watched_boards_platform'), 'watched_boards', type_='check')
    op.create_check_constraint(
        op.f('ck_watched_boards_platform'),
        'watched_boards',
        "platform in ('" + "','".join(_TWELVE) + "')",
    )


def downgrade() -> None:
    op.drop_constraint(op.f('ck_watched_boards_platform'), 'watched_boards', type_='check')
    op.create_check_constraint(
        op.f('ck_watched_boards_platform'),
        'watched_boards',
        "platform in ('" + "','".join(_ORIGINAL_FOUR) + "')",
    )
