"""board check error code check accepts duplicate_posting

Revision ID: 12fd7dd9c2a5
Revises: 3c540957b0d2

The Workable adapter reports `duplicate_posting` when a token-paged listing hands
back an id it has already collected without finishing -- a failure to make
progress. The code was added to `jfl_core.models.BoardCheckErrorCode` but not to
`_BOARD_CHECK_ERROR_CODES` or `ck_board_checks_error_code`, so a check that hit
it would have failed at INSERT: the check's own failure turned into a second,
unrecorded one. Same bug class as `3c540957b0d2` fixed for platforms.

Earlier migrations are deployed and are not edited. The code lists are written
out here rather than imported, because a migration is a snapshot: importing the
live tuple would silently change what an old revision does.
"""
from __future__ import annotations

from alembic import op

revision = '12fd7dd9c2a5'
down_revision = '3c540957b0d2'
branch_labels = None
depends_on = None

_BEFORE = (
    'not_found', 'http_client_error', 'rate_limited', 'server_error', 'timeout',
    'connection_error', 'malformed_response', 'unidentifiable_job', 'count_mismatch',
    'page_cap_reached', 'request_budget_exhausted', 'deadline_exceeded',
    'listing_ceiling', 'unsupported_board', 'drop_guard',
)
_AFTER = (
    'not_found', 'http_client_error', 'rate_limited', 'server_error', 'timeout',
    'connection_error', 'malformed_response', 'unidentifiable_job', 'count_mismatch',
    'page_cap_reached', 'duplicate_posting', 'request_budget_exhausted',
    'deadline_exceeded', 'listing_ceiling', 'unsupported_board', 'drop_guard',
)


def _recreate(codes: tuple[str, ...]) -> None:
    op.drop_constraint(op.f('ck_board_checks_error_code'), 'board_checks', type_='check')
    op.create_check_constraint(
        op.f('ck_board_checks_error_code'),
        'board_checks',
        "error_code is null or error_code in ('" + "','".join(codes) + "')",
    )


def upgrade() -> None:
    _recreate(_AFTER)


def downgrade() -> None:
    _recreate(_BEFORE)
