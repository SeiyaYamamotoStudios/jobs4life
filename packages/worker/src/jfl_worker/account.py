"""The user's Anthropic account, as the handlers see it.

Two moments matter, and every model-calling handler goes through both:

  * **a call was refused for an account-level reason** -- credits exhausted,
    key rejected, access denied. `blocked_by_account` reads the category off
    the `GenerateError` / `GateError` (classified once, at the call site, by
    `jfl_core.model_api`); `park_for_account` records it on the user's key
    health and raises `AccountBlockedError`, which the runner turns into a park.
    The handler leaves its own row waiting (never `failed`) before calling it.
  * **a call succeeded** -- `note_model_call` clears a blocked state, from each
    handler's `_RunRecorder`, since a `runs` row is the one thing every model
    call writes whether it worked or not.
"""

from __future__ import annotations

import datetime as dt
from typing import NoReturn

from jfl_core.model_api import AccountBlock, account_block_of
from jfl_core.models import RunRecord
from jfl_core.storage.api_key_health import PostgresApiKeyHealthRepository
from sqlalchemy.engine import Engine

from jfl_worker.registry import AccountBlockedError, TaskContext


def blocked_by_account(exc: BaseException) -> AccountBlock | None:
    """The account-level reason `exc` carries, or None for anything else."""
    return account_block_of(exc)


def park_for_account(ctx: TaskContext, block: AccountBlock) -> NoReturn:
    """Record the block on this user's key health, then raise
    `AccountBlockedError` for the runner to park the task.

    Its own transaction, for the reason every handler's `_fail` gives: the one
    the model call ran in may be rolling back, and the banner must survive it.
    `from None` so the SDK's exception is not chained into the traceback the
    runner logs.
    """
    with ctx.engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, ctx.user_id).mark_blocked(block, now=ctx.now)
    raise AccountBlockedError(block) from None


def note_model_call(engine: Engine, run: RunRecord) -> None:
    """Clear a blocked key health once a call has gone through.

    "Gone through" means the API accepted it: any outcome other than `error`,
    or an `error` that still reported usage (a truncated or unparseable answer
    is our problem, and it was billed, so the account works). One UPDATE that
    matches nothing unless the state was blocked.
    """
    if run.outcome == "error" and run.tokens_in is None:
        return
    with engine.begin() as conn:
        PostgresApiKeyHealthRepository(conn, run.user_id).mark_ok(now=dt.datetime.now(dt.UTC))
