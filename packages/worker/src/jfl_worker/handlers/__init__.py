"""Every kind this worker can run, in one readable list.

Adding a handler is two lines here and nothing anywhere else. Deleting one is a
line here -- and tasks of that kind then sit `pending` rather than failing,
because the claim query only asks for registered kinds.

`calls_model` is not optional and not guessable: it is what
`JFL_DISABLE_MODEL_CALLS` acts on, so a handler that spends a user's key and
says `calls_model=False` would make the incident lever a lie.

`build_registry` takes the settings because a model-calling handler needs two
things from the process environment -- the master key, to unseal the user's
stored API key, and which model to call. They are bound here, at wiring time,
rather than read inside a handler: `WorkerSettings.from_env` is this process's
one environment read, the same rule `RequestContext.from_env` follows at the
CLI boundary.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection

from jfl_worker.handlers.boards import KIND as CHECK_BOARD
from jfl_worker.handlers.boards import SCHEDULE_KIND as SCHEDULE_BOARD_CHECKS
from jfl_worker.handlers.boards import (
    TransportFactory,
    build_check_board,
    build_schedule_board_checks,
)
from jfl_worker.handlers.extraction import KIND as EXTRACT_JOB_AD
from jfl_worker.handlers.extraction import build_extract_job_ad
from jfl_worker.handlers.sessions import KIND as PURGE_EXPIRED_SESSIONS
from jfl_worker.handlers.sessions import purge_expired_sessions
from jfl_worker.registry import HandlerRegistry
from jfl_worker.settings import WorkerSettings

__all__ = [
    "CHECK_BOARD",
    "EXTRACT_JOB_AD",
    "PURGE_EXPIRED_SESSIONS",
    "SCHEDULE_BOARD_CHECKS",
    "build_check_board",
    "build_extract_job_ad",
    "build_registry",
    "build_schedule_board_checks",
    "purge_expired_sessions",
]


def build_registry(
    settings: WorkerSettings,
    *,
    board_transport: TransportFactory | None = None,
    board_owners: Collection[uuid.UUID] | None = None,
) -> HandlerRegistry:
    """The worker's handlers. `main()` passes settings and nothing else.

    The two keyword arguments exist so a test can run the REAL registry without
    it being able to reach a job board: `board_transport` replaces the httpx
    transport `check_board` would otherwise open, and `board_owners` confines the
    scheduling pass to the test's own users. Both default to production.
    """
    registry = HandlerRegistry()
    registry.register(
        PURGE_EXPIRED_SESSIONS,
        purge_expired_sessions,
        calls_model=False,
    )
    registry.register(
        CHECK_BOARD,
        build_check_board(transport_factory=board_transport),
        # False, and true: public ATS APIs and pure rules, no `anthropic` on the
        # path. The kill switch is for spending a user's key; this spends none.
        calls_model=False,
    )
    registry.register(
        SCHEDULE_BOARD_CHECKS,
        build_schedule_board_checks(only_owners=board_owners),
        calls_model=False,
    )
    registry.register(
        EXTRACT_JOB_AD,
        build_extract_job_ad(master_key=settings.master_key, model=settings.model),
        # True, and this is the line the kill switch acts on. Extraction is one
        # Anthropic call on the user's own key.
        calls_model=True,
    )
    return registry
