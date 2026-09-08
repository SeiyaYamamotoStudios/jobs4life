"""Every kind this worker can run, in one readable list.

Adding a handler is two lines here and nothing anywhere else. Deleting one is a
line here -- and tasks of that kind then sit `pending` rather than failing,
because the claim query only asks for registered kinds.

`calls_model` is not optional and not guessable: it is what
`JFL_DISABLE_MODEL_CALLS` acts on, so a handler that spends a user's key and
says `calls_model=False` would make the incident lever a lie.
"""

from __future__ import annotations

from jfl_worker.handlers.sessions import KIND as PURGE_EXPIRED_SESSIONS
from jfl_worker.handlers.sessions import purge_expired_sessions
from jfl_worker.registry import HandlerRegistry

__all__ = ["PURGE_EXPIRED_SESSIONS", "build_registry", "purge_expired_sessions"]


def build_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register(
        PURGE_EXPIRED_SESSIONS,
        purge_expired_sessions,
        calls_model=False,
    )
    return registry
