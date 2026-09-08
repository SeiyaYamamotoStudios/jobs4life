"""Handler registration, by `kind`, explicitly.

No decorators, no import-time side effects, no scanning a package for anything
that looks like a handler. `jfl_worker.handlers.build_registry()` is a function
with one `register(...)` line per kind, so the answer to "what can this worker
run?" is a list you can read, and a handler that was never wired up fails
loudly at review rather than silently at 3am.

Two consequences of the registry being the source of truth:

  * the claim query asks for exactly the kinds registered here, so a task whose
    kind no deployed worker knows is never claimed. It waits, `pending`, for a
    worker that knows it -- rather than being claimed and failed by an older
    container mid-rollout;
  * `calls_model` is a property of the handler, declared at registration. It is
    what the `JFL_DISABLE_MODEL_CALLS` kill switch acts on, so getting it wrong
    means an incident lever that does not stop the spending. Default it to
    nothing: every registration must say.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from jfl_core.models import Task
from sqlalchemy.engine import Engine


@dataclass(frozen=True, slots=True)
class TaskContext:
    """What a handler is given.

    `engine`, not a `Connection`: the worker's claim transaction is committed
    before a handler runs (see `jfl_core.storage.tasks`), so a handler opens its
    own transactions and owns their boundaries. `user_id` is on the task and is
    how a handler scopes itself -- a handler touching user content constructs a
    `TenantScopedRepository` with `ctx.task.user_id` and nothing else.

    `now` is passed rather than read from the clock inside the handler, so the
    whole of one attempt shares one timestamp and a test can pick it.
    """

    task: Task
    engine: Engine
    now: datetime

    @property
    def user_id(self) -> uuid.UUID:
        return self.task.user_id


# A handler returns whatever it wants recorded in the success log line -- counts,
# ids, a cost. None means "nothing worth saying". It raises to fail; the worker
# turns the exception into `last_error` and a retry.
Handler = Callable[[TaskContext], Mapping[str, object] | None]


@dataclass(frozen=True, slots=True)
class HandlerSpec:
    kind: str
    handler: Handler
    calls_model: bool


class DuplicateHandlerError(RuntimeError):
    """Two handlers registered for one kind.

    Raised rather than resolved: silently keeping the first would mean the
    running behaviour depends on import order, and silently keeping the last
    would mean a typo'd kind quietly replaces a working handler.
    """


class HandlerRegistry:
    """Kind -> handler. Small on purpose."""

    def __init__(self) -> None:
        self._specs: dict[str, HandlerSpec] = {}

    def register(self, kind: str, handler: Handler, *, calls_model: bool) -> None:
        if kind in self._specs:
            raise DuplicateHandlerError(f"a handler for {kind!r} is already registered")
        self._specs[kind] = HandlerSpec(kind=kind, handler=handler, calls_model=calls_model)

    def get(self, kind: str) -> HandlerSpec | None:
        return self._specs.get(kind)

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def runnable_kinds(self, *, allow_model_calls: bool) -> tuple[str, ...]:
        """The kinds this worker may claim right now.

        With the kill switch on, model-calling kinds are simply not asked for,
        which is what leaves them `pending` rather than claimed-and-refused. The
        dispatcher re-checks the switch anyway, for the case where it is thrown
        between the claim and the call.
        """
        return tuple(
            sorted(
                spec.kind
                for spec in self._specs.values()
                if allow_model_calls or not spec.calls_model
            )
        )
