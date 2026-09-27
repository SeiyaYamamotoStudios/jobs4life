"""The jobs4life background worker (slice B1).

A gate call takes ~2 minutes, so nothing that slow may run inside a request.
This package is the poll loop that runs it instead: claim as many tasks as
there are free slots (`JFL_WORKER_CONCURRENCY`, four by default), dispatch each
on its own thread to a handler registered against its `kind`, record what
happened, and sleep until a slot frees or the poll interval passes.

Nothing here calls a model. The queue is infrastructure; handlers that spend a
user's API key arrive with the slices that need them, and until then this
package has no `anthropic` dependency to call one with.
"""

from jfl_worker.registry import HandlerRegistry, HandlerSpec, TaskContext
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings, model_calls_disabled

__all__ = [
    "HandlerRegistry",
    "HandlerSpec",
    "TaskContext",
    "Worker",
    "WorkerSettings",
    "model_calls_disabled",
]
