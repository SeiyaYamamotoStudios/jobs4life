"""One button, several steps: queue the next task when this one has finished.

"Write the CV" needs the ad read and the job checked against the user's
confirmed facts first. When those are missing, the drafting screen queues the
first missing step and writes the rest into its payload as `then`:

    {"application_id": "...",
     "then": [{"kind": "generate_coverage", "payload": {"job_id": "..."}},
              {"kind": "generate_cv_draft", "payload": {"application_id": "...",
                                                        "kind": "cv_bullets"}}]}

A handler that succeeds calls `queue_next`, which queues `then[0]` with the rest
of the list carried forward and `"after": <this task's id>`. Nothing about what
any step does changes: each handler still refuses to run without its own
prerequisites, so a step whose predecessor failed is simply never queued, and a
chain that stops shows the user which step stopped it.

**Only a closed set of kinds may follow.** The payload is written by the web
app, but a task payload is not a place to take instructions from -- a chain can
only ever continue into the two drafting steps, never into an arbitrary kind.

**Idempotent under at-least-once delivery.** A redelivered task finds its own
follow-up (`PostgresTaskRepository.follow_up`) and queues nothing further, so
one press is one of each step, never two -- and every one of those steps is a
charge on the user's own key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jfl_core.storage.tasks import PostgresTaskRepository

from jfl_worker.registry import TaskContext

# The steps a chain may continue into. Extraction is never a follow-up: it is
# only ever the first step.
FOLLOW_UP_KINDS = frozenset({"generate_coverage", "generate_cv_draft"})

# Keys the chain itself owns, never passed through from a step's own payload.
_CHAIN_KEYS = frozenset({"then", "after"})


def _steps(payload: Mapping[str, object]) -> list[Mapping[str, Any]]:
    raw = payload.get("then")
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return []
    return [step for step in raw if isinstance(step, Mapping)]


def queue_next(ctx: TaskContext) -> None:
    """Queue the step after this one, if the payload names one and it has not
    already been queued. A malformed or unrecognised step ends the chain
    quietly rather than failing a task whose own work succeeded.
    """
    steps = _steps(ctx.task.payload)
    if not steps:
        return
    step, rest = steps[0], steps[1:]
    kind = step.get("kind")
    step_payload = step.get("payload")
    if kind not in FOLLOW_UP_KINDS or not isinstance(step_payload, Mapping):
        return
    payload: dict[str, Any] = {k: v for k, v in step_payload.items() if k not in _CHAIN_KEYS}
    payload["after"] = str(ctx.task.id)
    if rest:
        payload["then"] = [dict(s) for s in rest]
    with ctx.engine.begin() as conn:
        repo = PostgresTaskRepository(conn, ctx.user_id)
        if repo.follow_up(ctx.task.id) is not None:
            return
        repo.enqueue(kind=str(kind), payload=payload)
