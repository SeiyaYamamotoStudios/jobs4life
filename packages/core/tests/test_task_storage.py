"""Unit tests for the task queue's storage layer -- the parts that need no
database: the model's repr, error sanitising, and the tenancy split.

The SQL is tested against real Postgres in `tests/test_tasks_queue_integration.py`.
"""

from __future__ import annotations

import datetime as dt
import inspect
import uuid

from jfl_core.models import ReclaimResult, Task
from jfl_core.storage.tasks import (
    _MAX_ERROR_CHARS,
    PostgresTaskQueue,
    PostgresTaskRepository,
    _clean_error,
)
from jfl_core.storage.tenancy import PreAuthRepository, TenantScopedRepository

NOW = dt.datetime(2026, 9, 8, 12, 0, tzinfo=dt.UTC)


def _task(**overrides: object) -> Task:
    values: dict[str, object] = {
        "id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "kind": "draft_cv",
        "payload": {"anthropic_api_key": "sk-ant-should-never-appear"},
        "status": "pending",
        "attempts": 0,
        "max_attempts": 3,
        "scheduled_at": NOW,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return Task(**values)  # type: ignore[arg-type]


def test_a_tasks_repr_never_carries_its_payload() -> None:
    """A `Task` reaches log lines and tracebacks. A payload is meant to hold
    arguments only, but "meant to" is not a guarantee -- and this project holds
    other people's API keys, so the default repr leaves it out.
    """
    task = _task()
    printed = repr(task)
    assert "sk-ant-should-never-appear" not in printed
    assert "payload" not in printed
    assert str(task.id) in printed  # still useful
    assert "draft_cv" in printed
    # Available when someone actually asks for it.
    assert task.payload["anthropic_api_key"] == "sk-ant-should-never-appear"


def test_clean_error_removes_nul_bytes_postgres_would_reject() -> None:
    assert _clean_error("boom\x00boom") == "boomboom"


def test_clean_error_truncates_a_runaway_message() -> None:
    cleaned = _clean_error("x" * (_MAX_ERROR_CHARS * 2))
    assert len(cleaned) == _MAX_ERROR_CHARS
    assert cleaned.endswith("... [truncated]")


def test_a_short_error_is_left_exactly_as_it_is() -> None:
    assert _clean_error("RuntimeError: boom") == "RuntimeError: boom"


def test_reclaim_result_is_falsey_when_nothing_moved() -> None:
    assert not ReclaimResult()
    assert ReclaimResult(requeued=[uuid.uuid4()])


def test_the_task_repository_is_tenancy_scoped_with_no_override() -> None:
    """What `tests/test_tenancy_enforcement.py` checks globally, pinned locally:
    a request-side repository is bound to one user at construction.
    """
    assert issubclass(PostgresTaskRepository, TenantScopedRepository)
    parameters = inspect.signature(PostgresTaskRepository.__init__).parameters
    assert "user_id" in parameters
    assert parameters["user_id"].default is inspect.Parameter.empty
    for name, func in inspect.getmembers(PostgresTaskRepository, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        assert not any("user_id" in p for p in inspect.signature(func).parameters)


def test_the_worker_side_queue_is_deliberately_outside_the_tenancy_scheme() -> None:
    """A decision worth pinning rather than leaving to a reader's memory.

    `PostgresTaskQueue` claims across every tenant, because a worker is a daemon
    with no user in context -- so it is neither tenancy-scoped nor named
    `*Repository`, and `tests/test_tenancy_enforcement.py` therefore neither
    catches it by accident nor waves it through by accident. What keeps that
    honest is the narrowness of its job: it moves rows between the four queue
    states and returns tasks carrying the `user_id` a handler must then scope
    itself to.

    If someone renames it to `...Repository`, the tenancy test will demand it
    subclass one of the two bases and this test will fail alongside -- which is
    the conversation that should happen before that name changes.
    """
    assert not PostgresTaskQueue.__name__.endswith("Repository")
    assert not issubclass(PostgresTaskQueue, TenantScopedRepository | PreAuthRepository)

    # It offers nothing that reads a user's content: every public method is a
    # state transition on the queue itself.
    public = {
        name
        for name, _ in inspect.getmembers(PostgresTaskQueue, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert public == {"claim", "mark_succeeded", "mark_failed", "release", "reclaim_stale"}
