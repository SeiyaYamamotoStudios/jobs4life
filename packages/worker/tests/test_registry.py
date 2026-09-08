"""Handler registration is explicit, and `calls_model` is what the kill switch acts on."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from jfl_core.crypto.envelope import MasterKey
from jfl_worker.handlers import EXTRACT_JOB_AD, PURGE_EXPIRED_SESSIONS, build_registry
from jfl_worker.registry import DuplicateHandlerError, HandlerRegistry, TaskContext
from jfl_worker.settings import WorkerSettings


def _noop(ctx: TaskContext) -> Mapping[str, object] | None:
    return None


def test_register_and_get() -> None:
    registry = HandlerRegistry()
    registry.register("thing", _noop, calls_model=False)
    spec = registry.get("thing")
    assert spec is not None
    assert spec.handler is _noop
    assert spec.calls_model is False


def test_unknown_kind_returns_none_rather_than_raising() -> None:
    assert HandlerRegistry().get("nope") is None


def test_duplicate_registration_is_an_error() -> None:
    registry = HandlerRegistry()
    registry.register("thing", _noop, calls_model=False)
    with pytest.raises(DuplicateHandlerError):
        registry.register("thing", _noop, calls_model=True)


def test_runnable_kinds_drops_model_handlers_when_calls_are_disabled() -> None:
    registry = HandlerRegistry()
    registry.register("cheap", _noop, calls_model=False)
    registry.register("expensive", _noop, calls_model=True)

    assert registry.runnable_kinds(allow_model_calls=True) == ("cheap", "expensive")
    assert registry.runnable_kinds(allow_model_calls=False) == ("cheap",)


def test_the_shipped_registry_declares_calls_model_correctly_for_each_kind() -> None:
    """The kill switch is only as good as this line.

    B1 shipped infrastructure only and this test asserted the registry held one
    handler that called no model. B3 is the slice that adds one: `extract_job_ad`
    reads a pasted ad on the user's own key. What is worth asserting now is not
    the count but the flag -- a handler that spends money and says
    `calls_model=False` would make `JFL_DISABLE_MODEL_CALLS` a lever that does
    not stop the spending.
    """
    settings = WorkerSettings(database_url="x", master_key=MasterKey.generate())
    registry = build_registry(settings)

    assert registry.kinds() == (EXTRACT_JOB_AD, PURGE_EXPIRED_SESSIONS)

    purge = registry.get(PURGE_EXPIRED_SESSIONS)
    assert purge is not None and purge.calls_model is False

    extract = registry.get(EXTRACT_JOB_AD)
    assert extract is not None and extract.calls_model is True

    # And the switch actually removes it from what a worker will claim.
    assert registry.runnable_kinds(allow_model_calls=False) == (PURGE_EXPIRED_SESSIONS,)
