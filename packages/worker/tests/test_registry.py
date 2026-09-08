"""Handler registration is explicit, and `calls_model` is what the kill switch acts on."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from jfl_worker.handlers import PURGE_EXPIRED_SESSIONS, build_registry
from jfl_worker.registry import DuplicateHandlerError, HandlerRegistry, TaskContext


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


def test_the_shipped_registry_holds_the_one_handler_and_it_calls_no_model() -> None:
    """B1 ships infrastructure only. A model-calling handler appearing here
    without the slice that needs it is a regression, not a head start.
    """
    registry = build_registry()
    assert registry.kinds() == (PURGE_EXPIRED_SESSIONS,)
    spec = registry.get(PURGE_EXPIRED_SESSIONS)
    assert spec is not None and spec.calls_model is False
