"""The board handlers' decisions that need no database.

Everything that does -- recording checks, the scheduling pass, retries against
real rows -- is in `tests/test_boards_worker_integration.py`.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from jfl_core.models import Task
from jfl_intake.http import HttpxTransport, PoliteTransport, Transport
from jfl_worker.handlers.boards import KIND, build_check_board, polite_httpx_transport
from jfl_worker.registry import PermanentTaskError, TaskContext
from sqlalchemy import create_engine

NOW = dt.datetime(2026, 9, 10, 6, 0, tzinfo=dt.UTC)


def _ctx(payload: dict[str, Any]) -> TaskContext:
    task = Task(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        kind=KIND,
        payload=payload,
        status="running",
        attempts=1,
        max_attempts=3,
        scheduled_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    # Never connected to: `create_engine` is lazy, and a test that reached the
    # database would fail on connection refused rather than find a real one.
    engine = create_engine("postgresql+psycopg://nobody@127.0.0.1:1/none")
    return TaskContext(task=task, engine=engine, now=NOW)


@contextmanager
def _no_network() -> Iterator[Transport]:
    raise AssertionError("a bad payload must fail before any fetch")
    yield  # pragma: no cover


@pytest.mark.parametrize("payload", [{}, {"board_id": 7}, {"board_id": "not-a-uuid"}])
def test_a_bad_payload_fails_permanently_before_touching_anything(payload: dict[str, Any]) -> None:
    handler = build_check_board(transport_factory=_no_network)
    with pytest.raises(PermanentTaskError) as caught:
        handler(_ctx(payload))
    # A literal message: `last_error` is quoted into logs and admin queries.
    assert "payload" in str(caught.value)
    assert "not-a-uuid" not in str(caught.value)


def test_the_production_transport_is_polite_httpx_and_opens_no_connection() -> None:
    with polite_httpx_transport()() as transport:
        assert isinstance(transport, PoliteTransport)
        assert isinstance(transport._inner, HttpxTransport)
        assert transport.requests == 0
