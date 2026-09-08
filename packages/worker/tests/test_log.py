"""Structured logs, and the redaction that stops a payload reaching one.

CLAUDE.md's rule is absolute: a credential is never logged, never in `runs`,
never in a trace. Since users bring their own Anthropic keys, a logging bug is a
credential disclosure -- so redaction is mechanical here rather than a habit
someone has to keep.
"""

from __future__ import annotations

import io
import json
import logging

from jfl_worker.log import REDACTED, configure_logging, log_event


def _emit(**fields: object) -> dict[str, object]:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    log_event(logger, logging.INFO, "task.started", **fields)
    line: dict[str, object] = json.loads(stream.getvalue().strip())
    return line


def test_one_json_object_per_line() -> None:
    line = _emit(task_id="abc", kind="purge_expired_sessions", attempt=1)
    assert line["event"] == "task.started"
    assert line["level"] == "info"
    assert line["logger"] == "jfl_worker"
    assert line["task_id"] == "abc"
    assert line["attempt"] == 1
    assert "ts" in line


def test_payload_and_credential_shaped_fields_are_redacted() -> None:
    line = _emit(
        payload={"anthropic_api_key": "sk-ant-secret"},
        api_key="sk-ant-secret",
        session_token="deadbeef",
        SECRET="hunter2",
        kind="draft_cv",
    )
    assert line["payload"] == REDACTED
    assert line["api_key"] == REDACTED
    assert line["session_token"] == REDACTED
    assert line["SECRET"] == REDACTED
    assert line["kind"] == "draft_cv"  # ordinary fields survive
    assert "sk-ant-secret" not in json.dumps(line)


def test_a_field_cannot_overwrite_a_reserved_key() -> None:
    line = _emit(event="not-the-event", level="not-the-level")
    assert line["event"] == "task.started"
    assert line["level"] == "info"
    assert line["field_event"] == "not-the-event"


def test_an_unserialisable_value_degrades_to_a_repr_rather_than_crashing() -> None:
    line = _emit(thing=object())
    assert isinstance(line["thing"], str)
    assert "object object at" in line["thing"]


def test_exceptions_carry_a_traceback_field() -> None:
    stream = io.StringIO()
    logger = configure_logging(stream=stream)
    try:
        raise ValueError("boom")
    except ValueError:
        log_event(logger, logging.ERROR, "task.failed", exc_info=True, task_id="abc")
    line = json.loads(stream.getvalue().strip())
    assert "ValueError: boom" in line["traceback"]


def test_configure_logging_is_idempotent() -> None:
    """Called twice, it must not double every line."""
    stream = io.StringIO()
    configure_logging(stream=stream)
    logger = configure_logging(stream=stream)
    log_event(logger, logging.INFO, "worker.started")
    assert len(stream.getvalue().strip().splitlines()) == 1
