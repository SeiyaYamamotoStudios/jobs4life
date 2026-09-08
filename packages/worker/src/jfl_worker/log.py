"""Structured logging, one JSON object per line, on stdout.

Docker collects stdout, so there is no file to rotate and no handler to
configure on the VPS. One object per line means `docker compose logs worker |
jq 'select(.event == "task.failed")'` works without a log stack.

**Redaction is mechanical, not a habit.** `log_event` replaces the value of any
field whose name looks like a secret or a task payload, so a future handler that
logs `payload=task.payload` writes `"[redacted]"` instead of a user's API key.
The rule this enforces is CLAUDE.md's, and it is not negotiable: a key is never
logged, never in `runs`, never in a trace. Belt and braces -- `Task.payload`
already carries `repr=False` for the same reason.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

LOGGER_NAME = "jfl_worker"

REDACTED = "[redacted]"

# Matched as substrings against the field name, lowercased. Deliberately broad:
# over-redacting a log line costs an operator one debugging step, under-redacting
# one costs a user their credential.
_SECRET_HINTS = (
    "payload",
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "cookie",
    "authorization",
)

# The keys the formatter puts at the top level itself; a field of the same name
# would silently overwrite one.
_RESERVED = frozenset({"ts", "level", "logger", "event", "traceback"})


def _redact(fields: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for name, value in fields.items():
        key = f"field_{name}" if name in _RESERVED else name
        lowered = name.lower()
        if any(hint in lowered for hint in _SECRET_HINTS):
            clean[key] = REDACTED
        else:
            clean[key] = value
    return clean


class JsonFormatter(logging.Formatter):
    """One line, one object. Unserialisable values become their `repr`, so a
    logging mistake degrades to an ugly line rather than to a crashed worker.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "event_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=repr)


def configure_logging(level: int = logging.INFO, stream: Any = None) -> logging.Logger:
    """Attach the JSON formatter to the worker's logger. Idempotent."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    /,
    **fields: Any,
) -> None:
    """Emit one structured line. Field names are checked for secrets first.

    The first three parameters are positional-only so that a caller may pass a
    field called `event` or `level` -- exactly the collision that would
    otherwise be found at 3am, from a log line that is missing.

    `exc_info` is pulled out of `**fields` rather than declared as a keyword-only
    parameter for a duller reason: a declared one makes every
    `log_event(..., **some_dict)` call site a type error, and handlers return
    dictionaries of fields to log.
    """
    exc_info = bool(fields.pop("exc_info", False))
    logger.log(level, event, exc_info=exc_info, extra={"event_fields": _redact(fields)})
