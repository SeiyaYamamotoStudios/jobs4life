"""Slice C7a's background half, end to end against real Postgres: enqueue a
`suggest_titles` task, run it in the real worker loop, and check what lands in
the database. Needs `docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing-key case, the kill switch, the already-done case -- the root
`conftest.py` guard is what would raise if the handler reached the API anyway.
Mirrors `tests/test_extraction_integration.py` closely.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.job_filters import PostgresJobFilterRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_core.storage.title_suggestions import PostgresTitleSuggestionRepository
from jfl_intake.http import Transport
from jfl_intake.normalise import normalise
from jfl_worker.handlers import SUGGEST_TITLES, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

TITLES_PAYLOAD = {
    "titles": [
        {"title": "Senior Engineering Manager", "gloss": "a step up"},
        {"title": "Engineering Lead", "gloss": ""},
    ]
}


@contextmanager
def _never_fetch_a_board() -> Iterator[Transport]:
    raise AssertionError("a worker test tried to fetch a job board")
    yield  # pragma: no cover


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    created = create_engine(DATABASE_URL)
    yield created
    created.dispose()


@pytest.fixture
def user(engine: Engine) -> Iterator[uuid.UUID]:
    uid = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users_table).values(id=uid, email=f"{uid}@test.invalid"))
    try:
        yield uid
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id == uid))


@pytest.fixture
def master_key() -> MasterKey:
    return MasterKey.generate()


@pytest.fixture
def log_stream() -> io.StringIO:
    return io.StringIO()


class _FakeMessages:
    def __init__(self, response: Message | None, exception: Exception | None) -> None:
        self._response = response
        self._exception = exception
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        assert self._response is not None
        return self._response


class _FakeClient:
    def __init__(self, response: Message | None, exception: Exception | None) -> None:
        self.messages = _FakeMessages(response, exception)


def _message(payload: dict[str, Any]) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model="claude-haiku-4-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(
            input_tokens=300,
            output_tokens=60,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    exception: Exception | None = None,
) -> list[dict[str, Any]]:
    constructions: list[dict[str, Any]] = []
    client = _FakeClient(None if payload is None else _message(payload), exception)

    def _construct(**kwargs: Any) -> _FakeClient:
        constructions.append(kwargs)
        return client

    monkeypatch.setattr(anthropic, "Anthropic", _construct)
    return constructions


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def add_pending_suggestion(engine: Engine, user_id: uuid.UUID, phrase: str) -> uuid.UUID:
    with engine.begin() as conn:
        PostgresJobFilterRepository(conn, user_id).save_filter(
            workplaces=[], title_includes=phrase, title_excludes="", location=""
        )
        row = PostgresTitleSuggestionRepository(conn, user_id).create_pending(
            phrase=phrase, phrase_key=normalise(phrase)
        )
    assert row is not None
    return row.id


def enqueue(engine: Engine, user_id: uuid.UUID, suggestion_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=SUGGEST_TITLES, payload={"suggestion_id": str(suggestion_id)}
        )
    return task.id


def run_worker(
    engine: Engine,
    user_id: uuid.UUID,
    master_key: MasterKey,
    stream: io.StringIO,
    *,
    env: dict[str, str] | None = None,
) -> Worker:
    settings = WorkerSettings(
        database_url=DATABASE_URL, system_user_id=user_id, master_key=master_key
    )
    worker = Worker(
        registry=build_registry(
            settings, board_transport=_never_fetch_a_board, board_owners={user_id}
        ),
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, user_id),
        engine=engine,
        env=env or {},
        logger=configure_logging(stream=stream),
    )
    for _ in range(10):
        if worker.run_once() == 0:
            return worker
    raise AssertionError("the queue did not drain -- a task is looping")


def task_row(engine: Engine, task_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return conn.execute(select(tasks_table).where(tasks_table.c.id == task_id)).one()


def suggestion_row(engine: Engine, user_id: uuid.UUID, suggestion_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresTitleSuggestionRepository(conn, user_id).get(suggestion_id)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_suggests_and_marks_the_row_done(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    task_id = enqueue(engine, user, suggestion_id)

    install_fake_client(monkeypatch, payload=TITLES_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    row = suggestion_row(engine, user, suggestion_id)
    assert row is not None
    assert row.status == "done"
    assert row.error_code is None
    assert [s.title for s in row.suggestions] == [
        "Senior Engineering Manager",
        "Engineering Lead",
    ]


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    enqueue(engine, user, suggestion_id)

    install_fake_client(monkeypatch, payload=TITLES_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].stage == "suggest_titles"
    assert runs[0].model == "claude-haiku-4-5"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None


def test_a_redelivered_task_does_not_call_the_model_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    enqueue(engine, user, suggestion_id)

    calls = install_fake_client(monkeypatch, payload=TITLES_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)
    assert len(calls) == 1

    second_task = enqueue(engine, user, suggestion_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1, "a redelivered task called the model again"


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    task_id = enqueue(engine, user, suggestion_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "no Anthropic API key stored" in task.last_error

    row = suggestion_row(engine, user, suggestion_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "no_api_key"

    with engine.begin() as conn:
        assert conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all() == []


# --------------------------------------------------------------------------
# Failures from the model
# --------------------------------------------------------------------------


def test_a_rejected_key_fails_permanently(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    task_id = enqueue(engine, user, suggestion_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1

    row = suggestion_row(engine, user, suggestion_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "api_key_rejected"


def test_a_transient_failure_is_retried_rather_than_given_up_on(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    task_id = enqueue(engine, user, suggestion_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.APIStatusError(
            "overloaded", response=httpx2.Response(529, request=request), body=None
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.scheduled_at > dt.datetime.now(dt.UTC)

    row = suggestion_row(engine, user, suggestion_id)
    assert row is not None and row.error_code == "model_error"


# --------------------------------------------------------------------------
# The kill switch
# --------------------------------------------------------------------------


def test_the_kill_switch_leaves_the_task_pending_and_calls_nothing(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No fake client is installed: if the handler reached the API despite the
    switch, the root conftest guard would raise and this test would fail for
    the right reason.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    suggestion_id = add_pending_suggestion(engine, user, "engineering manager")
    task_id = enqueue(engine, user, suggestion_id)

    worker = run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})
    assert worker is not None

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.attempts == 0

    row = suggestion_row(engine, user, suggestion_id)
    assert row is not None and row.status == "pending"
