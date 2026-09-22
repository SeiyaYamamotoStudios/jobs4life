"""The `cluster_capabilities` handler, end to end against real Postgres:
enqueue the task, run it in the real worker loop, and check what lands in the
database. Needs `docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing key, the kill switch, the already-answered run -- the root
`conftest.py` guard is what would raise if the handler reached the API anyway.
Mirrors `tests/test_title_suggestions_worker_integration.py` closely.
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
from jfl_core.ids import fact_fingerprint, role_key
from jfl_core.models import ProposedFact
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.capability_clusters import PostgresCapabilityClusterRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import CLUSTER_CAPABILITIES, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

CLUSTER_PAYLOAD = {
    "capabilities": [
        {"label": "FX pricing platforms", "fact_ids": ["f1", "f2"]},
        # An id nobody sent. It must never reach the database.
        {"label": "Time travel", "fact_ids": ["f99"]},
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
            input_tokens=500,
            output_tokens=90,
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
    calls: list[dict[str, Any]] = []
    client = _FakeClient(None if payload is None else _message(payload), exception)
    original_create = client.messages.create

    def _recording_create(**kwargs: Any) -> Message:
        calls.append(kwargs)
        return original_create(**kwargs)

    client.messages.create = _recording_create  # type: ignore[method-assign]
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    return calls


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def confirm_facts(engine: Engine, user_id: uuid.UUID, texts: list[str]) -> list[uuid.UUID]:
    """Two confirmed facts under one role, the only kind a capability may cite."""
    role = "Acme Ltd -- Head of Engineering"
    with engine.begin() as conn:
        repo = PostgresCandidateFactRepository(conn, user_id)
        repo.add_proposed(
            [
                ProposedFact(
                    role_label=role,
                    role_key=role_key(role),
                    source_line=text,
                    fact_text=text,
                    fingerprint=fact_fingerprint(role_key(role), text),
                    ordinal=index,
                )
                for index, text in enumerate(texts)
            ]
        )
        ids = [f.id for f in repo.list_facts()]
        for fact_id in ids:
            repo.confirm(fact_id)
    return ids


def create_run(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        row = PostgresCapabilityClusterRepository(conn, user_id).create_pending(
            trace_id=uuid.uuid4()
        )
    return row.id


def enqueue(engine: Engine, user_id: uuid.UUID, cluster_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=CLUSTER_CAPABILITIES, payload={"cluster_id": str(cluster_id)}
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


def cluster_row(engine: Engine, user_id: uuid.UUID, cluster_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresCapabilityClusterRepository(conn, user_id).get(cluster_id)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_groups_the_facts_and_marks_the_run_done(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    fact_ids = confirm_facts(engine, user, ["Rebuilt FX pricing", "Ran the pricing platform"])
    cluster_id = create_run(engine, user)
    task_id = enqueue(engine, user, cluster_id)

    install_fake_client(monkeypatch, payload=CLUSTER_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    row = cluster_row(engine, user, cluster_id)
    assert row is not None
    assert row.status == "done"
    assert row.error_code is None
    assert row.fact_count == 2
    assert [p.label for p in row.proposals] == ["FX pricing platforms"], (
        "the capability built from an id nobody sent must not reach storage"
    )
    assert sorted(row.proposals[0].fact_ids) == sorted(fact_ids)
    assert len(row.proposals[0].span_ids) == 2
    assert row.unclustered_fact_ids == []
    assert row.omitted_fact_ids == []


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    confirm_facts(engine, user, ["Rebuilt FX pricing"])
    cluster_id = create_run(engine, user)
    enqueue(engine, user, cluster_id)

    install_fake_client(monkeypatch, payload=CLUSTER_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].stage == "cluster_capabilities"
    assert runs[0].model == "claude-haiku-4-5"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None
    # The run's own trace, so the screen can price it without this table
    # holding any money.
    row = cluster_row(engine, user, cluster_id)
    assert row is not None and runs[0].trace_id == row.trace_id


def test_a_redelivered_task_does_not_call_the_model_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    confirm_facts(engine, user, ["Rebuilt FX pricing"])
    cluster_id = create_run(engine, user)
    enqueue(engine, user, cluster_id)

    calls = install_fake_client(monkeypatch, payload=CLUSTER_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)
    assert len(calls) == 1

    second_task = enqueue(engine, user, cluster_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1, "a redelivered task called the model again"


def test_facts_that_do_not_fit_one_call_are_recorded_rather_than_dropped(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap's promise. Three confirmed facts, a limit of one: the other two
    come back as `omitted_fact_ids` and the screen shows them.
    """
    monkeypatch.setattr("jfl_worker.handlers.capability_clusters.MAX_FACTS", 1)
    store_key(engine, user, master_key, FAKE_KEY)
    confirm_facts(engine, user, ["One", "Two", "Three"])
    cluster_id = create_run(engine, user)
    enqueue(engine, user, cluster_id)

    install_fake_client(
        monkeypatch, payload={"capabilities": [{"label": "Something", "fact_ids": ["f1"]}]}
    )
    run_worker(engine, user, master_key, log_stream)

    row = cluster_row(engine, user, cluster_id)
    assert row is not None
    assert row.fact_count == 1
    assert len(row.omitted_fact_ids) == 2


def test_a_fact_the_model_placed_in_nothing_is_still_shown(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    confirm_facts(engine, user, ["One", "Two"])
    cluster_id = create_run(engine, user)
    enqueue(engine, user, cluster_id)

    install_fake_client(
        monkeypatch, payload={"capabilities": [{"label": "Something", "fact_ids": ["f1"]}]}
    )
    run_worker(engine, user, master_key, log_stream)

    row = cluster_row(engine, user, cluster_id)
    assert row is not None
    assert len(row.unclustered_fact_ids) == 1


def test_nothing_left_to_group_finishes_without_calling_the_model(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No confirmed facts at all. No fake client is installed: the conftest
    guard would raise if the handler reached the API.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    cluster_id = create_run(engine, user)
    task_id = enqueue(engine, user, cluster_id)

    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"
    row = cluster_row(engine, user, cluster_id)
    assert row is not None and row.status == "done" and row.proposals == []
    with engine.begin() as conn:
        assert conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all() == []


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    confirm_facts(engine, user, ["Rebuilt FX pricing"])
    cluster_id = create_run(engine, user)
    task_id = enqueue(engine, user, cluster_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "no Anthropic API key stored" in task.last_error

    row = cluster_row(engine, user, cluster_id)
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
    confirm_facts(engine, user, ["Rebuilt FX pricing"])
    cluster_id = create_run(engine, user)
    task_id = enqueue(engine, user, cluster_id)

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
    assert FAKE_KEY not in task.last_error

    row = cluster_row(engine, user, cluster_id)
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
    confirm_facts(engine, user, ["Rebuilt FX pricing"])
    cluster_id = create_run(engine, user)
    task_id = enqueue(engine, user, cluster_id)

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

    row = cluster_row(engine, user, cluster_id)
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
    confirm_facts(engine, user, ["Rebuilt FX pricing"])
    cluster_id = create_run(engine, user)
    task_id = enqueue(engine, user, cluster_id)

    run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.attempts == 0

    row = cluster_row(engine, user, cluster_id)
    assert row is not None and row.status == "pending"
