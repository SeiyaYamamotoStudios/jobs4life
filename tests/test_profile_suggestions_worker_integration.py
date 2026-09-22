"""The `suggest_profile_settings` handler, end to end against real Postgres:
enqueue the task, run it in the real worker loop, and check what lands in the
database. Needs `docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing key, the kill switch, the already-answered run -- the root
`conftest.py` guard is what would raise if the handler reached the API anyway.
Mirrors `tests/test_capability_clusters_worker_integration.py` closely.

Two things this handler owes that the clustering one does not, and both are
tested here: a kind outside the whitelist never reaches the database however the
model answers, and a suggestion the user has already turned down is never
proposed a second time.
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
from jfl_core.storage.profile_suggestions import PostgresProfileSuggestionRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import SUGGEST_PROFILE_SETTINGS, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

CV = """\
Jo Smith -- Engineering Manager

Northwind, London, 2021-2024
Ran platform engineering across three squads.
I no longer work on frontend.
"""

SUGGESTION_PAYLOAD = {
    "suggestions": [
        {
            "kind": "discipline",
            "value": "platform engineering",
            "source_line": "Ran platform engineering across three squads.",
        },
        {
            "kind": "location",
            "value": "London",
            "source_line": "Northwind, London, 2021-2024",
        },
        # A kind nobody asked for. It must never reach the database: a guessed
        # constraint would be read by scoring as the user's own requirement.
        {
            "kind": "comp_floor",
            "value": "£150,000",
            "source_line": "Ran platform engineering across three squads.",
        },
        # A quote that is in no CV. An invented line takes its suggestion with it.
        {
            "kind": "discipline",
            "value": "quantum cryptography",
            "source_line": "Led the quantum cryptography group.",
        },
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

    def create(self, **kwargs: Any) -> Message:
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
            input_tokens=2000,
            output_tokens=120,
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


def upload_cv(engine: Engine, user_id: uuid.UUID, text: str = CV) -> uuid.UUID:
    with engine.begin() as conn:
        return PostgresSentDocumentRepository(conn, user_id).add_cv(filename="cv.md", text=text).id


def create_run(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        row = PostgresProfileSuggestionRepository(conn, user_id).create_pending(
            trace_id=uuid.uuid4()
        )
    return row.id


def enqueue(engine: Engine, user_id: uuid.UUID, run_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=SUGGEST_PROFILE_SETTINGS, payload={"suggestion_run_id": str(run_id)}
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


def suggestion_row(engine: Engine, user_id: uuid.UUID, run_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresProfileSuggestionRepository(conn, user_id).get(run_id)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_reads_the_cvs_and_marks_the_run_done(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    task_id = enqueue(engine, user, run_id)

    install_fake_client(monkeypatch, payload=SUGGESTION_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    row = suggestion_row(engine, user, run_id)
    assert row is not None
    assert row.status == "done"
    assert row.error_code is None
    assert row.cv_count == 1
    kinds = {p.kind for p in row.proposals}
    assert kinds == {"discipline", "location"}, (
        "a kind outside the whitelist, or one quoting a line no CV holds, must not reach storage"
    )
    assert [p.values for p in row.proposals if p.kind == "discipline"] == [["platform engineering"]]
    assert all(p.source_lines for p in row.proposals), "a proposal with no CV words behind it"


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    enqueue(engine, user, run_id)

    install_fake_client(monkeypatch, payload=SUGGESTION_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        runs = conn.execute(
            select(runs_table).where(
                runs_table.c.user_id == user,
                runs_table.c.stage == "suggest_profile_settings",
            )
        ).all()
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].model == "claude-haiku-4-5"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None
    # The run's own trace, so the screen can price it without this table
    # holding any money.
    row = suggestion_row(engine, user, run_id)
    assert row is not None and runs[0].trace_id == row.trace_id


def test_a_redelivered_task_does_not_call_the_model_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    enqueue(engine, user, run_id)

    calls = install_fake_client(monkeypatch, payload=SUGGESTION_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)
    assert len(calls) == 1

    second_task = enqueue(engine, user, run_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    with engine.begin() as conn:
        runs = conn.execute(
            select(runs_table).where(
                runs_table.c.user_id == user,
                runs_table.c.stage == "suggest_profile_settings",
            )
        ).all()
    assert len(runs) == 1, "a redelivered task called the model again"


def test_a_suggestion_already_answered_is_not_proposed_again(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejecting sticks across runs. The key is content-derived, so the same
    CV read a second time folds the same suggestion to the same key.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    upload_cv(engine, user)
    first = create_run(engine, user)
    enqueue(engine, user, first)
    install_fake_client(monkeypatch, payload=SUGGESTION_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    row = suggestion_row(engine, user, first)
    assert row is not None
    with engine.begin() as conn:
        repo = PostgresProfileSuggestionRepository(conn, user)
        for proposal in row.proposals:
            repo.set_proposal_state(first, proposal.key, "rejected")

    second = create_run(engine, user)
    enqueue(engine, user, second)
    run_worker(engine, user, master_key, log_stream)

    again = suggestion_row(engine, user, second)
    assert again is not None
    assert again.status == "done"
    assert again.proposals == [], "a suggestion the user turned down was offered again"


def test_no_cv_uploaded_finishes_without_calling_the_model(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No fake client is installed: the conftest guard would raise if the
    handler reached the API.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    run_id = create_run(engine, user)
    task_id = enqueue(engine, user, run_id)

    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"
    row = suggestion_row(engine, user, run_id)
    assert row is not None and row.status == "done" and row.proposals == []
    with engine.begin() as conn:
        assert (
            conn.execute(
                select(runs_table).where(
                    runs_table.c.user_id == user,
                    runs_table.c.stage == "suggest_profile_settings",
                )
            ).all()
            == []
        )


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    task_id = enqueue(engine, user, run_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "no Anthropic API key stored" in task.last_error

    row = suggestion_row(engine, user, run_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "no_api_key"


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
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    task_id = enqueue(engine, user, run_id)

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

    row = suggestion_row(engine, user, run_id)
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
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    task_id = enqueue(engine, user, run_id)

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

    row = suggestion_row(engine, user, run_id)
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
    upload_cv(engine, user)
    run_id = create_run(engine, user)
    task_id = enqueue(engine, user, run_id)

    run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.attempts == 0

    row = suggestion_row(engine, user, run_id)
    assert row is not None and row.status == "pending"


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_a_run_belonging_to_someone_else_is_never_read(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
) -> None:
    """The repository is constructed with one user and has no per-call
    override, so a cross-tenant id simply is not there. No fake client: the
    guard would raise if the handler reached the API.
    """
    other = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users_table).values(id=other, email=f"{other}@test.invalid"))
    try:
        store_key(engine, user, master_key, FAKE_KEY)
        upload_cv(engine, user)
        theirs = create_run(engine, other)
        task_id = enqueue(engine, user, theirs)

        run_worker(engine, user, master_key, log_stream)

        assert task_row(engine, task_id).status == "succeeded"
        row = suggestion_row(engine, other, theirs)
        assert row is not None and row.status == "pending"
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id == other))
