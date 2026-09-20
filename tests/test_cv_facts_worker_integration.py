"""Slice B6's background half, end to end against real Postgres: enqueue an
`extract_cv_facts` task, run it in the real worker loop, and check what lands
in the database. Needs `docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing-key case, the kill switch, the already-read case -- the root
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
import httpx2
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.sent_documents import PostgresSentDocumentRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import EXTRACT_CV_FACTS, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

ACME = "Acme Ltd -- Engineering Manager, 2021-2024"

CV_TEXT = (
    "# Jane Doe\n\n"
    "## Acme Ltd -- Engineering Manager, 2021-2024\n\n"
    "- Led a team of 8 engineers across two squads.\n"
    "- Wrote Python and Go.\n"
)

SECOND_CV = CV_TEXT + "\n- Chaired the architecture review board.\n"

FACTS_PAYLOAD = {
    "facts": [
        {
            "role_label": ACME,
            "source_line": "Led a team of 8 engineers across two squads.",
            "fact_text": "Led a team of 8 engineers.",
            "probe": "How many people reported to you directly?",
        },
        {
            "role_label": ACME,
            "source_line": "Wrote Python and Go.",
            "fact_text": "Wrote Python and Go.",
            "probe": "",
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
            # Spans have no cascade from users, so they go first.
            conn.execute(delete(spans_table).where(spans_table.c.user_id == uid))
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


def _message(payload: dict[str, Any], *, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model="claude-opus-5",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=(
            RefusalStopDetails(type="refusal", category="cyber")
            if stop_reason == "refusal"
            else None
        ),
        type="message",
        usage=Usage(
            input_tokens=1200,
            output_tokens=300,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    exception: Exception | None = None,
    stop_reason: str = "end_turn",
) -> list[dict[str, Any]]:
    client = _FakeClient(
        None if payload is None else _message(payload, stop_reason=stop_reason), exception
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    return client.messages.calls


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def upload(engine: Engine, user_id: uuid.UUID, text: str = CV_TEXT) -> uuid.UUID:
    with engine.begin() as conn:
        stored = PostgresSentDocumentRepository(conn, user_id).add_cv(filename="cv.md", text=text)
    return stored.id


def enqueue(engine: Engine, user_id: uuid.UUID, document_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=EXTRACT_CV_FACTS, payload={"sent_document_id": str(document_id)}
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


def cv_row(engine: Engine, user_id: uuid.UUID, document_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresSentDocumentRepository(conn, user_id).get_cv(document_id)


def facts(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return PostgresCandidateFactRepository(conn, user_id).list_facts()


def runs_for(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return conn.execute(select(runs_table).where(runs_table.c.user_id == user_id)).all()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_proposes_facts_and_marks_the_cv_read(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

    install_fake_client(monkeypatch, payload=FACTS_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    cv = cv_row(engine, user, document_id)
    assert cv is not None
    assert cv.extraction_status == "done"
    assert cv.extraction_error_code is None
    assert cv.facts_proposed == 2

    proposed = facts(engine, user)
    assert [f.fact_text for f in proposed] == ["Led a team of 8 engineers.", "Wrote Python and Go."]
    # Every one of them is `proposed` and none carries a span: nothing a CV
    # claims grounds anything until the user confirms it.
    assert {f.state for f in proposed} == {"proposed"}
    assert all(f.span_id is None for f in proposed)
    # And the CV's own words are kept beside each fact, to quote back.
    assert proposed[0].source_line == "Led a team of 8 engineers across two squads."
    assert proposed[0].probe == "How many people reported to you directly?"
    assert proposed[1].probe is None


def test_nothing_the_handler_writes_reaches_the_corpus(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler's whole output is `proposed` rows. A worker that could
    ground on a CV would quietly undo the measurement this project is for.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    enqueue(engine, user, upload(engine, user))

    install_fake_client(monkeypatch, payload=FACTS_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        spans = conn.execute(select(spans_table).where(spans_table.c.user_id == user)).all()
    assert spans == []


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    enqueue(engine, user, upload(engine, user))

    install_fake_client(monkeypatch, payload=FACTS_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    runs = runs_for(engine, user)
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].stage == "extract_cv_facts"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None


def test_the_api_key_never_reaches_the_task_or_the_run_row(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    task_id = enqueue(engine, user, upload(engine, user))

    install_fake_client(monkeypatch, payload=FACTS_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert FAKE_KEY not in json.dumps(task.payload)
    assert FAKE_KEY not in (task.last_error or "")
    assert all(FAKE_KEY not in (r.error or "") for r in runs_for(engine, user))
    assert FAKE_KEY not in log_stream.getvalue()


def test_a_redelivered_task_does_not_call_the_model_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    document_id = upload(engine, user)
    enqueue(engine, user, document_id)

    calls = install_fake_client(monkeypatch, payload=FACTS_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)
    assert len(calls) == 1

    second_task = enqueue(engine, user, document_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    assert len(runs_for(engine, user)) == 1, "a redelivered task called the model again"


def test_a_second_cv_repeating_the_same_facts_adds_no_rows(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thirty-three near-identical CVs must collapse into one list to confirm.
    The second CV is still *read* -- it is a different document and may hold
    something new -- but what it repeats does not become a second row.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    enqueue(engine, user, upload(engine, user))
    install_fake_client(monkeypatch, payload=FACTS_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    second = upload(engine, user, SECOND_CV)
    enqueue(engine, user, second)
    run_worker(engine, user, master_key, log_stream)

    assert len(facts(engine, user)) == 2
    assert cv_row(engine, user, second).facts_proposed == 0  # nothing new in it


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "no Anthropic API key stored" in task.last_error

    cv = cv_row(engine, user, document_id)
    assert cv is not None and cv.extraction_status == "failed"
    assert cv.extraction_error_code == "no_api_key"
    assert runs_for(engine, user) == []


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
    store_key(engine, user, master_key, FAKE_KEY)
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

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
    assert cv_row(engine, user, document_id).extraction_error_code == "api_key_rejected"


def test_a_refusal_is_permanent(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

    install_fake_client(monkeypatch, payload={"facts": []}, stop_reason="refusal")
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "failed"
    assert cv_row(engine, user, document_id).extraction_error_code == "model_refused"


def test_a_cv_too_long_for_one_call_is_permanent(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

    install_fake_client(monkeypatch, payload={"facts": []}, stop_reason="max_tokens")
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "failed"
    assert cv_row(engine, user, document_id).extraction_error_code == "cv_too_long"


def test_a_transient_failure_is_retried_rather_than_given_up_on(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

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
    assert cv_row(engine, user, document_id).extraction_error_code == "model_error"


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
    document_id = upload(engine, user)
    task_id = enqueue(engine, user, document_id)

    run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.attempts == 0
    assert cv_row(engine, user, document_id).extraction_status == "pending"
