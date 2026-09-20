"""NEXT.md's task 4, end to end against real Postgres: enqueue a
`check_application_answer` or `draft_application_answer` task, run it in the
real worker loop, and check what lands in the database. Needs
`docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing-key case, the kill switch -- the root `conftest.py` guard is what
would raise if the handler reached the API anyway. Mirrors
`tests/test_title_suggestions_worker_integration.py` closely.

The fake client answers both SDK entry points this feature's two calls use:
`.messages.create(...)` (the assessment call and the draft call) and
`.messages.stream(...)` (the claim gate's own automatic pass, via
`jfl_gate.gate.check_text` -- see `packages/gate/tests/test_gate.py`'s fake
for why the gate streams).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import uuid
from collections.abc import Iterator
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import Job, JobRequirement
from jfl_core.storage.application_questions import PostgresApplicationQuestionRepository
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresJobRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import CHECK_APPLICATION_ANSWER, DRAFT_APPLICATION_ANSWER, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select, update
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

ASSESSMENT_PAYLOAD = {"assessment": "Answers the question directly.", "gaps": ""}
DRAFT_ANSWER_PAYLOAD = {"draft": "I want this role for its focus on reliability."}
# One sentence, one claim result -- keeps `_check_alignment` trivial to satisfy.
GATE_PAYLOAD = {
    "sentences": [
        {
            "index": 1,
            "kind": "framing",
            "verdict": "supported",
            "drift_label": "framing",
            "cited_span_ids": [],
            "evidence_note": "",
        }
    ]
}


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
    """Answers both `.create(...)` and `.stream(...)`. `create_response` feeds
    the assessment/draft call; `stream_response` feeds the claim gate's
    automatic pass -- see the module docstring.
    """

    def __init__(
        self,
        *,
        create_response: Message | None = None,
        create_exception: Exception | None = None,
        stream_response: Message | None = None,
        stream_exception: Exception | None = None,
    ) -> None:
        self._create_response = create_response
        self._create_exception = create_exception
        self._stream_response = stream_response
        self._stream_exception = stream_exception
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.create_calls.append(kwargs)
        if self._create_exception is not None:
            raise self._create_exception
        assert self._create_response is not None
        return self._create_response

    def stream(self, **kwargs: Any) -> _FakeStream:
        self.stream_calls.append(kwargs)
        return _FakeStream(self._stream_response, self._stream_exception)


class _FakeStream:
    def __init__(self, response: Message | None, exception: Exception | None) -> None:
        self._response = response
        self._exception = exception

    def __enter__(self) -> _FakeStream:
        if self._exception is not None:
            raise self._exception
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Message:
        assert self._response is not None
        return self._response


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


def _message(payload: dict[str, Any], *, stop_reason: str = "end_turn") -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model="claude-opus-5",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
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
    create_payload: dict[str, Any] | None = None,
    create_exception: Exception | None = None,
    stream_payload: dict[str, Any] | None = None,
    stream_stop_reason: str = "end_turn",
    stream_exception: Exception | None = None,
) -> _FakeMessages:
    messages = _FakeMessages(
        create_response=None if create_payload is None else _message(create_payload),
        create_exception=create_exception,
        stream_response=None
        if stream_payload is None
        else _message(stream_payload, stop_reason=stream_stop_reason),
        stream_exception=stream_exception,
    )
    client = _FakeClient(messages)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    return messages


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def add_application(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user_id).create_application(
            title="Engineering Manager, Acme"
        )
    return application.id


def add_job_with_requirements(
    engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID
) -> None:
    """A job with one requirement, linked to `application_id` -- bypasses the
    (model-calling) extraction path entirely, since these tests are about the
    two calls this feature itself makes, not about extraction.
    """
    job = Job(
        id=uuid.uuid4(),
        user_id=user_id,
        source="paste",
        employer="Acme",
        title="Engineering Manager",
        raw_text="ad text",
        content_hash="0" * 64,
    )
    requirement = JobRequirement(
        id=uuid.uuid4(),
        user_id=user_id,
        job_id=job.id,
        ordinal=0,
        text="Owns production reliability",
        necessity="essential",
    )
    with engine.begin() as conn:
        PostgresJobRepository(conn).upsert_job(job)
        PostgresJobRepository(conn).replace_requirements(user_id, job.id, [requirement])
        conn.execute(
            update(applications_table)
            .where(applications_table.c.id == application_id)
            .values(job_id=job.id)
        )


def add_question(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        question = PostgresApplicationQuestionRepository(conn, user_id).add_question(
            application_id, "Why do you want to work here?"
        )
    return question.id


def add_pending_check_answer(
    engine: Engine, user_id: uuid.UUID, question_id: uuid.UUID, answer_text: str = "Because I care."
) -> uuid.UUID:
    with engine.begin() as conn:
        answer = PostgresApplicationQuestionRepository(conn, user_id).create_user_answer(
            question_id, answer_text
        )
    assert answer is not None
    return answer.id


def add_pending_draft_answer(
    engine: Engine, user_id: uuid.UUID, question_id: uuid.UUID
) -> uuid.UUID:
    with engine.begin() as conn:
        answer = PostgresApplicationQuestionRepository(conn, user_id).create_draft_answer(
            question_id
        )
    assert answer is not None
    return answer.id


def enqueue(engine: Engine, user_id: uuid.UUID, kind: str, answer_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=kind, payload={"answer_id": str(answer_id)}
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

    def _never_fetch_a_board() -> Transport:
        raise AssertionError("a worker test tried to fetch a job board")

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


def answer_row(engine: Engine, user_id: uuid.UUID, answer_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresApplicationQuestionRepository(conn, user_id).get_answer(answer_id)


def run_rows(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return list(conn.execute(select(runs_table).where(runs_table.c.user_id == user_id)).all())


# --------------------------------------------------------------------------
# check_application_answer -- the happy path
# --------------------------------------------------------------------------


def test_check_answer_gates_and_assesses_and_writes_two_runs_rows(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    add_job_with_requirements(engine, user, application_id)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    task_id = enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    install_fake_client(monkeypatch, create_payload=ASSESSMENT_PAYLOAD, stream_payload=GATE_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    row = answer_row(engine, user, answer_id)
    assert row is not None
    assert row.status == "done"
    assert row.error_code is None
    assert row.answer_text == "Because I care."  # untouched -- the user's own words
    assert row.assessment == ASSESSMENT_PAYLOAD
    # Not exact dict equality: `check_text` fills in each sentence's own
    # `text` and `rule_flags` (see `jfl_gate.gate._assemble`/`apply_rules`),
    # neither of which the model response carries.
    assert row.gate_result is not None
    stored_sentences = row.gate_result["sentences"]
    assert [
        {
            k: s[k]
            for k in ("index", "kind", "verdict", "drift_label", "cited_span_ids", "evidence_note")
        }
        for s in stored_sentences
    ] == GATE_PAYLOAD["sentences"]
    assert row.model == "claude-opus-5"
    assert row.trace_id is not None

    runs = run_rows(engine, user)
    assert len(runs) == 2
    assert {r.stage for r in runs} == {"assess_answer", "baseline"}
    assert {r.trace_id for r in runs} == {row.trace_id}
    assert all(r.outcome == "ok" for r in runs)


def test_check_answer_without_a_linked_job_still_succeeds(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ad has been read yet -- `job` and `requirements` are absent, and the
    assessment call still runs (rendering that absence honestly, per
    `_format_question_job`/`_format_question_requirements`), never blocked.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    install_fake_client(monkeypatch, create_payload=ASSESSMENT_PAYLOAD, stream_payload=GATE_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    row = answer_row(engine, user, answer_id)
    assert row is not None and row.status == "done"


def test_a_redelivered_check_task_does_not_call_the_model_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    messages = install_fake_client(
        monkeypatch, create_payload=ASSESSMENT_PAYLOAD, stream_payload=GATE_PAYLOAD
    )
    run_worker(engine, user, master_key, log_stream)
    assert len(messages.create_calls) == 1
    assert len(messages.stream_calls) == 1

    second_task = enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    assert len(run_rows(engine, user)) == 2, "a redelivered task called the model again"


# --------------------------------------------------------------------------
# draft_application_answer -- the happy path
# --------------------------------------------------------------------------


def test_draft_answer_drafts_and_gates_and_fills_in_the_text(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    add_job_with_requirements(engine, user, application_id)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_draft_answer(engine, user, question_id)
    task_id = enqueue(engine, user, DRAFT_APPLICATION_ANSWER, answer_id)

    install_fake_client(
        monkeypatch, create_payload=DRAFT_ANSWER_PAYLOAD, stream_payload=GATE_PAYLOAD
    )
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    row = answer_row(engine, user, answer_id)
    assert row is not None
    assert row.status == "done"
    assert row.answer_text == "I want this role for its focus on reliability."
    assert row.assessment is None  # never asked to judge its own draft
    # Not exact dict equality: `check_text` fills in each sentence's own
    # `text` and `rule_flags` (see `jfl_gate.gate._assemble`/`apply_rules`),
    # neither of which the model response carries.
    assert row.gate_result is not None
    stored_sentences = row.gate_result["sentences"]
    assert [
        {
            k: s[k]
            for k in ("index", "kind", "verdict", "drift_label", "cited_span_ids", "evidence_note")
        }
        for s in stored_sentences
    ] == GATE_PAYLOAD["sentences"]

    runs = run_rows(engine, user)
    assert len(runs) == 2
    assert {r.stage for r in runs} == {"draft_answer", "baseline"}


def test_draft_answer_without_requirements_fails_permanently_without_calling_the_model(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No fake client is installed: if the handler reached the API despite
    having nothing to draft against, the root conftest guard would raise and
    this test would fail for the right reason.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)  # no job linked
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_draft_answer(engine, user, question_id)
    task_id = enqueue(engine, user, DRAFT_APPLICATION_ANSWER, answer_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1

    row = answer_row(engine, user, answer_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "no_requirements"
    assert run_rows(engine, user) == []


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    application_id = add_application(engine, user)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    task_id = enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "no Anthropic API key stored" in task.last_error

    row = answer_row(engine, user, answer_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "no_api_key"
    assert run_rows(engine, user) == []


# --------------------------------------------------------------------------
# Failures from the model
# --------------------------------------------------------------------------


def test_a_refusal_on_the_gate_pass_fails_permanently(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    task_id = enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    install_fake_client(
        monkeypatch,
        create_payload=ASSESSMENT_PAYLOAD,
        stream_payload=GATE_PAYLOAD,
        stream_stop_reason="refusal",
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1

    row = answer_row(engine, user, answer_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "model_refused"


def test_a_rejected_key_fails_permanently(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    task_id = enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        stream_exception=anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1

    row = answer_row(engine, user, answer_id)
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
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    question_id = add_question(engine, user, application_id)
    answer_id = add_pending_check_answer(engine, user, question_id)
    task_id = enqueue(engine, user, CHECK_APPLICATION_ANSWER, answer_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        stream_exception=anthropic.APIStatusError(
            "overloaded", response=httpx2.Response(529, request=request), body=None
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.scheduled_at > dt.datetime.now(dt.UTC)

    row = answer_row(engine, user, answer_id)
    assert row is not None
    assert row.status == "failed"  # not "done" -- a redelivery must retry it
    assert row.error_code == "model_error"


# --------------------------------------------------------------------------
# The kill switch
# --------------------------------------------------------------------------


def test_the_kill_switch_leaves_both_kinds_pending_and_calls_nothing(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No fake client is installed: if either handler reached the API despite
    the switch, the root conftest guard would raise and this test would fail
    for the right reason.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    add_job_with_requirements(engine, user, application_id)
    question_id = add_question(engine, user, application_id)
    check_id = add_pending_check_answer(engine, user, question_id)
    draft_id = add_pending_draft_answer(engine, user, question_id)
    check_task = enqueue(engine, user, CHECK_APPLICATION_ANSWER, check_id)
    draft_task = enqueue(engine, user, DRAFT_APPLICATION_ANSWER, draft_id)

    run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    assert task_row(engine, check_task).status == "pending"
    assert task_row(engine, draft_task).status == "pending"

    check_row = answer_row(engine, user, check_id)
    draft_row = answer_row(engine, user, draft_id)
    assert check_row is not None and check_row.status == "pending"
    assert draft_row is not None and draft_row.status == "pending"
