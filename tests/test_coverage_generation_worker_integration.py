"""B5's coverage-check button, end to end against real Postgres: enqueue a
`generate_coverage` task, run it in the real worker loop, and check what
lands in the database. Needs `docker compose up -d` and `alembic upgrade
head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing-key case, the kill switch -- the root `conftest.py` guard is what
would raise if the handler reached the API anyway. Mirrors
`tests/test_title_suggestions_worker_integration.py` closely.
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
from jfl_core.db.tables import users as users_table
from jfl_core.ids import content_hash, requirement_id
from jfl_core.ids import job_id as derive_job_id
from jfl_core.models import Job, JobRequirement
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresJobRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import GENERATE_COVERAGE, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.registry import TaskContext
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

COVERAGE_PAYLOAD = {
    "results": [
        {
            "status": "evidenced",
            "cited_span_ids": [],
            "evidence_note": "Traces to the corpus.",
            "question": None,
        }
    ]
}


@contextmanager
def _never_fetch_a_board() -> Iterator[Transport]:
    """No test in this file watches a job board. This user never adds one, so
    the recurring `schedule_board_checks` tick finds nothing to check -- but
    passing this transport anyway, the same way
    `tests/test_title_suggestions_worker_integration.py` does, means a change
    elsewhere that ever gave this test user a board fails loudly here rather
    than reaching for the network (the root conftest guard would also catch
    it, but this names the assumption).
    """
    raise AssertionError("a coverage-handler test tried to fetch a job board")
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
        model="claude-opus-5",
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


def add_job(engine: Engine, user_id: uuid.UUID, *, requirements: list[str]) -> uuid.UUID:
    raw_text = f"Senior Engineer at Acme. Remote. {uuid.uuid4()}"
    jid = derive_job_id(user_id, raw_text)
    job = Job(
        id=jid,
        user_id=user_id,
        source="paste",
        employer="Acme",
        title="Senior Engineer",
        location="Remote",
        raw_text=raw_text,
        content_hash=content_hash(raw_text),
    )
    with engine.begin() as conn:
        repo = PostgresJobRepository(conn)
        repo.upsert_job(job)
        repo.replace_requirements(
            user_id,
            jid,
            [
                JobRequirement(
                    id=requirement_id(jid, text),
                    user_id=user_id,
                    job_id=jid,
                    ordinal=i,
                    text=text,
                    necessity="essential",
                )
                for i, text in enumerate(requirements)
            ],
        )
    return jid


def enqueue(engine: Engine, user_id: uuid.UUID, job_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=GENERATE_COVERAGE, payload={"job_id": str(job_id)}
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


def get_task(engine: Engine, user_id: uuid.UUID, task_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresTaskRepository(conn, user_id).get_task(task_id)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_records_coverage_rows(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    task_id = enqueue(engine, user, job_id)

    install_fake_client(monkeypatch, payload=COVERAGE_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None and task.status == "succeeded"

    with engine.begin() as conn:
        rows = PostgresJobRepository(conn).latest_coverage(user, job_id)
    assert len(rows) == 1
    assert rows[0].status == "evidenced"


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    enqueue(engine, user, job_id)

    install_fake_client(monkeypatch, payload=COVERAGE_PAYLOAD)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].stage == "coverage"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None


def test_a_redelivered_task_does_not_call_the_model_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The idempotency guard is per-task, via `trace_id = task.id` (see the
    handler's module docstring) -- proved by invoking the handler directly on
    the same `TaskContext` twice, the same way
    `tests/test_worker_integration.py`'s feed-mark-purge test proves it for a
    model-free handler.
    """
    from jfl_worker.handlers.coverage_generation import build_generate_coverage

    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(
            kind=GENERATE_COVERAGE, payload={"job_id": str(job_id)}
        )
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    handler = build_generate_coverage(master_key=master_key, model="claude-opus-5")

    calls = install_fake_client(monkeypatch, payload=COVERAGE_PAYLOAD)
    first = handler(ctx)
    assert first is not None and "skipped" not in first
    assert len(calls) == 1

    monkeypatch.undo()  # back to the socket/API guard: a second real call would raise
    second = handler(ctx)
    assert second is not None and "skipped" in second

    with engine.begin() as conn:
        rows = PostgresJobRepository(conn).latest_coverage(user, job_id)
    assert len(rows) == 1  # not doubled


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    task_id = enqueue(engine, user, job_id)

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert task.attempts == 1
    assert "coverage generation failed permanently: no_api_key" in (task.last_error or "")

    with engine.begin() as conn:
        assert conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all() == []


# --------------------------------------------------------------------------
# Failures from the model, and from the job itself
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
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    task_id = enqueue(engine, user, job_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "coverage generation failed permanently: api_key_rejected" in (task.last_error or "")


def test_a_transient_failure_is_retried_rather_than_given_up_on(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    task_id = enqueue(engine, user, job_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.APIStatusError(
            "overloaded", response=httpx2.Response(529, request=request), body=None
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "pending"
    assert task.scheduled_at > dt.datetime.now(dt.UTC)


def test_a_job_with_no_requirements_fails_permanently_with_a_clear_code(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No model call is installed: if the handler reached the API despite there
    being nothing to check, the root conftest guard would raise.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=[])
    task_id = enqueue(engine, user, job_id)

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "coverage generation failed permanently: no_requirements" in (task.last_error or "")


def test_a_job_id_for_another_user_fails_permanently_as_no_job(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """Tenancy: `run_coverage` reads the job through `ctx.user_id`, which is
    this task's owner -- a job id belonging to someone else (or nobody) reads
    exactly like a missing job, never like a different failure.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    task_id = enqueue(engine, user, uuid.uuid4())

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "coverage generation failed permanently: no_job" in (task.last_error or "")


# --------------------------------------------------------------------------
# The kill switch
# --------------------------------------------------------------------------


def test_the_kill_switch_leaves_the_task_pending_and_calls_nothing(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    task_id = enqueue(engine, user, job_id)

    worker = run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})
    assert worker is not None

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "pending"
    assert task.attempts == 0


# --------------------------------------------------------------------------
# One press, several steps: the check queues the CV behind it, once
# --------------------------------------------------------------------------


def test_a_chained_check_queues_the_next_step_once_even_when_redelivered(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Write the CV" pressed before the check had run queues the check with
    the draft named in `then` (`jfl_web.routes.drafts.request_draft`). The
    handler queues the draft when it succeeds -- and a redelivered task, which
    skips the model, still queues nothing more: one press, one of each step.
    """
    from jfl_worker.handlers.coverage_generation import build_generate_coverage

    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    application_id = str(uuid.uuid4())
    draft_step = {
        "kind": "generate_cv_draft",
        "payload": {"application_id": application_id, "kind": "cv_bullets"},
    }
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(
            kind=GENERATE_COVERAGE, payload={"job_id": str(job_id), "then": [draft_step]}
        )
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    handler = build_generate_coverage(master_key=master_key, model="claude-opus-5")

    install_fake_client(monkeypatch, payload=COVERAGE_PAYLOAD)
    handler(ctx)
    monkeypatch.undo()
    handler(ctx)  # redelivery: skips the model, and must not queue a second draft

    with engine.begin() as conn:
        drafts = PostgresTaskRepository(conn, user).list_tasks(kind="generate_cv_draft")
    assert len(drafts) == 1
    assert drafts[0].payload == {
        "application_id": application_id,
        "kind": "cv_bullets",
        "after": str(task.id),
    }


def test_a_chain_only_continues_into_the_drafting_steps(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task payload is not a place to take instructions from: a `then` naming
    any other kind queues nothing."""
    from jfl_worker.handlers.coverage_generation import build_generate_coverage

    store_key(engine, user, master_key, FAKE_KEY)
    job_id = add_job(engine, user, requirements=["5+ years of Python"])
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(
            kind=GENERATE_COVERAGE,
            payload={"job_id": str(job_id), "then": [{"kind": "check_board", "payload": {}}]},
        )
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    install_fake_client(monkeypatch, payload=COVERAGE_PAYLOAD)
    build_generate_coverage(master_key=master_key, model="claude-opus-5")(ctx)

    with engine.begin() as conn:
        assert PostgresTaskRepository(conn, user).follow_up(task.id) is None


def test_a_read_that_finds_nothing_to_do_still_hands_on_to_the_check(
    engine: Engine, user: uuid.UUID, master_key: MasterKey
) -> None:
    """The extraction handler's "nothing to extract" path -- an ad already read,
    here simply no such application -- calls no model and still queues the
    chained check, which then says for itself whether there is anything to
    check. No fake client: the root guard would raise on any model call.
    """
    from jfl_worker.handlers.extraction import build_extract_job_ad

    job_id = uuid.uuid4()
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(
            kind="extract_job_ad",
            payload={
                "application_id": str(uuid.uuid4()),
                "then": [{"kind": GENERATE_COVERAGE, "payload": {"job_id": str(job_id)}}],
            },
        )
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    result = build_extract_job_ad(master_key=master_key, model="claude-opus-5")(ctx)
    assert result is not None and "skipped" in result

    with engine.begin() as conn:
        follow_up = PostgresTaskRepository(conn, user).follow_up(task.id)
    assert follow_up is not None
    assert follow_up.kind == GENERATE_COVERAGE
    assert follow_up.payload == {"job_id": str(job_id), "after": str(task.id)}
