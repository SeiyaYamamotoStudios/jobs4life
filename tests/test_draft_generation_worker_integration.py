"""B5's "Generate a CV" button, end to end against real Postgres: enqueue a
`generate_cv_draft` task, run it in the real worker loop, and check what
lands in the database. Needs `docker compose up -d` and `alembic upgrade
head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake for the draft call, and `jfl_generate.draft.check_text` (the automatic
claim-gate pass) is monkeypatched directly -- the same technique
`packages/generate/tests/test_draft.py` uses, so this file never has to fake
`.messages.stream()` as well as `.messages.create()`. In the tests that never
reach a model call at all -- missing coverage, missing requirements, no key,
the kill switch -- the root `conftest.py` guard is what would raise if the
handler reached the API anyway. Mirrors
`tests/test_coverage_generation_worker_integration.py` and
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
import jfl_generate.draft as draft_module
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import requirement_id
from jfl_core.models import JobRequirement, RequirementCoverage, RunRecord
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresJobRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_gate.pricing import MODEL
from jfl_gate.schema import GateOutput, SentenceResult
from jfl_intake.http import Transport
from jfl_worker.handlers import GENERATE_CV_DRAFT, build_registry
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

DRAFT_PAYLOAD = {"title": "CV bullets for Acme", "draft": "- Led the platform team at Acme."}

_SUPPORTED_GATE_OUTPUT = GateOutput(
    sentences=[
        SentenceResult(
            index=1,
            kind="claim",
            verdict="supported",
            drift_label="supported",
            cited_span_ids=[],
            evidence_note="Traces cleanly.",
            text="Led the platform team at Acme.",
        )
    ]
)


@contextmanager
def _never_fetch_a_board() -> Iterator[Transport]:
    """No test in this file watches a job board -- see the identical guard in
    `tests/test_coverage_generation_worker_integration.py`.
    """
    raise AssertionError("a draft-handler test tried to fetch a job board")
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


def install_fake_gate(
    monkeypatch: pytest.MonkeyPatch, output: GateOutput = _SUPPORTED_GATE_OUTPUT
) -> list[Any]:
    """Stands in for `jfl_gate.gate.check_text` -- see the module docstring.
    Writes a real `runs` row on the `run_repo` it is handed, the same way the
    real function does, so a draft's total cost still spans both calls.
    """
    calls: list[Any] = []

    def _fake(ctx: RequestContext, grounding_repo: object, run_repo: Any, text: str) -> GateOutput:
        calls.append(ctx)
        run_repo.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="gate",
                stage="baseline",
                model=MODEL,
                outcome="ok",
                started_at=dt.datetime.now(dt.UTC),
            )
        )
        return output

    monkeypatch.setattr(draft_module, "check_text", _fake)
    return calls


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


_DEFAULT_REQUIREMENTS = ["5+ years of Python"]


def add_application(
    engine: Engine,
    user_id: uuid.UUID,
    *,
    with_ad: bool = True,
    requirements: list[str] | None = None,
    with_coverage: bool = True,
) -> tuple[uuid.UUID, uuid.UUID | None]:
    """An application, optionally backed by a job with requirements and
    recorded coverage -- `generate_draft`'s three prerequisites (see its
    docstring), each individually switchable so a test can leave exactly one
    unmet. `requirements=None` means "the one default requirement";
    `requirements=[]` means "extracted, but none found".
    """
    if requirements is None:
        requirements = _DEFAULT_REQUIREMENTS
    raw_text = f"Senior Engineer at Acme. Remote. {uuid.uuid4()}" if with_ad else None
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user_id).create_application(
            title="Senior Engineer",
            raw_job_text=raw_text,
        )
    job_id = application.job_id
    if job_id is None:
        return application.id, job_id

    with engine.begin() as conn:
        job_repo = PostgresJobRepository(conn)
        job_repo.replace_requirements(
            user_id,
            job_id,
            [
                JobRequirement(
                    id=requirement_id(job_id, text),
                    user_id=user_id,
                    job_id=job_id,
                    ordinal=i,
                    text=text,
                    necessity="essential",
                )
                for i, text in enumerate(requirements)
            ],
        )
        if with_coverage and requirements:
            found = job_repo.get_job(user_id, job_id)
            assert found is not None
            for requirement in found[1]:
                job_repo.record_coverage(
                    RequirementCoverage(
                        user_id=user_id,
                        requirement_id=requirement.id,
                        trace_id=uuid.uuid4(),
                        status="evidenced",
                        cited_span_ids=[],
                        evidence_note="Traces to the corpus.",
                    )
                )
    return application.id, job_id


def enqueue(
    engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID, kind: str = "cv_bullets"
) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=GENERATE_CV_DRAFT, payload={"application_id": str(application_id), "kind": kind}
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


def test_the_handler_records_a_draft(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id, job_id = add_application(engine, user)
    assert job_id is not None
    task_id = enqueue(engine, user, application_id)

    install_fake_client(monkeypatch, payload=DRAFT_PAYLOAD)
    install_fake_gate(monkeypatch)
    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None and task.status == "succeeded"

    with engine.begin() as conn:
        drafts = PostgresJobRepository(conn).list_drafts(user, job_id)
    assert len(drafts) == 1
    assert drafts[0].kind == "cv_bullets"
    assert "Led the platform team at Acme." in drafts[0].text
    assert drafts[0].created_at is not None


def test_both_runs_rows_share_one_trace_id_so_cost_is_one_query(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id, job_id = add_application(engine, user)
    assert job_id is not None
    enqueue(engine, user, application_id)

    install_fake_client(monkeypatch, payload=DRAFT_PAYLOAD)
    install_fake_gate(monkeypatch)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        drafts = PostgresJobRepository(conn).list_drafts(user, job_id)
        runs = conn.execute(
            select(runs_table).where(runs_table.c.trace_id == drafts[0].trace_id)
        ).all()
    assert {r.stage for r in runs} == {"draft", "baseline"}
    assert all(r.user_id == user for r in runs)


def test_a_redelivered_task_does_not_generate_a_second_draft(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same idempotency proof as the coverage handler's: invoke the handler
    directly on the same `TaskContext` twice.
    """
    from jfl_worker.handlers.draft_generation import build_generate_cv_draft

    store_key(engine, user, master_key, FAKE_KEY)
    application_id, job_id = add_application(engine, user)
    assert job_id is not None
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(
            kind=GENERATE_CV_DRAFT,
            payload={"application_id": str(application_id), "kind": "cv_bullets"},
        )
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    handler = build_generate_cv_draft(master_key=master_key, model="claude-opus-5")

    calls = install_fake_client(monkeypatch, payload=DRAFT_PAYLOAD)
    install_fake_gate(monkeypatch)
    first = handler(ctx)
    assert first is not None and "skipped" not in first
    assert len(calls) == 1

    monkeypatch.undo()  # back to the socket/API guard for the draft call
    second = handler(ctx)
    assert second is not None and "skipped" in second
    assert second["draft_id"] == first["draft_id"]

    with engine.begin() as conn:
        assert len(PostgresJobRepository(conn).list_drafts(user, job_id)) == 1


# --------------------------------------------------------------------------
# Prerequisites are explicit, never silently satisfied
# --------------------------------------------------------------------------


def test_missing_coverage_fails_permanently_with_a_clear_code(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No model call is installed: `generate_draft` refuses before it would
    ever reach one, and the root conftest guard is the backstop.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id, _ = add_application(engine, user, with_coverage=False)
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "draft generation failed permanently: no_coverage" in (task.last_error or "")


def test_missing_requirements_fails_permanently_with_a_clear_code(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id, _ = add_application(engine, user, requirements=[], with_coverage=False)
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "draft generation failed permanently: no_requirements" in (task.last_error or "")


def test_no_job_linked_to_the_application_fails_permanently_with_a_clear_code(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id, job_id = add_application(engine, user, with_ad=False)
    assert job_id is None
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "draft generation failed permanently: no_job" in (task.last_error or "")


def test_an_application_for_another_user_fails_permanently(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    task_id = enqueue(engine, user, uuid.uuid4())

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert "no application for this user" in (task.last_error or "")


# --------------------------------------------------------------------------
# No key stored, and failures from the model
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    application_id, _ = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key, log_stream)

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "failed"
    assert task.attempts == 1
    assert "draft generation failed permanently: no_api_key" in (task.last_error or "")


def test_a_rejected_key_fails_permanently(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id, _ = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

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
    assert "draft generation failed permanently: api_key_rejected" in (task.last_error or "")


def test_a_transient_failure_is_retried_rather_than_given_up_on(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id, _ = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

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


# --------------------------------------------------------------------------
# The kill switch
# --------------------------------------------------------------------------


def test_the_kill_switch_leaves_the_task_pending_and_calls_nothing(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id, _ = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    worker = run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})
    assert worker is not None

    task = get_task(engine, user, task_id)
    assert task is not None
    assert task.status == "pending"
    assert task.attempts == 0
