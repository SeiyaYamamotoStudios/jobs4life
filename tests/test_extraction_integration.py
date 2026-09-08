"""Slice B3's background half, end to end against real Postgres: enqueue a
paste, run the `extract_job_ad` handler in the real worker loop, and check what
lands in the database.

Needs `docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call, and in the tests that do not
-- the missing-key case, the already-done case -- the root `conftest.py` guard
is what would raise if the handler reached the API anyway. That guard is not
weakened anywhere in this file; a real client still needs both the `e2e` marker
and `JFL_ALLOW_REAL_API=1`.

The user is a throwaway row per test, so everything created here disappears when
it is deleted: applications, jobs, tasks, credentials and runs all cascade.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import uuid
from collections.abc import Iterator
from typing import Any

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import job_requirements as job_requirements_table
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_worker.handlers import EXTRACT_JOB_AD, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = "postgresql+psycopg://jfl:jfl@localhost:5433/jfl"

# Shaped like a real key and obviously not one. Every test that stores a
# credential uses this exact string, which is what the leak test greps for.
FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

AD = """Senior Platform Engineer

Acme Corp, London. You will own the deployment pipeline.

Requirements:
- 5+ years of Python
- Kubernetes experience preferred
"""

EXTRACTED = {
    "employer": "Acme Corp",
    "title": "Senior Platform Engineer",
    "location": "London",
    "requirements": [
        {"text": "5+ years of Python", "necessity": "essential"},
        {"text": "Kubernetes experience", "necessity": "desirable"},
    ],
}


# --------------------------------------------------------------------------
# Fixtures and fakes
# --------------------------------------------------------------------------


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
            input_tokens=800,
            output_tokens=200,
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
    """Replace the Anthropic client with a fake. Returns the list the
    constructor's kwargs are appended to, so a test can see the key that was
    handed to it -- which is how "the key does reach the call" is proved
    alongside "the key reaches nothing else".
    """
    constructions: list[dict[str, Any]] = []
    client = _FakeClient(None if payload is None else _message(payload), exception)

    def _construct(**kwargs: Any) -> _FakeClient:
        constructions.append(kwargs)
        return client

    monkeypatch.setattr(anthropic, "Anthropic", _construct)
    return constructions


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def add_application(engine: Engine, user_id: uuid.UUID, *, ad: str = AD) -> uuid.UUID:
    """What the paste box does: a row with a provisional title, an ad stored
    verbatim, and an extraction waiting to happen.
    """
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user_id).create_application(
            title="Senior Platform Engineer",
            raw_job_text=ad,
            title_is_provisional=True,
            extraction_status="pending",
        )
    return application.id


def enqueue(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=EXTRACT_JOB_AD, payload={"application_id": str(application_id)}
        )
    return task.id


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def run_worker(
    engine: Engine, user_id: uuid.UUID, master_key: MasterKey, stream: io.StringIO
) -> Worker:
    """The real loop, the real registry, drained until nothing is due.

    Drained rather than one `run_once`, because the worker enqueues its own
    recurring session purge on the first iteration and `batch_size` is 1 -- so a
    single pass might spend itself on maintenance and never reach the task under
    test. A task backing off into the future is not due, so this still
    terminates on the retry cases.
    """
    settings = WorkerSettings(
        database_url=DATABASE_URL, system_user_id=user_id, master_key=master_key
    )
    worker = Worker(
        registry=build_registry(settings),
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, user_id),
        engine=engine,
        env={},
        logger=configure_logging(stream=stream),
    )
    for _ in range(10):
        if worker.run_once() == 0:
            return worker
    raise AssertionError("the queue did not drain -- a task is looping")


def task_row(engine: Engine, task_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return conn.execute(select(tasks_table).where(tasks_table.c.id == task_id)).one()


def application_row(engine: Engine, application_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return conn.execute(
            select(applications_table).where(applications_table.c.id == application_id)
        ).one()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_extracts_stores_requirements_and_updates_the_application(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    install_fake_client(monkeypatch, payload=EXTRACTED)
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"

    row = application_row(engine, application_id)
    assert row.extraction_status == "done"
    assert row.extraction_error_code is None
    assert row.extracted_at is not None
    # The provisional title was replaced by what the ad actually says, and it
    # stopped being provisional in the same write.
    assert row.title == "Senior Platform Engineer"
    assert row.title_is_provisional is False
    assert row.employer == "Acme Corp"

    with engine.begin() as conn:
        requirements = conn.execute(
            select(job_requirements_table.c.text, job_requirements_table.c.necessity)
            .where(job_requirements_table.c.user_id == user)
            .order_by(job_requirements_table.c.ordinal)
        ).all()
    assert [r.text for r in requirements] == ["5+ years of Python", "Kubernetes experience"]
    assert [r.necessity for r in requirements] == ["essential", "desirable"]


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`runs` is the cost-attribution table and the handler may not bypass it --
    the user is paying for this call with their own key.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    enqueue(engine, user, application_id)

    install_fake_client(monkeypatch, payload=EXTRACTED)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].stage == "extract_requirements"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None


def test_a_user_typed_employer_is_never_overwritten_by_extraction(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole product is about not putting words in someone's mouth. An
    extraction may fill a blank; it may not correct a person.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user).create_application(
            title="The title I typed myself",
            employer="The employer I typed myself",
            raw_job_text=AD,
            title_is_provisional=False,
            extraction_status="pending",
        )
    enqueue(engine, user, application.id)

    install_fake_client(monkeypatch, payload=EXTRACTED)
    run_worker(engine, user, master_key, log_stream)

    row = application_row(engine, application.id)
    assert row.extraction_status == "done"
    assert row.title == "The title I typed myself"
    assert row.employer == "The employer I typed myself"


def test_a_redelivered_task_does_not_extract_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delivery is at-least-once, and twice here means paying twice.

    The second run finds the extraction already `done` and returns without
    constructing a client at all -- which the root conftest guard would make
    loud, because the fake is deliberately not installed for it.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    enqueue(engine, user, application_id)

    calls = install_fake_client(monkeypatch, payload=EXTRACTED)
    run_worker(engine, user, master_key, log_stream)
    assert len(calls) == 1

    # Same work, delivered again -- as a reclaimed row would be.
    second_task = enqueue(engine, user, application_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
    assert len(runs) == 1, "a redelivered task called the model again"


# --------------------------------------------------------------------------
# No key stored: the failure the brief names
# --------------------------------------------------------------------------


def test_a_user_with_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No key stored is not weather. Nothing changes between attempts, so
    burning three of them buys twenty minutes of a spinner on a screen that
    should already say "add your API key".

    No fake client is installed: if the handler reached the API despite there
    being no key, the root conftest guard would raise and this test would fail
    for the right reason.
    """
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    worker = run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1  # one spent, two left unused
    assert task.finished_at is not None
    assert "no Anthropic API key stored" in task.last_error

    # And it is genuinely terminal: no later pass picks it up again, whatever
    # else the worker finds to do.
    worker.run_once()
    still = task_row(engine, task_id)
    assert still.status == "failed"
    assert still.attempts == 1

    row = application_row(engine, application_id)
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "no_api_key"

    # Nothing was charged, so nothing was recorded.
    with engine.begin() as conn:
        assert conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all() == []


def test_the_no_key_failure_reaches_the_page_with_a_link_to_settings(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """The message has to be actionable, not merely honest."""
    from jfl_web.jobads import extraction_failure

    application_id = add_application(engine, user)
    enqueue(engine, user, application_id)
    run_worker(engine, user, master_key, log_stream)

    with engine.begin() as conn:
        extraction = PostgresApplicationRepository(conn, user).get_extraction(application_id)
    assert extraction is not None and extraction.error_code == "no_api_key"

    failure = extraction_failure(extraction.error_code)
    assert failure.fix_url == "/settings"
    assert "API key" in failure.message


# --------------------------------------------------------------------------
# Failures that come back from the model
# --------------------------------------------------------------------------


def test_a_rejected_key_fails_permanently_and_points_at_settings(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key Anthropic refuses is not going to be accepted on the third try."""
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.AuthenticationError(
            "invalid x-api-key",
            response=httpx2.Response(401, request=request),
            body=None,
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1

    row = application_row(engine, application_id)
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "api_key_rejected"


def test_a_transient_api_failure_is_retried_rather_than_given_up_on(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the distinction: a 529 is weather, and waiting helps."""
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

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
    assert task.scheduled_at > dt.datetime.now(dt.UTC)  # backing off, not gone
    assert application_row(engine, application_id).extraction_error_code == "model_error"


# --------------------------------------------------------------------------
# Custody: the key reaches the call and nothing else
# --------------------------------------------------------------------------


def test_the_api_key_reaches_the_client_and_nothing_else(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-negotiable rule, checked everywhere the key could have landed.

    A logging bug in this app is a credential disclosure, which is why custody
    is tested rather than reviewed. The key is proved to reach the Anthropic
    client -- otherwise this test would pass on a handler that simply never
    loaded it -- and then proved absent from every row and every line the run
    produced.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    constructions = install_fake_client(monkeypatch, payload=EXTRACTED)
    run_worker(engine, user, master_key, log_stream)

    # It did reach the call. Without this the rest proves nothing.
    assert constructions == [{"api_key": FAKE_KEY}]

    task = task_row(engine, task_id)
    assert FAKE_KEY not in json.dumps(task.payload)
    assert set(task.payload) == {"application_id"}
    assert FAKE_KEY not in (task.last_error or "")

    # Every log line the worker wrote, including the started/succeeded pair.
    logged = log_stream.getvalue()
    assert logged, "the worker logged nothing, so this proves nothing"
    assert FAKE_KEY not in logged

    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all()
        application = conn.execute(
            select(applications_table).where(applications_table.c.id == application_id)
        ).one()
    assert FAKE_KEY not in json.dumps([str(dict(r._mapping)) for r in runs])
    assert FAKE_KEY not in str(dict(application._mapping))


def test_a_failing_run_puts_no_key_in_the_error_it_records(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The likeliest leak is a formatted exception: the SDK's errors carry the
    request that authenticated with the key. So the failure path is checked
    with an exception whose own message contains the key.
    """
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.AuthenticationError(
            f"invalid x-api-key: {FAKE_KEY}",
            response=httpx2.Response(401, request=request),
            body=None,
        ),
    )
    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert FAKE_KEY not in (task.last_error or "")
    assert FAKE_KEY not in log_stream.getvalue()
    assert FAKE_KEY not in str(dict(application_row(engine, application_id)._mapping))


# --------------------------------------------------------------------------
# Tenancy, at the handler
# --------------------------------------------------------------------------


def test_a_task_cannot_extract_against_another_users_application(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """The web route refuses this at the door (see
    `test_applications_web_integration.py`); this is the second lock, at the
    handler, where the only `user_id` in play is the claimed task's own.
    """
    other = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users_table).values(id=other, email=f"{other}@test.invalid"))
    try:
        victim = add_application(engine, other)
        store_key(engine, user, master_key, FAKE_KEY)
        # A task owned by `user`, naming `other`'s application. No client is
        # installed: reaching the model here would be the failure.
        task_id = enqueue(engine, user, victim)

        run_worker(engine, user, master_key, log_stream)

        # It succeeds by doing nothing -- the scoped read finds no such
        # application, so there is nothing to extract and nothing to say.
        assert task_row(engine, task_id).status == "succeeded"
        row = application_row(engine, victim)
        assert row.extraction_status == "pending"
        assert row.employer is None
        with engine.begin() as conn:
            assert conn.execute(select(runs_table).where(runs_table.c.user_id == user)).all() == []
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id == other))
