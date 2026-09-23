"""Slice B4's background half, end to end against real Postgres: enqueue a
`score_application` task, run it in the real worker loop, and check what lands
in the database. Needs `docker compose up -d` and `alembic upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is monkeypatched to a
fake in the tests that get as far as a model call; in the ones that do not --
the missing-key case, the kill switch, the already-scored case -- the root
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
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.fit import BREACH_CEILING
from jfl_core.ids import content_hash, job_id, requirement_id
from jfl_core.models import Job, JobRequirement
from jfl_core.profile import Constraint, Objective, Profile
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.postgres import PostgresJobRepository
from jfl_core.storage.profile import PostgresProfileRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.http import Transport
from jfl_worker.handlers import SCORE_APPLICATION, build_registry
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

AD = "Engineering Manager at Northwind. You will need five years of Python."

# No `want_it_score`: the model is not asked for one. It gives a verdict per
# constraint and per objective, and `jfl_core.fit` derives the number from
# those -- see `docs/profile-schema.md`.
SCORE_PAYLOAD: dict[str, Any] = {
    "could_get_score": 6,
    "could_get_assessment": "Your record evidences the Python.",
    "want_it_assessment": "The commute breaks what you said you would travel.",
    "constraint_verdicts": [
        {
            "index": 1,
            "verdict": "contradicted",
            "note": "On site five days a week in Manchester.",
        }
    ],
    "objective_verdicts": [
        {"rank": 1, "verdict": "silent", "note": "The ad does not say how hands-on it is."}
    ],
    "levers": [],
}

COVERAGE_PAYLOAD: dict[str, Any] = {
    "results": [
        {
            "status": "absent",
            "cited_span_ids": [],
            "evidence_note": "The corpus is silent on Python.",
            "question": "Have you worked in Python?",
        }
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


# -- fake client --------------------------------------------------------------


class _FakeMessages:
    def __init__(self, responses: list[Message], exception: Exception | None) -> None:
        self._responses = responses
        self._exception = exception
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        assert self._responses, "the handler made more calls than the test set up"
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[Message], exception: Exception | None) -> None:
        self.messages = _FakeMessages(responses, exception)


def _message(payload: dict[str, Any]) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model="claude-opus-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(
            input_tokens=2000,
            output_tokens=400,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payloads: list[dict[str, Any]] | None = None,
    exception: Exception | None = None,
) -> list[dict[str, Any]]:
    """Returns the list the fake appends each `messages.create` call to, so a
    test can count how many model calls the handler actually made.
    """
    client = _FakeClient([_message(p) for p in (payloads or [])], exception)

    def _construct(**kwargs: Any) -> _FakeClient:
        return client

    monkeypatch.setattr(anthropic, "Anthropic", _construct)
    return client.messages.calls


# -- fixtures in the database -------------------------------------------------


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey, key: str) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, key, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=key[-4:],
        )


def add_application(
    engine: Engine, user_id: uuid.UUID, *, with_requirements: bool = True
) -> uuid.UUID:
    """An application with a read ad behind it -- what "Score this application"
    is pressed on.
    """
    application_id = uuid.uuid4()
    jid = job_id(user_id, AD)
    with engine.begin() as conn:
        repo = PostgresJobRepository(conn)
        repo.upsert_job(
            Job(
                id=jid,
                user_id=user_id,
                source="paste",
                employer="Northwind",
                title="Engineering Manager",
                location="Manchester",
                raw_text=AD,
                content_hash=content_hash(AD),
            )
        )
        if with_requirements:
            repo.replace_requirements(
                user_id,
                jid,
                [
                    JobRequirement(
                        id=requirement_id(jid, "Five years of Python"),
                        user_id=user_id,
                        job_id=jid,
                        ordinal=0,
                        text="Five years of Python",
                        necessity="essential",
                    )
                ],
            )
        conn.execute(
            insert(applications_table).values(
                id=application_id,
                user_id=user_id,
                job_id=jid,
                title="Engineering Manager",
                extraction_status="done",
            )
        )
    return application_id


def add_profile(engine: Engine, user_id: uuid.UUID) -> None:
    """One constraint and one objective -- enough for the derivation to have
    something to derive from, and for the unfilled sections to be reported.
    """
    with engine.begin() as conn:
        PostgresProfileRepository(conn, user_id).save(
            Profile(
                constraints=[
                    Constraint(kind="workplace", stance="must", note="One day a week at most")
                ],
                objectives=[
                    Objective(
                        rank=1,
                        text="Back to hands-on platform work",
                        evidence_of_delivery="Ships weekly",
                    )
                ],
            )
        )


def add_pending_score(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        return PostgresScoreRepository(conn, user_id).create_pending(application_id).id


def enqueue(
    engine: Engine, user_id: uuid.UUID, score_id: uuid.UUID, *, max_attempts: int = 3
) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=SCORE_APPLICATION, payload={"score_id": str(score_id)}, max_attempts=max_attempts
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


def score_row(engine: Engine, user_id: uuid.UUID, score_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresScoreRepository(conn, user_id).get(score_id)


def runs_for(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return list(conn.execute(select(runs_table).where(runs_table.c.user_id == user_id)).all())


def record_coverage(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> None:
    """Coverage already recorded for this job, so the handler has no reason to
    run it again.
    """
    from jfl_core.models import RequirementCoverage

    with engine.begin() as conn:
        detail_job = conn.execute(
            select(applications_table.c.job_id).where(applications_table.c.id == application_id)
        ).scalar_one()
        repo = PostgresJobRepository(conn)
        found = repo.get_job(user_id, detail_job)
        assert found is not None
        for requirement in found[1]:
            repo.record_coverage(
                RequirementCoverage(
                    user_id=user_id,
                    requirement_id=requirement.id,
                    trace_id=uuid.uuid4(),
                    status="evidenced",
                    cited_span_ids=[],
                    evidence_note="Three roles document it.",
                )
            )


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_handler_scores_and_marks_the_row_done(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    add_profile(engine, user)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

    calls = install_fake_client(monkeypatch, payloads=[SCORE_PAYLOAD])
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"
    assert len(calls) == 1, "coverage was already recorded, so only one call was owed"

    row = score_row(engine, user, score_id)
    assert row is not None
    assert row.status == "done"
    assert row.error_code is None
    # Two numbers, both stored, neither derived from the other. The second is
    # derived from the verdicts stored beside it: the only `must` is broken and
    # the only objective is unanswered, so the ad evidences nothing of what
    # this person said matters.
    assert row.could_get_score == 6
    assert row.want_it_score is not None and row.want_it_score <= BREACH_CEILING
    assert row.want_it_score == 1
    assert row.could_get_assessment == "Your record evidences the Python."
    assert row.want_it_assessment.startswith("The commute")
    assert [v.verdict for v in row.constraint_verdicts] == ["contradicted"]
    assert [v.verdict for v in row.objective_verdicts] == ["silent"]
    # The breach is derived from the constraint verdict and stated in plain
    # words, never folded silently into the number.
    assert row.hard_gate_breaches[0].breach.startswith("On site five days a week")
    assert row.model == "claude-opus-5"
    assert row.cost_usd is not None and row.cost_usd > 0
    assert row.trace_id is not None
    # Everything the user has not filled in is named, not guessed.
    assert {n.question_key for n in row.not_stated} >= {"capabilities", "disciplines"}


def test_a_runs_row_is_written_for_the_model_call(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    enqueue(engine, user, score_id)

    install_fake_client(monkeypatch, payloads=[SCORE_PAYLOAD])
    run_worker(engine, user, master_key, log_stream)

    runs = runs_for(engine, user)
    assert len(runs) == 1
    assert runs[0].component == "generate"
    assert runs[0].stage == "score"
    assert runs[0].outcome == "ok"
    assert runs[0].cost_usd is not None

    row = score_row(engine, user, score_id)
    assert row is not None and row.cost_usd == runs[0].cost_usd
    assert row.trace_id == runs[0].trace_id


def test_with_no_coverage_recorded_it_runs_coverage_first_and_bills_both(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "Could I get this" is judged from coverage, so with none recorded there
    is nothing honest to judge from. The button that enqueued this says the
    check may run, so this is a told cost, not a silent one -- and what the row
    shows is the total of both calls.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    score_id = add_pending_score(engine, user, application_id)
    enqueue(engine, user, score_id)

    calls = install_fake_client(monkeypatch, payloads=[COVERAGE_PAYLOAD, SCORE_PAYLOAD])
    run_worker(engine, user, master_key, log_stream)

    assert len(calls) == 2
    runs = runs_for(engine, user)
    assert sorted(r.stage for r in runs) == ["coverage", "score"]

    row = score_row(engine, user, score_id)
    assert row is not None and row.status == "done"
    assert row.cost_usd == sum(r.cost_usd for r in runs)


def test_a_redelivered_task_does_not_score_a_second_time(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    enqueue(engine, user, score_id)

    calls = install_fake_client(monkeypatch, payloads=[SCORE_PAYLOAD])
    run_worker(engine, user, master_key, log_stream)
    assert len(calls) == 1

    second_task = enqueue(engine, user, score_id)
    monkeypatch.undo()  # back to the guard: any client construction now raises
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, second_task).status == "succeeded"
    assert len(runs_for(engine, user)) == 1, "a redelivered task scored again"


# --------------------------------------------------------------------------
# No key stored
# --------------------------------------------------------------------------


def test_no_api_key_fails_once_and_is_never_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    application_id = add_application(engine, user)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1
    assert "no Anthropic API key stored" in task.last_error
    assert FAKE_KEY not in (task.last_error or "")

    row = score_row(engine, user, score_id)
    assert row is not None
    assert row.status == "failed"
    assert row.error_code == "no_api_key"
    assert runs_for(engine, user) == []


# --------------------------------------------------------------------------
# Nothing to score against
# --------------------------------------------------------------------------


def test_an_unread_ad_fails_permanently_without_calling_anything(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, log_stream: io.StringIO
) -> None:
    """No fake client: a number for "could I get this" from an ad nobody has
    read would be a number with nothing behind it, and the guard would raise if
    the handler called the model anyway.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user, with_requirements=False)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

    run_worker(engine, user, master_key, log_stream)

    task = task_row(engine, task_id)
    assert task.status == "failed"
    assert task.attempts == 1

    row = score_row(engine, user, score_id)
    assert row is not None and row.status == "failed"
    assert row.error_code == "no_requirements"
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
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

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
    # A closed-set code, never SDK text from a call made with the user's key.
    assert "invalid x-api-key" not in (task.last_error or "")

    row = score_row(engine, user, score_id)
    assert row is not None and row.error_code == "api_key_rejected"


def test_a_transient_failure_is_retried_rather_than_given_up_on(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2

    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

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

    row = score_row(engine, user, score_id)
    # Retrying, not failed: the row stays `pending` so the retry is not
    # skipped as already finished, and carries the code so the page can say
    # "retrying" rather than show an error.
    assert row is not None and row.status == "pending"
    assert row.error_code == "model_error"


def _overloaded() -> Exception:
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(
        "overloaded", response=httpx2.Response(529, request=request), body=None
    )


def test_a_transient_failure_on_the_last_attempt_marks_the_row_failed(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_key(engine, user, master_key, FAKE_KEY)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id, max_attempts=1)

    install_fake_client(monkeypatch, exception=_overloaded())
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "failed"
    row = score_row(engine, user, score_id)
    assert row is not None and row.status == "failed"
    assert row.error_code == "model_error"


def test_the_retry_after_a_transient_failure_actually_scores(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug under the owner's "sometimes fails": the first attempt used to
    mark the row `failed`, so the queued retry found a finished row and skipped
    itself. Now the retry runs and lands the score.
    """
    store_key(engine, user, master_key, FAKE_KEY)
    add_profile(engine, user)
    application_id = add_application(engine, user)
    record_coverage(engine, user, application_id)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

    install_fake_client(monkeypatch, exception=_overloaded())
    run_worker(engine, user, master_key, log_stream)
    assert task_row(engine, task_id).status == "pending"

    # Make the retry due now, and let it succeed.
    with engine.begin() as conn:
        conn.execute(
            tasks_table.update()
            .where(tasks_table.c.id == task_id)
            .values(scheduled_at=dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1))
        )
    install_fake_client(monkeypatch, payloads=[SCORE_PAYLOAD])
    run_worker(engine, user, master_key, log_stream)

    assert task_row(engine, task_id).status == "succeeded"
    row = score_row(engine, user, score_id)
    assert row is not None and row.status == "done"
    assert row.error_code is None
    assert row.could_get_score == 6


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
    application_id = add_application(engine, user)
    score_id = add_pending_score(engine, user, application_id)
    task_id = enqueue(engine, user, score_id)

    run_worker(engine, user, master_key, log_stream, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    task = task_row(engine, task_id)
    assert task.status == "pending"
    assert task.attempts == 0

    row = score_row(engine, user, score_id)
    assert row is not None and row.status == "pending"


# --------------------------------------------------------------------------
# Tenancy
# --------------------------------------------------------------------------


def test_a_task_carrying_another_users_score_id_scores_nothing(
    engine: Engine,
    user: uuid.UUID,
    master_key: MasterKey,
    log_stream: io.StringIO,
) -> None:
    """The handler reads the row under the task's own `user_id`, so an id from
    another account simply resolves to nothing. No fake client: reaching the
    model here would be the guard's failure to report.
    """
    other = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(insert(users_table).values(id=other, email=f"{other}@test.invalid"))
    try:
        store_key(engine, user, master_key, FAKE_KEY)
        application_id = add_application(engine, other)
        score_id = add_pending_score(engine, other, application_id)
        task_id = enqueue(engine, user, score_id)

        run_worker(engine, user, master_key, log_stream)

        assert task_row(engine, task_id).status == "succeeded"
        row = score_row(engine, other, score_id)
        assert row is not None and row.status == "pending"
        assert runs_for(engine, user) == []
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id == other))
