"""Scoring starts the moment an application is added -- end to end against real
Postgres, through the real worker loop and the real registry.

The owner: *"scoring needs to kick off instantly with the application being
added."* Adding an application is the user choosing the job (CLAUDE.md,
2026-09-15), so a successful read of the ad queues the first score in the same
transaction that records the read. What is proved here:

  * a pasted ad chains read -> score, and exactly one score is queued;
  * a redelivered read queues nothing more (and calls nothing);
  * no key stored: the read fails with `no_api_key` and no score is queued;
  * an application that already has a score does not get a second one from a
    re-read -- re-scoring is the button;
  * "Track as application" chains too: fetch -> read -> score.

**No test here spends a penny.** The Anthropic client is a fake that answers
each call in turn; where the fake is removed, the root conftest guard would
raise if anything reached for the API.
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
import jfl_worker.handlers.description as description_module
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import application_scores as scores_table
from jfl_core.db.tables import applications as applications_table
from jfl_core.db.tables import tasks as tasks_table
from jfl_core.db.tables import users as users_table
from jfl_core.models import CheckPlan, ObservedJob
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.boards import PostgresBoardRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.scores import PostgresScoreRepository
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_intake.adapters.base import FetchResult
from jfl_intake.descriptions import DescriptionResult
from jfl_intake.engine import REPOST_WINDOW, plan_check
from jfl_intake.http import Transport
from jfl_intake.normalise import fingerprint
from jfl_worker.handlers import (
    EXTRACT_JOB_AD,
    FETCH_JOB_DESCRIPTION,
    SCORE_APPLICATION,
    build_registry,
)
from jfl_worker.log import configure_logging
from jfl_worker.queue import postgres_enqueuer_scope, postgres_queue_scope
from jfl_worker.runner import Worker
from jfl_worker.settings import WorkerSettings
from sqlalchemy import create_engine, delete, insert, select
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")

FAKE_KEY = "sk-ant-api03-NEVERLEAKTHISVALUE-0123456789abcdef"

AD = """Senior Platform Engineer

Acme Corp, London. You will own the deployment pipeline.

Requirements:
- 5+ years of Python
"""

EXTRACTED: dict[str, Any] = {
    "employer": "Acme Corp",
    "title": "Senior Platform Engineer",
    "location": "London",
    "requirements": [{"text": "5+ years of Python", "necessity": "essential"}],
}

COVERAGE: dict[str, Any] = {
    "results": [
        {
            "status": "absent",
            "cited_span_ids": [],
            "evidence_note": "The corpus is silent on Python.",
            "question": "Have you worked in Python?",
        }
    ]
}

SCORE: dict[str, Any] = {
    "could_get_score": 4,
    "could_get_assessment": "Nothing confirmed evidences the Python.",
    "want_it_assessment": "Nothing recorded to measure against.",
    "constraint_verdicts": [],
    "objective_verdicts": [],
    "levers": [],
}

DAY0 = dt.datetime(2026, 9, 1, 6, 0, tzinfo=dt.UTC)


# -- fixtures -----------------------------------------------------------------


@contextmanager
def _dummy_transport() -> Iterator[Transport]:
    yield object()  # type: ignore[misc]


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


# -- fakes ----------------------------------------------------------------------


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


class _FakeMessages:
    """Answers each call with the next payload, in order: the read, then the
    coverage check the scorer runs first, then the score itself."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        assert self._payloads, "the handler made more model calls than this test expected"
        return _message(self._payloads.pop(0))


class _FakeClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.messages = _FakeMessages(payloads)


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch, payloads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    client = _FakeClient(payloads)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    return client.messages.calls


# -- helpers --------------------------------------------------------------------


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, FAKE_KEY, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=FAKE_KEY[-4:],
        )


def paste_application(engine: Engine, user_id: uuid.UUID) -> uuid.UUID:
    """What `POST /applications` does: the row, and its read queued with it."""
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user_id).create_application(
            title="Senior Platform Engineer",
            raw_job_text=AD,
            title_is_provisional=True,
            extraction_status="pending",
        )
        PostgresTaskRepository(conn, user_id).enqueue(
            kind=EXTRACT_JOB_AD, payload={"application_id": str(application.id)}
        )
    return application.id


def run_worker(engine: Engine, user_id: uuid.UUID, master_key: MasterKey) -> None:
    settings = WorkerSettings(
        database_url=DATABASE_URL, system_user_id=user_id, master_key=master_key
    )
    worker = Worker(
        registry=build_registry(
            settings,
            board_transport=_dummy_transport,
            description_transport=_dummy_transport,
            board_owners={user_id},
        ),
        settings=settings,
        queue_scope=postgres_queue_scope(engine),
        enqueuer_scope=postgres_enqueuer_scope(engine, user_id),
        engine=engine,
        env={},
        logger=configure_logging(stream=io.StringIO()),
    )
    for _ in range(12):
        if worker.run_once() == 0:
            return
    raise AssertionError("the queue did not drain -- a task is looping")


def score_rows(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(scores_table).where(
                    scores_table.c.user_id == user_id,
                    scores_table.c.application_id == application_id,
                )
            ).all()
        )


def score_tasks(engine: Engine, user_id: uuid.UUID) -> list[Any]:
    with engine.begin() as conn:
        return list(
            conn.execute(
                select(tasks_table).where(
                    tasks_table.c.user_id == user_id, tasks_table.c.kind == SCORE_APPLICATION
                )
            ).all()
        )


def application_row(engine: Engine, application_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return conn.execute(
            select(applications_table).where(applications_table.c.id == application_id)
        ).one()


# -- the paste path -------------------------------------------------------------


def test_a_successful_read_queues_exactly_one_score_and_it_runs(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_key(engine, user, master_key)
    application_id = paste_application(engine, user)
    calls = install_fake_client(monkeypatch, [EXTRACTED, COVERAGE, SCORE])

    run_worker(engine, user, master_key)

    assert application_row(engine, application_id).extraction_status == "done"
    (row,) = score_rows(engine, user, application_id)
    assert row.status == "done"
    assert row.could_get_score == 4
    (task,) = score_tasks(engine, user)
    assert task.status == "succeeded"
    assert task.payload == {"score_id": str(row.id)}
    assert len(calls) == 3  # read, coverage, score -- and nothing else


def test_a_redelivered_read_queues_no_second_score(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At-least-once delivery: the same read, delivered again, finds the ad
    already `done` and returns before calling anything -- so it cannot queue a
    second score either. The fake is removed for the second run; the root
    conftest guard would raise if a client were constructed.
    """
    store_key(engine, user, master_key)
    application_id = paste_application(engine, user)
    install_fake_client(monkeypatch, [EXTRACTED, COVERAGE, SCORE])
    run_worker(engine, user, master_key)
    assert len(score_rows(engine, user, application_id)) == 1

    with engine.begin() as conn:
        PostgresTaskRepository(conn, user).enqueue(
            kind=EXTRACT_JOB_AD, payload={"application_id": str(application_id)}
        )
    monkeypatch.undo()
    run_worker(engine, user, master_key)

    assert len(score_rows(engine, user, application_id)) == 1
    assert len(score_tasks(engine, user)) == 1


def test_no_api_key_queues_no_score_and_the_read_says_why(
    engine: Engine, user: uuid.UUID, master_key: MasterKey
) -> None:
    """No fake installed: nothing may reach for the API."""
    application_id = paste_application(engine, user)

    run_worker(engine, user, master_key)

    row = application_row(engine, application_id)
    assert row.extraction_status == "failed"
    assert row.extraction_error_code == "no_api_key"
    assert score_rows(engine, user, application_id) == []
    assert score_tasks(engine, user) == []


def test_a_re_read_does_not_queue_a_second_score(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain is the *first* score. Once an application has any scoring run,
    a re-read of its ad (a button) does not score it again on its own --
    that is what Re-score is for, and it is the user's money.
    """
    store_key(engine, user, master_key)
    application_id = paste_application(engine, user)
    install_fake_client(monkeypatch, [EXTRACTED, COVERAGE, SCORE])
    run_worker(engine, user, master_key)

    with engine.begin() as conn:
        assert PostgresApplicationRepository(conn, user).request_extraction(application_id)
        PostgresTaskRepository(conn, user).enqueue(
            kind=EXTRACT_JOB_AD, payload={"application_id": str(application_id)}
        )
    calls = install_fake_client(monkeypatch, [EXTRACTED])
    run_worker(engine, user, master_key)

    assert len(calls) == 1  # the re-read, and no score
    assert len(score_rows(engine, user, application_id)) == 1
    assert len(score_tasks(engine, user)) == 1


# -- the tracked path -----------------------------------------------------------


def _add_board_job(engine: Engine, owner: uuid.UUID) -> Any:
    with engine.begin() as conn:
        boards = PostgresBoardRepository(conn, owner)
        board = boards.add_board(
            platform="greenhouse",
            board_url="https://boards.greenhouse.io/acme",
            board_key={"token": "acme"},
            label="Acme",
        )
        observed = ObservedJob(
            external_id="1",
            title="Senior Platform Engineer",
            location="London",
            url="https://job-boards.greenhouse.io/acme/jobs/1",
            fingerprint=fingerprint("Senior Platform Engineer", "London"),
        )
        result = FetchResult(status="complete", jobs=(observed,), expected_total=1)
        state = boards.lock_check_state(
            board.id, observed_external_ids=["1"], closed_since=DAY0 - REPOST_WINDOW
        )
        assert state is not None
        plan: CheckPlan = plan_check(state, result, observed_at=DAY0)
        boards.apply_check_plan(plan, started_at=DAY0 - dt.timedelta(minutes=1), finished_at=DAY0)
        (job,) = boards.list_jobs(board.id)
    return job


def test_track_as_application_chains_fetch_then_read_then_score(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_key(engine, user, master_key)
    job = _add_board_job(engine, user)
    with engine.begin() as conn:
        application = PostgresApplicationRepository(conn, user).create_application(
            title=job.title,
            source="Watched board",
            extraction_status="pending",
            board_job_id=job.id,
        )
        PostgresTaskRepository(conn, user).enqueue(
            kind=FETCH_JOB_DESCRIPTION, payload={"application_id": str(application.id)}
        )
    monkeypatch.setattr(
        description_module,
        "fetch_description",
        lambda *args, **kwargs: DescriptionResult(text=AD, error_code=None, requests=1),
    )
    install_fake_client(monkeypatch, [EXTRACTED, COVERAGE, SCORE])

    run_worker(engine, user, master_key)

    assert application_row(engine, application.id).extraction_status == "done"
    (row,) = score_rows(engine, user, application.id)
    assert row.status == "done"
    with engine.begin() as conn:
        latest = PostgresScoreRepository(conn, user).latest(application.id)
    assert latest is not None and latest.id == row.id
