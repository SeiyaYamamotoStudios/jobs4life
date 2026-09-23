"""The "Write the CV" button, end to end against real Postgres: a `generate_cv_draft` task
with `kind: cv_document` run in the real worker loop, and a complete CV stored
as a new `cv_documents` version. Needs `docker compose up -d` and `alembic
upgrade head`.

**No test here spends a penny.** `anthropic.Anthropic` is a fake for the CV
call and `jfl_generate.cv_document.check_text` is monkeypatched for the claim
gate's pass -- the same technique as
`tests/test_draft_generation_worker_integration.py`, which this mirrors. In the
tests that never reach a model call, the root `conftest.py` guard is what would
raise if the handler reached the API anyway.

The corpus is written through the real hosted write path
(`jfl_core.corpus_source.append_confirmed_fact`), so the skeleton here is read
from the same spans a CV-onboarded user has. Every fixture is fictional.
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
import jfl_generate.cv_document as cv_module
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.corpus_source import append_confirmed_fact
from jfl_core.crypto.envelope import MasterKey, seal
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import spans as spans_table
from jfl_core.db.tables import users as users_table
from jfl_core.ids import requirement_id
from jfl_core.models import JobRequirement, RequirementCoverage, RunRecord
from jfl_core.profile import CvHeaderLink, CvHeaderSettings, Profile
from jfl_core.storage.applications import PostgresApplicationRepository
from jfl_core.storage.credentials import ANTHROPIC_API_KEY, PostgresCredentialRepository
from jfl_core.storage.cv_documents import PostgresCvDocumentRepository
from jfl_core.storage.postgres import PostgresJobRepository
from jfl_core.storage.profile import PostgresProfileRepository, save_profile
from jfl_core.storage.tasks import PostgresTaskRepository
from jfl_core.storage.user_corpus import PostgresUserCorpusRepository
from jfl_gate.schema import GateOutput, SentenceResult
from jfl_intake.http import Transport
from jfl_worker.handlers import GENERATE_CV_DRAFT, build_registry
from jfl_worker.handlers.draft_generation import build_generate_cv_draft
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

ROLE = "Northwind Traders -- Head of Engineering, Nov 2021 - Present"

CV_PAYLOAD: dict[str, Any] = {
    "summary": ["Engineering leader for platform teams."],
    "skills": [{"label": "Platform", "text": "Runs platform teams."}],
    "roles": [
        {
            "index": 1,
            "title": "Chief Executive",  # ignored: the model never writes a title
            "descriptor": "",
            "bullets": ["Led a platform team of eight engineers."],
        }
    ],
}

GATE_OUTPUT = GateOutput(
    sentences=[
        SentenceResult(
            index=i,
            kind="claim",
            verdict=verdict,  # type: ignore[arg-type]
            drift_label="supported" if verdict == "supported" else "scope_inflation",
            cited_span_ids=[],
            evidence_note=note,
        )
        for i, (verdict, note) in enumerate(
            [("supported", "a"), ("review", "b"), ("unsupported", "Team size not stated.")],
            start=1,
        )
    ]
)


@contextmanager
def _never_fetch_a_board() -> Iterator[Transport]:
    raise AssertionError("a CV-handler test tried to fetch a job board")
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
        conn.execute(
            insert(users_table).values(
                id=uid, email=f"{uid}@test.invalid", display_name="Morgan Fictional"
            )
        )
    try:
        yield uid
    finally:
        with engine.begin() as conn:
            conn.execute(delete(users_table).where(users_table.c.id == uid))


@pytest.fixture
def master_key() -> MasterKey:
    return MasterKey.generate()


class _FakeClient:
    def __init__(self, response: Message | None, exception: Exception | None) -> None:
        self._response, self._exception = response, exception
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        assert self._response is not None
        return self._response


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    exception: Exception | None = None,
) -> _FakeClient:
    response = None
    if payload is not None:
        response = Message(
            id="msg_test",
            content=[TextBlock(type="text", text=json.dumps(payload))],
            model="claude-opus-5-5",
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
    client = _FakeClient(response, exception)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    return client


def install_fake_gate(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    gated: list[str] = []

    def _fake(ctx: RequestContext, grounding: object, run_repo: Any, text: str) -> GateOutput:
        gated.append(text)
        run_repo.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="gate",
                stage="baseline",
                model=ctx.gate_model,
                outcome="ok",
                started_at=dt.datetime.now(dt.UTC),
            )
        )
        return GATE_OUTPUT

    monkeypatch.setattr(cv_module, "check_text", _fake)
    return gated


def store_key(engine: Engine, user_id: uuid.UUID, master_key: MasterKey) -> None:
    with engine.begin() as conn:
        PostgresCredentialRepository(conn, user_id).store(
            provider=ANTHROPIC_API_KEY,
            sealed=seal(master_key, FAKE_KEY, user_id=user_id, provider=ANTHROPIC_API_KEY),
            key_hint=FAKE_KEY[-4:],
        )


def add_application(engine: Engine, user_id: uuid.UUID, *, with_coverage: bool = True) -> uuid.UUID:
    """An application with a job, one requirement, recorded coverage, and a
    small hosted corpus: one role with one confirmed fact, and an education line.
    """
    with engine.begin() as conn:
        append_confirmed_fact(
            conn, user_id, "Led a platform team of eight engineers.", section=ROLE
        )
        append_confirmed_fact(conn, user_id, "BSc Physics, 2008", section="Education")
        application = PostgresApplicationRepository(conn, user_id).create_application(
            title="Platform Lead", raw_job_text=f"Platform Lead at Acme. {uuid.uuid4()}"
        )
    job_id = application.job_id
    assert job_id is not None
    with engine.begin() as conn:
        jobs = PostgresJobRepository(conn)
        text = "Leads platform teams"
        jobs.replace_requirements(
            user_id,
            job_id,
            [
                JobRequirement(
                    id=requirement_id(job_id, text),
                    user_id=user_id,
                    job_id=job_id,
                    ordinal=0,
                    text=text,
                    necessity="essential",
                )
            ],
        )
        if with_coverage:
            jobs.record_coverage(
                RequirementCoverage(
                    user_id=user_id,
                    requirement_id=requirement_id(job_id, text),
                    trace_id=uuid.uuid4(),
                    status="evidenced",
                    cited_span_ids=[],
                    evidence_note="Traces.",
                )
            )
    return application.id


def enqueue(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> uuid.UUID:
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user_id).enqueue(
            kind=GENERATE_CV_DRAFT,
            payload={"application_id": str(application_id), "kind": "cv_document"},
        )
    return task.id


def run_worker(
    engine: Engine,
    user_id: uuid.UUID,
    master_key: MasterKey,
    *,
    env: dict[str, str] | None = None,
) -> None:
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
        logger=configure_logging(stream=io.StringIO()),
    )
    for _ in range(10):
        if worker.run_once() == 0:
            return
    raise AssertionError("the queue did not drain -- a task is looping")


def get_task(engine: Engine, user_id: uuid.UUID, task_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresTaskRepository(conn, user_id).get_task(task_id)


def latest(engine: Engine, user_id: uuid.UUID, application_id: uuid.UUID) -> Any:
    with engine.begin() as conn:
        return PostgresCvDocumentRepository(conn, user_id).latest(application_id)


def test_the_handler_stores_a_complete_gated_cv(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_key(engine, user, master_key)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)
    client = install_fake_client(monkeypatch, payload=CV_PAYLOAD)
    gated = install_fake_gate(monkeypatch)

    run_worker(engine, user, master_key)

    task = get_task(engine, user, task_id)
    assert task.status == "succeeded"
    version = latest(engine, user, application_id)
    assert version is not None and version.status == "generated"
    assert version.trace_id == task_id
    document = version.document
    assert document.header.name == "Morgan Fictional"  # the account's display name
    (role,) = document.roles
    assert (role.employer, role.title, role.dates) == (
        "Northwind Traders",
        "Head of Engineering",
        "Nov 2021 - Present",
    )
    assert role.bullets[0].verdict == "unsupported"  # flagged, and still there
    assert [line.text for line in document.education] == ["BSc Physics, 2008"]
    assert "BSc Physics" not in gated[0]
    assert len(client.calls) == 1

    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.trace_id == task_id)).all()
    assert sorted(r.stage for r in runs) == ["baseline", "cv_document"]
    assert {r.model for r in runs} == {"claude-opus-5-5", "claude-opus-5"}


def test_the_header_and_interests_come_from_the_profile_and_reach_no_model(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name, tagline, contact, links and interests are profile settings: they
    land on the generated CV and are never sent to the model, the claim gate
    or the corpus."""
    store_key(engine, user, master_key)
    application_id = add_application(engine, user)
    profile = Profile(
        cv_header=CvHeaderSettings(
            name="Morgan Q. Fictional",
            tagline="Head of Engineering | Logistics",
            phone="07700 900123",
            email="morgan@fictional.invalid.example",
            location="Leeds, UK",
            links=[CvHeaderLink(label="github.com/morganq", url="https://github.com/morganq")],
        ),
        interests=["Orienteering", "Bell ringing"],
    )
    with engine.begin() as conn:
        save_profile(
            PostgresProfileRepository(conn, user), PostgresUserCorpusRepository(conn, user), profile
        )
    enqueue(engine, user, application_id)
    client = install_fake_client(monkeypatch, payload=CV_PAYLOAD)
    gated = install_fake_gate(monkeypatch)

    run_worker(engine, user, master_key)

    version = latest(engine, user, application_id)
    assert version is not None
    header = version.document.header
    assert header.name == "Morgan Q. Fictional"  # the profile's, over the account's
    assert header.tagline == "Head of Engineering | Logistics"
    assert header.contact == ["07700 900123", "morgan@fictional.invalid.example", "Leeds, UK"]
    assert [(link.label, link.url) for link in header.links] == [
        ("github.com/morganq", "https://github.com/morganq")
    ]
    assert version.document.interests == ["Orienteering", "Bell ringing"]

    sent_to_model = json.dumps(client.calls, default=str)
    settings = ["07700 900123", "morgan@fictional", "Leeds, UK", "morganq", "Orienteering"]
    for value in settings + ["Morgan Q. Fictional"]:
        assert value not in sent_to_model, value
        assert all(value not in text for text in gated), value
    with engine.begin() as conn:
        span_texts = (
            conn.execute(select(spans_table.c.text).where(spans_table.c.user_id == user))
            .scalars()
            .all()
        )
    assert not any(value in text for value in settings for text in span_texts)


def test_a_redelivered_task_does_not_write_a_second_version(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_key(engine, user, master_key)
    application_id = add_application(engine, user)
    with engine.begin() as conn:
        task = PostgresTaskRepository(conn, user).enqueue(
            kind=GENERATE_CV_DRAFT,
            payload={"application_id": str(application_id), "kind": "cv_document"},
        )
    ctx = TaskContext(task=task, engine=engine, now=dt.datetime.now(dt.UTC))
    handler = build_generate_cv_draft(master_key=master_key, model="claude-opus-5-5")

    install_fake_client(monkeypatch, payload=CV_PAYLOAD)
    install_fake_gate(monkeypatch)
    first = handler(ctx)
    assert first is not None and "skipped" not in first

    monkeypatch.undo()  # back to the API guard: a second model call would raise
    second = handler(ctx)
    assert second is not None and second["cv_document_id"] == first["cv_document_id"]
    assert "skipped" in second

    with engine.begin() as conn:
        versions = PostgresCvDocumentRepository(conn, user).list_versions(application_id)
    assert len(versions) == 1


def test_no_api_key_fails_once_and_permanently(
    engine: Engine, user: uuid.UUID, master_key: MasterKey
) -> None:
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key)

    task = get_task(engine, user, task_id)
    assert (task.status, task.attempts) == ("failed", 1)
    assert "draft generation failed permanently: no_api_key" in (task.last_error or "")
    assert latest(engine, user, application_id) is None


def test_missing_coverage_fails_permanently_before_any_call(
    engine: Engine, user: uuid.UUID, master_key: MasterKey
) -> None:
    store_key(engine, user, master_key)
    application_id = add_application(engine, user, with_coverage=False)
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key)

    task = get_task(engine, user, task_id)
    assert task.status == "failed"
    assert "draft generation failed permanently: no_coverage" in (task.last_error or "")


def test_a_rejected_key_fails_permanently_with_one_runs_row(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_key(engine, user, master_key)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
        ),
    )

    run_worker(engine, user, master_key)

    task = get_task(engine, user, task_id)
    assert task.status == "failed"
    assert "draft generation failed permanently: api_key_rejected" in (task.last_error or "")
    assert FAKE_KEY not in (task.last_error or "")
    with engine.begin() as conn:
        runs = conn.execute(select(runs_table).where(runs_table.c.trace_id == task_id)).all()
    assert [(r.stage, r.outcome) for r in runs] == [("cv_document", "error")]
    assert all(FAKE_KEY not in (r.error or "") for r in runs)


def test_a_transient_failure_is_retried(
    engine: Engine, user: uuid.UUID, master_key: MasterKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_key(engine, user, master_key)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    install_fake_client(
        monkeypatch,
        exception=anthropic.APIStatusError(
            "overloaded", response=httpx2.Response(529, request=request), body=None
        ),
    )

    run_worker(engine, user, master_key)

    task = get_task(engine, user, task_id)
    assert task.status == "pending"
    assert task.scheduled_at > dt.datetime.now(dt.UTC)
    assert latest(engine, user, application_id) is None


def test_the_kill_switch_leaves_the_task_pending_and_calls_nothing(
    engine: Engine, user: uuid.UUID, master_key: MasterKey
) -> None:
    store_key(engine, user, master_key)
    application_id = add_application(engine, user)
    task_id = enqueue(engine, user, application_id)

    run_worker(engine, user, master_key, env={"JFL_DISABLE_MODEL_CALLS": "1"})

    task = get_task(engine, user, task_id)
    assert (task.status, task.attempts) == ("pending", 0)
    assert latest(engine, user, application_id) is None
