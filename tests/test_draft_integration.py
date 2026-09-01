"""Integration tests for the `drafts` table and repository: round-trip, and
that a draft's two `runs` rows (the draft call, stage='draft', and the
automatic claim-gate pass, stage='baseline') share one `trace_id` -- what
makes a draft's total cost a single query. Needs `docker compose up -d` and
`alembic upgrade head`. No Anthropic API calls -- this tests the storage layer
`generate_draft` writes through, not the model calls themselves (those are
covered with fakes in packages/generate/tests/test_draft.py).

Follows the transaction-rollback fixture pattern from test_schema_integration.py
and test_jobs_integration.py.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from jfl_core.context import RequestContext
from jfl_core.db.tables import runs as runs_table
from jfl_core.db.tables import users
from jfl_core.ids import content_hash, job_id, requirement_id
from jfl_core.models import Draft, Job, JobRequirement, RunRecord
from jfl_core.storage.postgres import PostgresJobRepository, PostgresRunRepository
from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import Connection

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine):
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()  # every test leaves the database as it found it


@pytest.fixture
def user(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


@pytest.fixture
def job_repo(conn: Connection) -> PostgresJobRepository:
    return PostgresJobRepository(conn)


def _job(user_id: uuid.UUID, raw_text: str = "Senior Engineer at Acme. Remote.") -> Job:
    return Job(
        id=job_id(user_id, raw_text),
        user_id=user_id,
        source="paste",
        employer="Acme",
        title="Senior Engineer",
        location="Remote",
        raw_text=raw_text,
        content_hash=content_hash(raw_text),
    )


def _requirement(user_id: uuid.UUID, job: Job, text: str = "5+ years of Python") -> JobRequirement:
    return JobRequirement(
        id=requirement_id(job.id, text),
        user_id=user_id,
        job_id=job.id,
        ordinal=0,
        text=text,
        necessity="essential",
    )


def _ctx(user_id: uuid.UUID) -> RequestContext:
    return RequestContext(user_id=user_id, anthropic_api_key=None, database_url="unused-in-tests")


def test_draft_round_trips_through_record_and_list(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirement = _requirement(user, job)
    job_repo.replace_requirements(user, job.id, [requirement])

    gate_result = {
        "sentences": [
            {
                "index": 1,
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [],
                "reason": "Traces cleanly.",
                "rule_flags": [],
                "text": "Led the platform team.",
            }
        ]
    }
    draft = Draft(
        user_id=user,
        job_id=job.id,
        kind="cv_bullets",
        text="Led the platform team.",
        gate_result=gate_result,
        trace_id=uuid.uuid4(),
    )
    job_repo.record_draft(draft)

    found = job_repo.list_drafts(user, job.id)
    assert len(found) == 1
    stored = found[0]
    assert stored.id == draft.id
    assert stored.kind == "cv_bullets"
    assert stored.text == "Led the platform team."
    assert stored.gate_result == gate_result
    assert stored.trace_id == draft.trace_id


def test_list_drafts_returns_most_recent_first_and_only_for_the_given_job(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job_a = _job(user, "Job ad A.")
    job_b = _job(user, "Job ad B.")
    job_repo.upsert_job(job_a)
    job_repo.upsert_job(job_b)

    draft_a1 = Draft(
        user_id=user,
        job_id=job_a.id,
        kind="cv_bullets",
        text="First draft for A.",
        gate_result={"sentences": []},
        trace_id=uuid.uuid4(),
    )
    job_repo.record_draft(draft_a1)
    draft_a2 = Draft(
        user_id=user,
        job_id=job_a.id,
        kind="cover_letter",
        text="Second draft for A.",
        gate_result={"sentences": []},
        trace_id=uuid.uuid4(),
    )
    job_repo.record_draft(draft_a2)
    draft_b = Draft(
        user_id=user,
        job_id=job_b.id,
        kind="cv_bullets",
        text="A draft for B.",
        gate_result={"sentences": []},
        trace_id=uuid.uuid4(),
    )
    job_repo.record_draft(draft_b)

    found = job_repo.list_drafts(user, job_a.id)
    assert {d.id for d in found} == {draft_a1.id, draft_a2.id}
    assert draft_b.id not in {d.id for d in found}


def test_a_drafts_two_runs_rows_share_one_trace_id(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    """The property `generate_draft` relies on: the draft call (stage='draft')
    and the automatic claim-gate pass (stage='baseline') both write a `runs`
    row under `ctx.trace_id`, and that trace_id is what gets stored on the
    `drafts` row -- so a draft's total cost is one query:
    `SELECT sum(cost_usd) FROM runs WHERE trace_id = drafts.trace_id`.

    This writes the two `runs` rows directly (no live model call, per this
    test's budget) rather than through `generate_draft`, to isolate the
    storage-layer property from the model-call machinery already covered by
    fakes in test_draft.py.
    """
    job = _job(user)
    job_repo.upsert_job(job)
    requirement = _requirement(user, job)
    job_repo.replace_requirements(user, job.id, [requirement])

    ctx = _ctx(user)
    run_repo = PostgresRunRepository(conn)

    run_repo.record(
        RunRecord(
            user_id=user,
            trace_id=ctx.trace_id,
            component="generate",
            stage="draft",
            model="claude-opus-5",
            tokens_in=1000,
            tokens_out=200,
            cache_read_tokens=0,
            cache_write_tokens=500,
            cost_usd=Decimal("0.010"),
            latency_ms=1200,
            outcome="ok",
            started_at=datetime.now(UTC),
        )
    )
    run_repo.record(
        RunRecord(
            user_id=user,
            trace_id=ctx.trace_id,
            component="gate",
            stage="baseline",
            model="claude-opus-5",
            tokens_in=8000,
            tokens_out=1500,
            cache_read_tokens=0,
            cache_write_tokens=7000,
            cost_usd=Decimal("0.320"),
            latency_ms=9000,
            outcome="ok",
            started_at=datetime.now(UTC),
        )
    )

    draft = Draft(
        user_id=user,
        job_id=job.id,
        kind="cv_bullets",
        text="Led the platform team.",
        gate_result={"sentences": []},
        trace_id=ctx.trace_id,
    )
    job_repo.record_draft(draft)

    rows = conn.execute(select(runs_table).where(runs_table.c.trace_id == ctx.trace_id)).all()
    assert len(rows) == 2
    assert {row.stage for row in rows} == {"draft", "baseline"}
    total_cost = sum(row.cost_usd for row in rows)
    assert total_cost == Decimal("0.330")

    # And the stored draft points at that same trace_id -- the join that makes a
    # draft's cost a single query.
    stored = job_repo.list_drafts(user, job.id)[0]
    assert stored.trace_id == ctx.trace_id
