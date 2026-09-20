"""Integration tests for slice 2a's schema and repository: jobs,
job_requirements, requirement_coverage, gap_questions. Needs `docker compose
up -d` and `alembic upgrade head`. No Anthropic API calls -- these test the
storage layer the two model calls write through, not the calls themselves.

Follows the transaction-rollback fixture pattern from test_schema_integration.py.
"""

from __future__ import annotations

import datetime
import os
import uuid
from pathlib import Path

import pytest
from jfl_core.context import RequestContext
from jfl_core.db.tables import requirement_coverage as requirement_coverage_table
from jfl_core.db.tables import users
from jfl_core.ids import (
    adjudicated_span_id_from_answer,
    content_hash,
    gap_question_id,
    job_id,
    requirement_id,
)
from jfl_core.ingest.gap_answers import gap_answer_span_id
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.models import GapQuestion, Job, JobRequirement, RequirementCoverage, Span
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresIngestRepository,
    PostgresJobRepository,
)
from jfl_generate.jobs import answer_question
from sqlalchemy import create_engine, insert
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


def _requirements(user_id: uuid.UUID, job: Job, texts: list[str]) -> list[JobRequirement]:
    return [
        JobRequirement(
            id=requirement_id(job.id, text),
            user_id=user_id,
            job_id=job.id,
            ordinal=i,
            text=text,
            necessity="essential",
        )
        for i, text in enumerate(texts)
    ]


# --- round-trip each new table --------------------------------------------------


def test_job_round_trips_through_upsert_and_get(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    created = job_repo.upsert_job(job)
    assert created is True

    found = job_repo.get_job(user, job.id)
    assert found is not None
    stored_job, requirements = found
    assert stored_job.id == job.id
    assert stored_job.employer == "Acme"
    assert stored_job.raw_text == job.raw_text
    assert requirements == []


def test_re_pasting_the_same_ad_is_idempotent(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    first = job_repo.upsert_job(job)
    updated = job.model_copy(update={"employer": "Acme (renamed)"})
    second = job_repo.upsert_job(updated)

    assert first is True
    assert second is False  # same deterministic id -> update, not a new row

    found = job_repo.get_job(user, job.id)
    assert found is not None
    assert found[0].employer == "Acme (renamed)"


def test_job_requirements_round_trip_and_preserve_ordinal_order(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python", "Kubernetes", "SQL"])
    job_repo.replace_requirements(user, job.id, requirements)

    found = job_repo.get_job(user, job.id)
    assert found is not None
    _, stored = found
    assert [r.text for r in stored] == ["Python", "Kubernetes", "SQL"]
    assert [r.ordinal for r in stored] == [0, 1, 2]


def test_replace_requirements_swaps_rather_than_accumulates(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    job_repo.replace_requirements(user, job.id, _requirements(user, job, ["Python"]))
    job_repo.replace_requirements(user, job.id, _requirements(user, job, ["Python", "Go"]))

    found = job_repo.get_job(user, job.id)
    assert found is not None
    _, stored = found
    assert [r.text for r in stored] == ["Python", "Go"]


def test_list_jobs_reports_requirement_count_and_summary_fields(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    job_repo.replace_requirements(user, job.id, _requirements(user, job, ["Python", "Go"]))

    summaries = job_repo.list_jobs(user)
    assert len(summaries) == 1
    assert summaries[0].id == job.id
    assert summaries[0].employer == "Acme"
    assert summaries[0].requirement_count == 2


def test_requirement_coverage_round_trips(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python"])
    job_repo.replace_requirements(user, job.id, requirements)

    coverage = RequirementCoverage(
        user_id=user,
        requirement_id=requirements[0].id,
        trace_id=uuid.uuid4(),
        status="evidenced",
        cited_span_ids=[uuid.uuid4(), uuid.uuid4()],
        evidence_note="Corpus documents 5 years of Python.",
    )
    job_repo.record_coverage(coverage)

    latest = job_repo.latest_coverage(user, job.id)
    assert len(latest) == 1
    assert latest[0].status == "evidenced"
    assert latest[0].evidence_note == coverage.evidence_note
    assert set(latest[0].cited_span_ids) == set(coverage.cited_span_ids)


def test_gap_question_round_trips(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    question = GapQuestion(
        id=gap_question_id(requirements[0].id),
        user_id=user,
        requirement_id=requirements[0].id,
        question="Have you operated a Kubernetes cluster in production?",
    )
    job_repo.upsert_gap_question(question)

    fetched = job_repo.get_question(user, question.id)
    assert fetched is not None
    assert fetched.status == "open"
    assert fetched.question == question.question


# --- DISTINCT ON latest-coverage query -------------------------------------------


def test_latest_coverage_returns_the_most_recent_row_per_requirement(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    """The demonstration the whole slice is built for: re-running coverage must
    show a status change, and `latest_coverage` must report the newer one.

    The older row is inserted with an explicit backdated `created_at` rather
    than relying on wall-clock progress between two `record_coverage` calls:
    within one transaction, Postgres's `now()` is the transaction start time
    and does not advance between statements, so two rows written back-to-back
    in the same test transaction would otherwise tie.
    """
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python"])
    job_repo.replace_requirements(user, job.id, requirements)

    older_id = uuid.uuid4()
    conn.execute(
        insert(requirement_coverage_table).values(
            id=older_id,
            user_id=user,
            requirement_id=requirements[0].id,
            trace_id=uuid.uuid4(),
            status="absent",
            cited_span_ids=[],
            evidence_note="No evidence yet.",
            created_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        )
    )
    newer = RequirementCoverage(
        user_id=user,
        requirement_id=requirements[0].id,
        trace_id=uuid.uuid4(),
        status="evidenced",
        cited_span_ids=[uuid.uuid4()],
        evidence_note="Answered gap question now documents this.",
    )
    job_repo.record_coverage(newer)

    latest = job_repo.latest_coverage(user, job.id)
    assert len(latest) == 1  # one row per requirement, not one per history entry
    assert latest[0].id == newer.id
    assert latest[0].status == "evidenced"


def test_latest_coverage_returns_one_row_per_requirement_not_per_job(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python", "Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    for requirement in requirements:
        job_repo.record_coverage(
            RequirementCoverage(
                user_id=user,
                requirement_id=requirement.id,
                trace_id=uuid.uuid4(),
                status="absent",
                cited_span_ids=[],
                evidence_note="No evidence.",
            )
        )

    latest = job_repo.latest_coverage(user, job.id)
    assert {row.requirement_id for row in latest} == {r.id for r in requirements}


# --- append-only coverage history ------------------------------------------------


def test_coverage_history_is_append_only(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    """Two runs against the same requirement must leave both rows in place --
    the before/after comparison is the point of the slice.
    """
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python"])
    job_repo.replace_requirements(user, job.id, requirements)

    first = RequirementCoverage(
        user_id=user,
        requirement_id=requirements[0].id,
        trace_id=uuid.uuid4(),
        status="absent",
        cited_span_ids=[],
        evidence_note="run 1",
    )
    second = RequirementCoverage(
        user_id=user,
        requirement_id=requirements[0].id,
        trace_id=uuid.uuid4(),
        status="evidenced",
        cited_span_ids=[],
        evidence_note="run 2",
    )
    job_repo.record_coverage(first)
    job_repo.record_coverage(second)

    rows = conn.execute(
        requirement_coverage_table.select().where(
            requirement_coverage_table.c.requirement_id == requirements[0].id
        )
    ).all()
    assert len(rows) == 2  # both runs survive; the second did not overwrite the first
    assert {row.id for row in rows} == {first.id, second.id}


# --- gap-question upsert never clobbers an answered row --------------------------


def test_gap_question_upsert_refreshes_text_while_open(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python"])
    job_repo.replace_requirements(user, job.id, requirements)

    qid = gap_question_id(requirements[0].id)
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid, user_id=user, requirement_id=requirements[0].id, question="First wording?"
        )
    )
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid, user_id=user, requirement_id=requirements[0].id, question="Second wording?"
        )
    )

    fetched = job_repo.get_question(user, qid)
    assert fetched is not None
    assert fetched.question == "Second wording?"
    assert fetched.id == qid  # still one stable row, not a new one


def test_gap_question_upsert_never_overwrites_an_answered_row(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    """Uses `add_adjudicated_span` directly to get an answered row on the
    board quickly -- that repository method is not what `jfl answer` uses any
    more (see "the answer path" section below), but it remains the write path
    for the unbuilt review-items flow, so exercising it here is still real
    coverage, not a stand-in for a path this test isn't actually testing.
    """
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Python"])
    job_repo.replace_requirements(user, job.id, requirements)

    grounding_repo = PostgresGroundingRepository(conn)
    qid = gap_question_id(requirements[0].id)
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid,
            user_id=user,
            requirement_id=requirements[0].id,
            question="Have you used Python in production?",
        )
    )

    answer_text = "Yes, I used Python for 5 years across two roles."
    span_id = adjudicated_span_id_from_answer(user, answer_text, qid)
    grounding_repo.add_adjudicated_span(
        user,
        Span(
            id=span_id,
            user_id=user,
            document_id=None,
            provenance="adjudicated",
            kind="paragraph",
            section_path="Answered questions",
            text=answer_text,
            content_hash=content_hash(answer_text),
        ),
    )
    job_repo.mark_question_answered(user, qid, answer_text, span_id)

    # A later coverage run tries to refresh the same question -- must be a no-op.
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid,
            user_id=user,
            requirement_id=requirements[0].id,
            question="This should never be written.",
        )
    )

    fetched = job_repo.get_question(user, qid)
    assert fetched is not None
    assert fetched.status == "answered"
    assert fetched.question == "Have you used Python in production?"
    assert fetched.answer_text == answer_text
    assert fetched.resulting_span_id == span_id

    # And it must have dropped out of the open-questions list.
    assert job_repo.list_open_questions(user, job.id) == []


# --- add_adjudicated_span repository mechanics (review-items flow, not `jfl answer`) ---


def test_answered_gap_question_span_is_visible_to_the_next_coverage_check(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    """Not a model call -- just the DB-level proof that `add_adjudicated_span`
    plus `mark_question_answered` wire together correctly: the new span shows
    up in `all_spans` (what the next coverage call reads).

    This is repository-mechanics coverage for the still-present
    `add_adjudicated_span` path, kept for the unbuilt review-items flow (see
    CLAUDE.md's decisions log, "A gap answer lands in corpus markdown, not the
    database"). `jfl answer` itself no longer calls this method -- see
    `test_answer_question_writes_to_markdown_and_ingests_a_document_span`
    below for that path.
    """
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    grounding_repo = PostgresGroundingRepository(conn)
    qid = gap_question_id(requirements[0].id)
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid,
            user_id=user,
            requirement_id=requirements[0].id,
            question="Have you operated Kubernetes in production?",
        )
    )

    before = grounding_repo.all_spans(user)
    assert before == []

    answer_text = "I ran a production Kubernetes cluster for two years at Acme."
    span_id = adjudicated_span_id_from_answer(user, answer_text, qid)
    grounding_repo.add_adjudicated_span(
        user,
        Span(
            id=span_id,
            user_id=user,
            document_id=None,
            provenance="adjudicated",
            kind="paragraph",
            section_path="Answered questions",
            text=answer_text,
            content_hash=content_hash(answer_text),
        ),
    )
    job_repo.mark_question_answered(user, qid, answer_text, span_id)

    after = grounding_repo.all_spans(user)
    assert len(after) == 1
    assert after[0].id == span_id
    assert after[0].text == answer_text
    assert after[0].provenance == "adjudicated"


# --- the answer path: jfl_generate.jobs.answer_question against live Postgres ----


def _ctx(user_id: uuid.UUID) -> RequestContext:
    return RequestContext(user_id=user_id, anthropic_api_key=None, database_url="unused-in-tests")


def test_answer_question_writes_to_markdown_and_ingests_a_document_span(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository, tmp_path: Path
) -> None:
    """The real `jfl answer` path, end to end against live Postgres: the
    question moves to `answered`, `resulting_span_id` points at a real
    `provenance='document'` span row (not `adjudicated`), and the fact lives
    in `corpus/answered-questions.md` where the author can read, edit, or
    delete it.
    """
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    qid = gap_question_id(requirements[0].id)
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid,
            user_id=user,
            requirement_id=requirements[0].id,
            question="Have you operated Kubernetes in production?",
        )
    )

    ingest_repo = PostgresIngestRepository(conn)
    grounding_repo = PostgresGroundingRepository(conn)
    answer_text = "I ran a production Kubernetes cluster for two years at Acme."

    span_id = answer_question(
        _ctx(user), ingest_repo, job_repo, qid, answer_text, corpus_dir=tmp_path
    )

    # The markdown file gained the fact, verbatim.
    content = (tmp_path / "answered-questions.md").read_text(encoding="utf-8")
    assert f"- {answer_text}" in content
    assert span_id == gap_answer_span_id(user, answer_text)

    # The question moved to answered, pointing at that span.
    fetched = job_repo.get_question(user, qid)
    assert fetched is not None
    assert fetched.status == "answered"
    assert fetched.resulting_span_id == span_id

    # A real span row exists, document-provenance, and it is the only one --
    # no adjudicated span was created by this path.
    spans = grounding_repo.all_spans(user)
    matching = [s for s in spans if s.id == span_id]
    assert len(matching) == 1
    assert matching[0].provenance == "document"
    assert matching[0].text == answer_text
    assert all(s.provenance != "adjudicated" for s in spans)


def test_answer_question_reingest_is_idempotent(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository, tmp_path: Path
) -> None:
    """Re-running `jfl ingest` after an answer must not duplicate the span --
    content-addressed ids make the second pass a pure no-op."""
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    qid = gap_question_id(requirements[0].id)
    job_repo.upsert_gap_question(
        GapQuestion(
            id=qid,
            user_id=user,
            requirement_id=requirements[0].id,
            question="Have you operated Kubernetes in production?",
        )
    )

    ingest_repo = PostgresIngestRepository(conn)
    grounding_repo = PostgresGroundingRepository(conn)
    answer_question(
        _ctx(user),
        ingest_repo,
        job_repo,
        qid,
        "I ran a production Kubernetes cluster for two years at Acme.",
        corpus_dir=tmp_path,
    )
    before = {s.id for s in grounding_repo.all_spans(user)}

    summary = run_ingestion(_ctx(user), ingest_repo, corpus_dir=tmp_path)

    after = {s.id for s in grounding_repo.all_spans(user)}
    assert summary.spans_created == 0
    assert after == before


# --- coverage_run_exists (B5's idempotency check for generate_coverage) --------


def test_coverage_run_exists_is_true_after_a_run_and_false_before(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    trace_id = uuid.uuid4()
    assert job_repo.coverage_run_exists(user, trace_id) is False

    job_repo.record_coverage(
        RequirementCoverage(
            user_id=user,
            requirement_id=requirements[0].id,
            trace_id=trace_id,
            status="evidenced",
            cited_span_ids=[],
            evidence_note="Traces cleanly.",
        )
    )
    assert job_repo.coverage_run_exists(user, trace_id) is True
    # A different trace -- a fresh "check again" -- is a different question.
    assert job_repo.coverage_run_exists(user, uuid.uuid4()) is False


def test_coverage_run_exists_is_scoped_to_the_given_user(
    conn: Connection, user: uuid.UUID, job_repo: PostgresJobRepository
) -> None:
    other_user = uuid.uuid4()
    conn.execute(insert(users).values(id=other_user, email=f"{other_user}@test.invalid"))

    job = _job(user)
    job_repo.upsert_job(job)
    requirements = _requirements(user, job, ["Kubernetes"])
    job_repo.replace_requirements(user, job.id, requirements)

    trace_id = uuid.uuid4()
    job_repo.record_coverage(
        RequirementCoverage(
            user_id=user,
            requirement_id=requirements[0].id,
            trace_id=trace_id,
            status="evidenced",
            cited_span_ids=[],
            evidence_note="Traces cleanly.",
        )
    )

    assert job_repo.coverage_run_exists(user, trace_id) is True
    assert job_repo.coverage_run_exists(other_user, trace_id) is False
