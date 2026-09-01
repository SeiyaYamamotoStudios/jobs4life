"""Unit tests for the orchestration in jfl_generate.jobs: `add_job`,
`run_coverage`, `answer_question`. No live API, no database -- the two model
calls (`extract_requirements`, `check_coverage`) are monkeypatched directly
rather than faking the Anthropic client a second time; that plumbing is
already covered by test_extract.py and test_coverage.py. Repositories are
trivial in-memory stand-ins for the Protocols in jfl_core.repositories.
"""

from __future__ import annotations

import uuid

import anthropic
import jfl_generate.jobs as jobs_module
import pytest
from jfl_core.context import RequestContext
from jfl_core.ids import adjudicated_span_id_from_answer, content_hash, gap_question_id
from jfl_core.models import (
    GapQuestion,
    Job,
    JobRequirement,
    JobSummary,
    RequirementCoverage,
    Span,
    SpanCandidate,
)
from jfl_generate.errors import GenerateError
from jfl_generate.schema import (
    CoverageOutput,
    ExtractedRequirement,
    ExtractOutput,
    RequirementCoverageResult,
)

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


# --- fakes -------------------------------------------------------------------


class _FakeGroundingRepo:
    def __init__(self, spans: list[Span] | None = None) -> None:
        self._spans = list(spans or [])
        self.added: list[Span] = []

    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None:
        return next((s for s in self._spans if s.id == span_id), None)

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]:
        raise NotImplementedError

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        return self._spans

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID:
        self.added.append(span)
        self._spans.append(span)
        return span.id


class _FakeRunRepo:
    def __init__(self) -> None:
        self.recorded: list[object] = []

    def record(self, run: object) -> None:
        self.recorded.append(run)


class _FakeJobRepository:
    """In-memory stand-in for `JobRepository` -- enough behaviour to exercise
    `jobs.py`'s orchestration without a database.
    """

    def __init__(self) -> None:
        self.jobs: dict[uuid.UUID, Job] = {}
        self.requirements: dict[uuid.UUID, list[JobRequirement]] = {}
        self.coverage: list[RequirementCoverage] = []
        self.questions: dict[uuid.UUID, GapQuestion] = {}

    def upsert_job(self, job: Job) -> bool:
        created = job.id not in self.jobs
        self.jobs[job.id] = job
        return created

    def replace_requirements(
        self, user_id: uuid.UUID, job_id: uuid.UUID, requirements: list[JobRequirement]
    ) -> None:
        self.requirements[job_id] = list(requirements)

    def list_jobs(self, user_id: uuid.UUID) -> list[JobSummary]:
        raise NotImplementedError

    def get_job(
        self, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> tuple[Job, list[JobRequirement]] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return job, self.requirements.get(job_id, [])

    def record_coverage(self, coverage: RequirementCoverage) -> None:
        self.coverage.append(coverage)

    def latest_coverage(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[RequirementCoverage]:
        raise NotImplementedError

    def upsert_gap_question(self, question: GapQuestion) -> None:
        existing = self.questions.get(question.id)
        if existing is None:
            self.questions[question.id] = question
            return
        if existing.status == "open":
            self.questions[question.id] = existing.model_copy(
                update={"question": question.question}
            )

    def list_open_questions(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[GapQuestion]:
        req_ids = {r.id for r in self.requirements.get(job_id, [])}
        return [
            q for q in self.questions.values() if q.status == "open" and q.requirement_id in req_ids
        ]

    def get_question(self, user_id: uuid.UUID, question_id: uuid.UUID) -> GapQuestion | None:
        return self.questions.get(question_id)

    def mark_question_answered(
        self,
        user_id: uuid.UUID,
        question_id: uuid.UUID,
        answer_text: str,
        resulting_span_id: uuid.UUID,
    ) -> None:
        existing = self.questions[question_id]
        self.questions[question_id] = existing.model_copy(
            update={
                "status": "answered",
                "answer_text": answer_text,
                "resulting_span_id": resulting_span_id,
            }
        )


def _ctx() -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key="test-key", database_url="unused")


def _fail_if_model_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Used by the answer-path tests: no model call is allowed anywhere in that
    path, so constructing a client at all is a bug.
    """

    def _boom(**kwargs: object) -> None:
        raise AssertionError("no model call should happen in the answer path")

    monkeypatch.setattr(anthropic, "Anthropic", _boom)


# --- add_job -------------------------------------------------------------------


def test_add_job_stores_job_and_requirements_with_deterministic_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = ExtractOutput(
        employer="Acme Corp",
        title="Senior Engineer",
        location="Remote",
        requirements=[
            ExtractedRequirement(text="5+ years of Python", necessity="essential"),
            ExtractedRequirement(text="Kubernetes experience", necessity="desirable"),
        ],
    )
    monkeypatch.setattr(
        jobs_module, "extract_requirements", lambda ctx, run_repo, raw_text: extracted
    )

    job_repo = _FakeJobRepository()
    run_repo = _FakeRunRepo()
    ctx = _ctx()
    raw_text = "Senior Engineer at Acme Corp. Remote. 5+ years of Python. Kubernetes a plus."

    job, requirements = jobs_module.add_job(ctx, run_repo, job_repo, raw_text)

    assert job.employer == "Acme Corp"
    assert job.content_hash == content_hash(raw_text)
    assert job.id in job_repo.jobs
    assert len(requirements) == 2
    assert [r.necessity for r in requirements] == ["essential", "desirable"]
    assert [r.ordinal for r in requirements] == [0, 1]
    assert job_repo.requirements[job.id] == requirements


def test_re_adding_the_same_ad_text_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = ExtractOutput(employer="", title="", location="", requirements=[])
    monkeypatch.setattr(
        jobs_module, "extract_requirements", lambda ctx, run_repo, raw_text: extracted
    )

    job_repo = _FakeJobRepository()
    run_repo = _FakeRunRepo()
    ctx = _ctx()
    raw_text = "Same ad, pasted twice."

    job_a, _ = jobs_module.add_job(ctx, run_repo, job_repo, raw_text)
    job_b, _ = jobs_module.add_job(ctx, run_repo, job_repo, raw_text)

    assert job_a.id == job_b.id
    assert len(job_repo.jobs) == 1


# --- run_coverage ----------------------------------------------------------------


def _seeded_job(
    job_repo: _FakeJobRepository, requirement_texts: list[str]
) -> tuple[Job, list[JobRequirement]]:
    job = Job(
        id=uuid.uuid4(),
        user_id=USER,
        source="paste",
        raw_text="ad text",
        content_hash="0" * 64,
    )
    job_repo.jobs[job.id] = job
    requirements = [
        JobRequirement(
            id=uuid.uuid4(),
            user_id=USER,
            job_id=job.id,
            ordinal=i,
            text=text,
            necessity="essential",
        )
        for i, text in enumerate(requirement_texts)
    ]
    job_repo.requirements[job.id] = requirements
    return job, requirements


def test_run_coverage_records_one_row_per_requirement_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_repo = _FakeJobRepository()
    job, requirements = _seeded_job(job_repo, ["Python", "Kubernetes"])

    output = CoverageOutput(
        results=[
            RequirementCoverageResult(
                status="evidenced", cited_span_ids=[], reason="documented", question=None
            ),
            RequirementCoverageResult(
                status="absent",
                cited_span_ids=[],
                reason="not mentioned",
                question="Have you used K8s?",
            ),
        ]
    )
    seen_requirements: list[list[str]] = []

    def _fake_check_coverage(
        ctx: object, grounding_repo: object, run_repo: object, texts: list[str]
    ) -> CoverageOutput:
        seen_requirements.append(list(texts))
        return output

    monkeypatch.setattr(jobs_module, "check_coverage", _fake_check_coverage)

    ctx = _ctx()
    rows = jobs_module.run_coverage(ctx, _FakeGroundingRepo(), _FakeRunRepo(), job_repo, job.id)

    assert seen_requirements == [["Python", "Kubernetes"]]
    assert len(rows) == 2
    assert rows[0].requirement_id == requirements[0].id
    assert rows[0].status == "evidenced"
    assert rows[1].requirement_id == requirements[1].id
    assert rows[1].status == "absent"
    assert all(r.trace_id == ctx.trace_id for r in rows)
    assert job_repo.coverage == rows


def test_run_coverage_creates_a_gap_question_only_for_absent_and_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_repo = _FakeJobRepository()
    job, requirements = _seeded_job(job_repo, ["A", "B", "C", "D"])

    output = CoverageOutput(
        results=[
            RequirementCoverageResult(
                status="evidenced", cited_span_ids=[], reason="r", question=None
            ),
            RequirementCoverageResult(
                status="partial", cited_span_ids=[], reason="r", question="Tell me about B?"
            ),
            RequirementCoverageResult(
                status="absent", cited_span_ids=[], reason="r", question="Tell me about C?"
            ),
            RequirementCoverageResult(
                status="contradicted", cited_span_ids=[], reason="r", question=None
            ),
        ]
    )
    monkeypatch.setattr(jobs_module, "check_coverage", lambda *a, **k: output)

    jobs_module.run_coverage(_ctx(), _FakeGroundingRepo(), _FakeRunRepo(), job_repo, job.id)

    open_ids = {q.requirement_id for q in job_repo.questions.values()}
    assert open_ids == {requirements[1].id, requirements[2].id}
    for q in job_repo.questions.values():
        assert q.id == gap_question_id(q.requirement_id)
        assert q.status == "open"


def test_run_coverage_raises_for_a_job_that_does_not_exist() -> None:
    with pytest.raises(GenerateError):
        jobs_module.run_coverage(
            _ctx(), _FakeGroundingRepo(), _FakeRunRepo(), _FakeJobRepository(), uuid.uuid4()
        )


def test_run_coverage_raises_for_a_job_with_no_requirements() -> None:
    job_repo = _FakeJobRepository()
    job, _ = _seeded_job(job_repo, [])
    with pytest.raises(GenerateError, match="no requirements"):
        jobs_module.run_coverage(_ctx(), _FakeGroundingRepo(), _FakeRunRepo(), job_repo, job.id)


# --- answer_question -------------------------------------------------------------


def test_answer_question_writes_a_verbatim_adjudicated_span_with_no_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_model_called(monkeypatch)

    job_repo = _FakeJobRepository()
    requirement_id = uuid.uuid4()
    question = GapQuestion(
        id=gap_question_id(requirement_id),
        user_id=USER,
        requirement_id=requirement_id,
        question="Have you led a migration?",
    )
    job_repo.questions[question.id] = question

    grounding = _FakeGroundingRepo()
    answer_text = "I led the Q3 database migration and was on-call for the cutover."

    span_id = jobs_module.answer_question(_ctx(), grounding, job_repo, question.id, answer_text)

    assert span_id == adjudicated_span_id_from_answer(USER, answer_text, question.id)
    assert len(grounding.added) == 1
    span = grounding.added[0]
    assert span.text == answer_text  # stored word-for-word, no rewriting
    assert span.provenance == "adjudicated"
    assert span.kind == "paragraph"
    assert span.section_path == "Answered questions"
    assert span.document_id is None

    stored = job_repo.questions[question.id]
    assert stored.status == "answered"
    assert stored.answer_text == answer_text
    assert stored.resulting_span_id == span_id


def test_answer_question_raises_for_a_question_that_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_model_called(monkeypatch)
    with pytest.raises(GenerateError):
        jobs_module.answer_question(
            _ctx(), _FakeGroundingRepo(), _FakeJobRepository(), uuid.uuid4(), "answer"
        )


def test_answer_question_raises_for_an_already_answered_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_model_called(monkeypatch)

    job_repo = _FakeJobRepository()
    requirement_id = uuid.uuid4()
    question = GapQuestion(
        id=gap_question_id(requirement_id),
        user_id=USER,
        requirement_id=requirement_id,
        question="Have you led a migration?",
        status="answered",
        answer_text="Already answered.",
    )
    job_repo.questions[question.id] = question

    with pytest.raises(GenerateError, match="already"):
        jobs_module.answer_question(
            _ctx(), _FakeGroundingRepo(), job_repo, question.id, "a new answer"
        )
