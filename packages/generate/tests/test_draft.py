"""Unit tests for `generate_draft`: the fixed control flow around the draft
call plus the automatic claim-gate pass. No live API, no database.

The draft call's own model-call machinery (client construction, exception
ladder, refusal/parse-failure handling, one `runs` row per path) mirrors
`check_coverage`'s and is tested the same way test_coverage.py tests it: a
fake Anthropic client patched onto `anthropic.Anthropic`. The claim-gate pass
is `jfl_gate.gate.check_text`, imported directly into `jfl_generate.draft` --
rather than faking the Anthropic client a second time to drive both calls
through one fake, `check_text` is monkeypatched directly on the `draft`
module, same pattern test_jobs.py uses for `check_coverage` inside
`run_coverage`. `jfl_gate/tests/test_gate.py` already covers `check_text`'s
own machinery.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import anthropic
import httpx2
import jfl_generate.draft as draft_module
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.models import (
    Draft,
    GapQuestion,
    Job,
    JobRequirement,
    JobSummary,
    RequirementCoverage,
    RunRecord,
    Span,
    SpanCandidate,
)
from jfl_gate.gate import GateError
from jfl_gate.pricing import MODEL, compute_cost_usd
from jfl_gate.schema import GateOutput, SentenceResult
from jfl_generate.draft import generate_draft
from jfl_generate.errors import GenerateError

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


# --- fakes -------------------------------------------------------------------


class _FakeGroundingRepo:
    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans
        self.all_spans_calls: list[uuid.UUID] = []

    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None:
        raise NotImplementedError

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]:
        raise NotImplementedError

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        self.all_spans_calls.append(user_id)
        return self._spans

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID:
        raise NotImplementedError


class _FakeRunRepo:
    def __init__(self) -> None:
        self.recorded: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self.recorded.append(run)


class _FakeJobRepository:
    def __init__(
        self,
        job: Job | None = None,
        requirements: list[JobRequirement] | None = None,
        coverage: list[RequirementCoverage] | None = None,
    ) -> None:
        self._job = job
        self._requirements = requirements or []
        self._coverage = coverage or []
        self.drafts: list[Draft] = []

    def upsert_job(self, job: Job) -> bool:
        raise NotImplementedError

    def replace_requirements(
        self, user_id: uuid.UUID, job_id: uuid.UUID, requirements: list[JobRequirement]
    ) -> None:
        raise NotImplementedError

    def list_jobs(self, user_id: uuid.UUID) -> list[JobSummary]:
        raise NotImplementedError

    def get_job(
        self, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> tuple[Job, list[JobRequirement]] | None:
        if self._job is None or self._job.id != job_id:
            return None
        return self._job, self._requirements

    def record_coverage(self, coverage: RequirementCoverage) -> None:
        raise NotImplementedError

    def latest_coverage(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[RequirementCoverage]:
        return self._coverage

    def upsert_gap_question(self, question: GapQuestion) -> None:
        raise NotImplementedError

    def list_open_questions(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[GapQuestion]:
        raise NotImplementedError

    def get_question(self, user_id: uuid.UUID, question_id: uuid.UUID) -> GapQuestion | None:
        raise NotImplementedError

    def mark_question_answered(
        self,
        user_id: uuid.UUID,
        question_id: uuid.UUID,
        answer_text: str,
        resulting_span_id: uuid.UUID,
    ) -> None:
        raise NotImplementedError

    def record_draft(self, draft: Draft) -> None:
        self.drafts.append(draft)

    def list_drafts(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[Draft]:
        return [d for d in self.drafts if d.job_id == job_id]


class _FakeMessages:
    def __init__(self, response: Message | None = None, exception: Exception | None = None):
        self._response = response
        self._exception = exception
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        assert self._response is not None
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: Message | None = None, exception: Exception | None = None):
        self.messages = _FakeMessages(response, exception)


# --- helpers -------------------------------------------------------------------


def _span(text: str = "Led the platform team") -> Span:
    return Span(
        id=uuid.uuid4(),
        user_id=USER,
        document_id=uuid.uuid4(),
        provenance="document",
        kind="bullet",
        section_path="Northwind",
        ordinal=0,
        text=text,
        content_hash="0" * 64,
    )


def _job() -> Job:
    return Job(
        id=uuid.uuid4(),
        user_id=USER,
        source="paste",
        employer="Acme",
        title="Senior Engineer",
        location="Remote",
        raw_text="Senior Engineer at Acme. Remote. 5+ years of Python.",
        content_hash="0" * 64,
    )


def _requirement(job: Job, text: str = "5+ years of Python") -> JobRequirement:
    return JobRequirement(
        id=uuid.uuid4(), user_id=USER, job_id=job.id, ordinal=0, text=text, necessity="essential"
    )


def _coverage_row(requirement: JobRequirement) -> RequirementCoverage:
    return RequirementCoverage(
        user_id=USER,
        requirement_id=requirement.id,
        trace_id=uuid.uuid4(),
        status="evidenced",
        cited_span_ids=[],
        reason="Corpus documents this.",
    )


def _ctx(api_key: str | None = "test-key") -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key=api_key, database_url="unused")


def _draft_response(
    draft_text: str,
    *,
    stop_reason: str = "end_turn",
    stop_details: RefusalStopDetails | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps({"draft": draft_text}))],
        model=MODEL,
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=stop_details,
        type="message",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        ),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeAnthropicClient) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)


_SUPPORTED_GATE_OUTPUT = GateOutput(
    sentences=[
        SentenceResult(
            index=1,
            kind="claim",
            verdict="supported",
            drift_label="supported",
            cited_span_ids=[],
            evidence_note="Traces cleanly.",
        )
    ]
)


def _fake_check_text(
    monkeypatch: pytest.MonkeyPatch,
    output: GateOutput = _SUPPORTED_GATE_OUTPUT,
    *,
    record_a_run: bool = True,
) -> list[tuple[Any, ...]]:
    """Stands in for `jfl_gate.gate.check_text`. Records a `runs` row on the same
    run_repo passed to it, mirroring what the real function does, so tests can
    assert both calls share one trace_id.
    """
    calls: list[tuple[Any, ...]] = []

    def _fake(ctx: RequestContext, grounding_repo: object, run_repo: Any, text: str) -> GateOutput:
        calls.append((ctx, grounding_repo, run_repo, text))
        if record_a_run:
            run_repo.record(
                RunRecord(
                    user_id=ctx.user_id,
                    trace_id=ctx.trace_id,
                    component="gate",
                    stage="baseline",
                    model=MODEL,
                    outcome="ok",
                    started_at=datetime.now(),
                )
            )
        return output

    monkeypatch.setattr(draft_module, "check_text", _fake)
    return calls


# --- credential resolution -------------------------------------------------------


class TestCredentialResolution:
    @staticmethod
    def _record_construction(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        client = _FakeAnthropicClient(_draft_response("A drafted bullet."))

        def _construct(**kwargs: Any) -> _FakeAnthropicClient:
            calls.append(kwargs)
            return client

        monkeypatch.setattr(anthropic, "Anthropic", _construct)
        return calls

    def test_explicit_key_is_passed_to_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._record_construction(monkeypatch)
        _fake_check_text(monkeypatch)
        job = _job()
        requirement = _requirement(job)
        job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])

        generate_draft(
            _ctx(api_key="sk-ant-explicit"),
            job_repo,
            _FakeGroundingRepo([_span()]),
            _FakeRunRepo(),
            job.id,
            "cv_bullets",
        )
        assert calls == [{"api_key": "sk-ant-explicit"}]

    def test_absent_key_constructs_a_bare_client_for_profile_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_construction(monkeypatch)
        _fake_check_text(monkeypatch)
        job = _job()
        requirement = _requirement(job)
        job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])

        generate_draft(
            _ctx(api_key=None),
            job_repo,
            _FakeGroundingRepo([_span()]),
            _FakeRunRepo(),
            job.id,
            "cv_bullets",
        )
        assert calls == [{}]


# --- preconditions -----------------------------------------------------------


def test_missing_job_raises_without_calling_the_api() -> None:
    job_repo = _FakeJobRepository()
    with pytest.raises(GenerateError, match="no job"):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([]), _FakeRunRepo(), uuid.uuid4(), "cv_bullets"
        )


def test_job_with_no_requirements_raises_without_calling_the_api() -> None:
    job = _job()
    job_repo = _FakeJobRepository(job, [])
    with pytest.raises(GenerateError, match="requirements"):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([]), _FakeRunRepo(), job.id, "cv_bullets"
        )


def test_job_with_no_coverage_recorded_raises_a_clear_message_and_does_not_run_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No silent coverage run -- that would be a second, unbudgeted model call.
    Constructing an Anthropic client at all here is a bug.
    """

    def _boom(**kwargs: object) -> None:
        raise AssertionError("no model call should happen when coverage is missing")

    monkeypatch.setattr(anthropic, "Anthropic", _boom)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], coverage=[])

    with pytest.raises(GenerateError, match="jfl job coverage"):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([]), _FakeRunRepo(), job.id, "cv_bullets"
        )


# --- successful path -----------------------------------------------------------


def test_successful_draft_calls_the_claim_gate_and_persists_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _draft_response(
        "Led the platform team at Acme.",
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=500,
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)
    gate_calls = _fake_check_text(monkeypatch)

    job = _job()
    requirement = _requirement(job)
    span = _span()
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    run_repo = _FakeRunRepo()
    ctx = _ctx()

    result = generate_draft(
        ctx, job_repo, _FakeGroundingRepo([span]), run_repo, job.id, "cv_bullets"
    )

    # The claim gate was called on exactly the drafted text.
    assert len(gate_calls) == 1
    assert gate_calls[0][3] == "Led the platform team at Acme."

    # Both the draft's own `runs` row and the gate's share one trace_id -- what
    # makes a draft's total cost a single query.
    assert len(run_repo.recorded) == 2
    draft_run, gate_run = run_repo.recorded
    assert draft_run.trace_id == gate_run.trace_id == ctx.trace_id
    assert draft_run.component == "generate"
    assert draft_run.stage == "draft"
    assert draft_run.outcome == "ok"
    assert draft_run.tokens_in == 1000
    assert draft_run.tokens_out == 200
    assert draft_run.cache_read_tokens == 500
    assert draft_run.cost_usd == compute_cost_usd(MODEL, 1000, 200, 500, 0)
    assert isinstance(draft_run.started_at, datetime)

    # Both are persisted on the job repository.
    assert result.text == "Led the platform team at Acme."
    assert result.kind == "cv_bullets"
    assert result.job_id == job.id
    assert result.trace_id == ctx.trace_id
    assert result.gate_result == _SUPPORTED_GATE_OUTPUT.model_dump(mode="json")
    assert job_repo.drafts == [result]


def test_grounding_input_is_the_corpus_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """`all_spans` is the only grounding read -- nothing here should reach for
    sent_documents/sent_spans (not even reachable through this Protocol; see
    CLAUDE.md's architectural constraints).
    """
    client = _FakeAnthropicClient(response=_draft_response("A drafted bullet."))
    _patch_client(monkeypatch, client)
    _fake_check_text(monkeypatch)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    grounding_repo = _FakeGroundingRepo([_span()])
    ctx = _ctx()

    generate_draft(ctx, job_repo, grounding_repo, _FakeRunRepo(), job.id, "cv_bullets")

    assert grounding_repo.all_spans_calls == [ctx.user_id]


def test_request_caches_the_corpus_and_keeps_job_and_requirements_volatile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAnthropicClient(response=_draft_response("A drafted bullet."))
    _patch_client(monkeypatch, client)
    _fake_check_text(monkeypatch)

    span = _span("A distinctive corpus fact about the platform team.")
    job = _job()
    requirement = _requirement(job, "A distinctive requirement about rockets.")
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])

    generate_draft(
        _ctx(), job_repo, _FakeGroundingRepo([span]), _FakeRunRepo(), job.id, "cv_bullets"
    )

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]

    assert kwargs["model"] == MODEL
    system_blocks = kwargs["system"]
    # Two blocks -- instructions, then corpus -- with the cache breakpoint on the
    # corpus block (cache="corpus"), so the cached prefix is instructions+corpus,
    # exactly what the pre-split single-string prompt cached.
    assert len(system_blocks) == 2
    assert "cache_control" not in system_blocks[0]
    assert system_blocks[1]["cache_control"] == {"type": "ephemeral"}
    combined_system_text = "".join(b["text"] for b in system_blocks)
    assert str(span.id) in combined_system_text
    assert span.text in combined_system_text

    # Job/requirements are volatile -- belongs in `messages`, not the cached
    # `system` block, or every distinct job would bust the cache.
    assert requirement.text not in combined_system_text
    user_content = kwargs["messages"][0]["content"]
    assert requirement.text in user_content

    assert kwargs["output_config"]["format"]["type"] == "json_schema"


def test_a_flagged_draft_is_still_returned_not_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim gate informs, it never blocks -- see CLAUDE.md, 'How the claim
    gate behaves.' A flagged verdict must not stop the draft from being
    persisted or returned.
    """
    client = _FakeAnthropicClient(response=_draft_response("Owned the FX pricing platform."))
    _patch_client(monkeypatch, client)

    flagged_output = GateOutput(
        sentences=[
            SentenceResult(
                index=1,
                kind="claim",
                verdict="unsupported",
                drift_label="invented_quantity",
                cited_span_ids=[],
                evidence_note="No such number in the corpus.",
            )
        ]
    )
    _fake_check_text(monkeypatch, output=flagged_output)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])

    result = generate_draft(
        _ctx(), job_repo, _FakeGroundingRepo([_span()]), _FakeRunRepo(), job.id, "cv_bullets"
    )

    assert result.text == "Owned the FX pricing platform."
    assert result.gate_result == flagged_output.model_dump(mode="json")
    assert job_repo.drafts == [result]


# --- draft call failure paths -----------------------------------------------------


@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda: anthropic.RateLimitError(
            "rate limited",
            response=httpx2.Response(
                429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            body=None,
        ),
        lambda: anthropic.BadRequestError(
            "bad request",
            response=httpx2.Response(
                400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            ),
            body=None,
        ),
        lambda: anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
    ],
)
def test_api_errors_record_an_error_run_and_raise_generate_error(
    monkeypatch: pytest.MonkeyPatch, exception_factory: Any
) -> None:
    client = _FakeAnthropicClient(exception=exception_factory())
    _patch_client(monkeypatch, client)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    run_repo = _FakeRunRepo()

    with pytest.raises(GenerateError):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([_span()]), run_repo, job.id, "cv_bullets"
        )

    assert len(run_repo.recorded) == 1
    run = run_repo.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert run.tokens_in is None
    assert job_repo.drafts == []  # nothing persisted on a failed draft call


def test_refusal_records_a_refused_run_and_raises_generate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _draft_response(
        "", stop_reason="refusal", stop_details=RefusalStopDetails(type="refusal", category="cyber")
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    run_repo = _FakeRunRepo()

    with pytest.raises(GenerateError, match="refus"):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([_span()]), run_repo, job.id, "cv_bullets"
        )

    assert len(run_repo.recorded) == 1
    assert run_repo.recorded[0].outcome == "refused"
    assert job_repo.drafts == []


def test_max_tokens_truncation_records_an_error_run_and_names_the_real_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated response would otherwise surface as a JSONDecodeError -- see
    jfl_gate.gate's max_tokens check, which this mirrors: the check must run
    before any attempt to parse the (truncated, likely invalid) response body,
    so the error names the real cause instead of a misleading parse failure.
    """
    response = _draft_response("", stop_reason="max_tokens")
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    run_repo = _FakeRunRepo()

    with pytest.raises(GenerateError, match="max_tokens"):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([_span()]), run_repo, job.id, "cv_bullets"
        )

    assert len(run_repo.recorded) == 1
    run = run_repo.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert "max_tokens" in run.error
    assert job_repo.drafts == []


def test_malformed_json_records_an_error_run_and_raises_generate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Message(
        id="msg_test",
        content=[TextBlock(type="text", text="not valid json{{{")],
        model=MODEL,
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    run_repo = _FakeRunRepo()

    with pytest.raises(GenerateError):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([_span()]), run_repo, job.id, "cv_bullets"
        )

    assert run_repo.recorded[0].outcome == "error"
    assert job_repo.drafts == []


# --- claim-gate failure path -------------------------------------------------------


def test_gate_error_propagates_after_the_draft_calls_own_run_is_already_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GateError here is a real API failure, not a flagged claim -- the draft
    call's own `runs` row has already committed by the time it is raised, and
    nothing is persisted to `drafts`.
    """
    client = _FakeAnthropicClient(response=_draft_response("A drafted bullet."))
    _patch_client(monkeypatch, client)

    def _raise(*args: object, **kwargs: object) -> GateOutput:
        raise GateError("model refused to respond: cyber")

    monkeypatch.setattr(draft_module, "check_text", _raise)

    job = _job()
    requirement = _requirement(job)
    job_repo = _FakeJobRepository(job, [requirement], [_coverage_row(requirement)])
    run_repo = _FakeRunRepo()

    with pytest.raises(GateError):
        generate_draft(
            _ctx(), job_repo, _FakeGroundingRepo([_span()]), run_repo, job.id, "cv_bullets"
        )

    assert len(run_repo.recorded) == 1  # the draft call's own row, recorded before the gate ran
    assert run_repo.recorded[0].outcome == "ok"
    assert job_repo.drafts == []
