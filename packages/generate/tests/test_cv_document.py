"""Unit tests for `generate_cv_document`: the skeleton from the corpus, one model
call for the claims, one claim-gate pass mapped back onto lines. No live API, no
database -- a fake Anthropic client for the CV call, and `check_text`
monkeypatched on the module, the same pattern `test_draft.py` uses. Every
fixture is fictional.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx2
import jfl_generate.cv_document as cv_module
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.cv_document import CvLine
from jfl_core.ingest.parser import parse_document
from jfl_core.models import (
    Draft,
    Job,
    JobRequirement,
    RequirementCoverage,
    RunRecord,
    Span,
    SpanCandidate,
)
from jfl_core.profile import Capability
from jfl_gate.pricing import MODEL
from jfl_gate.schema import GateOutput, SentenceResult
from jfl_generate.cv_document import (
    apply_gate_output,
    clean_line,
    gate_text,
    generate_cv_document,
)
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import CV_DOCUMENT_OUTPUT_SCHEMA

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")

CORPUS = """\
# Morgan Fictional — Career Record

## Northwind Traders

### Head of Engineering, Nov 2021 – Present
- Led a platform team of eight engineers.

### Senior Engineer, Mar 2018 – Oct 2021
- Built the pricing service.

## Education
- BSc Computer Science, University of Nowhere, 2008

## Things stated as NOT true
- Morgan has never managed a budget.
"""


# --- fakes -------------------------------------------------------------------


class _FakeGroundingRepo:
    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None:
        raise NotImplementedError

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]:
        raise NotImplementedError

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        return self._spans

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID:
        raise NotImplementedError


class _FakeRunRepo:
    def __init__(self) -> None:
        self.recorded: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self.recorded.append(run)


class _FakeJobRepo:
    def __init__(
        self, job: Job, requirements: list[JobRequirement], coverage: list[RequirementCoverage]
    ) -> None:
        self._job, self._requirements, self._coverage = job, requirements, coverage

    def get_job(
        self, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> tuple[Job, list[JobRequirement]] | None:
        return (self._job, self._requirements) if job_id == self._job.id else None

    def latest_coverage(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[RequirementCoverage]:
        return self._coverage

    def record_draft(self, draft: Draft) -> None:  # pragma: no cover - never called
        raise AssertionError("a CV document is not a draft row")


class _FakeClient:
    def __init__(self, response: Message | None = None, exception: Exception | None = None):
        self.calls: list[dict[str, Any]] = []
        self._response, self._exception = response, exception
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        assert self._response is not None
        return self._response


def _message(
    payload: dict[str, Any],
    *,
    stop_reason: str = "end_turn",
    stop_details: RefusalStopDetails | None = None,
) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model=MODEL,
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=stop_details,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


PAYLOAD: dict[str, Any] = {
    "summary": ["Engineering leader who builds platform teams. Enjoys hard problems."],
    "skills": [{"label": "Platform", "text": "Built and ran a platform team."}],
    "roles": [
        {"index": 1, "descriptor": "", "bullets": ["- Led a platform team of eight engineers."]},
        {"index": 2, "descriptor": "", "bullets": ["Built the pricing service."]},
    ],
}


def _claim(verdict: str, note: str) -> SentenceResult:
    return SentenceResult(
        index=1,
        kind="claim",
        verdict=verdict,  # type: ignore[arg-type]
        drift_label="supported" if verdict == "supported" else "scope_inflation",
        cited_span_ids=[],
        evidence_note=note,
    )


def _framing(note: str = "Framing.") -> SentenceResult:
    return SentenceResult(
        index=1,
        kind="framing",
        verdict="supported",
        drift_label="framing",
        cited_span_ids=[],
        evidence_note=note,
    )


# summary (2 sentences), skill (1), bullet 1 (1), bullet 2 (1)
GATE_OUTPUT = GateOutput(
    sentences=[
        _claim("supported", "Summary claim traces."),
        _framing("Motivation."),
        _claim("review", "Team size not stated."),
        _claim("supported", "Traces."),
        _claim("unsupported", "No pricing service in the corpus."),
    ]
)


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any] | None = None,
    *,
    response: Message | None = None,
    exception: Exception | None = None,
    gate_output: GateOutput = GATE_OUTPUT,
) -> tuple[_FakeClient, list[str], Any, _FakeGroundingRepo, _FakeRunRepo, Job]:
    # `Any` for the job repo: it implements only the three methods this path
    # reads, not the whole `JobRepository` protocol.
    client = _FakeClient(
        response or (None if exception else _message(payload or PAYLOAD)), exception
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)
    gated: list[str] = []

    def _fake_check(ctx: RequestContext, grounding: object, runs: Any, text: str) -> GateOutput:
        gated.append(text)
        runs.record(
            RunRecord(
                user_id=ctx.user_id,
                trace_id=ctx.trace_id,
                component="gate",
                stage="baseline",
                model=ctx.gate_model,
                outcome="ok",
                started_at=datetime.now(UTC),
            )
        )
        return gate_output

    monkeypatch.setattr(cv_module, "check_text", _fake_check)
    job = Job(
        id=uuid.uuid4(),
        user_id=USER,
        source="paste",
        employer="Acme",
        title="Platform Lead",
        raw_text="Platform Lead at Acme.",
        content_hash="0" * 64,
    )
    requirement = JobRequirement(
        id=uuid.uuid4(),
        user_id=USER,
        job_id=job.id,
        ordinal=0,
        text="Led teams",
        necessity="essential",
    )
    coverage = RequirementCoverage(
        user_id=USER,
        requirement_id=requirement.id,
        trace_id=uuid.uuid4(),
        status="evidenced",
        cited_span_ids=[],
        evidence_note="Yes.",
    )
    spans = parse_document("file:corpus/record.md", CORPUS, USER).spans
    return (
        client,
        gated,
        _FakeJobRepo(job, [requirement], [coverage]),
        _FakeGroundingRepo(spans),
        _FakeRunRepo(),
        job,
    )


def _ctx() -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key="test-key", database_url="unused")


NOW = datetime(2026, 9, 23, 10, 30, tzinfo=UTC)


def test_the_skeleton_is_the_corpus_and_the_claims_are_the_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, jobs, grounding, runs, job = _setup(monkeypatch)
    result = generate_cv_document(_ctx(), jobs, grounding, runs, job.id, name="Morgan", now=NOW)
    document = result.document
    assert document.header.name == "Morgan"
    assert [(r.title, r.employer, r.dates) for r in document.roles] == [
        ("Head of Engineering", "Northwind Traders", "Nov 2021 – Present"),
        ("Senior Engineer", "Northwind Traders", "Mar 2018 – Oct 2021"),
    ]
    assert [b.text for b in document.roles[0].bullets] == [
        "Led a platform team of eight engineers."
    ]
    assert document.education == [
        CvLine(text="BSc Computer Science, University of Nowhere, 2008", origin="fact")
    ]


def test_a_response_that_writes_a_title_or_date_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        **PAYLOAD,
        "roles": [
            {
                "index": 1,
                "title": "Chief Technology Officer",
                "employer": "Megacorp",
                "dates": "2001 – Present",
                "descriptor": "",
                "bullets": ["Led a platform team of eight engineers."],
            }
        ],
    }
    _, _, jobs, grounding, runs, job = _setup(
        monkeypatch,
        payload,
        gate_output=GateOutput(
            sentences=[
                _claim("supported", "a"),
                _framing(),
                _claim("supported", "b"),
                _claim("supported", "c"),
            ]
        ),
    )
    document = generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW).document
    first = document.roles[0]
    assert (first.title, first.employer, first.dates) == (
        "Head of Engineering",
        "Northwind Traders",
        "Nov 2021 – Present",
    )
    assert "Megacorp" not in document.model_dump_json()


def test_an_out_of_range_role_index_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        **PAYLOAD,
        "roles": [
            {"index": 0, "descriptor": "", "bullets": ["Zero is not a role."]},
            {"index": 7, "descriptor": "", "bullets": ["Seven is not a role."]},
            {
                "index": 2,
                "descriptor": "A tea wholesaler.",
                "bullets": ["Built the pricing service."],
            },
        ],
    }
    gate = GateOutput(
        sentences=[
            _claim("supported", "a"),
            _framing(),
            _claim("supported", "b"),
            _claim("supported", "c"),
        ]
    )
    _, gated, jobs, grounding, runs, job = _setup(monkeypatch, payload, gate_output=gate)
    document = generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW).document
    assert document.roles[0].bullets == []
    assert document.roles[1].descriptor == "A tea wholesaler."
    assert "Seven is not a role." not in gated[0]
    assert "Zero is not a role." not in gated[0]


def test_the_gate_verdicts_land_on_the_right_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    _, gated, jobs, grounding, runs, job = _setup(monkeypatch)
    document = generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW).document

    # The summary's two sentences: a supported claim and framing -> supported.
    assert document.summary[0].verdict == "supported"
    assert document.summary[0].note == "Summary claim traces."
    assert document.skills[0].text.verdict == "review"
    assert document.skills[0].text.note == "Team size not stated."
    assert document.roles[0].bullets[0].verdict == "supported"
    # A flagged line stays in the document.
    assert document.roles[1].bullets[0].verdict == "unsupported"
    assert document.roles[1].bullets[0].text == "Built the pricing service."

    # One gate pass, over generated lines only: no fact line, no title, no date,
    # no descriptor, no boundary.
    assert len(gated) == 1
    assert "BSc Computer Science" not in gated[0]
    assert "Nov 2021" not in gated[0]
    assert "Head of Engineering" not in gated[0]
    assert "never managed a budget" not in gated[0]
    assert all(line.verdict is None for line in document.education)


def test_a_line_that_is_only_framing_stays_not_checked() -> None:
    lines = [CvLine(text="I love building teams."), CvLine(text="Led eight engineers.")]
    apply_gate_output(
        lines, GateOutput(sentences=[_framing("Motivation."), _claim("supported", "Traces.")])
    )
    assert lines[0].verdict == "framing"
    assert lines[1].verdict == "supported"


def test_a_line_takes_its_worst_sentence() -> None:
    lines = [CvLine(text="Led eight engineers. Doubled revenue.")]
    apply_gate_output(
        lines,
        GateOutput(sentences=[_claim("supported", "ok"), _claim("unsupported", "No number.")]),
    )
    assert (lines[0].verdict, lines[0].note) == ("unsupported", "No number.")


def test_a_misaligned_gate_output_is_an_error_not_a_guess() -> None:
    with pytest.raises(GenerateError, match="does not line up"):
        apply_gate_output(
            [CvLine(text="One. Two.")], GateOutput(sentences=[_claim("supported", "")])
        )


def test_generated_lines_never_carry_markup_into_the_gate() -> None:
    assert clean_line("  - Led\n the team. ") == "Led the team."
    assert clean_line("# A title-looking line") == "A title-looking line"
    assert gate_text([CvLine(text="A."), CvLine(text="B.")]) == "A.\n\nB."


def test_the_prompt_carries_the_clock_the_ceiling_and_the_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, jobs, grounding, runs, job = _setup(monkeypatch)
    capabilities = [
        Capability(label="Kubernetes", tier="working"),
        Capability(label="Budgeting", tier="absent"),
    ]
    generate_cv_document(_ctx(), jobs, grounding, runs, job.id, capabilities=capabilities, now=NOW)
    (call,) = client.calls
    message = call["messages"][0]["content"]
    assert "2026-09-23T10:30:00+00:00" in message
    assert "Kubernetes: " in message
    assert "Budgeting: this person says they do NOT have this" in message
    assert "Do not write a claim above the depth listed here." in message
    assert "Morgan has never managed a budget." in message
    assert "Role 1: Head of Engineering -- Northwind Traders (Nov 2021 – Present)" in message
    # The clock is volatile, so it is kept out of the cached system prefix.
    assert "2026-09-23" not in json.dumps(call["system"])
    assert call["output_config"]["effort"] == "high"
    assert call["model"] == "claude-opus-5-5"


def test_the_schema_names_nothing_reason_and_gives_roles_no_title_or_date() -> None:
    dumped = json.dumps(CV_DOCUMENT_OUTPUT_SCHEMA)
    assert '"reason"' not in dumped
    role = CV_DOCUMENT_OUTPUT_SCHEMA["properties"]["roles"]["items"]["properties"]  # type: ignore[index]
    assert set(role) == {"index", "descriptor", "bullets"}


def test_both_calls_write_one_runs_row_each_under_one_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, jobs, grounding, runs, job = _setup(monkeypatch)
    ctx = _ctx()
    result = generate_cv_document(ctx, jobs, grounding, runs, job.id, now=NOW)
    assert [(r.component, r.stage, r.outcome) for r in runs.recorded] == [
        ("generate", "cv_document", "ok"),
        ("gate", "baseline", "ok"),
    ]
    assert {r.trace_id for r in runs.recorded} == {ctx.trace_id} == {result.trace_id}
    assert runs.recorded[0].cost_usd is not None


def test_a_refusal_writes_one_row_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    refused = _message(
        PAYLOAD,
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="cyber", explanation=None),
    )
    _, gated, jobs, grounding, runs, job = _setup(monkeypatch, response=refused)
    with pytest.raises(GenerateError, match="model refused to respond: cyber"):
        generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW)
    assert [(r.outcome, r.error) for r in runs.recorded] == [("refused", "refusal: cyber")]
    assert gated == []


def test_an_api_error_writes_one_row_and_raises_with_the_classifiable_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError(
        "bad key", response=httpx2.Response(401, request=request), body=None
    )
    _, _, jobs, grounding, runs, job = _setup(monkeypatch, exception=error)
    with pytest.raises(GenerateError, match="^authentication_error"):
        generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW)
    assert len(runs.recorded) == 1 and runs.recorded[0].outcome == "error"


def test_missing_coverage_refuses_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, jobs, grounding, runs, job = _setup(monkeypatch)
    jobs._coverage = []
    with pytest.raises(GenerateError, match="^no coverage recorded for job"):
        generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW)
    assert client.calls == [] and runs.recorded == []


def test_nothing_generated_means_no_gate_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _, gated, jobs, grounding, runs, job = _setup(
        monkeypatch, {"summary": [], "skills": [], "roles": []}
    )
    result = generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW)
    assert gated == []
    assert len(result.document.roles) == 2
    assert [r.stage for r in runs.recorded] == ["cv_document"]


def test_the_header_falls_back_to_the_corpus_title(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, jobs, grounding, runs, job = _setup(monkeypatch)
    document = generate_cv_document(_ctx(), jobs, grounding, runs, job.id, now=NOW).document
    assert document.header.name == "Morgan Fictional"
    assert document.header.contact == [] and document.header.links == []


def test_the_cv_speaks_in_the_first_person() -> None:
    """The owner's CVs speak as him: "I" in the summary, verb-led bullets with the
    "I" implied. The corpus is written about him in the third person, and a model
    left to itself mirrors that -- so the instruction says whose voice it is, and
    never frames the task as describing "the candidate" from outside."""
    from jfl_generate.prompts import _CV_DOCUMENT_INSTRUCTIONS

    text = _CV_DOCUMENT_INSTRUCTIONS
    assert "first person" in text
    assert "Never refer to the candidate by name" in text
    assert "introducing the candidate" not in text
