"""Unit tests for `jfl_generate.answers`: the assessment call and the draft
call behind application questions (NEXT.md's task 4). Mirrors
`packages/generate/tests/test_titles.py`'s approach: `anthropic.Anthropic` is
monkeypatched to a fake client, and the run repository is a trivial in-memory
stand-in for the Protocol in `jfl_core.repositories`. No live API, no
database.

The gate pass itself (`jfl_gate.gate.check_text`) is not called from this
module at all -- see `jfl_generate.answers`'s docstring: gating a checked or
drafted answer is the caller's job (the worker handler), the same split
`jfl_generate.draft.generate_draft` does not use, but which keeps this file
from having to fake the Anthropic client twice to drive two calls through one.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import anthropic
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.models import Job, JobRequirement, RunRecord, Span, SpanCandidate
from jfl_gate.pricing import compute_cost_usd
from jfl_generate.answers import assess_answer, draft_application_answer
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    ASSESS_ANSWER_OUTPUT_SCHEMA,
    DRAFT_ANSWER_OUTPUT_SCHEMA,
    build_assess_answer_prompt,
    build_draft_answer_system_blocks,
)

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


# --- fakes -------------------------------------------------------------------


class _FakeRunRepo:
    def __init__(self) -> None:
        self.recorded: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self.recorded.append(run)


class _FakeGroundingRepo:
    def __init__(self, spans: list[Span] | None = None) -> None:
        self._spans = spans or []
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


def _ctx(api_key: str | None = "test-key", model: str = "claude-opus-5") -> RequestContext:
    return RequestContext(
        user_id=USER, anthropic_api_key=api_key, database_url="unused", model=model
    )


def _job() -> Job:
    return Job(
        id=uuid.uuid4(),
        user_id=USER,
        source="paste",
        employer="Acme",
        title="Senior Engineer",
        raw_text="ad text",
        content_hash="0" * 64,
    )


def _requirement(text: str, necessity: str = "essential") -> JobRequirement:
    return JobRequirement(
        id=uuid.uuid4(),
        user_id=USER,
        job_id=uuid.uuid4(),
        ordinal=0,
        text=text,
        necessity=necessity,  # type: ignore[arg-type]
    )


def _response(
    payload: dict[str, Any],
    *,
    model: str = "claude-opus-5",
    stop_reason: str = "end_turn",
    stop_details: RefusalStopDetails | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model=model,
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


_ASSESSMENT_PAYLOAD = {"assessment": "Covers the requirements well.", "gaps": "No metric given."}
_DRAFT_PAYLOAD = {"draft": "I want this role because of its focus on reliability."}


# --- schema --------------------------------------------------------------------


def _property_names(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                found.update(value.keys())
            found.update(_property_names(value))
    elif isinstance(node, list):
        for item in node:
            found.update(_property_names(item))
    return found


def test_assess_schema_has_no_property_named_reason() -> None:
    """CLAUDE.md's 2026-09-02 decision: a schema property named `reason`,
    combined with a labelling system prompt, has tripped the API's
    reverse-engineering/duplication classifier before.
    """
    assert "reason" not in _property_names(ASSESS_ANSWER_OUTPUT_SCHEMA)


def test_assess_schema_requires_assessment_and_gaps_only() -> None:
    assert ASSESS_ANSWER_OUTPUT_SCHEMA["required"] == ["assessment", "gaps"]
    assert ASSESS_ANSWER_OUTPUT_SCHEMA["additionalProperties"] is False


def test_draft_answer_schema_has_no_property_named_reason() -> None:
    assert "reason" not in _property_names(DRAFT_ANSWER_OUTPUT_SCHEMA)


# --- prompt builders -------------------------------------------------------------


class TestAssessAnswerPrompt:
    def test_carries_the_question_text_verbatim(self) -> None:
        prompt = build_assess_answer_prompt(
            job=_job(),
            requirements=[_requirement("Five years of Python")],
            question_text="Why do you want to work here?",
            answer_text="Because of the mission.",
            now=NOW,
        )
        assert "Why do you want to work here?" in prompt
        assert "Because of the mission." in prompt

    def test_carries_the_requirements(self) -> None:
        prompt = build_assess_answer_prompt(
            job=_job(),
            requirements=[_requirement("Five years of Python"), _requirement("Owns on-call")],
            question_text="Q",
            answer_text="A",
            now=NOW,
        )
        assert "Five years of Python" in prompt
        assert "Owns on-call" in prompt

    def test_carries_the_job(self) -> None:
        prompt = build_assess_answer_prompt(
            job=_job(), requirements=[], question_text="Q", answer_text="A", now=NOW
        )
        assert "Senior Engineer" in prompt
        assert "Acme" in prompt

    def test_no_requirements_reads_as_a_statement_not_a_blank(self) -> None:
        prompt = build_assess_answer_prompt(
            job=None, requirements=[], question_text="Q", answer_text="A", now=NOW
        )
        assert "no requirements extracted" in prompt
        assert "no job ad linked" in prompt

    def test_includes_the_timestamp(self) -> None:
        prompt = build_assess_answer_prompt(
            job=None, requirements=[], question_text="Q", answer_text="A", now=NOW
        )
        assert NOW.isoformat() in prompt


class TestDraftAnswerSystemBlocks:
    def test_carries_the_question_and_requirements_and_corpus(self) -> None:
        span = Span(
            id=uuid.uuid4(),
            user_id=USER,
            provenance="document",
            kind="bullet",
            text="Owned the reliability programme.",
            content_hash="0" * 64,
        )
        blocks = build_draft_answer_system_blocks(
            [span],
            job=_job(),
            requirements=[_requirement("Owns reliability")],
            question_text="Why this role?",
            now=NOW,
            cache="corpus",
        )
        whole = "".join(b["text"] for b in blocks)
        assert "Why this role?" in whole
        assert "Owns reliability" in whole
        assert "Owned the reliability programme." in whole


# --- assess_answer ---------------------------------------------------------------


def test_assess_answer_empty_text_raises_without_calling_the_api() -> None:
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="answer text"):
        assess_answer(
            _ctx(),
            runs,
            question_text="Q",
            answer_text="   ",
            job=None,
            requirements=[],
            now=NOW,
        )
    assert runs.recorded == []


def test_assess_answer_successful_call_parses_result_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(_ASSESSMENT_PAYLOAD, input_tokens=400, output_tokens=90)
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    ctx = _ctx()

    result = assess_answer(
        ctx,
        runs,
        question_text="Why this role?",
        answer_text="Because of the mission.",
        job=_job(),
        requirements=[_requirement("Five years of Python")],
        now=NOW,
    )

    assert result.assessment == "Covers the requirements well."
    assert result.gaps == "No metric given."

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "generate"
    assert run.stage == "assess_answer"
    assert run.model == "claude-opus-5"
    assert run.user_id == ctx.user_id
    assert run.trace_id == ctx.trace_id
    assert run.tokens_in == 400
    assert run.tokens_out == 90
    assert run.cost_usd == compute_cost_usd("claude-opus-5", 400, 90, 0, 0)
    assert run.error is None


def test_assess_answer_uses_ctx_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeAnthropicClient(response=_response(_ASSESSMENT_PAYLOAD, model="claude-sonnet-5"))
    _patch_client(monkeypatch, client)

    ctx = _ctx(model="claude-sonnet-5")
    runs = _FakeRunRepo()
    assess_answer(ctx, runs, question_text="Q", answer_text="A", job=None, requirements=[], now=NOW)

    assert client.messages.calls[0]["model"] == "claude-sonnet-5"
    assert runs.recorded[0].model == "claude-sonnet-5"


def test_assess_answer_refusal_raises_and_records_a_refused_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        _ASSESSMENT_PAYLOAD,
        stop_reason="refusal",
        stop_details=RefusalStopDetails(category="reasoning_extraction", type="refusal"),
    )
    _patch_client(monkeypatch, _FakeAnthropicClient(response=response))

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="refused"):
        assess_answer(
            _ctx(), runs, question_text="Q", answer_text="A", job=None, requirements=[], now=NOW
        )
    assert runs.recorded[0].outcome == "refused"


def test_assess_answer_unparseable_json_raises_and_records_an_error_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = Message(
        id="msg_test",
        content=[TextBlock(type="text", text="not json")],
        model="claude-opus-5",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    _patch_client(monkeypatch, _FakeAnthropicClient(response=bad))

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="parse"):
        assess_answer(
            _ctx(), runs, question_text="Q", answer_text="A", job=None, requirements=[], now=NOW
        )
    assert runs.recorded[0].outcome == "error"


# --- draft_application_answer -----------------------------------------------------


def test_draft_application_answer_reads_the_whole_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _FakeAnthropicClient(response=_response(_DRAFT_PAYLOAD)))

    grounding = _FakeGroundingRepo([])
    ctx = _ctx()
    draft_application_answer(
        ctx,
        grounding,
        _FakeRunRepo(),
        question_text="Why this role?",
        job=_job(),
        requirements=[_requirement("Owns reliability")],
        now=NOW,
    )
    assert grounding.all_spans_calls == [ctx.user_id]


def test_draft_application_answer_successful_call_returns_text_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(_DRAFT_PAYLOAD, input_tokens=600, output_tokens=120)
    _patch_client(monkeypatch, _FakeAnthropicClient(response=response))

    runs = _FakeRunRepo()
    ctx = _ctx()
    text = draft_application_answer(
        ctx,
        _FakeGroundingRepo([]),
        runs,
        question_text="Why this role?",
        job=_job(),
        requirements=[_requirement("Owns reliability")],
        now=NOW,
    )

    assert text == "I want this role because of its focus on reliability."
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "generate"
    assert run.stage == "draft_answer"
    assert run.tokens_in == 600
    assert run.tokens_out == 120


def test_draft_application_answer_empty_draft_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _FakeAnthropicClient(response=_response({"draft": "   "})))

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="empty"):
        draft_application_answer(
            _ctx(),
            _FakeGroundingRepo([]),
            runs,
            question_text="Q",
            job=_job(),
            requirements=[_requirement("X")],
            now=NOW,
        )
    # The model call itself still recorded an ok run -- the emptiness is
    # caught after parsing, same as `generate_draft`'s equivalent checks.
    assert runs.recorded[0].outcome == "ok"


def test_draft_application_answer_truncated_output_raises_and_records_an_error_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(_DRAFT_PAYLOAD, stop_reason="max_tokens")
    _patch_client(monkeypatch, _FakeAnthropicClient(response=response))

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="truncated"):
        draft_application_answer(
            _ctx(),
            _FakeGroundingRepo([]),
            runs,
            question_text="Q",
            job=_job(),
            requirements=[_requirement("X")],
            now=NOW,
        )
    assert runs.recorded[0].outcome == "error"
