"""Unit tests for `check_text`: the fixed control flow around the one model call.

No live API, no database -- `anthropic.Anthropic` is monkeypatched to a fake client,
and the grounding/run repositories are trivial in-memory stand-ins for the Protocols
in jfl_core.repositories.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.models import RunRecord, Span, SpanCandidate
from jfl_gate.gate import GateError, check_text
from jfl_gate.pricing import MODEL, compute_cost_usd

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
        section_path="Kaluza",
        ordinal=0,
        text=text,
        content_hash="0" * 64,
    )


def _ctx(api_key: str | None = "test-key") -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key=api_key, database_url="unused")


def _response(
    sentences_payload: list[dict[str, Any]],
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
        content=[TextBlock(type="text", text=json.dumps({"sentences": sentences_payload}))],
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


_SUPPORTED_ITEM = {
    "text": "Led the platform team.",
    "kind": "claim",
    "verdict": "supported",
    "drift_label": "supported",
    "cited_span_ids": [],
    "reason": "Matches the corpus.",
}


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeAnthropicClient) -> None:
    # jfl_gate.gate does `import anthropic` -- patching the attribute on the shared
    # module object (not a copy) is what makes gate.py's own reference see the fake.
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)


# --- tests -------------------------------------------------------------------


class TestCredentialResolution:
    """No key in the context means "use this machine's ambient credential".

    `ant auth login` writes an OAuth profile the SDK resolves on its own -- the
    same profile resolution Claude Code uses -- so an absent key must reach a
    bare client rather than being rejected up front. An explicit key still wins.
    """

    @staticmethod
    def _record_construction(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        client = _FakeAnthropicClient(_response([_SUPPORTED_ITEM]))

        def _construct(**kwargs: Any) -> _FakeAnthropicClient:
            calls.append(kwargs)
            return client

        monkeypatch.setattr(anthropic, "Anthropic", _construct)
        return calls

    def test_explicit_key_is_passed_to_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._record_construction(monkeypatch)
        check_text(
            _ctx(api_key="sk-ant-explicit"),
            _FakeGroundingRepo([_span()]),
            _FakeRunRepo(),
            "Led the platform team.",
        )
        assert calls == [{"api_key": "sk-ant-explicit"}]

    def test_absent_key_constructs_a_bare_client_for_profile_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_construction(monkeypatch)
        check_text(
            _ctx(api_key=None),
            _FakeGroundingRepo([_span()]),
            _FakeRunRepo(),
            "Led the platform team.",
        )
        # No api_key kwarg at all: passing api_key=None would suppress profile
        # resolution rather than defer to it.
        assert calls == [{}]


def test_empty_input_raises_without_calling_the_api_or_recording_a_run() -> None:
    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GateError, match="sentences"):
        check_text(_ctx(), grounding, runs, "   ")

    assert runs.recorded == []


def test_successful_call_parses_result_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        [_SUPPORTED_ITEM],
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=500,
        cache_creation_input_tokens=0,
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()
    ctx = _ctx()

    result = check_text(ctx, grounding, runs, "Led the platform team.")

    assert len(result.sentences) == 1
    assert result.sentences[0].verdict == "supported"

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "gate"
    assert run.stage == "baseline"
    assert run.model == MODEL
    assert run.user_id == ctx.user_id
    assert run.trace_id == ctx.trace_id
    assert run.tokens_in == 1000
    assert run.tokens_out == 200
    assert run.cache_read_tokens == 500
    assert run.cache_write_tokens == 0
    assert run.cost_usd == compute_cost_usd(1000, 200, 500, 0)
    assert run.error is None
    assert run.latency_ms is not None and run.latency_ms >= 0
    assert isinstance(run.started_at, datetime)


def test_request_caches_the_corpus_and_keeps_sentences_out_of_the_cached_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response([_SUPPORTED_ITEM])
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    span = _span("A distinctive corpus fact about the platform team.")
    grounding = _FakeGroundingRepo([span])
    runs = _FakeRunRepo()

    check_text(_ctx(), grounding, runs, "Led the platform team.")

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]

    assert kwargs["model"] == MODEL
    system_blocks = kwargs["system"]
    assert len(system_blocks) == 1
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert str(span.id) in system_blocks[0]["text"]
    assert span.text in system_blocks[0]["text"]

    # The sentence under test is volatile -- it belongs in `messages`, not in the
    # cached `system` block, or every distinct input would bust the cache.
    assert "Led the platform team." not in system_blocks[0]["text"]
    user_content = kwargs["messages"][0]["content"]
    assert "Led the platform team." in user_content

    assert kwargs["output_config"]["format"]["type"] == "json_schema"


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
def test_api_errors_record_an_error_run_and_raise_gate_error(
    monkeypatch: pytest.MonkeyPatch, exception_factory: Any
) -> None:
    client = _FakeAnthropicClient(exception=exception_factory())
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GateError):
        check_text(_ctx(), grounding, runs, "Led the platform team.")

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert run.tokens_in is None  # the call never returned usage


def test_refusal_records_a_refused_run_and_raises_gate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        [],
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="cyber"),
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GateError, match="refus"):
        check_text(_ctx(), grounding, runs, "Led the platform team.")

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "refused"
    assert run.tokens_in is not None  # usage is still billed and recorded


def test_malformed_json_records_an_error_run_and_raises_gate_error(
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

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GateError):
        check_text(_ctx(), grounding, runs, "Led the platform team.")

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "error"


def test_response_with_no_text_block_records_an_error_run_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Message(
        id="msg_test",
        content=[],
        model=MODEL,
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GateError):
        check_text(_ctx(), grounding, runs, "Led the platform team.")

    assert runs.recorded[0].outcome == "error"


def test_loads_all_non_retired_spans_for_the_context_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response([_SUPPORTED_ITEM])
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()
    ctx = _ctx()

    check_text(ctx, grounding, runs, "Led the platform team.")

    assert grounding.all_spans_calls == [ctx.user_id]
