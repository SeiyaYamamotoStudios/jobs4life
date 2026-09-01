"""Unit tests for `extract_requirements`: the fixed control flow around the one
model call. No live API, no database -- mirrors jfl_gate/tests/test_gate.py's
approach closely: `anthropic.Anthropic` is monkeypatched to a fake client, and
the run repository is a trivial in-memory stand-in for the Protocol in
jfl_core.repositories.
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
from jfl_core.models import RunRecord
from jfl_gate.pricing import MODEL, compute_cost_usd
from jfl_generate.errors import GenerateError
from jfl_generate.extract import extract_requirements

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


# --- fakes -------------------------------------------------------------------


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


def _ctx(api_key: str | None = "test-key") -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key=api_key, database_url="unused")


_EXTRACT_PAYLOAD = {
    "employer": "Acme Corp",
    "title": "Senior Engineer",
    "location": "Remote",
    "requirements": [
        {"text": "5+ years of Python", "necessity": "essential"},
        {"text": "Kubernetes experience", "necessity": "desirable"},
    ],
}


def _response(
    payload: dict[str, Any],
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
        content=[TextBlock(type="text", text=json.dumps(payload))],
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


# --- tests -------------------------------------------------------------------


class TestCredentialResolution:
    @staticmethod
    def _record_construction(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        client = _FakeAnthropicClient(_response(_EXTRACT_PAYLOAD))

        def _construct(**kwargs: Any) -> _FakeAnthropicClient:
            calls.append(kwargs)
            return client

        monkeypatch.setattr(anthropic, "Anthropic", _construct)
        return calls

    def test_explicit_key_is_passed_to_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._record_construction(monkeypatch)
        extract_requirements(_ctx(api_key="sk-ant-explicit"), _FakeRunRepo(), "Some job ad text.")
        assert calls == [{"api_key": "sk-ant-explicit"}]

    def test_absent_key_constructs_a_bare_client_for_profile_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_construction(monkeypatch)
        extract_requirements(_ctx(api_key=None), _FakeRunRepo(), "Some job ad text.")
        assert calls == [{}]


def test_empty_input_raises_without_calling_the_api_or_recording_a_run() -> None:
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="text"):
        extract_requirements(_ctx(), runs, "   ")
    assert runs.recorded == []


def test_successful_call_parses_result_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        _EXTRACT_PAYLOAD,
        input_tokens=1000,
        output_tokens=200,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    ctx = _ctx()

    result = extract_requirements(
        ctx, runs, "Senior Engineer at Acme Corp. Remote. Requires Python."
    )

    assert result.employer == "Acme Corp"
    assert len(result.requirements) == 2

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "generate"
    assert run.stage == "extract_requirements"
    assert run.model == MODEL
    assert run.user_id == ctx.user_id
    assert run.trace_id == ctx.trace_id
    assert run.tokens_in == 1000
    assert run.tokens_out == 200
    assert run.cost_usd == compute_cost_usd(1000, 200, 0, 0)
    assert run.error is None
    assert run.latency_ms is not None and run.latency_ms >= 0
    assert isinstance(run.started_at, datetime)


def test_request_sends_no_corpus_and_no_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extraction reads only the ad -- see the module docstring: no corpus to
    cache, so `system` is a plain string with no `cache_control` block.
    """
    client = _FakeAnthropicClient(response=_response(_EXTRACT_PAYLOAD))
    _patch_client(monkeypatch, client)

    extract_requirements(
        _ctx(), _FakeRunRepo(), "A distinctive job ad about spaceship engineering."
    )

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]
    assert kwargs["model"] == MODEL
    assert isinstance(kwargs["system"], str)
    user_content = kwargs["messages"][0]["content"]
    assert "A distinctive job ad about spaceship engineering." in user_content
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
def test_api_errors_record_an_error_run_and_raise_generate_error(
    monkeypatch: pytest.MonkeyPatch, exception_factory: Any
) -> None:
    client = _FakeAnthropicClient(exception=exception_factory())
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError):
        extract_requirements(_ctx(), runs, "Some job ad text.")

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert run.tokens_in is None  # the call never returned usage


def test_refusal_records_a_refused_run_and_raises_generate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        {},
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="cyber"),
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="refus"):
        extract_requirements(_ctx(), runs, "Some job ad text.")

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "refused"
    assert run.tokens_in is not None  # usage is still billed and recorded


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

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError):
        extract_requirements(_ctx(), runs, "Some job ad text.")

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

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError):
        extract_requirements(_ctx(), runs, "Some job ad text.")

    assert runs.recorded[0].outcome == "error"
