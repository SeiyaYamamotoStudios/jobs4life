"""Unit tests for `check_coverage`: the fixed control flow around the one model
call. No live API, no database -- mirrors jfl_gate/tests/test_gate.py closely,
including the corpus-caching assertions, since coverage caches the corpus the
same way the claim gate does.
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
from jfl_gate.pricing import MODEL, compute_cost_usd
from jfl_generate.coverage import check_coverage
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


def _ctx(api_key: str | None = "test-key") -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key=api_key, database_url="unused")


_EVIDENCED_RESULT = {
    "status": "evidenced",
    "cited_span_ids": [],
    "evidence_note": "Matches the corpus.",
    "question": "",
}


def _response(
    results_payload: list[dict[str, Any]],
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
        content=[TextBlock(type="text", text=json.dumps({"results": results_payload}))],
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
        client = _FakeAnthropicClient(_response([_EVIDENCED_RESULT]))

        def _construct(**kwargs: Any) -> _FakeAnthropicClient:
            calls.append(kwargs)
            return client

        monkeypatch.setattr(anthropic, "Anthropic", _construct)
        return calls

    def test_explicit_key_is_passed_to_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._record_construction(monkeypatch)
        check_coverage(
            _ctx(api_key="sk-ant-explicit"),
            _FakeGroundingRepo([_span()]),
            _FakeRunRepo(),
            ["Python"],
        )
        assert calls == [{"api_key": "sk-ant-explicit"}]

    def test_absent_key_constructs_a_bare_client_for_profile_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_construction(monkeypatch)
        check_coverage(
            _ctx(api_key=None), _FakeGroundingRepo([_span()]), _FakeRunRepo(), ["Python"]
        )
        assert calls == [{}]


def test_empty_requirements_raises_without_calling_the_api_or_recording_a_run() -> None:
    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="requirements"):
        check_coverage(_ctx(), grounding, runs, [])
    assert runs.recorded == []


def test_successful_call_parses_result_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        [_EVIDENCED_RESULT],
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

    result = check_coverage(ctx, grounding, runs, ["Python experience"])

    assert len(result.results) == 1
    assert result.results[0].status == "evidenced"

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "generate"
    assert run.stage == "coverage"
    assert run.model == MODEL
    assert run.tokens_in == 1000
    assert run.tokens_out == 200
    assert run.cache_read_tokens == 500
    assert run.cost_usd == compute_cost_usd(MODEL, 1000, 200, 500, 0)
    assert run.error is None
    assert run.latency_ms is not None and run.latency_ms >= 0
    assert isinstance(run.started_at, datetime)


def test_request_caches_the_corpus_and_keeps_requirements_out_of_the_cached_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response([_EVIDENCED_RESULT])
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    span = _span("A distinctive corpus fact about the platform team.")
    grounding = _FakeGroundingRepo([span])
    runs = _FakeRunRepo()

    check_coverage(_ctx(), grounding, runs, ["Led a platform team"])

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

    # The requirement under check is volatile -- belongs in `messages`, not the
    # cached `system` block, or every distinct job would bust the cache.
    assert "Led a platform team" not in combined_system_text
    user_content = kwargs["messages"][0]["content"]
    assert "Led a platform team" in user_content

    assert kwargs["output_config"]["format"]["type"] == "json_schema"


def test_loads_all_non_retired_spans_for_the_context_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response([_EVIDENCED_RESULT])
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()
    ctx = _ctx()

    check_coverage(ctx, grounding, runs, ["Python"])

    assert grounding.all_spans_calls == [ctx.user_id]


def test_result_count_mismatch_records_an_error_run_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No safe way to guess which requirement a stray or missing result belongs
    to -- a count mismatch is treated as a parse failure, not silently zipped.
    """
    response = _response([_EVIDENCED_RESULT])  # one result for two requirements
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError, match="2"):
        check_coverage(_ctx(), grounding, runs, ["Python", "Kubernetes"])

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "error"


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

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError):
        check_coverage(_ctx(), grounding, runs, ["Python"])

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert run.tokens_in is None


def test_refusal_records_a_refused_run_and_raises_generate_error(
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

    with pytest.raises(GenerateError, match="refus"):
        check_coverage(_ctx(), grounding, runs, ["Python"])

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "refused"


def test_max_tokens_truncation_records_an_error_run_and_names_the_real_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated response would otherwise surface as a JSONDecodeError -- see
    jfl_gate.gate's max_tokens check, which this mirrors: the check must run
    before any attempt to parse the (truncated, likely invalid) response body,
    so the error names the real cause instead of a misleading parse failure.
    """
    response = _response([], stop_reason="max_tokens")
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError, match="max_tokens"):
        check_coverage(_ctx(), grounding, runs, ["Python"])

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert "max_tokens" in run.error


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

    grounding = _FakeGroundingRepo([_span()])
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError):
        check_coverage(_ctx(), grounding, runs, ["Python"])

    assert runs.recorded[0].outcome == "error"


# --- one bad citation never voids a run (production, 2026-09-23) -------------


def _result(
    status: str, cited: list[str], note: str = "note", question: str = ""
) -> dict[str, Any]:
    return {
        "status": status,
        "cited_span_ids": cited,
        "evidence_note": note,
        "question": question,
    }


def test_a_malformed_citation_is_dropped_and_the_rest_of_the_run_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production failure, reproduced: `results.4.cited_span_ids.1` was not
    a uuid and strict parsing threw away every requirement's verdict. Now the
    one id is dropped, the requirement keeps its status on the citation that
    does check out, and every other result is untouched.
    """
    span = _span()
    results = [_result("evidenced", [str(span.id)]) for _ in range(4)]
    results.append(_result("evidenced", [str(span.id), "span-12"]))
    _patch_client(monkeypatch, _FakeAnthropicClient(response=_response(results)))
    runs = _FakeRunRepo()

    output = check_coverage(_ctx(), _FakeGroundingRepo([span]), runs, [f"r{i}" for i in range(5)])

    assert [r.status for r in output.results] == ["evidenced"] * 5
    assert output.results[4].cited_span_ids == [span.id]
    assert output.results[4].unparseable_citations == []
    assert output.dropped_citations == 1
    assert output.downgraded_requirements == 0
    (run,) = runs.recorded
    assert run.outcome == "ok"
    assert run.attributes == {"dropped_citations": 1, "downgraded_requirements": 0}


def test_a_citation_naming_no_span_in_the_corpus_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _span()
    stranger = uuid.uuid4()
    results = [_result("evidenced", [str(span.id), str(stranger)])]
    _patch_client(monkeypatch, _FakeAnthropicClient(response=_response(results)))
    runs = _FakeRunRepo()

    output = check_coverage(_ctx(), _FakeGroundingRepo([span]), runs, ["Python"])

    assert output.results[0].cited_span_ids == [span.id]
    assert output.results[0].status == "evidenced"
    assert output.dropped_citations == 1
    assert runs.recorded[0].attributes == {"dropped_citations": 1, "downgraded_requirements": 0}


def test_a_status_left_resting_on_nothing_is_downgraded_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evidenced -> partial, partial -> absent, never to contradicted -- a
    citation that could not be checked is an absence of evidence, not evidence
    of the opposite. The note says why, and the model's own words stay.
    """
    span = _span()
    results = [
        _result("evidenced", ["not-a-uuid"], note="Corpus documents Python."),
        _result("partial", [str(uuid.uuid4())], note="Some Kubernetes.", question="More?"),
        _result("contradicted", ["bogus"], note="Corpus says otherwise."),
        _result("absent", [], note="Silent.", question="Have you?"),
    ]
    _patch_client(monkeypatch, _FakeAnthropicClient(response=_response(results)))
    runs = _FakeRunRepo()

    output = check_coverage(
        _ctx(), _FakeGroundingRepo([span]), runs, ["Python", "K8s", "Go", "Rust"]
    )

    evidenced, partial, contradicted, absent = output.results
    assert evidenced.status == "partial"
    assert evidenced.cited_span_ids == []
    assert evidenced.evidence_note.startswith("Corpus documents Python.")
    assert "could not be checked" in evidenced.evidence_note
    assert partial.status == "absent"
    assert partial.question == "More?"
    assert "could not be checked" in partial.evidence_note
    # Not a status that rests on a citation, so dropping one leaves it be.
    assert contradicted.status == "contradicted"
    assert absent.status == "absent"
    assert absent.evidence_note == "Silent."
    assert output.dropped_citations == 3
    assert output.downgraded_requirements == 2
    assert runs.recorded[0].attributes == {"dropped_citations": 3, "downgraded_requirements": 2}


def test_an_evidenced_result_the_model_left_uncited_is_not_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The downgrade fires only when dropping emptied the list -- a run with no
    bad citation must come out exactly as the model gave it."""
    _patch_client(monkeypatch, _FakeAnthropicClient(response=_response([_EVIDENCED_RESULT])))
    runs = _FakeRunRepo()

    output = check_coverage(_ctx(), _FakeGroundingRepo([_span()]), runs, ["Python"])

    assert output.results[0].status == "evidenced"
    assert output.dropped_citations == 0
    assert runs.recorded[0].attributes == {"dropped_citations": 0, "downgraded_requirements": 0}
