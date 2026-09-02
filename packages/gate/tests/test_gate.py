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
from jfl_gate.gate import EFFORT, GateError, check_text, sentences_from_text, split_blocks
from jfl_gate.pricing import MODEL, compute_cost_usd

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
# Fixed rather than random so a response fixture can cite this span by id.
_SPAN_ID = uuid.uuid5(uuid.NAMESPACE_OID, "jfl-test-span")


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

    def stream(self, **kwargs: Any) -> _FakeStream:
        """Mirrors `client.messages.stream(...)`: a context manager yielding an object
        whose `get_final_message()` returns the assembled Message. The gate streams
        because its `max_tokens` is too high for a non-streaming request, but it
        consumes no partial output, so the double only needs the final message.

        An API error is raised on entering the stream, which is where the real SDK
        raises it too -- the request is issued by `__enter__`, not by `stream()`.
        """
        self.calls.append(kwargs)
        return _FakeStream(self._response, self._exception)


class _FakeStream:
    def __init__(self, response: Message | None, exception: Exception | None):
        self._response = response
        self._exception = exception

    def __enter__(self) -> _FakeStream:
        if self._exception is not None:
            raise self._exception
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Message:
        assert self._response is not None
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: Message | None = None, exception: Exception | None = None):
        self.messages = _FakeMessages(response, exception)


# --- helpers -------------------------------------------------------------------


def _span(text: str = "Led the platform team") -> Span:
    return Span(
        id=_SPAN_ID,
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


# A `supported` claim cites the span it traces to. It must: the rule tier escalates
# an uncited `supported` claim to `review`, because "traces cleanly to the corpus"
# with nothing to trace to is unverifiable rather than verified.
_SUPPORTED_ITEM = {
    "index": 1,
    "kind": "claim",
    "verdict": "supported",
    "drift_label": "supported",
    "cited_span_ids": [str(_SPAN_ID)],
    "reason": "Matches the corpus.",
}


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeAnthropicClient) -> None:
    # jfl_gate.gate does `import anthropic` -- patching the attribute on the shared
    # module object (not a copy) is what makes gate.py's own reference see the fake.
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)


# --- splitting -------------------------------------------------------------------


class TestSplitBlocks:
    """Bullets and paragraphs, mirroring jfl_core.ingest.parser's block model --
    a blank line or a fresh bullet marker starts a new block, everything else
    extends whatever block is already open.
    """

    def test_a_bullet_with_no_terminal_punctuation_is_one_block(self) -> None:
        assert split_blocks("- Did the thing") == ["Did the thing"]

    def test_a_new_bullet_marker_always_starts_a_fresh_block(self) -> None:
        text = "- First\n- Second\n- Third"
        assert split_blocks(text) == ["First", "Second", "Third"]

    def test_consecutive_bullets_never_merge_even_without_a_blank_line(self) -> None:
        text = "- First point\n- Second point"
        assert len(split_blocks(text)) == 2

    def test_blank_line_separates_two_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        assert split_blocks(text) == ["First paragraph.", "Second paragraph."]

    def test_wrapped_paragraph_lines_with_no_blank_line_merge_into_one_block(self) -> None:
        """Not a PDF artifact here -- ordinary hand-wrapped prose, which stays one
        block exactly as it does in corpus markdown.
        """
        text = "This is a paragraph\nthat wraps onto a second line."
        assert split_blocks(text) == ["This is a paragraph that wraps onto a second line."]

    def test_a_document_of_only_headings_is_one_block_per_heading(self) -> None:
        text = "WHAT I BRING\n\nEDUCATION\n\nPERSONAL INTERESTS"
        assert split_blocks(text) == ["WHAT I BRING", "EDUCATION", "PERSONAL INTERESTS"]

    def test_empty_text_has_no_blocks(self) -> None:
        assert split_blocks("") == []

    def test_whitespace_only_text_has_no_blocks(self) -> None:
        assert split_blocks("   \n\n  \n") == []


class TestSentencesFromText:
    """`sentences_from_text` composes `split_blocks` with the existing
    corpus sentence splitter -- the point of both is that a unit never crosses
    a block boundary, so two unrelated claims are never fused into one verdict.
    """

    def test_a_bullet_with_no_terminal_punctuation_is_one_unit(self) -> None:
        assert sentences_from_text("- Did the thing") == ["Did the thing"]

    def test_a_bullet_containing_two_sentences_becomes_two_units(self) -> None:
        result = sentences_from_text("- Did one thing. Then did another thing.")
        assert result == ["Did one thing.", "Then did another thing."]

    def test_plain_prose_is_unchanged(self) -> None:
        """The property CLAUDE.md calls out explicitly: `jfl check` on a plain
        two-sentence string must still yield exactly two units.
        """
        assert sentences_from_text("Two sentences. Like this.") == [
            "Two sentences.",
            "Like this.",
        ]

    def test_a_heading_is_its_own_unit_not_fused_with_a_bullet_below_it(self) -> None:
        text = "EDUCATION\n\n- Completed a course"
        assert sentences_from_text(text) == ["EDUCATION", "Completed a course"]

    def test_consecutive_bullets_never_merge(self) -> None:
        text = "- First point\n- Second point"
        assert sentences_from_text(text) == ["First point", "Second point"]

    def test_blank_line_separated_paragraphs_stay_separate(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        assert sentences_from_text(text) == ["First paragraph.", "Second paragraph."]

    def test_empty_text_yields_no_sentences(self) -> None:
        assert sentences_from_text("") == []

    def test_whitespace_only_text_yields_no_sentences(self) -> None:
        assert sentences_from_text("   \n\n  ") == []

    def test_a_document_of_only_headings_yields_one_unit_per_heading(self) -> None:
        text = "WHAT I BRING\n\nEDUCATION\n\nPERSONAL INTERESTS"
        assert sentences_from_text(text) == ["WHAT I BRING", "EDUCATION", "PERSONAL INTERESTS"]


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
    # Never asked of the model (see SentenceResult.text) -- filled in from the input
    # sentence list once `_check_alignment` confirms the returned index lines up.
    assert result.sentences[0].text == "Led the platform team."

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
    assert run.cost_usd == compute_cost_usd(MODEL, 1000, 200, 500, 0)
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
    # Two blocks -- instructions, then corpus -- with the cache breakpoint on the
    # corpus block (cache="corpus"), so the cached prefix is instructions+corpus,
    # exactly what the pre-split single-string prompt cached.
    assert len(system_blocks) == 2
    assert "cache_control" not in system_blocks[0]
    assert system_blocks[1]["cache_control"] == {"type": "ephemeral"}
    combined_system_text = "".join(b["text"] for b in system_blocks)
    assert str(span.id) in combined_system_text
    assert span.text in combined_system_text

    # The sentence under test is volatile -- it belongs in `messages`, not in the
    # cached `system` block, or every distinct input would bust the cache.
    assert "Led the platform team." not in combined_system_text
    user_content = kwargs["messages"][0]["content"]
    assert "Led the platform team." in user_content

    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["effort"] == EFFORT


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


def test_rule_tier_runs_after_parsing_and_records_escalations_on_the_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule tier (jfl_gate.rules.apply_rules) makes no model call of its own, so
    it must not add a second `runs` row -- and its firing rate belongs in the one row
    `check_text` already writes, as `attributes["rule_escalations"]`.
    """
    boundary_span = Span(
        id=uuid.uuid4(),
        user_id=USER,
        document_id=uuid.uuid4(),
        provenance="document",
        kind="bullet",
        section_path="Things stated explicitly as NOT true, or as boundaries to hold",
        ordinal=0,
        text="Has not personally operated a self-managed kayelisk cluster.",
        content_hash="0" * 64,
    )
    item = {
        "index": 1,
        "kind": "claim",
        "verdict": "supported",
        "drift_label": "supported",
        "cited_span_ids": [],
        "reason": "Matches the corpus.",
    }
    client = _FakeAnthropicClient(response=_response([item]))
    _patch_client(monkeypatch, client)

    grounding = _FakeGroundingRepo([boundary_span])
    runs = _FakeRunRepo()

    result = check_text(
        _ctx(), grounding, runs, "Personally operated the kayelisk cluster end to end."
    )

    # The rule tier escalated the sentence -- verdict moved to "review" and it
    # carries at least one rule flag, but drift_label is untouched.
    assert result.sentences[0].verdict == "review"
    assert result.sentences[0].rule_flags
    assert result.sentences[0].drift_label == "supported"

    assert len(runs.recorded) == 1  # still exactly one row, not two
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.attributes == {"rule_escalations": 1}


class TestAlignment:
    """`SentenceResult` carries a 1-based `index` instead of an echoed sentence text
    (see schema.py) -- these tests are the deterministic replacement for what echoed
    text used to catch: a dropped, duplicated, or reordered result silently attaching
    a verdict to the wrong sentence. `_check_alignment` must catch every shape of
    that and only that; a correct response must still populate `.text` for every
    sentence from the input list, in order.
    """

    _MULTI_TEXT = "First point. Second point. Third point."

    @staticmethod
    def _item(index: int, text_for_reason: str = "ok") -> dict[str, Any]:
        return {
            "index": index,
            "kind": "claim",
            "verdict": "supported",
            "drift_label": "supported",
            "cited_span_ids": [],
            "reason": text_for_reason,
        }

    def test_correct_indices_populate_text_from_the_input_list_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        items = [self._item(1), self._item(2), self._item(3)]
        client = _FakeAnthropicClient(response=_response(items))
        _patch_client(monkeypatch, client)

        result = check_text(_ctx(), _FakeGroundingRepo([_span()]), _FakeRunRepo(), self._MULTI_TEXT)

        assert [s.text for s in result.sentences] == [
            "First point.",
            "Second point.",
            "Third point.",
        ]

    def test_missing_result_raises_and_records_an_error_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only 2 results for 3 input sentences.
        items = [self._item(1), self._item(2)]
        client = _FakeAnthropicClient(response=_response(items))
        _patch_client(monkeypatch, client)

        runs = _FakeRunRepo()
        with pytest.raises(GateError, match="misaligned"):
            check_text(_ctx(), _FakeGroundingRepo([_span()]), runs, self._MULTI_TEXT)

        assert len(runs.recorded) == 1
        assert runs.recorded[0].outcome == "error"
        assert runs.recorded[0].error is not None
        assert "misaligned" in runs.recorded[0].error

    def test_duplicated_index_raises_and_records_an_error_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Right count, but index 2 repeated and index 3 missing.
        items = [self._item(1), self._item(2), self._item(2)]
        client = _FakeAnthropicClient(response=_response(items))
        _patch_client(monkeypatch, client)

        runs = _FakeRunRepo()
        with pytest.raises(GateError, match="misaligned"):
            check_text(_ctx(), _FakeGroundingRepo([_span()]), runs, self._MULTI_TEXT)

        assert runs.recorded[0].outcome == "error"

    def test_reordered_indices_raise_even_with_the_right_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same set {1, 2, 3}, wrong order -- must still be caught, since a
        # position-based zip against a merely-correct *set* would silently attach
        # verdicts to the wrong sentences.
        items = [self._item(1), self._item(3), self._item(2)]
        client = _FakeAnthropicClient(response=_response(items))
        _patch_client(monkeypatch, client)

        runs = _FakeRunRepo()
        with pytest.raises(GateError, match="misaligned"):
            check_text(_ctx(), _FakeGroundingRepo([_span()]), runs, self._MULTI_TEXT)

        assert runs.recorded[0].outcome == "error"
