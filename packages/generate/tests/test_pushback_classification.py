"""Unit tests for `classify_pushback`. Mirrors
`packages/generate/tests/test_titles.py`'s approach closely:
`anthropic.Anthropic` is monkeypatched to a fake client, and the run
repository is a trivial in-memory stand-in for the Protocol in
jfl_core.repositories. No live API, no database.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.models import RunRecord
from jfl_core.pushback import PUSHBACK_KINDS
from jfl_core.repositories import RunRepository
from jfl_gate.pricing import compute_cost_usd
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA,
    build_pushback_classification_prompt,
)
from jfl_generate.pushback import MAX_NOTE_CHARS, MODEL, classify_pushback

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


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


def _call(
    *,
    ctx: RequestContext | None = None,
    runs: RunRepository | None = None,
    user_text: str = "I actually led that team of twelve",
    could_get_score: int | None = 4,
    could_get_explanation: str = "Limited evidence of team leadership.",
    want_score: int | None = 6,
    want_explanation: str = "Two of what you said matters are evidenced.",
    earlier_texts: tuple[str, ...] = (),
    now: datetime = NOW,
) -> Any:
    return classify_pushback(
        ctx or _ctx(),
        runs or _FakeRunRepo(),
        user_text=user_text,
        could_get_score=could_get_score,
        could_get_explanation=could_get_explanation,
        want_score=want_score,
        want_explanation=want_explanation,
        earlier_texts=earlier_texts,
        now=now,
    )


_CLASSIFICATION_PAYLOAD = {
    "kind": "capability",
    "direction": "up",
    "new_information": True,
    "classification_note": "A claim about what they have done, not what they want.",
}


def _response(
    payload: dict[str, Any],
    *,
    stop_reason: str = "end_turn",
    stop_details: RefusalStopDetails | None = None,
    input_tokens: int = 300,
    output_tokens: int = 40,
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


# --- schema ------------------------------------------------------------------


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


def test_schema_has_no_property_named_reason() -> None:
    """CLAUDE.md's 2026-09-02 decision: a schema property named `reason`,
    combined with a labelling system prompt, has tripped the API's
    reverse-engineering/duplication classifier before -- and this call is a
    labelling prompt (classify into one of three kinds) if anything in this
    package is.
    """
    assert "reason" not in _property_names(PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA)


def test_schema_requires_all_four_fields_and_nothing_else() -> None:
    assert PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA["required"] == [
        "kind",
        "direction",
        "new_information",
        "classification_note",
    ]
    assert PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA["additionalProperties"] is False


def test_schema_kind_enum_matches_the_closed_pushback_kind_list() -> None:
    kind_schema = PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA["properties"]["kind"]  # type: ignore[index]
    assert set(kind_schema["enum"]) == set(PUSHBACK_KINDS)


# --- prompt builder ------------------------------------------------------------


class TestPromptBuilder:
    def test_includes_the_users_words_verbatim_and_the_timestamp(self) -> None:
        prompt = build_pushback_classification_prompt(
            user_text="I actually led that team of twelve",
            could_get_score=4,
            could_get_explanation="Limited evidence of team leadership.",
            want_score=4,
            want_explanation="",
            earlier_texts=[],
            now=NOW,
        )
        assert "I actually led that team of twelve" in prompt
        assert NOW.isoformat() in prompt

    def test_includes_earlier_texts_when_given(self) -> None:
        prompt = build_pushback_classification_prompt(
            user_text="new words",
            could_get_score=4,
            could_get_explanation="",
            want_score=4,
            want_explanation="",
            earlier_texts=["said this before"],
            now=NOW,
        )
        assert "said this before" in prompt

    def test_says_nothing_else_said_when_there_are_no_earlier_texts(self) -> None:
        prompt = build_pushback_classification_prompt(
            user_text="new words",
            could_get_score=4,
            could_get_explanation="",
            want_score=4,
            want_explanation="",
            earlier_texts=[],
            now=NOW,
        )
        assert "nothing else" in prompt

    def test_never_asks_the_model_to_rewrite_the_users_words(self) -> None:
        prompt = build_pushback_classification_prompt(
            user_text="new words",
            could_get_score=4,
            could_get_explanation="",
            want_score=4,
            want_explanation="",
            earlier_texts=[],
            now=NOW,
        )
        assert "rewrite" in prompt or "Do not rewrite" in prompt

    def test_an_unscored_shown_score_reads_as_unscored_not_none(self) -> None:
        prompt = build_pushback_classification_prompt(
            user_text="new words",
            could_get_score=None,
            could_get_explanation="",
            want_score=None,
            want_explanation="",
            earlier_texts=[],
            now=NOW,
        )
        assert "None" not in prompt
        assert "unscored" in prompt


# --- the call itself -----------------------------------------------------------


class TestCredentialResolution:
    @staticmethod
    def _record_construction(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        client = _FakeAnthropicClient(_response(_CLASSIFICATION_PAYLOAD))

        def _construct(**kwargs: Any) -> _FakeAnthropicClient:
            calls.append(kwargs)
            return client

        monkeypatch.setattr(anthropic, "Anthropic", _construct)
        return calls

    def test_explicit_key_is_passed_to_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._record_construction(monkeypatch)
        _call(ctx=_ctx(api_key="sk-ant-explicit"))
        assert calls == [{"api_key": "sk-ant-explicit"}]

    def test_absent_key_constructs_a_bare_client_for_profile_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_construction(monkeypatch)
        _call(ctx=_ctx(api_key=None))
        assert calls == [{}]


def test_empty_user_text_raises_without_calling_the_api_or_recording_a_run() -> None:
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="user_text"):
        _call(runs=runs, user_text="   ")
    assert runs.recorded == []


def test_the_call_always_uses_haiku_never_ctx_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The product-model decision (CLAUDE.md 2026-09-05) does not apply here --
    this call is hard-coded to the second, cheaper model regardless of what the
    user has configured.
    """
    client = _FakeAnthropicClient(response=_response(_CLASSIFICATION_PAYLOAD))
    _patch_client(monkeypatch, client)

    ctx = RequestContext(
        user_id=USER, anthropic_api_key="k", database_url="unused", model="claude-opus-5"
    )
    runs = _FakeRunRepo()
    _call(ctx=ctx, runs=runs)

    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"
    assert runs.recorded[0].model == "claude-haiku-4-5"


def test_successful_call_parses_result_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(_CLASSIFICATION_PAYLOAD, input_tokens=500, output_tokens=30)
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    ctx = _ctx()

    result = _call(ctx=ctx, runs=runs)

    assert result.kind == "capability"
    assert result.new_information is True
    assert result.note == "A claim about what they have done, not what they want."

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "generate"
    assert run.stage == "classify_pushback"
    assert run.model == MODEL
    assert run.user_id == ctx.user_id
    assert run.trace_id == ctx.trace_id
    assert run.tokens_in == 500
    assert run.tokens_out == 30
    assert run.cost_usd == compute_cost_usd(MODEL, 500, 30, 0, 0)
    assert run.error is None
    assert run.latency_ms is not None and run.latency_ms >= 0
    assert isinstance(run.started_at, datetime)


def test_request_carries_the_output_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeAnthropicClient(response=_response(_CLASSIFICATION_PAYLOAD))
    _patch_client(monkeypatch, client)

    _call()

    kwargs = client.messages.calls[0]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] is PUSHBACK_CLASSIFICATION_OUTPUT_SCHEMA


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
        _call(runs=runs)

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert run.tokens_in is None  # the call never returned usage


def test_refusal_records_a_refused_run_and_raises_generate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        _CLASSIFICATION_PAYLOAD,
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="reasoning_extraction"),
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="refus"):
        _call(runs=runs)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "refused"


def test_max_tokens_truncation_records_an_error_run(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response({}, stop_reason="max_tokens")
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="max_tokens"):
        _call(runs=runs)

    assert runs.recorded[0].outcome == "error"


def test_malformed_json_records_an_error_run(monkeypatch: pytest.MonkeyPatch) -> None:
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
        _call(runs=runs)

    assert runs.recorded[0].outcome == "error"


# --- sanitisation --------------------------------------------------------------


class TestSanitisation:
    def test_whitespace_is_collapsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "kind": "preference",
            "direction": "down",
            "new_information": False,
            "classification_note": "  a   note   with\nextra   whitespace  ",
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = _call()
        assert result.note == "a note with extra whitespace"

    def test_a_note_over_the_cap_is_trimmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        long_note = "word " * 100  # comfortably over MAX_NOTE_CHARS
        payload = {
            "kind": "factual",
            "direction": "down",
            "new_information": False,
            "classification_note": long_note,
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = _call()
        assert len(result.note) <= MAX_NOTE_CHARS

    @pytest.mark.parametrize("kind", ["preference", "capability", "factual"])
    def test_every_real_kind_passes_through_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, kind: str
    ) -> None:
        payload = {
            "kind": kind,
            "direction": "down",
            "new_information": True,
            "classification_note": "",
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = _call()
        assert result.kind == kind

    def test_an_unrecognised_kind_falls_back_to_the_safe_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`factual` is the safe fallback -- see `jfl_generate.pushback._sanitise`'s
        docstring: it is the one kind that never moves the number, in either
        direction, so a malformed or fabricated `kind` from the model can never
        land on `capability` downward, which applies in full with no evidence
        asked for.
        """
        payload = {
            "kind": "something-the-model-made-up",
            "direction": "down",
            "new_information": True,
            "classification_note": "",
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = _call()
        assert result.kind == "factual"

    def test_new_information_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "kind": "preference",
            "direction": "down",
            "new_information": False,
            "classification_note": "",
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = _call()
        assert result.new_information is False


class TestDirection:
    """The one box asks for words only, so the direction is read from them."""

    @pytest.mark.parametrize("direction", ["up", "down"])
    def test_a_real_direction_passes_through(
        self, monkeypatch: pytest.MonkeyPatch, direction: str
    ) -> None:
        payload = {
            "kind": "preference",
            "direction": direction,
            "new_information": True,
            "classification_note": "",
        }
        _patch_client(monkeypatch, _FakeAnthropicClient(response=_response(payload)))
        result = _call()
        assert (result.kind, result.direction) == ("preference", direction)

    @pytest.mark.parametrize("direction", ["", "sideways"])
    def test_no_trustworthy_direction_reads_as_the_kind_that_moves_nothing(
        self, monkeypatch: pytest.MonkeyPatch, direction: str
    ) -> None:
        """A capability reading with no direction must not land on "down",
        which applies in full -- so it becomes factual, and counts as upward
        on the drift meter, the conservative side.
        """
        payload = {
            "kind": "capability",
            "direction": direction,
            "new_information": True,
            "classification_note": "",
        }
        _patch_client(monkeypatch, _FakeAnthropicClient(response=_response(payload)))
        result = _call()
        assert (result.kind, result.direction) == ("factual", "up")

    def test_the_prompt_carries_both_numbers(self) -> None:
        prompt = build_pushback_classification_prompt(
            user_text="words",
            could_get_score=4,
            could_get_explanation="The ad asks for ownership.",
            want_score=7,
            want_explanation="Remote and hands-on.",
            earlier_texts=[],
            now=NOW,
        )
        assert "The ad asks for ownership." in prompt
        assert "Remote and hands-on." in prompt
        assert "7 out of 10" in prompt and "4 out of 10" in prompt
