"""Unit tests for `suggest_titles` -- slice C7a. Mirrors
`packages/generate/tests/test_extract.py`'s approach closely: `anthropic.Anthropic`
is monkeypatched to a fake client, and the run repository is a trivial
in-memory stand-in for the Protocol in jfl_core.repositories. No live API, no
database.
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
from jfl_gate.pricing import compute_cost_usd
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    TITLE_SUGGESTIONS_OUTPUT_SCHEMA,
    build_title_suggestion_prompt,
)
from jfl_generate.titles import MODEL, suggest_titles

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


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


_TITLES_PAYLOAD = {
    "titles": [
        {"title": "Engineering Manager", "gloss": ""},
        {"title": "Head of Engineering", "gloss": "a step up"},
    ]
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


# --- schema ------------------------------------------------------------------


def test_schema_has_no_property_named_reason() -> None:
    """CLAUDE.md's 2026-09-02 decision: a schema property named `reason`,
    combined with a labelling system prompt, has tripped the API's
    reverse-engineering/duplication classifier before.
    """

    def _names(node: object) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    found.update(value.keys())
                found.update(_names(value))
        elif isinstance(node, list):
            for item in node:
                found.update(_names(item))
        return found

    assert "reason" not in _names(TITLE_SUGGESTIONS_OUTPUT_SCHEMA)


def test_schema_requires_title_and_gloss_only() -> None:
    item_schema = TITLE_SUGGESTIONS_OUTPUT_SCHEMA["properties"]["titles"]["items"]  # type: ignore[index]
    assert item_schema["required"] == ["title", "gloss"]
    assert item_schema["additionalProperties"] is False


# --- prompt / context builder --------------------------------------------------


class TestPromptBuilder:
    def test_includes_the_phrase_and_timestamp(self) -> None:
        prompt = build_title_suggestion_prompt(
            "engineering manager",
            other_includes=[],
            excludes=[],
            application_titles=[],
            now=NOW,
        )
        assert '"engineering manager"' in prompt
        assert NOW.isoformat() in prompt

    def test_includes_other_includes_and_excludes_and_application_titles(self) -> None:
        prompt = build_title_suggestion_prompt(
            "SEM",
            other_includes=["engineering manager"],
            excludes=["marketing"],
            application_titles=["Head of Platform Engineering"],
            now=NOW,
        )
        assert "engineering manager" in prompt
        assert "marketing" in prompt
        assert "Head of Platform Engineering" in prompt

    def test_empty_context_reads_as_none_rather_than_blank(self) -> None:
        prompt = build_title_suggestion_prompt(
            "engineering manager",
            other_includes=[],
            excludes=[],
            application_titles=[],
            now=NOW,
        )
        assert "(none)" in prompt


# --- the call itself -----------------------------------------------------------


class TestCredentialResolution:
    @staticmethod
    def _record_construction(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        client = _FakeAnthropicClient(_response(_TITLES_PAYLOAD))

        def _construct(**kwargs: Any) -> _FakeAnthropicClient:
            calls.append(kwargs)
            return client

        monkeypatch.setattr(anthropic, "Anthropic", _construct)
        return calls

    def test_explicit_key_is_passed_to_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._record_construction(monkeypatch)
        suggest_titles(
            _ctx(api_key="sk-ant-explicit"), _FakeRunRepo(), phrase="engineering manager", now=NOW
        )
        assert calls == [{"api_key": "sk-ant-explicit"}]

    def test_absent_key_constructs_a_bare_client_for_profile_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_construction(monkeypatch)
        suggest_titles(_ctx(api_key=None), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert calls == [{}]


def test_empty_phrase_raises_without_calling_the_api_or_recording_a_run() -> None:
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="phrase"):
        suggest_titles(_ctx(), runs, phrase="   ", now=NOW)
    assert runs.recorded == []


def test_the_call_always_uses_haiku_never_ctx_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The product-model decision (CLAUDE.md 2026-09-05) does not apply here --
    this call is hard-coded to the second, cheaper model regardless of what the
    user has configured.
    """
    client = _FakeAnthropicClient(response=_response(_TITLES_PAYLOAD))
    _patch_client(monkeypatch, client)

    ctx = RequestContext(
        user_id=USER, anthropic_api_key="k", database_url="unused", model="claude-opus-5"
    )
    runs = _FakeRunRepo()
    suggest_titles(ctx, runs, phrase="engineering manager", now=NOW)

    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"
    assert runs.recorded[0].model == "claude-haiku-4-5"


def test_successful_call_parses_result_and_records_an_ok_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(_TITLES_PAYLOAD, input_tokens=500, output_tokens=80)
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    ctx = _ctx()

    result = suggest_titles(ctx, runs, phrase="engineering manager", now=NOW)

    assert [s.title for s in result] == ["Head of Engineering"]  # "Engineering
    # Manager" itself is dropped: it matches the phrase's own key.

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "ok"
    assert run.component == "generate"
    assert run.stage == "suggest_titles"
    assert run.model == MODEL
    assert run.user_id == ctx.user_id
    assert run.trace_id == ctx.trace_id
    assert run.tokens_in == 500
    assert run.tokens_out == 80
    assert run.cost_usd == compute_cost_usd(MODEL, 500, 80, 0, 0)
    assert run.error is None
    assert run.latency_ms is not None and run.latency_ms >= 0
    assert isinstance(run.started_at, datetime)


def test_request_carries_the_output_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeAnthropicClient(response=_response(_TITLES_PAYLOAD))
    _patch_client(monkeypatch, client)

    suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)

    assert len(client.messages.calls) == 1
    kwargs = client.messages.calls[0]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] is TITLE_SUGGESTIONS_OUTPUT_SCHEMA


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
        suggest_titles(_ctx(), runs, phrase="engineering manager", now=NOW)

    assert len(runs.recorded) == 1
    run = runs.recorded[0]
    assert run.outcome == "error"
    assert run.error is not None
    assert run.tokens_in is None  # the call never returned usage


def test_refusal_records_a_refused_run_and_raises_generate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response(
        {"titles": []},
        stop_reason="refusal",
        stop_details=RefusalStopDetails(type="refusal", category="cyber"),
    )
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="refus"):
        suggest_titles(_ctx(), runs, phrase="engineering manager", now=NOW)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "refused"


def test_max_tokens_truncation_records_an_error_run(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response({}, stop_reason="max_tokens")
    client = _FakeAnthropicClient(response=response)
    _patch_client(monkeypatch, client)

    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="max_tokens"):
        suggest_titles(_ctx(), runs, phrase="engineering manager", now=NOW)

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
        suggest_titles(_ctx(), runs, phrase="engineering manager", now=NOW)

    assert runs.recorded[0].outcome == "error"


# --- sanitisation ----------------------------------------------------------


class TestSanitisation:
    def test_a_comma_in_a_title_is_replaced_not_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A comma would silently split one suggested title into two filter
        alternatives (jfl_intake.filtering.parse_terms) -- see the module
        docstring.
        """
        payload = {"titles": [{"title": "Head of Engineering, Platform", "gloss": ""}]}
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert result[0].title == "Head of Engineering Platform"
        assert "," not in result[0].title

    def test_an_empty_title_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "titles": [
                {"title": "   ", "gloss": ""},
                {"title": "Engineering Lead", "gloss": ""},
            ]
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert [s.title for s in result] == ["Engineering Lead"]

    def test_a_title_over_the_length_limit_is_dropped_not_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_title = "Engineering " * 10  # comfortably over 80 chars
        payload = {"titles": [{"title": long_title, "gloss": ""}]}
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert result == []

    def test_a_suggestion_matching_the_phrase_itself_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {"titles": [{"title": "Manager, Engineering", "gloss": ""}]}
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert result == []

    def test_a_suggestion_matching_an_existing_include_phrase_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {"titles": [{"title": "Head of Engineering", "gloss": ""}]}
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(
            _ctx(),
            _FakeRunRepo(),
            phrase="engineering manager",
            other_includes=["head of engineering"],
            now=NOW,
        )
        assert result == []

    def test_a_suggestion_the_filter_already_matches_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shape the owner saw offered: "Technical Lead Manager", glossed
        by the model itself as "already in filter", with "technical lead"
        saved. The filter matches a title when all of an alternative's words
        are in it, so every Technical Lead Manager posting already matches
        "technical lead" -- keys compared only for equality let it through.
        Likewise "Senior Engineering Manager" under "engineering manager".
        A title that merely shares words ("Lead Engineer") is still offered.
        """
        payload = {
            "titles": [
                {
                    "title": "Technical Lead Manager",
                    "gloss": "Already in filter; equivalent seniority",
                },
                {"title": "Senior Engineering Manager", "gloss": "a step up"},
                {"title": "Lead Engineer", "gloss": ""},
            ]
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(
            _ctx(),
            _FakeRunRepo(),
            phrase="engineering manager",
            other_includes=["tech lead", "technical lead"],
            now=NOW,
        )
        assert [s.title for s in result] == ["Lead Engineer"]

    def test_duplicate_suggestions_within_one_response_are_deduped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "titles": [
                {"title": "Engineering Lead", "gloss": "first"},
                {"title": "engineering lead", "gloss": "same thing, different case"},
            ]
        }
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert len(result) == 1
        assert result[0].gloss == "first"

    def test_results_are_capped_at_ten(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {"titles": [{"title": f"Title Number {i}", "gloss": ""} for i in range(20)]}
        client = _FakeAnthropicClient(response=_response(payload))
        _patch_client(monkeypatch, client)

        result = suggest_titles(_ctx(), _FakeRunRepo(), phrase="engineering manager", now=NOW)
        assert len(result) == 10
