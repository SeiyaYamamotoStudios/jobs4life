"""Unit tests for `extract_cv_facts` -- slice B6. Mirrors
`packages/generate/tests/test_titles.py`: `anthropic.Anthropic` is
monkeypatched to a fake client and the run repository is an in-memory
stand-in. No live API, no database.
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
from jfl_core.cv_limits import MAX_CV_READ_CHARS
from jfl_core.ids import fact_fingerprint
from jfl_core.models import RunRecord
from jfl_generate.cv_facts import (
    MAX_FACT_CHARS,
    MAX_TOKENS,
    extract_cv_facts,
    needs_probe,
    to_proposed_facts,
)
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import CV_FACTS_OUTPUT_SCHEMA, build_cv_facts_prompt
from jfl_generate.schema import CvFactItem

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 20, 9, 30, tzinfo=UTC)
ROLE = "Acme Ltd -- Engineering Manager, 2021-2024"

CV_TEXT = "# Jane Doe\n\n## Acme Ltd -- Engineering Manager, 2021-2024\n\n- Led a team of 8.\n"


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


def _ctx(api_key: str | None = "test-key") -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key=api_key, database_url="unused")


_PAYLOAD = {
    "facts": [
        {
            "role_label": ROLE,
            "source_line": "Led a team of 8 engineers across two squads.",
            "fact_text": "Led a team of 8 engineers.",
            "probe": "How many people reported to you directly?",
        },
        {
            "role_label": ROLE,
            "source_line": "Owned the payments platform.",
            "fact_text": "Owned the payments platform.",
            "probe": "",
        },
        {
            "role_label": ROLE,
            "source_line": "Wrote Python and Go.",
            "fact_text": "Wrote Python and Go.",
            "probe": "",
        },
    ]
}


def _response(
    payload: dict[str, Any],
    *,
    stop_reason: str = "end_turn",
    stop_details: RefusalStopDetails | None = None,
) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model="claude-opus-5",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=stop_details,
        type="message",
        usage=Usage(
            input_tokens=900,
            output_tokens=400,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
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
    """CLAUDE.md's 2026-09-02 decision: a long labelling system prompt plus a
    schema demanding a label and a `reason` per item reads to the API as a
    distillation harvest, and every call comes back refused with an empty body
    while the cache write still bills. This prompt is exactly that shape, so
    the check belongs here and not only in the gate.
    """
    assert "reason" not in _property_names(CV_FACTS_OUTPUT_SCHEMA)


def test_schema_asks_for_the_four_fields_and_nothing_else() -> None:
    item = CV_FACTS_OUTPUT_SCHEMA["properties"]["facts"]["items"]  # type: ignore[index]
    assert item["required"] == ["role_label", "source_line", "fact_text", "probe"]
    assert item["additionalProperties"] is False


# --- the probe guarantee -------------------------------------------------------


class TestProbes:
    @pytest.mark.parametrize(
        "fact",
        [
            "Led a team of six engineers.",
            "Owned the payments platform.",
            "Drove the migration to Kubernetes.",
            "Delivered the rewrite.",
            "Cut p99 latency by 40%.",
            "Managed a budget of 2m.",
            "Worked there for 3 years.",
        ],
    )
    def test_numeric_and_ownership_shapes_need_one(self, fact: str) -> None:
        assert needs_probe(fact)

    @pytest.mark.parametrize(
        "fact",
        [
            "Wrote Python and Go.",
            "Member of the architecture review board.",
            "MSc in Computer Science.",
        ],
    )
    def test_ordinary_facts_do_not(self, fact: str) -> None:
        assert not needs_probe(fact)

    def test_a_missing_probe_is_supplied_for_a_shape_that_needs_one(self) -> None:
        """The prompt asks; a rule guarantees. Same reasoning as the gate's
        deterministic tier -- a prompt cannot promise this and a rule can.
        """
        facts = to_proposed_facts(
            [
                CvFactItem(
                    role_label=ROLE,
                    source_line="Owned the payments platform.",
                    fact_text="Owned the payments platform.",
                    probe=None,
                )
            ]
        )
        assert facts[0].probe is not None
        assert "?" in (facts[0].probe or "")

    def test_the_models_own_probe_is_kept_when_it_gave_one(self) -> None:
        facts = to_proposed_facts(
            [
                CvFactItem(
                    role_label=ROLE,
                    source_line="Led a team of 8.",
                    fact_text="Led a team of 8 engineers.",
                    probe="How many people reported to you directly?",
                )
            ]
        )
        assert facts[0].probe == "How many people reported to you directly?"

    def test_a_fact_that_needs_no_probe_gets_none(self) -> None:
        facts = to_proposed_facts(
            [
                CvFactItem(
                    role_label=ROLE,
                    source_line="Wrote Python and Go.",
                    fact_text="Wrote Python and Go.",
                    probe=None,
                )
            ]
        )
        assert facts[0].probe is None


# --- sanitising ----------------------------------------------------------------


class TestToProposedFacts:
    def test_fingerprint_and_role_key_are_derived_not_taken_from_the_model(self) -> None:
        facts = to_proposed_facts(
            [CvFactItem(role_label=ROLE, source_line="line", fact_text="Ran the rota.")]
        )
        assert facts[0].fingerprint == fact_fingerprint(ROLE, "Ran the rota.")
        assert facts[0].role_key == facts[0].role_key.lower()

    def test_the_same_fact_twice_in_one_response_becomes_one_row(self) -> None:
        """Otherwise the insert conflicts with itself mid-statement."""
        item = CvFactItem(role_label=ROLE, source_line="line", fact_text="Ran the rota.")
        assert len(to_proposed_facts([item, item])) == 1

    def test_entries_missing_a_role_line_or_fact_are_dropped(self) -> None:
        items = [
            CvFactItem(role_label="", source_line="line", fact_text="fact"),
            CvFactItem(role_label=ROLE, source_line="  ", fact_text="fact"),
            CvFactItem(role_label=ROLE, source_line="line", fact_text=""),
        ]
        assert to_proposed_facts(items) == []

    def test_an_implausibly_long_fact_is_dropped_rather_than_cut(self) -> None:
        """A cut fact is a different fact, and this one is about to be shown to
        its subject as their own words.
        """
        items = [
            CvFactItem(role_label=ROLE, source_line="line", fact_text="x" * (MAX_FACT_CHARS + 1))
        ]
        assert to_proposed_facts(items) == []

    def test_the_sent_document_is_carried_through(self) -> None:
        document = uuid.uuid4()
        facts = to_proposed_facts(
            [CvFactItem(role_label=ROLE, source_line="line", fact_text="fact")],
            sent_document_id=document,
        )
        assert facts[0].sent_document_id == document

    def test_ordinals_follow_the_cvs_order(self) -> None:
        items = [
            CvFactItem(role_label=ROLE, source_line="a", fact_text="First."),
            CvFactItem(role_label=ROLE, source_line="b", fact_text="Second."),
        ]
        assert [f.ordinal for f in to_proposed_facts(items)] == [0, 1]


# --- the call ------------------------------------------------------------------


class TestExtractCvFacts:
    def test_a_successful_read_returns_the_facts_and_records_one_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(_response(_PAYLOAD))
        _patch_client(monkeypatch, client)
        runs = _FakeRunRepo()

        facts = extract_cv_facts(_ctx(), runs, cv_text=CV_TEXT, now=NOW)

        assert [f.fact_text for f in facts] == [
            "Led a team of 8 engineers.",
            "Owned the payments platform.",
            "Wrote Python and Go.",
        ]
        assert len(runs.recorded) == 1
        run = runs.recorded[0]
        assert (run.component, run.stage, run.outcome) == ("generate", "extract_cv_facts", "ok")
        assert run.tokens_in == 900 and run.tokens_out == 400

    def test_the_cv_goes_in_the_user_message_and_the_clock_in_the_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLAUDE.md's 2026-09-07 decision: every model call is told what time
        it is. A CV is full of relative dates and an extraction that cannot
        locate itself in time is guessing.
        """
        client = _FakeAnthropicClient(_response(_PAYLOAD))
        _patch_client(monkeypatch, client)

        extract_cv_facts(_ctx(), _FakeRunRepo(), cv_text=CV_TEXT, now=NOW)

        call = client.messages.calls[0]
        assert call["system"] == build_cv_facts_prompt(now=NOW)
        assert NOW.isoformat() in call["system"]
        assert call["messages"] == [{"role": "user", "content": CV_TEXT}]
        assert call["max_tokens"] == MAX_TOKENS
        assert call["output_config"]["format"]["schema"] == CV_FACTS_OUTPUT_SCHEMA

    def test_a_long_cv_is_cut_to_the_read_ceiling_before_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CV is stored whole and read in part -- `jfl_core.cv_limits`. The cut
        happens here because this is the line above a call billed to the user's
        own key, and no caller can forget it.
        """
        client = _FakeAnthropicClient(_response(_PAYLOAD))
        _patch_client(monkeypatch, client)
        long_cv = "Led a platform team of eight engineers.\n" * 5_000
        assert len(long_cv) > MAX_CV_READ_CHARS

        extract_cv_facts(_ctx(), _FakeRunRepo(), cv_text=long_cv, now=NOW)

        sent = client.messages.calls[0]["messages"][0]["content"]
        assert len(sent) <= MAX_CV_READ_CHARS
        assert long_cv.startswith(sent)

    def test_a_cv_inside_the_read_ceiling_is_sent_whole(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(_response(_PAYLOAD))
        _patch_client(monkeypatch, client)

        extract_cv_facts(_ctx(), _FakeRunRepo(), cv_text=CV_TEXT, now=NOW)

        assert client.messages.calls[0]["messages"][0]["content"] == CV_TEXT

    def test_an_empty_cv_never_reaches_the_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeAnthropicClient(_response(_PAYLOAD))
        _patch_client(monkeypatch, client)
        with pytest.raises(GenerateError, match="no text found in the CV"):
            extract_cv_facts(_ctx(), _FakeRunRepo(), cv_text="   \n ", now=NOW)
        assert client.messages.calls == []

    def test_a_refusal_is_recorded_as_refused_and_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(
            _response(
                {"facts": []},
                stop_reason="refusal",
                stop_details=RefusalStopDetails(type="refusal", category="cyber"),
            )
        )
        _patch_client(monkeypatch, client)
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="model refused to respond"):
            extract_cv_facts(_ctx(), runs, cv_text=CV_TEXT, now=NOW)

        assert [r.outcome for r in runs.recorded] == ["refused"]

    def test_truncation_says_the_cv_is_too_long_and_still_records_the_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(_response({"facts": []}, stop_reason="max_tokens"))
        _patch_client(monkeypatch, client)
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="model output was truncated"):
            extract_cv_facts(_ctx(), runs, cv_text=CV_TEXT, now=NOW)

        assert [r.outcome for r in runs.recorded] == ["error"]
        assert runs.recorded[0].cost_usd is not None

    def test_an_api_error_still_writes_exactly_one_run_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(
            exception=anthropic.RateLimitError(
                "slow down",
                response=httpx2.Response(
                    429, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
                ),
                body=None,
            )
        )
        _patch_client(monkeypatch, client)
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="rate_limited"):
            extract_cv_facts(_ctx(), runs, cv_text=CV_TEXT, now=NOW)

        assert len(runs.recorded) == 1
        assert runs.recorded[0].outcome == "error"

    def test_unparseable_output_is_an_error_run_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = _response({"facts": []})
        message.content = [TextBlock(type="text", text="not json")]
        _patch_client(monkeypatch, _FakeAnthropicClient(message))
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="could not parse structured output"):
            extract_cv_facts(_ctx(), runs, cv_text=CV_TEXT, now=NOW)

        assert [r.outcome for r in runs.recorded] == ["error"]
