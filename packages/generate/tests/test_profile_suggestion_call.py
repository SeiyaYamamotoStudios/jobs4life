"""Unit tests for `suggest_profile_settings`. Mirrors
`test_capability_clustering.py`'s approach: `anthropic.Anthropic` is
monkeypatched to a fake client and the run repository is a trivial in-memory
stand-in. No live API, no database.

The interesting half of this module is `to_proposals`, because it holds the two
guarantees a prompt cannot give -- a kind outside the whitelist never becomes a
proposal, and a quote that is in none of the CVs we sent takes its whole
suggestion with it. The first is what makes "never propose comp, contract type,
right to work, notice or a categorical no" structural rather than a rule
someone has to remember; the second is what makes every proposal's evidence
real. Those get the bulk of the tests; the call itself gets the same `runs`-row
coverage every other call site has.
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
from jfl_core.ids import setting_key
from jfl_core.models import RunRecord
from jfl_generate.errors import GenerateError
from jfl_generate.profile_suggestions import (
    MAX_CV_CHARS,
    MAX_CVS,
    MAX_SUGGESTIONS,
    MAX_VALUE_CHARS,
    MODEL,
    build_cvs_message,
    select_cv_texts,
    suggest_profile_settings,
    to_proposals,
)
from jfl_generate.prompts import (
    PROFILE_SUGGESTIONS_OUTPUT_SCHEMA,
    build_profile_suggestions_prompt,
)
from jfl_generate.schema import ProposedSettingItem

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 22, 9, 30, tzinfo=UTC)

CV = """\
Jo Smith -- Engineering Manager

Northwind, London, 2021-2024
Engineering manager for the payments platform.
Ran platform engineering across three squads.
I no longer work on frontend.

Acme Ltd, Bristol, 2018-2021
Senior software engineer on FX pricing.
"""


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


def _item(kind: str, value: str, source_line: str) -> ProposedSettingItem:
    return ProposedSettingItem(kind=kind, value=value, source_line=source_line)


def _response(
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
            input_tokens=4000,
            output_tokens=120,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeAnthropicClient) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)


# --- the schema ---------------------------------------------------------------


def test_schema_has_no_property_named_reason() -> None:
    """CLAUDE.md's 2026-09-02 decision: a schema property named `reason`,
    combined with a labelling system prompt, has tripped the API's
    reverse-engineering/duplication classifier before -- and this call is
    exactly that shape, a labelling prompt over a list of items.
    """

    def _names(node: object) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    found.update(value)
                found.update(_names(value))
        elif isinstance(node, list):
            for item in node:
                found.update(_names(item))
        return found

    assert "reason" not in _names(PROFILE_SUGGESTIONS_OUTPUT_SCHEMA)


def test_schema_asks_for_a_kind_a_value_and_the_cv_line_and_nothing_else() -> None:
    item = PROFILE_SUGGESTIONS_OUTPUT_SCHEMA["properties"]["suggestions"]["items"]  # type: ignore[index]
    assert set(item["properties"]) == {"kind", "value", "source_line"}
    assert item["additionalProperties"] is False


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_is_told_what_time_it_is() -> None:
    """CLAUDE.md's 2026-09-07 decision -- every model call is told what time it
    is, and `now` is the caller's clock rather than one read in here.
    """
    prompt = build_profile_suggestions_prompt(max_suggestions=30, now=NOW)
    assert NOW.isoformat() in prompt


def test_the_prompt_names_the_four_kinds_and_forbids_the_rest() -> None:
    prompt = build_profile_suggestions_prompt(max_suggestions=30, now=NOW)
    for kind in ("discipline", "not_discipline", "location", "level"):
        assert f"- {kind}:" in prompt
    assert "Use only those four kinds." in prompt
    assert "Do not suggest pay, contract type, right to work, notice period" in prompt


def test_the_prompt_says_a_level_is_an_observation_not_a_floor() -> None:
    prompt = build_profile_suggestions_prompt(max_suggestions=30, now=NOW)
    assert "never phrased as a requirement, a floor or a minimum" in prompt


def test_the_prompt_refuses_a_not_this_read_off_an_absence() -> None:
    prompt = build_profile_suggestions_prompt(max_suggestions=30, now=NOW)
    assert "silence is not a statement" in prompt


def test_the_prompt_carries_the_ceiling_it_is_actually_given() -> None:
    assert "at most 7 suggestions" in build_profile_suggestions_prompt(max_suggestions=7, now=NOW)


def test_the_cvs_message_carries_every_cv_verbatim() -> None:
    message = build_cvs_message([CV, "Another CV entirely."])
    assert CV in message
    assert "Another CV entirely." in message
    assert "--- CV 1 ---" in message and "--- CV 2 ---" in message


# --- which CVs one call reads -------------------------------------------------


def test_no_more_cvs_than_the_cap() -> None:
    chosen = select_cv_texts([f"CV number {i}" for i in range(MAX_CVS + 5)])
    assert len(chosen) == MAX_CVS


def test_a_cv_that_does_not_fit_is_left_out_whole_never_cut() -> None:
    """Half a CV quotes lines that are not in it and reads as a document its
    author never wrote.
    """
    chosen = select_cv_texts(["x" * (MAX_CV_CHARS + 1), "a short one"])
    assert chosen == ["a short one"]


def test_an_empty_cv_is_skipped() -> None:
    assert select_cv_texts(["   \n ", CV]) == [CV]


# --- the whitelist, and therefore the exclusions ------------------------------


@pytest.mark.parametrize(
    "kind",
    ["comp_floor", "contract", "right_to_work", "notice", "categorical_no", "workplace"],
)
def test_a_forbidden_kind_is_dropped_even_when_the_model_returns_it(kind: str) -> None:
    """The exclusions are the complement of a whitelist, so they hold however
    the model answers. A guessed constraint of any of these kinds would be read
    by scoring as the user's own requirement.
    """
    items = [
        _item(kind, "£120,000", "Engineering manager for the payments platform."),
        _item("discipline", "engineering management", "Engineering manager for the payments"),
    ]
    proposals = to_proposals(items, cv_texts=[CV])
    assert [p.kind for p in proposals] == ["discipline"]


def test_an_unknown_kind_is_dropped() -> None:
    assert to_proposals([_item("vibes", "good", "Jo Smith")], cv_texts=[CV]) == []


def test_the_kind_is_matched_case_insensitively() -> None:
    proposals = to_proposals(
        [_item("Discipline", "platform engineering", "Ran platform engineering")], cv_texts=[CV]
    )
    assert [p.kind for p in proposals] == ["discipline"]


# --- fabricated and malformed proposals ---------------------------------------


def test_a_source_line_that_is_in_no_cv_takes_its_suggestion_with_it() -> None:
    """Definitional, not statistical: we know exactly what text went into the
    prompt, so "is this quote in it" is a lookup rather than a guess.
    """
    items = [
        _item("discipline", "quantum cryptography", "Led the quantum cryptography group."),
        _item("discipline", "platform engineering", "Ran platform engineering across three"),
    ]
    proposals = to_proposals(items, cv_texts=[CV])
    assert [p.values[0] for p in proposals] == ["platform engineering"]


def test_a_not_this_read_off_an_absence_has_no_line_to_quote_and_is_dropped() -> None:
    items = [_item("not_discipline", "frontend", "")]
    assert to_proposals(items, cv_texts=[CV]) == []


def test_a_not_this_the_cv_actually_states_survives() -> None:
    items = [_item("not_discipline", "frontend", "I no longer work on frontend.")]
    proposals = to_proposals(items, cv_texts=[CV])
    assert [(p.kind, p.values[0]) for p in proposals] == [("not_discipline", "frontend")]


def test_a_blank_value_is_dropped() -> None:
    assert to_proposals([_item("discipline", "   ", "Jo Smith")], cv_texts=[CV]) == []


def test_an_over_long_value_is_dropped_rather_than_truncated() -> None:
    long = "x" * (MAX_VALUE_CHARS + 1)
    assert to_proposals([_item("discipline", long, "Jo Smith")], cv_texts=[CV]) == []


def test_a_value_is_trimmed_and_its_whitespace_collapsed() -> None:
    proposals = to_proposals(
        [_item("discipline", "  platform   engineering ", "Ran platform engineering")],
        cv_texts=[CV],
    )
    assert proposals[0].values == ["platform engineering"]


# --- shaping ------------------------------------------------------------------


def test_locations_collapse_into_one_ordered_proposal() -> None:
    """The profile's location constraint is a single ordered list, so accepting
    it once is what the user actually wants to do. Order is the model's, which
    the prompt asks to be most recent first.
    """
    items = [
        _item("location", "London", "Northwind, London, 2021-2024"),
        _item("location", "Bristol", "Acme Ltd, Bristol, 2018-2021"),
    ]
    proposals = to_proposals(items, cv_texts=[CV])
    assert len(proposals) == 1
    assert proposals[0].kind == "location"
    assert proposals[0].values == ["London", "Bristol"]
    assert len(proposals[0].source_lines) == 2


def test_a_repeated_place_appears_once() -> None:
    items = [
        _item("location", "London", "Northwind, London, 2021-2024"),
        _item("location", "london", "Northwind, London, 2021-2024"),
    ]
    proposals = to_proposals(items, cv_texts=[CV])
    assert proposals[0].values == ["London"]


def test_only_one_level_observation_survives() -> None:
    """The profile holds one level setting, and two observations would be a
    choice nobody asked for.
    """
    items = [
        _item("level", "has been operating at engineering-manager level", "Jo Smith"),
        _item("level", "has been operating at director level", "Jo Smith"),
    ]
    proposals = to_proposals(items, cv_texts=[CV])
    assert [p.values[0] for p in proposals] == ["has been operating at engineering-manager level"]


def test_two_values_that_fold_to_one_are_deduplicated() -> None:
    items = [
        _item("discipline", "Platform Engineering", "Ran platform engineering"),
        _item("discipline", "platform  engineering", "Ran platform engineering"),
    ]
    assert len(to_proposals(items, cv_texts=[CV])) == 1


def test_no_more_proposals_than_the_ceiling() -> None:
    items = [
        _item("discipline", f"discipline {i}", "Engineering manager for the payments platform.")
        for i in range(MAX_SUGGESTIONS + 10)
    ]
    assert len(to_proposals(items, cv_texts=[CV])) == MAX_SUGGESTIONS


def test_every_proposal_carries_the_cv_words_behind_it() -> None:
    items = [_item("discipline", "platform engineering", "Ran platform engineering")]
    proposals = to_proposals(items, cv_texts=[CV])
    assert proposals[0].source_lines == ["Ran platform engineering"]


# --- a rejection sticks -------------------------------------------------------


def test_a_suggestion_already_answered_is_never_proposed_again() -> None:
    """The key is content-derived, so the same suggestion from the same CV --
    or from a later one saying the same thing -- folds to the same key.
    """
    answered = [setting_key("discipline", ["platform engineering"])]
    items = [_item("discipline", "Platform Engineering", "Ran platform engineering")]
    assert to_proposals(items, cv_texts=[CV], answered_keys=answered) == []


def test_an_answered_location_list_is_not_proposed_again() -> None:
    answered = [setting_key("location", ["London", "Bristol"])]
    items = [
        _item("location", "London", "Northwind, London, 2021-2024"),
        _item("location", "Bristol", "Acme Ltd, Bristol, 2018-2021"),
    ]
    assert to_proposals(items, cv_texts=[CV], answered_keys=answered) == []


# --- the call itself ----------------------------------------------------------

_PAYLOAD = {
    "suggestions": [
        {
            "kind": "discipline",
            "value": "engineering management",
            "source_line": "Engineering manager for the payments platform.",
        },
        {
            "kind": "comp_floor",
            "value": "£150,000",
            "source_line": "Engineering manager for the payments platform.",
        },
    ]
}


def test_a_successful_call_writes_exactly_one_runs_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAnthropicClient(_response(_PAYLOAD))
    _patch_client(monkeypatch, client)
    runs = _FakeRunRepo()

    items = suggest_profile_settings(_ctx(), runs, cv_texts=[CV], now=NOW)

    assert [item.kind for item in items] == ["discipline", "comp_floor"]
    assert len(runs.recorded) == 1
    row = runs.recorded[0]
    assert row.component == "generate"
    assert row.stage == "suggest_profile_settings"
    assert row.model == MODEL
    assert row.outcome == "ok"
    assert row.cost_usd is not None


def test_the_cv_text_is_what_goes_over_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeAnthropicClient(_response(_PAYLOAD))
    _patch_client(monkeypatch, client)
    suggest_profile_settings(_ctx(), _FakeRunRepo(), cv_texts=[CV], now=NOW)
    sent = client.messages.calls[0]["messages"][0]["content"]
    assert CV in sent


def test_the_call_always_uses_the_cheap_model_never_the_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAnthropicClient(_response(_PAYLOAD))
    _patch_client(monkeypatch, client)
    ctx = RequestContext(
        user_id=USER,
        anthropic_api_key="k",
        database_url="unused",
        model="claude-opus-5",
    )
    suggest_profile_settings(ctx, _FakeRunRepo(), cv_texts=[CV], now=NOW)
    assert client.messages.calls[0]["model"] == MODEL == "claude-haiku-4-5"


def test_an_api_error_still_writes_a_runs_row(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = _FakeAnthropicClient(
        exception=anthropic.AuthenticationError(
            "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
        )
    )
    _patch_client(monkeypatch, client)
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError):
        suggest_profile_settings(_ctx(), runs, cv_texts=[CV], now=NOW)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "error"
    assert runs.recorded[0].error is not None
    assert runs.recorded[0].error.startswith("authentication_error")


def test_a_refusal_still_writes_a_runs_row(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeAnthropicClient(
        _response(
            {"suggestions": []},
            stop_reason="refusal",
            stop_details=RefusalStopDetails(type="refusal", category="reasoning_extraction"),
        )
    )
    _patch_client(monkeypatch, client)
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError, match="model refused"):
        suggest_profile_settings(_ctx(), runs, cv_texts=[CV], now=NOW)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "refused"


def test_unparseable_output_still_writes_a_runs_row(monkeypatch: pytest.MonkeyPatch) -> None:
    message = _response({"suggestions": []})
    message.content = [TextBlock(type="text", text="not json at all")]
    _patch_client(monkeypatch, _FakeAnthropicClient(message))
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError, match="could not parse"):
        suggest_profile_settings(_ctx(), runs, cv_texts=[CV], now=NOW)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "error"


def test_no_cv_text_never_reaches_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call that would return nothing is not one worth charging for. No fake
    client is installed: the conftest guard would raise if this reached the SDK.
    """
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="no CV text"):
        suggest_profile_settings(_ctx(), runs, cv_texts=["  "], now=NOW)
    assert runs.recorded == []
