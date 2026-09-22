"""Unit tests for `cluster_capabilities`. Mirrors `test_titles.py`'s approach:
`anthropic.Anthropic` is monkeypatched to a fake client and the run repository
is a trivial in-memory stand-in. No live API, no database.

The interesting half of this module is `sanitise`, because it holds the three
guarantees a prompt cannot give -- an invented id never reaches storage, a fact
belongs to at most one capability, and a label that is just a role we sent is
refused. Those get the bulk of the tests; the call itself gets the same
`runs`-row coverage every other call site has.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.models import CandidateFact, RunRecord
from jfl_generate.capabilities import (
    MAX_CAPABILITIES,
    MAX_LABEL_CHARS,
    MODEL,
    build_facts_message,
    cluster_capabilities,
    fact_wire_id,
    sanitise,
    unplaced_facts,
)
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import (
    CAPABILITY_CLUSTER_OUTPUT_SCHEMA,
    build_capability_cluster_prompt,
)
from jfl_generate.schema import ClusteredCapabilityItem

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = datetime(2026, 9, 22, 9, 30, tzinfo=UTC)


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


def _fact(
    text: str = "Rebuilt the FX pricing platform",
    *,
    role: str = "Acme Ltd -- Head of Engineering",
    state: str = "confirmed",
    span_id: uuid.UUID | None = None,
    probe_answer: str | None = None,
) -> CandidateFact:
    when = dt.datetime(2026, 9, 1, tzinfo=UTC)
    return CandidateFact(
        id=uuid.uuid4(),
        user_id=USER,
        role_label=role,
        role_key=role.casefold(),
        source_line=text,
        fact_text=text,
        probe_answer=probe_answer,
        state=state,  # type: ignore[arg-type]
        span_id=span_id if span_id is not None or state != "confirmed" else uuid.uuid4(),
        fingerprint="f" * 64,
        created_at=when,
        updated_at=when,
    )


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
            input_tokens=400,
            output_tokens=80,
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
    exactly that shape, a long labelling prompt over a list of items.
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

    assert "reason" not in _names(CAPABILITY_CLUSTER_OUTPUT_SCHEMA)


def test_schema_asks_for_a_label_and_the_ids_it_covers_and_nothing_else() -> None:
    item = CAPABILITY_CLUSTER_OUTPUT_SCHEMA["properties"]["capabilities"]["items"]  # type: ignore[index]
    assert set(item["properties"]) == {"label", "fact_ids"}
    assert item["additionalProperties"] is False


# --- the prompt ---------------------------------------------------------------


def test_the_prompt_is_told_what_time_it_is() -> None:
    """CLAUDE.md's 2026-09-07 decision -- every model call is told what time it
    is, and `now` is the caller's clock rather than one read in here.
    """
    prompt = build_capability_cluster_prompt(max_capabilities=25, now=NOW)
    assert NOW.isoformat() in prompt


def test_the_prompt_forbids_role_titles_employers_and_invented_ids() -> None:
    prompt = build_capability_cluster_prompt(max_capabilities=25, now=NOW)
    assert "Never a job title, a seniority, or an employer's name." in prompt
    assert "Do not invent one." in prompt
    assert "A fact belongs to at most one capability." in prompt


def test_the_prompt_carries_the_ceiling_it_is_actually_given() -> None:
    assert "at most 7 capabilities" in build_capability_cluster_prompt(max_capabilities=7, now=NOW)


def test_the_facts_message_carries_every_fact_its_id_and_its_role() -> None:
    facts = [_fact("Ran the FX desk's pricing"), _fact("Hired six managers", role="Northwind")]
    message = build_facts_message(facts)
    for index, fact in enumerate(facts):
        assert fact_wire_id(index) in message
        assert fact.corpus_text in message
        assert fact.role_label in message


def test_a_probe_answer_travels_with_its_fact() -> None:
    """The answer is the half that matters -- "led how many?" -> "nine
    engineers" -- so it is in the text the model groups on, not only in the row.
    """
    fact = _fact("Led the platform team", probe_answer="nine engineers")
    assert "nine engineers" in build_facts_message([fact])


# --- sanitising ---------------------------------------------------------------


def _item(label: str, ids: list[str]) -> ClusteredCapabilityItem:
    return ClusteredCapabilityItem(label=label, fact_ids=ids)


def test_a_fabricated_fact_id_is_dropped() -> None:
    """The guarantee the prompt cannot give. An id we never sent is not
    resolved to the nearest fact -- it is dropped.
    """
    facts = [_fact()]
    proposals = sanitise([_item("FX pricing platforms", ["f1", "f99"])], facts)
    assert len(proposals) == 1
    assert proposals[0].fact_ids == [facts[0].id]
    assert proposals[0].span_ids == [facts[0].span_id]


def test_a_capability_whose_every_id_was_invented_is_dropped_entirely() -> None:
    proposals = sanitise([_item("Time travel", ["f40", "nonsense", ""])], [_fact()])
    assert proposals == []


def test_a_fact_belongs_to_at_most_one_capability() -> None:
    facts = [_fact("a"), _fact("b")]
    proposals = sanitise([_item("First thing", ["f1", "f2"]), _item("Second thing", ["f1"])], facts)
    assert [p.label for p in proposals] == ["First thing"]


def test_a_label_that_is_a_role_we_sent_is_rejected() -> None:
    """A role is not a capability. Definitional rather than statistical: the
    role labels are the strings we put in the prompt, so this is a lookup.
    """
    facts = [_fact(role="Acme Ltd -- Head of Engineering")]
    proposals = sanitise(
        [
            _item("Acme Ltd -- Head of Engineering", ["f1"]),
            _item("FX pricing platforms", ["f1"]),
        ],
        facts,
    )
    assert [p.label for p in proposals] == ["FX pricing platforms"]


def test_an_employer_half_of_a_role_label_is_rejected_too() -> None:
    facts = [_fact(role="Acme Ltd -- Head of Engineering")]
    assert sanitise([_item("Acme Ltd", ["f1"])], facts) == []
    assert sanitise([_item("head of engineering", ["f1"])], facts) == []


def test_a_capability_that_merely_contains_a_role_word_is_kept() -> None:
    """The counter-example the deleted heuristics would have failed on:
    "hiring engineering managers" is a capability, and a word list would flag it.
    """
    facts = [_fact(role="Acme Ltd -- Engineering Manager")]
    assert [p.label for p in sanitise([_item("Hiring engineering managers", ["f1"])], facts)] == [
        "Hiring engineering managers"
    ]


def test_a_label_is_trimmed_and_its_whitespace_collapsed() -> None:
    proposals = sanitise([_item("  FX   pricing\nplatforms ", ["f1"])], [_fact()])
    assert proposals[0].label == "FX pricing platforms"


def test_a_comma_in_a_label_becomes_a_space() -> None:
    """A comma reads as two capabilities wherever a list of them is rendered,
    the same hazard the title suggestions handle.
    """
    proposals = sanitise([_item("FX pricing, and rates", ["f1"])], [_fact()])
    assert proposals[0].label == "FX pricing and rates"


def test_an_over_long_label_is_dropped_rather_than_truncated() -> None:
    """A cut label is a different, wrong label."""
    assert sanitise([_item("x" * (MAX_LABEL_CHARS + 1), ["f1"])], [_fact()]) == []
    assert len(sanitise([_item("x" * MAX_LABEL_CHARS, ["f1"])], [_fact()])) == 1


def test_a_blank_label_is_dropped() -> None:
    assert sanitise([_item("   ", ["f1"])], [_fact()]) == []


def test_two_labels_that_fold_to_one_row_are_deduplicated() -> None:
    facts = [_fact("a"), _fact("b")]
    proposals = sanitise([_item("FX pricing", ["f1"]), _item("fx  Pricing", ["f2"])], facts)
    assert len(proposals) == 1


def test_no_more_capabilities_than_the_ceiling() -> None:
    facts = [_fact(f"fact {i}") for i in range(MAX_CAPABILITIES + 5)]
    items = [_item(f"Capability {i}", [f"f{i + 1}"]) for i in range(MAX_CAPABILITIES + 5)]
    assert len(sanitise(items, facts)) == MAX_CAPABILITIES


def test_a_fact_with_no_span_is_never_cited() -> None:
    """Only a confirmed fact carries a span, and a proposal with no span is a
    claim rather than evidence.
    """
    fact = _fact(state="proposed", span_id=None)
    assert sanitise([_item("Something", ["f1"])], [fact]) == []


def test_unplaced_facts_names_everything_no_proposal_covers() -> None:
    facts = [_fact("a"), _fact("b"), _fact("c")]
    proposals = sanitise([_item("First thing", ["f1", "f3"])], facts)
    assert unplaced_facts(facts, proposals) == [facts[1].id]


# --- the call -----------------------------------------------------------------

_PAYLOAD = {
    "capabilities": [
        {"label": "FX pricing platforms", "fact_ids": ["f1"]},
        {"label": "Hiring engineering managers", "fact_ids": ["f2"]},
    ]
}


def test_a_successful_call_writes_exactly_one_runs_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAnthropicClient(_response(_PAYLOAD))
    _patch_client(monkeypatch, client)
    runs = _FakeRunRepo()

    facts = [_fact("Ran FX pricing"), _fact("Hired six managers", role="Northwind")]
    proposals = cluster_capabilities(_ctx(), runs, facts=facts, now=NOW)

    assert [p.label for p in proposals] == ["FX pricing platforms", "Hiring engineering managers"]
    assert len(runs.recorded) == 1
    row = runs.recorded[0]
    assert row.component == "generate"
    assert row.stage == "cluster_capabilities"
    assert row.model == MODEL
    assert row.outcome == "ok"
    assert row.cost_usd is not None


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
    cluster_capabilities(ctx, _FakeRunRepo(), facts=[_fact()], now=NOW)
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
        cluster_capabilities(_ctx(), runs, facts=[_fact()], now=NOW)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "error"
    assert runs.recorded[0].error is not None
    assert runs.recorded[0].error.startswith("authentication_error")


def test_a_refusal_still_writes_a_runs_row(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeAnthropicClient(
        _response(
            {"capabilities": []},
            stop_reason="refusal",
            stop_details=RefusalStopDetails(type="refusal", category="reasoning_extraction"),
        )
    )
    _patch_client(monkeypatch, client)
    runs = _FakeRunRepo()

    with pytest.raises(GenerateError, match="model refused"):
        cluster_capabilities(_ctx(), runs, facts=[_fact()], now=NOW)

    assert len(runs.recorded) == 1
    assert runs.recorded[0].outcome == "refused"


def test_no_facts_never_reaches_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call that would return nothing is not one worth charging for. No fake
    client is installed: the conftest guard would raise if this reached the SDK.
    """
    runs = _FakeRunRepo()
    with pytest.raises(GenerateError, match="no confirmed facts"):
        cluster_capabilities(_ctx(), runs, facts=[], now=NOW)
    assert runs.recorded == []
