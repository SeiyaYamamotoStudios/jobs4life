"""Unit tests for `score_application`: the fixed control flow around one model
call, and the pure mapping from one response to two stored scores.

No live API and no database -- the client is a fake, and the root `conftest.py`
guard is what would raise if one of these reached the real one.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message, RefusalStopDetails, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.models import (
    Job,
    JobRequirement,
    RequirementCoverage,
    RunRecord,
)
from jfl_core.profile import Objective, Profile
from jfl_gate.pricing import MODEL
from jfl_generate.errors import GenerateError
from jfl_generate.prompts import ProposedFactView, ScoreInputs
from jfl_generate.schema import ScoreOutput
from jfl_generate.scoring import MAX_TOKENS, build_result, score_application

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = dt.datetime(2026, 9, 20, 11, 30, tzinfo=dt.UTC)


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


# --- helpers -----------------------------------------------------------------


def _ctx(api_key: str | None = "test-key") -> RequestContext:
    return RequestContext(user_id=USER, anthropic_api_key=api_key, database_url="unused")


def _job() -> Job:
    return Job(
        id=uuid.uuid4(),
        user_id=USER,
        source="paste",
        employer="Northwind",
        title="Engineering Manager",
        location="Manchester",
        raw_text="We are hiring an engineering manager.",
        content_hash="0" * 64,
    )


def _requirement(text: str = "Five years of Python") -> JobRequirement:
    return JobRequirement(
        id=uuid.uuid4(),
        user_id=USER,
        job_id=uuid.uuid4(),
        ordinal=0,
        text=text,
        necessity="essential",
    )


def _inputs(**kw: object) -> ScoreInputs:
    requirement = _requirement()
    defaults: dict[str, object] = {
        "job": _job(),
        "requirements": [requirement],
        "coverage": [
            RequirementCoverage(
                user_id=USER,
                requirement_id=requirement.id,
                trace_id=uuid.uuid4(),
                status="evidenced",
                cited_span_ids=[],
                evidence_note="Documented.",
            )
        ],
        "now": NOW,
    }
    defaults.update(kw)
    return ScoreInputs(**defaults)  # type: ignore[arg-type]


PAYLOAD: dict[str, Any] = {
    "could_get_score": 6,
    "could_get_assessment": "Your record evidences the Python and the team size. ",
    "want_it_score": 3,
    "want_it_assessment": "The commute breaks what you said you would travel.",
    "objective_verdicts": [],
    "hard_gate_breaches": [
        {"gate": "location", "breach": "On site five days a week in Manchester."}
    ],
    "levers": [],
}


def _response(
    payload: dict[str, Any] | str = PAYLOAD,
    *,
    stop_reason: str = "end_turn",
    stop_details: RefusalStopDetails | None = None,
) -> Message:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=text)],
        model=MODEL,
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_details=stop_details,
        type="message",
        usage=Usage(
            input_tokens=1200,
            output_tokens=400,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=800,
        ),
    )


def _patch(monkeypatch: pytest.MonkeyPatch, client: _FakeAnthropicClient) -> None:
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: client)


# --- the call ----------------------------------------------------------------


class TestTheCall:
    def test_it_returns_two_scores_and_writes_one_runs_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(_response())
        _patch(monkeypatch, client)
        runs = _FakeRunRepo()

        result = score_application(_ctx(), runs, _inputs())

        assert result.could_get_score == 6
        assert result.want_it_score == 3
        assert result.could_get_assessment == (
            "Your record evidences the Python and the team size."
        )
        assert len(runs.recorded) == 1
        row = runs.recorded[0]
        assert row.component == "generate"
        assert row.stage == "score"
        assert row.outcome == "ok"
        assert row.model == MODEL
        assert row.cost_usd is not None and row.cost_usd > 0

    def test_the_instructions_are_cached_and_the_context_is_not(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Everything volatile goes in `messages`; a byte of it in `system`
        would invalidate the cached prefix on every call.
        """
        client = _FakeAnthropicClient(_response())
        _patch(monkeypatch, client)
        score_application(_ctx(), _FakeRunRepo(), _inputs())

        sent = client.messages.calls[0]
        assert sent["max_tokens"] == MAX_TOKENS
        assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "Northwind" not in sent["system"][0]["text"]
        assert "Northwind" in sent["messages"][0]["content"]

    def test_a_job_with_no_requirements_never_reaches_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A number for "could I get this" from an ad nobody has read is a
        number with nothing behind it. No call, and so no `runs` row either.
        """
        client = _FakeAnthropicClient(_response())
        _patch(monkeypatch, client)
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="no requirements"):
            score_application(_ctx(), runs, _inputs(requirements=[], coverage=[]))

        assert client.messages.calls == []
        assert runs.recorded == []

    def test_a_refusal_is_recorded_before_it_is_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _FakeAnthropicClient(
            _response(
                stop_reason="refusal",
                stop_details=RefusalStopDetails(type="refusal", category="reasoning_extraction"),
            )
        )
        _patch(monkeypatch, client)
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="model refused to respond"):
            score_application(_ctx(), runs, _inputs())

        assert len(runs.recorded) == 1
        assert runs.recorded[0].outcome == "refused"
        # The money was spent, so the cost is attributed.
        assert runs.recorded[0].cost_usd is not None

    def test_an_api_error_is_recorded_before_it_is_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        client = _FakeAnthropicClient(
            exception=anthropic.AuthenticationError(
                "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
            )
        )
        _patch(monkeypatch, client)
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="authentication_error"):
            score_application(_ctx(), runs, _inputs())

        assert len(runs.recorded) == 1
        assert runs.recorded[0].outcome == "error"

    def test_truncated_output_is_recorded_and_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _FakeAnthropicClient(_response(stop_reason="max_tokens")))
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError, match="model output was truncated"):
            score_application(_ctx(), runs, _inputs())

        assert runs.recorded[0].outcome == "error"

    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            json.dumps({**PAYLOAD, "could_get_score": 0}),
            json.dumps({**PAYLOAD, "want_it_score": 11}),
            json.dumps({k: v for k, v in PAYLOAD.items() if k != "want_it_assessment"}),
        ],
    )
    def test_an_unusable_response_is_a_parse_failure_with_a_runs_row(
        self, monkeypatch: pytest.MonkeyPatch, payload: str
    ) -> None:
        """A score outside 1-10 is a parse failure, not a stored score: the
        CHECK constraint would otherwise reject it at INSERT, after the money
        had been spent and with nothing saying why.
        """
        _patch(monkeypatch, _FakeAnthropicClient(_response(payload)))
        runs = _FakeRunRepo()

        with pytest.raises(GenerateError):
            score_application(_ctx(), runs, _inputs())

        assert len(runs.recorded) == 1
        assert runs.recorded[0].outcome == "error"


# --- mapping one response onto two stored scores -----------------------------


def _objective(rank: int, what: str) -> Objective:
    return Objective(rank=rank, text=what)


def _output(**kw: Any) -> ScoreOutput:
    return ScoreOutput.model_validate({**PAYLOAD, **kw})


class TestBuildResult:
    def test_both_numbers_are_carried_through_untouched(self) -> None:
        result = build_result(_output(), _inputs())
        assert (result.could_get_score, result.want_it_score) == (6, 3)

    def test_a_breach_is_kept_as_written(self) -> None:
        result = build_result(_output(), _inputs())
        assert [(b.gate, b.breach) for b in result.hard_gate_breaches] == [
            ("location", "On site five days a week in Manchester.")
        ]

    def test_an_empty_breach_is_dropped_rather_than_shown_blank(self) -> None:
        output = _output(hard_gate_breaches=[{"gate": "comp", "breach": "  "}])
        assert build_result(output, _inputs()).hard_gate_breaches == []

    def test_each_objective_keeps_its_own_words_and_its_own_verdict(self) -> None:
        objectives = [_objective(1, "Back to hands-on work"), _objective(2, "Stop commuting")]
        output = _output(
            objective_verdicts=[
                {"ordinal": 2, "verdict": "Fully remote, so yes."},
                {"ordinal": 1, "verdict": "An EM role, so probably not."},
            ]
        )
        result = build_result(output, _inputs(profile=Profile(objectives=objectives)))
        assert [(v.ordinal, v.objective) for v in result.objective_verdicts] == [
            (1, "Back to hands-on work"),
            (2, "Stop commuting"),
        ]
        assert result.objective_verdicts[0].verdict == "An EM role, so probably not."

    def test_a_verdict_for_an_objective_the_user_never_wrote_is_dropped(self) -> None:
        output = _output(objective_verdicts=[{"ordinal": 4, "verdict": "Invented."}])
        result = build_result(
            output, _inputs(profile=Profile(objectives=[_objective(1, "Back to hands-on")]))
        )
        assert result.objective_verdicts == []

    def test_a_lever_carries_the_stored_fact_verbatim(self) -> None:
        facts = [ProposedFactView(fact_text="Ran a team of 12", role_label="Northwind")]
        output = _output(
            levers=[{"fact_index": 1, "would_move_to": 8, "note": "Covers requirement 1."}]
        )
        result = build_result(output, _inputs(proposed_facts=facts))
        assert len(result.levers) == 1
        lever = result.levers[0]
        assert lever.fact_text == "Ran a team of 12"
        assert lever.role_label == "Northwind"
        assert lever.would_move_to == 8

    def test_a_lever_naming_no_stored_fact_is_dropped(self) -> None:
        output = _output(levers=[{"fact_index": 9, "would_move_to": 8, "note": "Invented."}])
        result = build_result(output, _inputs(proposed_facts=[]))
        assert result.levers == []

    @pytest.mark.parametrize("would_move_to", [0, 11, 6, 2])
    def test_a_move_that_is_not_a_move_keeps_the_note_and_loses_the_number(
        self, would_move_to: int
    ) -> None:
        """Out of range, or not above the score it moves from. The note is
        still worth showing; the arithmetic is not.
        """
        facts = [ProposedFactView(fact_text="Ran a team of 12")]
        output = _output(
            levers=[{"fact_index": 1, "would_move_to": would_move_to, "note": "Covers it."}]
        )
        result = build_result(output, _inputs(proposed_facts=facts))
        assert result.levers[0].would_move_to is None
        assert result.levers[0].note == "Covers it."

    def test_unfilled_profile_sections_come_back_as_not_stated(self) -> None:
        profile = Profile(objectives=[_objective(1, "Back to hands-on work")])
        result = build_result(_output(), _inputs(profile=profile))
        keys = {n.question_key for n in result.not_stated}
        assert "objectives" not in keys
        assert "constraints" in keys
        assert "capabilities" in keys
        assert all(n.wording for n in result.not_stated)

    def test_the_result_carries_no_combined_number(self) -> None:
        result = build_result(_output(), _inputs())
        fields = set(vars(type(result)).get("__slots__", ()))
        for forbidden in ("overall", "combined", "composite", "average", "total"):
            assert not any(forbidden in field for field in fields)
