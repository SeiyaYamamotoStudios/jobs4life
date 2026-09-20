"""What one scoring call is actually told -- PLAN.md B4's context builder.

No model, no network, no database: these assemble the prompt and read it back.
The properties worth pinning are the ones a defect would be invisible in --
an unanswered profile question silently becoming a guess, two objectives being
merged, an unconfirmed CV claim reaching the prompt as though it were evidence,
or a schema property called `reason` coming back into existence.
"""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import (
    Job,
    JobRequirement,
    ProfileAnswer,
    ProfileObjective,
    ProfileRuledOut,
    RequirementCoverage,
)
from jfl_core.profile_questions import QUESTION_KEYS, QUESTIONS
from jfl_generate.prompts import (
    NOT_STATED,
    SCORE_OUTPUT_SCHEMA,
    ProposedFactView,
    ScoreInputs,
    build_score_system_blocks,
    build_score_system_prompt,
    build_score_user_message,
    unanswered_questions,
)

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
NOW = dt.datetime(2026, 9, 20, 11, 30, tzinfo=dt.UTC)


def _job(**kw: object) -> Job:
    fields: dict[str, object] = {
        "id": uuid.uuid4(),
        "user_id": USER,
        "source": "paste",
        "employer": "Northwind",
        "title": "Engineering Manager",
        "location": "Manchester, on-site",
        "raw_text": "We are hiring an engineering manager. On site five days a week.",
        "content_hash": "0" * 64,
    }
    fields.update(kw)
    return Job.model_validate(fields)


def _requirement(text: str, ordinal: int = 0, necessity: str = "essential") -> JobRequirement:
    return JobRequirement(
        id=uuid.uuid4(),
        user_id=USER,
        job_id=uuid.uuid4(),
        ordinal=ordinal,
        text=text,
        necessity=necessity,  # type: ignore[arg-type]
    )


def _coverage(requirement: JobRequirement, status: str, note: str) -> RequirementCoverage:
    return RequirementCoverage(
        user_id=USER,
        requirement_id=requirement.id,
        trace_id=uuid.uuid4(),
        status=status,  # type: ignore[arg-type]
        cited_span_ids=[],
        evidence_note=note,
    )


def _answer(key: str, text: str, structured: dict[str, object] | None = None) -> ProfileAnswer:
    return ProfileAnswer(
        id=uuid.uuid4(),
        question_key=key,  # type: ignore[arg-type]
        answer_text=text,
        structured=structured,
        created_at=NOW,
    )


def _objective(ordinal: int, what: str, evidence: str) -> ProfileObjective:
    return ProfileObjective(
        id=uuid.uuid4(),
        ordinal=ordinal,
        objective_text=what,
        evidence_text=evidence,
        created_at=NOW,
    )


def _inputs(**kw: object) -> ScoreInputs:
    requirement = _requirement("Five years of Python")
    defaults: dict[str, object] = {
        "job": _job(),
        "requirements": [requirement],
        "coverage": [_coverage(requirement, "evidenced", "Three roles document it.")],
        "answers": {},
    }
    defaults.update(kw)
    return ScoreInputs(**defaults)  # type: ignore[arg-type]


# -- the instructions ---------------------------------------------------------


class TestInstructions:
    def test_the_system_prompt_is_constant_so_it_caches(self) -> None:
        """Everything volatile is in the user message, so this block is
        byte-identical across every scoring call any user makes.
        """
        assert build_score_system_prompt() == build_score_system_prompt()
        blocks = build_score_system_blocks()
        assert len(blocks) == 1
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert blocks[0]["text"] == build_score_system_prompt()

    def test_it_forbids_combining_the_two_axes_in_so_many_words(self) -> None:
        prompt = build_score_system_prompt()
        assert "Never combine, average" in prompt
        assert "never return a third number" in prompt

    def test_it_tells_the_model_not_to_guess_an_unanswered_question(self) -> None:
        assert f'"{NOT_STATED}"' in build_score_system_prompt()

    def test_it_says_unconfirmed_claims_are_not_evidence(self) -> None:
        prompt = build_score_system_prompt()
        assert "not evidence" in prompt
        assert "did not count towards could_get_score" in prompt


# -- the schema ---------------------------------------------------------------


def _property_names(node: object) -> set[str]:
    names: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                names |= set(value)
            names |= _property_names(value)
    elif isinstance(node, list):
        for item in node:
            names |= _property_names(item)
    return names


class TestSchema:
    def test_no_property_is_named_reason_anywhere_in_it(self) -> None:
        """CLAUDE.md's 2026-09-02 decision: a long labelling prompt plus a
        schema demanding a label and a `reason` per item reads to the API as a
        distillation harvest, and every call refuses. The paragraph here is an
        `assessment`.
        """
        assert "reason" not in _property_names(SCORE_OUTPUT_SCHEMA)

    def test_it_carries_two_scores_and_no_composite(self) -> None:
        properties = SCORE_OUTPUT_SCHEMA["properties"]
        assert isinstance(properties, dict)
        assert "could_get_score" in properties
        assert "want_it_score" in properties
        for forbidden in ("overall", "overall_score", "combined", "composite", "average"):
            assert forbidden not in properties

    def test_a_lever_names_a_fact_by_index_never_by_text(self) -> None:
        """So a lever cannot quietly paraphrase what the user's CV said."""
        properties = SCORE_OUTPUT_SCHEMA["properties"]
        assert isinstance(properties, dict)
        lever_properties = properties["levers"]["items"]["properties"]
        assert set(lever_properties) == {"fact_index", "would_move_to", "note"}


# -- profile answers ----------------------------------------------------------


class TestProfileAnswers:
    def test_every_question_appears_with_the_answer_given(self) -> None:
        answers = {
            "location_commute": _answer("location_commute", "Sheffield, one day a week at most"),
            "levels": _answer("levels", "EM or above", {"levels": ["em", "above_em"]}),
        }
        message = build_score_user_message(_inputs(answers=answers))
        assert "Sheffield, one day a week at most" in message
        assert '"levels": ["em", "above_em"]' in message
        for question in QUESTIONS:
            assert question.wording in message

    def test_an_unanswered_question_is_reported_not_stated_and_never_guessed(self) -> None:
        answers = {"location_commute": _answer("location_commute", "Sheffield")}
        message = build_score_user_message(_inputs(answers=answers))
        lines = message.splitlines()
        for question in QUESTIONS:
            if question.key == "location_commute":
                continue
            index = lines.index(f"- {question.wording}")
            assert lines[index + 1].strip() == NOT_STATED

    def test_a_cleared_answer_counts_as_unanswered(self) -> None:
        """The storage layer keeps a blank row to record that a previous answer
        was cleared. A cleared answer is not an answer.
        """
        answers = {"trajectory": _answer("trajectory", "   ")}
        assert "trajectory" in {q.key for q in unanswered_questions(answers)}

    def test_unanswered_questions_lists_every_question_when_nothing_is_answered(self) -> None:
        assert tuple(q.key for q in unanswered_questions({})) == QUESTION_KEYS


# -- objectives ---------------------------------------------------------------


class TestObjectives:
    def test_each_objective_is_rendered_separately_with_its_ordinal(self) -> None:
        objectives = [
            _objective(1, "Get back to hands-on platform work", "A team that ships weekly"),
            _objective(2, "Stop commuting", "Two days a month at most"),
        ]
        message = build_score_user_message(_inputs(objectives=objectives))
        assert "- ordinal 1" in message
        assert "- ordinal 2" in message
        assert "Get back to hands-on platform work" in message
        assert "Stop commuting" in message
        # Neither objective's text has been folded into the other's block.
        first = message.index("- ordinal 1")
        second = message.index("- ordinal 2")
        assert "Stop commuting" not in message[first:second]

    def test_no_objectives_reads_as_not_stated(self) -> None:
        message = build_score_user_message(_inputs(objectives=[]))
        heading = "## Objectives for this move, each to be judged on its own"
        after = message[message.index(heading) + len(heading) :]
        assert after.lstrip().startswith(NOT_STATED)


# -- unconfirmed CV claims ----------------------------------------------------


class TestProposedFacts:
    def test_they_are_numbered_and_labelled_as_not_evidence(self) -> None:
        facts = [
            ProposedFactView(fact_text="Ran a team of 12", role_label="Northwind, EM"),
            ProposedFactView(fact_text="Owned the FX pricing platform", role_label="Contoso"),
        ]
        message = build_score_user_message(_inputs(proposed_facts=facts))
        assert "NOT evidence" in message
        assert "1. [Northwind, EM] Ran a team of 12" in message
        assert "2. [Contoso] Owned the FX pricing platform" in message

    def test_none_stored_means_no_levers_to_offer(self) -> None:
        message = build_score_user_message(_inputs(proposed_facts=[]))
        heading = "## Unconfirmed claims from this person's own CVs -- NOT evidence"
        assert "(none)" in message[message.index(heading) :]


# -- requirements, coverage, time ---------------------------------------------


class TestJobContext:
    def test_each_requirement_carries_its_recorded_coverage_verdict(self) -> None:
        one = _requirement("Five years of Python", 0)
        two = _requirement("Kubernetes in production", 1, "desirable")
        inputs = _inputs(
            requirements=[one, two],
            coverage=[
                _coverage(one, "evidenced", "Three roles document it."),
                _coverage(two, "absent", "The corpus is silent on Kubernetes."),
            ],
        )
        message = build_score_user_message(inputs)
        assert "1. [essential] Five years of Python" in message
        assert "coverage: evidenced -- Three roles document it." in message
        assert "2. [desirable] Kubernetes in production" in message
        assert "coverage: absent -- The corpus is silent on Kubernetes." in message

    def test_a_requirement_with_no_coverage_row_says_so_rather_than_guessing(self) -> None:
        one = _requirement("Five years of Python")
        message = build_score_user_message(_inputs(requirements=[one], coverage=[]))
        assert "coverage: not checked -- (no coverage recorded)" in message

    def test_the_call_is_told_what_time_it_is(self) -> None:
        """CLAUDE.md, 2026-09-07: every model call is told what time it is --
        a judgement about notice periods or start dates is guessing without it.
        """
        message = build_score_user_message(_inputs(now=NOW))
        assert NOW.isoformat() in message

    def test_a_ruled_out_decision_that_was_reopened_is_left_out(self) -> None:
        ruled_out = [
            ProfileRuledOut(id=uuid.uuid4(), decision_text="No more agency work", recorded_at=NOW),
            ProfileRuledOut(
                id=uuid.uuid4(),
                decision_text="No more startups",
                recorded_at=NOW,
                reopened_at=NOW,
            ),
        ]
        message = build_score_user_message(_inputs(ruled_out=ruled_out))
        assert "No more agency work" in message
        assert "No more startups" not in message
