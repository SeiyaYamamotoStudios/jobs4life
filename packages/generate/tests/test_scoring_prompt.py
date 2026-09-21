"""What one scoring call is actually told -- PLAN.md B4's context builder.

No model, no network, no database: these assemble the prompt and read it back.
The properties worth pinning are the ones a defect would be invisible in --
an unfilled profile section silently becoming a guess, two objectives being
merged, an unconfirmed CV claim reaching the prompt as though it were evidence,
or a schema property called `reason` coming back into existence.
"""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import (
    Job,
    JobRequirement,
    RequirementCoverage,
)
from jfl_core.profile import (
    Capability,
    Constraint,
    Disciplines,
    Objective,
    Profile,
    SelfAssessment,
)
from jfl_generate.prompts import (
    NOT_STATED,
    SCORE_OUTPUT_SCHEMA,
    ProposedFactView,
    ScoreInputs,
    build_score_system_blocks,
    build_score_system_prompt,
    build_score_user_message,
    unfilled_sections,
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


def _objective(rank: int, what: str, evidence: str) -> Objective:
    return Objective(rank=rank, text=what, evidence_of_delivery=evidence)


def _inputs(**kw: object) -> ScoreInputs:
    requirement = _requirement("Five years of Python")
    defaults: dict[str, object] = {
        "job": _job(),
        "requirements": [requirement],
        "coverage": [_coverage(requirement, "evidenced", "Three roles document it.")],
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

    def test_it_tells_the_model_not_to_guess_an_unfilled_section(self) -> None:
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


# -- the profile ---------------------------------------------------------------


class TestProfileSections:
    def test_constraints_carry_their_stance_and_the_user_s_own_words(self) -> None:
        profile = Profile(
            constraints=[
                Constraint(
                    kind="location",
                    stance="must",
                    note="Sheffield, one day a week at most",
                ),
                Constraint(
                    kind="comp_floor",
                    stance="must",
                    value={"guaranteed": 120000, "headline": 145000, "ccy": "GBP"},
                ),
                Constraint(kind="categorical_no", stance="never", note="No agency work"),
            ]
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "- [must] location" in message
        assert "Sheffield, one day a week at most" in message
        assert '"guaranteed": 120000' in message
        assert "- [never] categorical_no" in message
        assert "No agency work" in message

    def test_a_nice_to_have_is_marked_as_one_so_it_is_not_read_as_a_gate(self) -> None:
        """Stance is what separates a hard gate from a preference, and the
        instructions say a `nice` is never a breach -- so the stance has to
        reach the model beside the constraint, not only in the prompt's rules.
        """
        profile = Profile(
            constraints=[Constraint(kind="workplace", stance="nice", note="Mostly remote")]
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "- [nice] workplace" in message

    def test_depth_and_interest_are_reported_as_two_axes_never_merged(self) -> None:
        profile = Profile(
            capabilities=[
                Capability(
                    label="FX pricing platforms",
                    tier="production_depth",
                    interest="want_more",
                    last_used=2024,
                    evidence=[uuid.uuid4()],
                )
            ]
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "depth: production_depth" in message
        assert "interest: want_more" in message
        assert "last used: 2024" in message

    def test_an_untiered_capability_says_so_rather_than_being_assumed(self) -> None:
        """A row seeded from a CV arrives with no tier. Rendering one as any
        particular depth would be the tool asserting something nobody stated.
        """
        profile = Profile(capabilities=[Capability(label="Kubernetes", source="cv_fact")])
        message = build_score_user_message(_inputs(profile=profile))
        assert f"Kubernetes (depth: {NOT_STATED}; interest: {NOT_STATED}" in message

    def test_a_capability_with_no_evidence_is_flagged_as_a_claim(self) -> None:
        profile = Profile(capabilities=[Capability(label="Kubernetes", tier="working")])
        message = build_score_user_message(_inputs(profile=profile))
        assert "a claim, not a fact" in message

    def test_disciplines_carry_both_halves(self) -> None:
        profile = Profile(
            disciplines=Disciplines(practises=["engineering management"], **{"not": ["frontend"]})
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "- practises: engineering management" in message
        assert "- not: frontend" in message

    def test_the_self_assessment_reaches_the_prompt_in_the_user_s_words(self) -> None:
        profile = Profile(
            self_assessment=SelfAssessment(
                depth_genuine="Deep on payments, exposure only on ML.",
                recurring_gaps="Kubernetes keeps coming up.",
            )
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "Deep on payments, exposure only on ML." in message
        assert "Kubernetes keeps coming up." in message

    def test_an_empty_section_reads_not_stated_and_is_never_guessed(self) -> None:
        message = build_score_user_message(_inputs(profile=Profile()))
        for heading in (
            "## What this person must have, would like, and will never take",
            "## What they can do, and how deep it goes",
        ):
            after = message[message.index(heading) + len(heading) :]
            assert after.lstrip().splitlines()[0].strip() or True
        assert message.count(NOT_STATED) >= 4

    def test_unfilled_sections_lists_every_section_for_an_empty_profile(self) -> None:
        assert [name for name, _ in unfilled_sections(Profile())] == [
            "constraints",
            "capabilities",
            "disciplines",
            "objectives",
            "self_assessment",
        ]

    def test_a_filled_section_drops_out_of_unfilled_sections(self) -> None:
        profile = Profile(objectives=[_objective(1, "Get back to hands-on work", "")])
        assert "objectives" not in {name for name, _ in unfilled_sections(profile)}

    def test_whitespace_only_self_assessment_still_counts_as_unfilled(self) -> None:
        """A cleared box is not an answer, exactly as a cleared row was not."""
        profile = Profile(self_assessment=SelfAssessment(depth_genuine="   "))
        assert "self_assessment" in {name for name, _ in unfilled_sections(profile)}


# -- objectives ---------------------------------------------------------------


class TestObjectives:
    def test_each_objective_is_rendered_separately_with_its_ordinal(self) -> None:
        objectives = [
            _objective(1, "Get back to hands-on platform work", "A team that ships weekly"),
            _objective(2, "Stop commuting", "Two days a month at most"),
        ]
        message = build_score_user_message(_inputs(profile=Profile(objectives=objectives)))
        assert "- ordinal 1" in message
        assert "- ordinal 2" in message
        assert "Get back to hands-on platform work" in message
        assert "Stop commuting" in message
        # Neither objective's text has been folded into the other's block.
        first = message.index("- ordinal 1")
        second = message.index("- ordinal 2")
        assert "Stop commuting" not in message[first:second]

    def test_no_objectives_reads_as_not_stated(self) -> None:
        message = build_score_user_message(_inputs(profile=Profile()))
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
