"""What one scoring call is actually told -- PLAN.md B4's context builder,
rebuilt on the 2026-09-21 profile.

No model, no network, no database: these assemble the prompt and read it back.
The properties worth pinning are the ones a defect would be invisible in -- a
constraint reaching the prompt without the stance that gives it its meaning, an
unfilled section silently becoming a guess, two objectives being merged, an
unevidenced claim reaching the prompt as though it were evidence, or a schema
property called `reason` coming back into existence.
"""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import FIT_VERDICTS, Job, JobRequirement, RequirementCoverage
from jfl_core.profile import (
    Capability,
    Constraint,
    Disciplines,
    Objective,
    Profile,
)
from jfl_generate.prompts import (
    NOT_STATED,
    SCORE_OUTPUT_SCHEMA,
    ProposedFactView,
    ScoreInputs,
    build_score_system_blocks,
    build_score_system_prompt,
    build_score_user_message,
    claimed_items,
    constraint_label,
    not_stated_sections,
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


def _constraint(kind: str, stance: str, **kw: object) -> Constraint:
    return Constraint.model_validate({"kind": kind, "stance": stance, **kw})


def _capability(label: str, tier: str, evidence: list[uuid.UUID] | None = None) -> Capability:
    return Capability.model_validate({"label": label, "tier": tier, "evidence": evidence or []})


def _objective(rank: int, text: str, evidence: str = "") -> Objective:
    return Objective(rank=rank, text=text, evidence_of_delivery=evidence)


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

    def test_it_says_the_second_number_is_not_the_models_to_give(self) -> None:
        """The whole point of the 2026-09-21 change: the model gives verdicts,
        and `jfl_core.fit` derives the number from them.
        """
        prompt = build_score_system_prompt()
        assert 'You are not asked for a "do I want this" number.' in prompt
        assert "State no number: you are not given one." in prompt

    def test_it_names_all_four_verdicts_and_says_silence_is_a_question(self) -> None:
        prompt = build_score_system_prompt()
        for word in FIT_VERDICTS:
            assert f"`{word}`" in prompt
        assert "a silence is a question to ask at interview" in prompt
        assert "Never invent a breach out of a silence." in prompt

    def test_it_makes_the_tier_the_bridge_and_keeps_claims_out_of_evidence(self) -> None:
        prompt = build_score_system_prompt()
        assert "Only an evidenced capability counts as evidence." in prompt
        assert "did not count towards could_get_score" in prompt

    def test_it_asks_for_sentences_not_paragraphs(self) -> None:
        """The owner asked for a score plus one or two sentences per axis. The
        per-constraint verdicts carry the detail.
        """
        assert build_score_system_prompt().count("ONE OR TWO SENTENCES") == 2


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
        distillation harvest, and every call refuses. The sentences here are an
        `assessment` and a `note`.
        """
        assert "reason" not in _property_names(SCORE_OUTPUT_SCHEMA)

    def test_it_carries_one_number_and_no_composite(self) -> None:
        """`want_it_score` is absent on purpose -- it is derived from the
        verdicts, so there is nowhere for the model to return one that
        disagrees with them.
        """
        properties = SCORE_OUTPUT_SCHEMA["properties"]
        assert isinstance(properties, dict)
        assert "could_get_score" in properties
        assert "want_it_score" not in properties
        for forbidden in ("overall", "overall_score", "combined", "composite", "average"):
            assert forbidden not in properties

    def test_a_verdict_is_one_of_exactly_four_words(self) -> None:
        properties = SCORE_OUTPUT_SCHEMA["properties"]
        assert isinstance(properties, dict)
        for key in ("constraint_verdicts", "objective_verdicts"):
            item_properties = properties[key]["items"]["properties"]
            assert item_properties["verdict"]["enum"] == list(FIT_VERDICTS)

    def test_a_verdict_names_its_subject_by_index_never_by_text(self) -> None:
        """So a verdict cannot quietly restate what the user said mattered."""
        properties = SCORE_OUTPUT_SCHEMA["properties"]
        assert isinstance(properties, dict)
        assert set(properties["constraint_verdicts"]["items"]["properties"]) == {
            "index",
            "verdict",
            "note",
        }
        assert set(properties["objective_verdicts"]["items"]["properties"]) == {
            "rank",
            "verdict",
            "note",
        }

    def test_a_lever_names_a_claim_by_index_never_by_text(self) -> None:
        properties = SCORE_OUTPUT_SCHEMA["properties"]
        assert isinstance(properties, dict)
        lever_properties = properties["levers"]["items"]["properties"]
        assert set(lever_properties) == {"claim_index", "would_move_to", "note"}


# -- constraints --------------------------------------------------------------


class TestConstraints:
    def test_every_constraint_is_numbered_and_carries_its_stance(self) -> None:
        profile = Profile(
            constraints=[
                _constraint("workplace", "must", note="Remote, or one day a week at most"),
                _constraint("comp_floor", "nice", value={"guaranteed": 120000, "ccy": "GBP"}),
                _constraint("categorical_no", "never", note="No defence work"),
            ]
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "1. [must] working arrangement -- Remote, or one day a week at most" in message
        assert '2. [nice] lowest package -- {"ccy": "GBP", "guaranteed": 120000}' in message
        assert "3. [never] categorically will not do -- No defence work" in message

    def test_the_comp_floor_keeps_guaranteed_and_headline_apart(self) -> None:
        """A headline number is not an offer. Flattening the two into one here
        would assert something the user did not.
        """
        constraint = _constraint(
            "comp_floor", "must", value={"guaranteed": 120000, "headline": 145000, "ccy": "GBP"}
        )
        label = constraint_label(constraint)
        assert '"guaranteed": 120000' in label
        assert '"headline": 145000' in label

    def test_no_constraints_reads_as_not_stated(self) -> None:
        message = build_score_user_message(_inputs(profile=Profile()))
        heading = "## Constraints -- give a verdict for every one of these, by index"
        after = message[message.index(heading) + len(heading) :]
        assert after.lstrip().startswith(NOT_STATED)


# -- capabilities and the tier bridge ------------------------------------------


class TestCapabilities:
    def test_only_an_evidenced_capability_is_shown_as_evidence(self) -> None:
        profile = Profile(
            capabilities=[
                _capability("FX pricing platforms", "production_depth", [uuid.uuid4()]),
                _capability("Kubernetes", "working"),
            ]
        )
        message = build_score_user_message(_inputs(profile=profile))
        evidence_block = message[
            message.index("## Capabilities with corpus evidence behind them") : message.index(
                "## What this person says they do NOT have"
            )
        ]
        assert "FX pricing platforms" in evidence_block
        assert "production depth" in evidence_block
        assert "Kubernetes" not in evidence_block

    def test_an_unevidenced_capability_is_a_claim_and_can_become_a_lever(self) -> None:
        profile = Profile(capabilities=[_capability("Kubernetes", "working")])
        message = build_score_user_message(_inputs(profile=profile))
        assert "1. [claimed at working, no evidence] Kubernetes" in message
        claims = claimed_items(_inputs(profile=profile))
        assert [(c.text, c.claim_kind, c.tier) for c in claims] == [
            ("Kubernetes", "capability", "working")
        ]

    def test_a_capability_the_person_says_is_absent_is_never_a_lever(self) -> None:
        """They are saying they do not have it. Offering to "confirm" it would
        be the tool arguing with them about their own record.
        """
        profile = Profile(capabilities=[_capability("Frontend", "absent")])
        message = build_score_user_message(_inputs(profile=profile))
        assert claimed_items(_inputs(profile=profile)) == []
        absent_block = message[
            message.index("## What this person says they do NOT have") : message.index(
                "## What this person practises"
            )
        ]
        assert "Frontend" in absent_block

    def test_the_not_this_disciplines_are_shown_beside_absent_capabilities(self) -> None:
        profile = Profile(disciplines=Disciplines.model_validate({"not": ["frontend"]}))
        message = build_score_user_message(_inputs(profile=profile))
        assert "- frontend" in message


# -- objectives ---------------------------------------------------------------


class TestObjectives:
    def test_each_objective_is_rendered_separately_with_its_rank(self) -> None:
        profile = Profile(
            objectives=[
                _objective(1, "Get back to hands-on platform work", "A team that ships weekly"),
                _objective(2, "Stop commuting", "Two days a month at most"),
            ]
        )
        message = build_score_user_message(_inputs(profile=profile))
        assert "- rank 1" in message
        assert "- rank 2" in message
        assert "Get back to hands-on platform work" in message
        assert "Stop commuting" in message
        # Neither objective's text has been folded into the other's block.
        first = message.index("- rank 1")
        second = message.index("- rank 2")
        assert "Stop commuting" not in message[first:second]

    def test_no_objectives_reads_as_not_stated(self) -> None:
        message = build_score_user_message(_inputs(profile=Profile()))
        heading = "## Objectives -- give a verdict for every one of these, by rank"
        after = message[message.index(heading) + len(heading) :]
        assert after.lstrip().startswith(NOT_STATED)


# -- unfilled sections ---------------------------------------------------------


class TestNotStated:
    def test_an_empty_profile_reports_every_section_and_guesses_at_none(self) -> None:
        sections = not_stated_sections(Profile())
        assert {s.question_key for s in sections} == {
            "constraints",
            "capabilities",
            "disciplines",
            "objectives",
        }
        assert all(s.wording for s in sections)

    def test_a_filled_section_is_not_reported(self) -> None:
        profile = Profile(constraints=[_constraint("location", "must", note="Sheffield")])
        assert "constraints" not in {s.question_key for s in not_stated_sections(profile)}

    def test_only_a_not_this_list_still_counts_as_a_filled_discipline(self) -> None:
        profile = Profile(disciplines=Disciplines.model_validate({"not": ["frontend"]}))
        assert "disciplines" not in {s.question_key for s in not_stated_sections(profile)}


# -- unevidenced claims --------------------------------------------------------


class TestClaimedNotEvidence:
    def test_cv_facts_are_numbered_and_labelled_as_not_evidence(self) -> None:
        facts = [
            ProposedFactView(fact_text="Ran a team of 12", role_label="Northwind, EM"),
            ProposedFactView(fact_text="Owned the FX pricing platform", role_label="Contoso"),
        ]
        message = build_score_user_message(_inputs(proposed_facts=facts))
        assert "CLAIMED, NOT EVIDENCE" in message
        assert "1. [Northwind, EM] Ran a team of 12" in message
        assert "2. [Contoso] Owned the FX pricing platform" in message

    def test_capabilities_are_numbered_before_cv_facts_in_one_list(self) -> None:
        """One list, so a lever's index means the same thing whichever kind of
        claim it points at.
        """
        profile = Profile(capabilities=[_capability("Kubernetes", "working")])
        facts = [ProposedFactView(fact_text="Ran a team of 12")]
        claims = claimed_items(_inputs(profile=profile, proposed_facts=facts))
        assert [c.claim_kind for c in claims] == ["capability", "cv_fact"]

    def test_none_stored_means_no_levers_to_offer(self) -> None:
        message = build_score_user_message(_inputs(proposed_facts=[]))
        assert "(none)" in message[message.index("## CLAIMED, NOT EVIDENCE") :]


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
