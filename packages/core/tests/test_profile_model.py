"""The profile model -- `docs/profile-schema.md`, 2026-09-21.

`profiles.data` is JSONB and carries no CHECK constraint, so
`jfl_core.profile.Profile` is the only thing standing between a typo and a
stored value nothing can read. The design names that cost and accepts it; these
tests are what makes the acceptance honest.

Three properties, and a defect in any of them is invisible until much later:

  * every closed set rejects what is not in it, and the `Literal` the code may
    produce holds exactly the values the screens offer -- so the web layer
    cannot invent a fifth tier;
  * a profile survives the round trip through JSON, `not` alias and span-id
    UUIDs included, because a section that fails to parse back is a section the
    user silently loses;
  * an **unconfirmed** CV fact never seeds a capability. A capability seeded
    from an unconfirmed claim would be a row the corpus cannot evidence, which
    is the exact failure the 2026-09-18 decision exists to prevent.

No database and no model call: everything here is pure.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, get_args

import pytest
from jfl_core.models import CandidateFact
from jfl_core.profile import (
    CAPABILITY_SOURCES,
    CAPABILITY_TIERS,
    CONSTRAINT_KINDS,
    CORPUS_SECTIONS,
    INTERESTS,
    MAX_OBJECTIVES,
    STANCES,
    Capability,
    CapabilitySource,
    CapabilityTier,
    Constraint,
    ConstraintKind,
    Disciplines,
    Interest,
    Objective,
    Profile,
    SelfAssessment,
    Stance,
    capability_key,
    comp_value,
    location_value,
    propose_capabilities,
    self_assessment_corpus_lines,
    text_value,
)
from pydantic import ValidationError

NOW = dt.datetime(2026, 9, 21, 10, 0, tzinfo=dt.UTC)


def _fact(
    *,
    role: str = "Acme Ltd -- Engineering Manager",
    role_key_value: str = "acme",
    text: str = "Led a team of eight",
    state: str = "confirmed",
    span_id: uuid.UUID | None = None,
) -> CandidateFact:
    return CandidateFact(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role_label=role,
        role_key=role_key_value,
        source_line=f"- {text}",
        fact_text=text,
        state=state,  # type: ignore[arg-type]
        span_id=span_id if span_id is not None or state != "confirmed" else uuid.uuid4(),
        fingerprint=text,
        created_at=NOW,
        updated_at=NOW,
    )


# -- the closed sets -----------------------------------------------------------


class TestValueLists:
    """The drift guard that stands in for a CHECK constraint.

    Elsewhere a `Literal`, a tuple in `db.tables` and a CHECK in a migration
    must agree and two tests enforce it. JSONB has no CHECK, so the pairing is
    `Literal` against the tuple the screens read their options from -- named
    explicitly in the design as weaker, and accepted.
    """

    @pytest.mark.parametrize(
        ("literal", "values"),
        [
            (Stance, STANCES),
            (ConstraintKind, CONSTRAINT_KINDS),
            (CapabilityTier, CAPABILITY_TIERS),
            (Interest, INTERESTS),
            (CapabilitySource, CAPABILITY_SOURCES),
        ],
    )
    def test_the_literal_and_the_exported_tuple_hold_the_same_values(
        self, literal: object, values: tuple[str, ...]
    ) -> None:
        in_code = set(get_args(literal))
        offered = set(values)
        assert in_code == offered, (
            f"only in the model: {sorted(in_code - offered)}; "
            f"only offered by the screens: {sorted(offered - in_code)}"
        )

    @pytest.mark.parametrize(
        "values",
        [STANCES, CONSTRAINT_KINDS, CAPABILITY_TIERS, INTERESTS, CAPABILITY_SOURCES],
    )
    def test_no_exported_tuple_repeats_a_value(self, values: tuple[str, ...]) -> None:
        assert len(values) == len(set(values)), values

    def test_the_design_s_own_values_are_the_ones_shipped(self) -> None:
        """Pinned literally, not derived, so a rename has to be a decision
        rather than a diff nobody reads.
        """
        assert STANCES == ("must", "nice", "never")
        assert CAPABILITY_TIERS == (
            "production_depth",
            "working",
            "oversight_only",
            "absent",
        )
        assert CONSTRAINT_KINDS == (
            "location",
            "workplace",
            "level_floor",
            "comp_floor",
            "contract",
            "right_to_work",
            "notice",
            "categorical_no",
        )
        assert "want_more" in INTERESTS


# -- rejection -----------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [("kind", "salary"), ("stance", "maybe")],
    )
    def test_a_constraint_rejects_a_value_outside_its_set(self, field: str, value: str) -> None:
        payload: dict[str, Any] = {"kind": "comp_floor", "stance": "must", field: value}
        with pytest.raises(ValidationError):
            Constraint(**payload)

    @pytest.mark.parametrize("tier", ["expert", "deep", "", "PRODUCTION_DEPTH"])
    def test_a_capability_rejects_an_invented_tier(self, tier: str) -> None:
        with pytest.raises(ValidationError):
            Capability(label="Kubernetes", tier=tier)  # type: ignore[arg-type]

    @pytest.mark.parametrize("interest", ["keen", "meh", ""])
    def test_a_capability_rejects_an_invented_interest(self, interest: str) -> None:
        with pytest.raises(ValidationError):
            Capability(label="Kubernetes", interest=interest)  # type: ignore[arg-type]

    def test_a_capability_needs_a_label(self) -> None:
        with pytest.raises(ValidationError):
            Capability(label="   ")

    def test_an_unknown_field_is_rejected_rather_than_dropped(self) -> None:
        """Pydantic's default is to discard what it does not recognise, which
        in a store with no CHECK means a typo'd key is accepted, lost on the
        next save, and reported to nobody.
        """
        with pytest.raises(ValidationError):
            Profile.model_validate({"constraint": []})
        with pytest.raises(ValidationError):
            Capability.model_validate({"label": "Kubernetes", "teir": "working"})

    def test_an_objective_rank_is_one_to_four(self) -> None:
        for rank in (1, MAX_OBJECTIVES):
            assert Objective(rank=rank).rank == rank
        for rank in (0, MAX_OBJECTIVES + 1, -1):
            with pytest.raises(ValidationError):
                Objective(rank=rank)

    def test_two_objectives_cannot_share_a_rank(self) -> None:
        """A verdict is keyed by rank, so two objectives at one rank makes
        "objective 2's verdict" ambiguous -- the merge the design forbids,
        arriving by the back door.
        """
        with pytest.raises(ValidationError):
            Profile(objectives=[Objective(rank=1), Objective(rank=1)])

    def test_at_most_four_objectives(self) -> None:
        with pytest.raises(ValidationError):
            Profile(objectives=[Objective(rank=i) for i in range(1, 6)])

    def test_a_capability_evidence_entry_must_be_a_span_id(self) -> None:
        with pytest.raises(ValidationError):
            Capability(label="Kubernetes", evidence=["not-a-uuid"])  # type: ignore[list-item]

    def test_an_untiered_capability_is_legal_and_stays_untiered(self) -> None:
        """A row proposed from a CV arrives with no tier, and nothing may
        invent one: a tier nobody chose is exactly the claim this project
        measures the distance to.
        """
        assert Capability(label="Kubernetes").tier is None


# -- the row key, and what counts as evidence ----------------------------------


class TestCapabilityKey:
    """The screens name a capability row in a form action and an anchor, and
    there is no id column to name it by -- the whole profile is one document.
    """

    def test_spelling_folds_but_meaning_does_not(self) -> None:
        assert capability_key("FX pricing") == capability_key("fx  pricing")
        assert capability_key("FX pricing") != capability_key("FX pricing platforms")

    def test_a_row_carries_the_key_its_label_implies(self) -> None:
        assert Capability(label="FX pricing").key == capability_key("FX pricing")

    def test_a_key_is_safe_in_a_url_path(self) -> None:
        """A raw label would not be: "CI/CD" splits the route and tiers
        nothing, which is a 404 on a form the page itself rendered.
        """
        assert capability_key("CI/CD pipelines").isalnum()


class TestCapabilityEvidence:
    def test_a_tier_with_no_span_behind_it_is_a_claim(self) -> None:
        """The same status a CV line has before confirmation. Scoring reads
        this to decide whether a row is evidence or only a lever.
        """
        assert not Capability(label="Kubernetes", tier="working").has_evidence

    def test_a_row_carrying_a_span_is_evidenced(self) -> None:
        assert Capability(label="Kubernetes", evidence=[uuid.uuid4()]).has_evidence


# -- the constraint value shapes -----------------------------------------------


class TestConstraintValues:
    def test_locations_are_an_ordered_list_not_a_relocate_boolean(self) -> None:
        assert location_value(["London", "Bristol"]) == {"places": ["London", "Bristol"]}

    def test_comp_keeps_guaranteed_and_headline_apart(self) -> None:
        """A headline number is not an offer, and neither figure is derived
        from the other -- so a floor stated without a headline is a complete
        answer rather than half of one.
        """
        assert comp_value(120000, 145000) == {
            "ccy": "GBP",
            "guaranteed": 120000,
            "headline": 145000,
        }
        assert comp_value(120000, None) == {"ccy": "GBP", "guaranteed": 120000}

    def test_empty_text_is_no_value_at_all(self) -> None:
        """Not `{"text": ""}`, which reads back indistinguishably from an
        answer the user actually gave.
        """
        assert text_value("") == {}
        assert text_value("Permanent only") == {"text": "Permanent only"}


# -- looking one section up ----------------------------------------------------


class TestLookups:
    def test_a_kind_nobody_stated_reads_as_none(self) -> None:
        """None is a real answer the screens render as "not stated", never a
        blank field that reads like an empty one.
        """
        profile = Profile(constraints=[Constraint(kind="notice", stance="must")])
        assert profile.constraint("notice") is not None
        assert profile.constraint("comp_floor") is None

    def test_an_objective_is_found_by_the_slot_it_was_typed_into(self) -> None:
        """Rank is the slot, so an empty rank 1 does not shuffle rank 2 up
        underneath the user.
        """
        profile = Profile(objectives=[Objective(rank=2, text="Bigger scope")])
        assert profile.objective(1) is None
        assert profile.objective(2) is not None
        assert profile.objective(2).text == "Bigger scope"  # type: ignore[union-attr]


# -- the round trip ------------------------------------------------------------


def _full_profile() -> Profile:
    return Profile(
        constraints=[
            Constraint(
                kind="comp_floor",
                stance="must",
                value=comp_value(120000, 145000),
                note="base + pension, ignoring equity",
            ),
            Constraint(
                kind="location",
                stance="nice",
                value=location_value(["Sheffield", "Leeds", "Manchester"]),
            ),
            Constraint(kind="categorical_no", stance="never", note="No agency work"),
        ],
        capabilities=[
            Capability(
                label="FX pricing platforms",
                tier="production_depth",
                interest="want_more",
                last_used=2024,
                evidence=[uuid.uuid4(), uuid.uuid4()],
                source="cv_fact",
            )
        ],
        disciplines=Disciplines(
            practises=["engineering management", "platform engineering"],
            **{"not": ["frontend"]},
        ),
        objectives=[
            Objective(rank=1, text="Back to hands-on work", evidence_of_delivery="Ships weekly"),
            Objective(rank=2, text="Stop commuting", evidence_of_delivery="Two days a month"),
        ],
        self_assessment=SelfAssessment(
            depth_genuine="Deep on payments, exposure only on ML.",
            recurring_gaps="Kubernetes keeps coming up.",
        ),
    )


class TestRoundTrip:
    def test_a_full_profile_survives_json_and_back(self) -> None:
        original = _full_profile()
        assert Profile.model_validate(original.as_json()) == original

    def test_as_json_is_json_safe(self) -> None:
        """Span ids are UUIDs in the model and strings in JSONB. A `psycopg`
        adapter would raise on the way in, at write time, in the worker -- far
        from whatever put a UUID there.
        """
        import json

        json.dumps(_full_profile().as_json())

    def test_the_disciplines_key_is_not_and_reads_back(self) -> None:
        """`not` is a Python keyword, so the field is aliased. The alias has to
        hold in both directions or a whole section silently empties.
        """
        data = _full_profile().as_json()
        assert data["disciplines"]["not"] == ["frontend"]
        assert "not_practised" not in data["disciplines"]
        assert Profile.model_validate(data).disciplines.not_practised == ["frontend"]

    def test_an_empty_profile_round_trips_and_reads_as_empty(self) -> None:
        empty = Profile()
        assert empty.is_empty
        assert Profile.model_validate(empty.as_json()) == empty

    def test_a_profile_with_anything_in_it_is_not_empty(self) -> None:
        assert not Profile(objectives=[Objective(rank=1)]).is_empty


# -- the self-assessment's corpus half -----------------------------------------


class TestSelfAssessmentCorpusLines:
    def test_each_answer_maps_to_its_own_corpus_section(self) -> None:
        profile = Profile(
            self_assessment=SelfAssessment(
                depth_genuine="Deep on payments.", recurring_gaps="Kubernetes."
            )
        )
        assert self_assessment_corpus_lines(profile) == {
            "Depth and exposure": ["Deep on payments."],
            "Recurring gaps": ["Kubernetes."],
        }

    def test_a_cleared_answer_empties_its_section_rather_than_vanishing(self) -> None:
        """An empty list clears the section, which retires the span. Dropping
        the heading instead would leave the old statement grounding claims the
        user has withdrawn.
        """
        lines = self_assessment_corpus_lines(Profile())
        assert set(lines) == set(CORPUS_SECTIONS.values())
        assert all(v == [] for v in lines.values())

    def test_whitespace_only_text_is_not_a_corpus_fact(self) -> None:
        profile = Profile(self_assessment=SelfAssessment(depth_genuine="   \n  "))
        assert self_assessment_corpus_lines(profile)["Depth and exposure"] == []

    def test_the_section_headings_are_unchanged_from_the_retired_questions(self) -> None:
        """A span id derives from its section path and its content. Renaming a
        heading would retire every statement recorded under the old one and
        mint a new span for the same sentence.
        """
        assert CORPUS_SECTIONS == {
            "depth_genuine": "Depth and exposure",
            "recurring_gaps": "Recurring gaps",
        }


# -- seeding capabilities from confirmed facts ---------------------------------


class TestProposeCapabilities:
    def test_one_row_per_role_carrying_every_confirmed_span_as_evidence(self) -> None:
        one, two = uuid.uuid4(), uuid.uuid4()
        facts = [
            _fact(text="Led a team of eight", span_id=one),
            _fact(text="Owned the payments platform", span_id=two),
        ]
        proposed = propose_capabilities(facts)
        assert len(proposed) == 1
        assert proposed[0].label == "Acme Ltd -- Engineering Manager"
        assert proposed[0].evidence == [one, two]

    def test_a_proposed_row_arrives_untiered_and_marked_as_cv_derived(self) -> None:
        proposed = propose_capabilities([_fact()])
        assert proposed[0].tier is None
        assert proposed[0].interest is None
        assert proposed[0].source == "cv_fact"

    @pytest.mark.parametrize("state", ["proposed", "rejected"])
    def test_an_unconfirmed_fact_never_seeds_a_capability(self, state: str) -> None:
        """A CV's claim is not the user's. Seeding from one would put a row on
        the profile that no corpus span evidences -- the drift the 2026-09-18
        decision exists to prevent.
        """
        assert propose_capabilities([_fact(state=state, span_id=None)]) == []

    def test_a_confirmed_fact_with_no_span_is_skipped(self) -> None:
        """`evidence` is what makes a capability citable. A confirmed row whose
        span is missing has nothing to cite, so proposing it would be the same
        unevidenced claim by another route.
        """
        fact = _fact().model_copy(update={"span_id": None})
        assert propose_capabilities([fact]) == []

    def test_roles_keep_the_order_their_facts_arrived_in(self) -> None:
        facts = [
            _fact(role="Acme Ltd", role_key_value="acme", text="One"),
            _fact(role="Northwind", role_key_value="northwind", text="Two"),
            _fact(role="Acme Ltd", role_key_value="acme", text="Three"),
        ]
        assert [c.label for c in propose_capabilities(facts)] == ["Acme Ltd", "Northwind"]

    def test_a_capability_the_user_already_has_is_not_proposed_again(self) -> None:
        """Offering this twice must never overwrite a tier someone chose."""
        existing = [Capability(label="acme  ltd", tier="working")]
        facts = [_fact(role="Acme Ltd", role_key_value="acme")]
        assert propose_capabilities(facts, existing=existing) == []

    def test_the_earliest_fact_s_spelling_of_a_role_wins(self) -> None:
        facts = [
            _fact(role="Acme Ltd -- EM", role_key_value="acme", text="One"),
            _fact(role="ACME LIMITED, EM", role_key_value="acme", text="Two"),
        ]
        assert [c.label for c in propose_capabilities(facts)] == ["Acme Ltd -- EM"]

    def test_a_fact_with_no_role_label_is_skipped(self) -> None:
        assert propose_capabilities([_fact(role="   ")]) == []

    def test_the_same_span_is_never_listed_twice_as_evidence(self) -> None:
        span = uuid.uuid4()
        facts = [_fact(text="One", span_id=span), _fact(text="Two", span_id=span)]
        assert propose_capabilities(facts)[0].evidence == [span]

    def test_proposed_rows_validate_as_part_of_a_profile(self) -> None:
        """Seeding is only useful if what it returns can be saved."""
        proposed = propose_capabilities([_fact()])
        profile = Profile(capabilities=proposed)
        assert Profile.model_validate(profile.as_json()) == profile
