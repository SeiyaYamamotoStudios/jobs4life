"""The profile model and the CV-fact seeding. No database, no model call.

`jfl_core.profile` is the only write path into `profiles.data`, so what it
refuses is the whole of what the schema refuses -- there is no CHECK constraint
behind it (`docs/profile-schema.md`, "What this costs us"). These tests are that
guard's other half; `packages/web/tests/test_profile.py` holds the first.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from jfl_core.models import CandidateFact
from jfl_core.profile import (
    Capability,
    Constraint,
    Disciplines,
    Objective,
    Profile,
    capability_key,
    seed_capabilities_from_facts,
)
from pydantic import ValidationError

_NOW = dt.datetime(2026, 9, 21, 12, 0, tzinfo=dt.UTC)


def fact(
    text: str,
    *,
    state: str = "confirmed",
    span_id: uuid.UUID | None = None,
    role: str = "Acme -- Engineering Manager",
) -> CandidateFact:
    return CandidateFact(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        role_label=role,
        role_key=role.casefold(),
        source_line=text,
        fact_text=text,
        state=state,  # type: ignore[arg-type]
        confirmed_text=text if state == "confirmed" else None,
        span_id=span_id,
        fingerprint=uuid.uuid4().hex,
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_an_empty_profile_is_all_absent() -> None:
    """ "Not stated" is the resting state of every section, and it is a real
    value rather than a missing one.
    """
    profile = Profile()
    assert profile.is_empty()
    assert profile.constraints == []
    assert profile.capabilities == []
    assert profile.objectives == []
    assert profile.disciplines == Disciplines()
    assert profile.self_assessment.depth_genuine == ""


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "location", "stance": "maybe"},
        {"kind": "salary", "stance": "must"},
    ],
)
def test_a_constraint_outside_the_agreed_values_is_refused(payload: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        Constraint.model_validate(payload)


def test_a_tier_outside_the_agreed_values_is_refused() -> None:
    with pytest.raises(ValidationError):
        Capability.model_validate({"label": "FX pricing", "tier": "expert"})


def test_a_capability_starts_untiered_and_unevidenced() -> None:
    capability = Capability(label="FX pricing")
    assert capability.tier is None
    assert capability.interest is None
    assert not capability.evidenced


def test_there_is_no_fifth_objective_slot() -> None:
    with pytest.raises(ValidationError):
        Objective(rank=5, text="...")


def test_a_capability_key_folds_spelling_but_not_meaning() -> None:
    assert capability_key("FX pricing") == capability_key("fx  pricing")
    assert capability_key("FX pricing") != capability_key("FX pricing platform")


def test_the_profile_round_trips_through_the_documented_json() -> None:
    profile = Profile(
        constraints=[Constraint(kind="notice", stance="must", value={"text": "3 months"})],
        disciplines=Disciplines(practises=["engineering management"], **{"not": ["frontend"]}),
    )
    dumped = profile.model_dump(mode="json", by_alias=True)
    assert dumped["disciplines"] == {"practises": ["engineering management"], "not": ["frontend"]}
    assert Profile.model_validate(dumped) == profile


# -- seeding -------------------------------------------------------------------


def test_only_confirmed_facts_are_proposed_as_capabilities() -> None:
    """A proposed fact is a CV's claim. Seeding from it would put a capability
    in front of the user that they never said was true, which is the drift this
    whole flow exists to stop.
    """
    seeded = seed_capabilities_from_facts(
        [
            fact("Ran the FX pricing platform", span_id=uuid.uuid4()),
            fact("Doubled revenue single-handedly", state="proposed"),
            fact("Rejected thing", state="rejected"),
        ]
    )
    assert [c.label for c in seeded] == ["Ran the FX pricing platform"]


def test_a_seeded_capability_arrives_untiered_and_carrying_its_span() -> None:
    span = uuid.uuid4()
    seeded = seed_capabilities_from_facts([fact("Ran the FX pricing platform", span_id=span)])
    assert seeded[0].tier is None
    assert seeded[0].source == "cv_fact"
    assert seeded[0].evidence == [span]


def test_one_capability_gathers_the_spans_of_identical_facts() -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    seeded = seed_capabilities_from_facts(
        [
            fact("Ran the FX pricing platform", span_id=first),
            fact("ran the FX pricing platform", span_id=second, role="Other -- EM"),
        ]
    )
    assert len(seeded) == 1
    assert seeded[0].evidence == [first, second]


def test_a_long_fact_becomes_a_readable_label() -> None:
    long_fact = "Led the migration of " + "a very large system " * 10
    seeded = seed_capabilities_from_facts([fact(long_fact, span_id=uuid.uuid4())])
    label = seeded[0].label
    assert len(label) <= 84
    assert label.endswith("...")
    assert label.startswith("Led the migration of")
