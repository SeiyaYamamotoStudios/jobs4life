"""The clustering panel's display and merge rules, without a database.

The rule under test everywhere here is the same one: **a proposal never
overwrites what the user has already said.** Accepting one adds a row or adds
evidence to a row; it never touches a tier, an interest or a label the user
chose. The route-level half is in `tests/test_capability_clusters_web_integration.py`.
"""

from __future__ import annotations

import datetime as dt
import uuid

from jfl_core.models import CapabilityCluster, ProposedCapability
from jfl_core.profile import Capability, Profile, capability_key
from jfl_web.capabilityclusters import accepted_capabilities, cluster_failure, cluster_view

WHEN = dt.datetime(2026, 9, 22, tzinfo=dt.UTC)


def _cluster(
    proposals: list[ProposedCapability] | None = None,
    *,
    status: str = "done",
    unclustered: list[uuid.UUID] | None = None,
    omitted: list[uuid.UUID] | None = None,
    error_code: str | None = None,
    fact_count: int = 0,
) -> CapabilityCluster:
    return CapabilityCluster(
        id=uuid.uuid4(),
        status=status,  # type: ignore[arg-type]
        trace_id=uuid.uuid4(),
        proposals=proposals or [],
        fact_count=fact_count,
        unclustered_fact_ids=unclustered or [],
        omitted_fact_ids=omitted or [],
        error_code=error_code,  # type: ignore[arg-type]
        created_at=WHEN,
        updated_at=WHEN,
    )


def _proposal(label: str = "FX pricing platforms") -> tuple[ProposedCapability, uuid.UUID]:
    fact_id = uuid.uuid4()
    return (
        ProposedCapability(label=label, fact_ids=[fact_id], span_ids=[uuid.uuid4()]),
        fact_id,
    )


# -- the view -----------------------------------------------------------------


def test_no_run_is_a_view_with_nothing_in_it() -> None:
    view = cluster_view(None, {})
    assert view.cluster is None
    assert view.proposals == []


def test_a_proposal_shows_the_facts_behind_it_in_the_users_own_words() -> None:
    proposal, fact_id = _proposal()
    view = cluster_view(_cluster([proposal]), {str(fact_id): "Rebuilt the FX pricing platform"})
    assert [p.label for p in view.proposals] == ["FX pricing platforms"]
    assert view.proposals[0].evidence == ["Rebuilt the FX pricing platform"]
    assert view.proposals[0].key == capability_key("FX pricing platforms")


def test_an_answered_proposal_is_not_asked_again() -> None:
    proposal, _ = _proposal()
    cluster = _cluster([proposal.model_copy(update={"state": "accepted"})])
    assert cluster_view(cluster, {}).proposals == []


def test_facts_the_run_did_not_place_are_still_shown() -> None:
    """Nothing is silently dropped -- both the facts the model placed in nothing
    and the ones that did not fit in one bounded call appear on the page.
    """
    unplaced, overflow = uuid.uuid4(), uuid.uuid4()
    view = cluster_view(
        _cluster(unclustered=[unplaced], omitted=[overflow]),
        {str(unplaced): "Ran the on-call rota", str(overflow): "Wrote the incident policy"},
    )
    assert view.unplaced == ["Ran the on-call rota", "Wrote the incident policy"]


def test_a_failed_run_carries_a_sentence_and_where_to_fix_it() -> None:
    view = cluster_view(_cluster(status="failed", error_code="no_api_key"), {})
    assert view.failure is not None
    assert view.failure.fix_url == "/settings"


def test_an_unrecognised_error_code_still_gets_a_sentence() -> None:
    assert cluster_failure(None).message


# -- the merge ----------------------------------------------------------------


def test_accepting_a_proposal_adds_a_clustered_row_with_its_evidence() -> None:
    proposal, _ = _proposal()
    rows = accepted_capabilities(Profile(), proposal)
    assert [r.label for r in rows] == ["FX pricing platforms"]
    assert rows[0].source == "clustered"
    assert rows[0].evidence == proposal.span_ids
    assert rows[0].tier is None, "a proposal never arrives carrying a depth nobody chose"


def test_a_rename_wins_and_is_stored_exactly_as_typed() -> None:
    proposal, _ = _proposal()
    rows = accepted_capabilities(Profile(), proposal, label="FX rates and pricing")
    assert [r.label for r in rows] == ["FX rates and pricing"]


def test_a_blank_rename_keeps_the_proposed_label() -> None:
    proposal, _ = _proposal()
    assert accepted_capabilities(Profile(), proposal, label="   ")[0].label == proposal.label


def test_a_row_the_user_tiered_is_never_overwritten() -> None:
    """The rule the whole merge exists for."""
    proposal, _ = _proposal()
    saved = Capability(
        label="fx  PRICING platforms",
        tier="production_depth",
        interest="want_more",
        source="user",
    )
    rows = accepted_capabilities(Profile(capabilities=[saved]), proposal)
    assert len(rows) == 1
    assert rows[0].label == "fx  PRICING platforms", "the user's own wording survived"
    assert rows[0].tier == "production_depth"
    assert rows[0].interest == "want_more"
    assert rows[0].source == "user"
    # ...but the evidence it had none of is adopted, because that is a fact
    # about the corpus rather than a choice the user made.
    assert rows[0].evidence == proposal.span_ids


def test_accepting_the_same_proposal_twice_is_stable() -> None:
    proposal, _ = _proposal()
    once = accepted_capabilities(Profile(), proposal)
    twice = accepted_capabilities(Profile(capabilities=once), proposal)
    assert once == twice


def test_accepting_a_second_proposal_leaves_the_first_alone() -> None:
    first, _ = _proposal("FX pricing platforms")
    second, _ = _proposal("Hiring engineering managers")
    profile = Profile(capabilities=accepted_capabilities(Profile(), first))
    rows = accepted_capabilities(profile, second)
    assert [r.label for r in rows] == ["FX pricing platforms", "Hiring engineering managers"]
