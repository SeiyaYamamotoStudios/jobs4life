"""Display and merge rules for capability clustering -- the `/profile` panel.

No SQL and no model call here: storage is
`jfl_core.storage.capability_clusters`, the call is
`jfl_generate.capabilities`. This module turns a run into what the panel shows
and holds the one rule that decides what a user's answer does to their profile.

**A proposal is never applied silently.** The panel offers accept, rename and
reject; nothing on this page writes a capability the user did not press. That
is the same discipline `jfl_web.titlesuggestions` applies to filter phrases and
`jfl_web.routes.candidate_facts` applies to the corpus, and for the same reason:
a tool that quietly recorded claims on your behalf would be the tool this
project exists to oppose.

**The user's words win permanently.** A rename is stored on the proposal and
becomes the capability's label. A later run proposing the same grouping under
the model's own wording merges by `capability_key`, which is derived from the
label -- so the renamed row is a different key and the model's wording arrives
as a separate proposal the user can reject once, rather than as an overwrite
they never see.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from jfl_core.models import CapabilityCluster, CapabilityClusterErrorCode, ProposedCapability
from jfl_core.profile import Capability, Profile, capability_key

from jfl_web.profile import merge_capabilities

# A renamed label is still a label: the same ceiling the "add a capability" box
# applies, rejected rather than truncated.
MAX_LABEL = 200


@dataclass(frozen=True, slots=True)
class CapabilityClusterFailure:
    """What to tell the user. Mirrors `jfl_web.titlesuggestions`'s failures --
    a rejected or unreadable key is the one case worth pointing at Settings;
    the rest is not actionable beyond "try again".
    """

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[CapabilityClusterErrorCode, CapabilityClusterFailure] = {
    "no_api_key": CapabilityClusterFailure(
        "This needs your own Anthropic API key -- grouping your facts is a model "
        "call, billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": CapabilityClusterFailure(
        "Anthropic rejected the API key stored here.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "model_refused": CapabilityClusterFailure(
        "The model declined to group these facts. Nothing was changed."
    ),
    "model_error": CapabilityClusterFailure("Grouping your facts failed. Nothing was changed."),
    "credential_unreadable": CapabilityClusterFailure(
        "Your stored API key could not be unlocked on the server.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
}

# Anything unrecognised -- a code added to the database before this module
# caught up -- still gets a sentence rather than a blank panel.
_UNKNOWN = CapabilityClusterFailure("Grouping your facts failed. Nothing was changed.")


def cluster_failure(code: CapabilityClusterErrorCode | None) -> CapabilityClusterFailure:
    return _UNKNOWN if code is None else _FAILURES.get(code, _UNKNOWN)


@dataclass(frozen=True, slots=True)
class ProposalView:
    """One proposal as the panel renders it: its key for the form action, its
    label, and the confirmed facts it would carry as evidence -- in the user's
    own words, because "3 spans" tells nobody whether the grouping is right.
    """

    key: str
    label: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ClusterView:
    """One run's panel state -- what `_capability_clusters.html` renders,
    whether reached inline from `/profile` or from the polling route.
    """

    cluster: CapabilityCluster | None
    proposals: list[ProposalView] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)
    failure: CapabilityClusterFailure | None = None
    cost: object = None


def _fact_texts(fact_ids: Sequence[uuid.UUID], texts: dict[str, str]) -> list[str]:
    return [texts[str(fact_id)] for fact_id in fact_ids if str(fact_id) in texts]


def cluster_view(
    cluster: CapabilityCluster | None,
    fact_texts: dict[str, str],
    *,
    cost: object = None,
) -> ClusterView:
    """One run's view, whether or not there is a run.

    `fact_texts` maps a candidate fact's id to the words the user confirmed.
    Facts the run did not place -- the ones the model grouped into nothing and
    the ones that did not fit in a bounded call -- are shown together: both are
    still the user's confirmed facts, and the panel's job is to say so rather
    than let them disappear.
    """
    if cluster is None:
        return ClusterView(cluster=None)
    failure = cluster_failure(cluster.error_code) if cluster.status == "failed" else None
    proposals = [
        ProposalView(
            key=capability_key(proposal.label),
            label=proposal.label,
            evidence=_fact_texts(proposal.fact_ids, fact_texts),
        )
        for proposal in cluster.open_proposals
    ]
    unplaced = _fact_texts([*cluster.unclustered_fact_ids, *cluster.omitted_fact_ids], fact_texts)
    return ClusterView(
        cluster=cluster,
        proposals=proposals,
        unplaced=unplaced,
        failure=failure,
        cost=cost,
    )


def accepted_capabilities(
    profile: Profile, proposal: ProposedCapability, *, label: str | None = None
) -> list[Capability]:
    """The profile's capability rows with this proposal accepted onto them.

    `label` is the user's rename and wins over the model's wording. The merge
    is `jfl_web.profile.merge_capabilities`, unchanged and shared with the
    per-role seeds: **a saved row always beats an incoming one**, so accepting
    a proposal whose label matches something the user has already tiered or
    edited changes nothing about their row except adding the evidence, if it
    had none. Their words and their choices are never overwritten.
    """
    name = (label or proposal.label).strip() or proposal.label
    incoming = Capability(label=name, evidence=list(proposal.span_ids), source="clustered")
    return merge_capabilities(profile.capabilities, [incoming])


__all__ = [
    "MAX_LABEL",
    "CapabilityClusterFailure",
    "ClusterView",
    "ProposalView",
    "accepted_capabilities",
    "cluster_failure",
    "cluster_view",
]
