"""Capability clustering runs: one row per grouping call, tenancy-scoped.

A role is not a capability and a single confirmed fact is not one either -- the
first gives one row per employer, the second gives dozens of near-duplicates
across thirty-three overlapping CVs. Grouping is the one part of this worth a
model call, so one cheap call proposes labels and **this table holds the
proposals until the user answers them**.

Three rules hold this module up.

**A proposal is not a profile row.** Nothing here writes to `profiles`. The
only path from a proposal to a `Capability` is the user accepting it, in
`jfl_web.routes.profile`, and their rename wins permanently from that point --
`set_proposal_state` stores the label they typed, so a later run that proposes
the same grouping under the model's own wording merges by
`jfl_core.profile.capability_key` and leaves their words alone.

**No confirmed fact is silently dropped.** A run records what it was given
(`fact_count`), what the model placed in nothing (`unclustered_fact_ids`) and
what did not fit in one bounded call (`omitted_fact_ids`). All three are shown
on the screen, and the next run is handed whatever is still unplaced.

**Cost lives in `runs`, not here.** The row carries the `trace_id` the call ran
under -- minted when the run is created, so a *failed* run can still be priced
-- and the screen asks `RunRepository.cost_for_trace`.

No model call anywhere near this module, and no SQL above it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select, update

from jfl_core.db.tables import capability_clusters as table
from jfl_core.models import (
    CapabilityCluster,
    CapabilityClusterErrorCode,
    CapabilityProposalState,
    ProposedCapability,
)
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.status,
    table.c.trace_id,
    table.c.proposals,
    table.c.fact_count,
    table.c.unclustered_fact_ids,
    table.c.omitted_fact_ids,
    table.c.error_code,
    table.c.dismissed_at,
    table.c.created_at,
    table.c.updated_at,
)

# Statuses a run is still waiting in. One at a time per user: pressing the
# button twice must not buy two calls on the same facts.
UNFINISHED_STATUSES: tuple[str, ...] = ("pending",)


def _ids(raw: Any) -> list[uuid.UUID]:
    return [uuid.UUID(str(value)) for value in (raw or [])]


def _from_row(row: Any) -> CapabilityCluster:
    return CapabilityCluster(
        id=row.id,
        status=row.status,
        trace_id=row.trace_id,
        proposals=[ProposedCapability.model_validate(item) for item in (row.proposals or [])],
        fact_count=row.fact_count,
        unclustered_fact_ids=_ids(row.unclustered_fact_ids),
        omitted_fact_ids=_ids(row.omitted_fact_ids),
        error_code=row.error_code,
        dismissed_at=row.dismissed_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _json_ids(ids: Sequence[uuid.UUID]) -> list[str]:
    return [str(value) for value in ids]


class PostgresCapabilityClusterRepository(TenantScopedRepository):
    """This user's clustering runs, and no one else's."""

    def get(self, cluster_id: uuid.UUID) -> CapabilityCluster | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == cluster_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def latest(self) -> CapabilityCluster | None:
        """The newest run this user has not dismissed -- what the profile screen
        shows. Dismissed runs stay in the table: the call already ran, and
        paying for it again is not the fix for not wanting to look at it.
        """
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id, table.c.dismissed_at.is_(None))
            .order_by(table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def pending(self) -> CapabilityCluster | None:
        """A run already in flight, if there is one.

        Read by the route before it enqueues, so pressing the button twice
        costs one call. A read-then-write, not a lock: at-least-once delivery
        already requires the handler to be safe to run twice, and the handler
        itself refuses a run that is already `done`.
        """
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(
                table.c.user_id == self._user_id,
                table.c.status.in_(UNFINISHED_STATUSES),
                table.c.dismissed_at.is_(None),
            )
            .order_by(table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def create_pending(self, *, trace_id: uuid.UUID) -> CapabilityCluster:
        """A new `pending` run, carrying the trace the call will run under.

        The trace is minted here rather than in the handler so that a run which
        never reaches the model -- no key, an unsealable credential -- is still
        a row the screen can price at nothing rather than one it cannot price
        at all.
        """
        row = self._conn.execute(
            table.insert()
            .values(id=uuid.uuid4(), user_id=self._user_id, trace_id=trace_id)
            .returning(*_COLUMNS)
        ).one()
        return _from_row(row)

    def mark_done(
        self,
        cluster_id: uuid.UUID,
        proposals: Sequence[ProposedCapability],
        *,
        fact_count: int,
        unclustered_fact_ids: Sequence[uuid.UUID] = (),
        omitted_fact_ids: Sequence[uuid.UUID] = (),
    ) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == cluster_id, table.c.user_id == self._user_id)
            .values(
                status="done",
                proposals=[p.model_dump(mode="json") for p in proposals],
                fact_count=fact_count,
                unclustered_fact_ids=_json_ids(unclustered_fact_ids),
                omitted_fact_ids=_json_ids(omitted_fact_ids),
                error_code=None,
            )
        )

    def mark_failed(self, cluster_id: uuid.UUID, code: CapabilityClusterErrorCode) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == cluster_id, table.c.user_id == self._user_id)
            .values(status="failed", error_code=code)
        )

    def set_proposal_state(
        self,
        cluster_id: uuid.UUID,
        key: str,
        state: CapabilityProposalState,
        *,
        label: str | None = None,
    ) -> ProposedCapability | None:
        """Answer one proposal, by the `capability_key` of its current label.

        `label` is the user's rename and is stored verbatim -- from here on the
        row is in their words, and nothing regenerates over it. None if this
        user has no such run, or no open proposal under that key: both read the
        same way, because a repository scoped to one user cannot tell "not
        yours" from "not there".

        Read-modify-write inside the caller's transaction rather than a JSONB
        path update: the list is a handful of items, and the Pydantic model
        stays the only thing that writes its shape.
        """
        # `jfl_core.profile` imports nothing from storage, so the import is
        # local and one-directional rather than a cycle at module scope.
        from jfl_core.profile import capability_key

        row = self._conn.execute(
            select(table.c.proposals).where(
                table.c.id == cluster_id, table.c.user_id == self._user_id
            )
        ).first()
        if row is None:
            return None
        proposals = [ProposedCapability.model_validate(item) for item in (row.proposals or [])]
        answered: ProposedCapability | None = None
        updated: list[ProposedCapability] = []
        for proposal in proposals:
            if answered is None and capability_key(proposal.label) == key:
                answered = proposal.model_copy(
                    update={"state": state, "label": label or proposal.label}
                )
                updated.append(answered)
            else:
                updated.append(proposal)
        if answered is None:
            return None
        self._conn.execute(
            update(table)
            .where(table.c.id == cluster_id, table.c.user_id == self._user_id)
            .values(proposals=[p.model_dump(mode="json") for p in updated])
        )
        return answered

    def dismiss(self, cluster_id: uuid.UUID) -> bool:
        """Hide the run from the profile screen. The row is kept -- see
        `latest`.
        """
        row = self._conn.execute(
            update(table)
            .where(table.c.id == cluster_id, table.c.user_id == self._user_id)
            .values(dismissed_at=func.now())
            .returning(table.c.id)
        ).first()
        return row is not None


__all__ = ["PostgresCapabilityClusterRepository"]
