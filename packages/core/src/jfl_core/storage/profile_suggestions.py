"""Profile-suggestion runs: one row per CV-reading call, tenancy-scoped.

An uploaded CV states two different kinds of thing. It makes **claims about the
world** -- "rebuilt the FX pricing platform" -- which become `candidate_facts`
and are confirmed one at a time, because the corpus is what claims get measured
against and grounding on a CV would switch that measurement off silently
(CLAUDE.md, 2026-09-18). It also states plain **settings**: which disciplines
someone practises, where they have worked, the level the CV describes. Those
are not claims to be measured, so they can be proposed straight into the
profile -- and accepted with a click.

Three rules hold this module up, and they are the same three the capability
clusters follow.

**A proposal is not a profile row.** Nothing here writes to `profiles`. The
only path from a proposal to a saved setting is the user accepting it, in
`jfl_web.routes.profile`, and a setting they have already stated always wins.

**A rejection sticks.** `answered_keys` returns every proposal key this user has
ever accepted or rejected, across runs. The handler filters those out before
storing, so a suggestion answered once is never offered again -- from the same
CV or any later one. The key is content-derived
(`jfl_core.ids.setting_key`), which is what makes that true across runs.

**Cost lives in `runs`, not here.** The row carries the `trace_id` the call ran
under -- minted when the run is created, so a *failed* run can still be priced
-- and the screen asks `RunRepository.cost_for_trace`.

No model call anywhere near this module, and no SQL above it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select, update

from jfl_core.db.tables import profile_suggestions as table
from jfl_core.models import (
    ProfileSuggestionErrorCode,
    ProfileSuggestionRun,
    ProposedSetting,
    SuggestionState,
)
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.status,
    table.c.trace_id,
    table.c.proposals,
    table.c.cv_count,
    table.c.error_code,
    table.c.created_at,
    table.c.updated_at,
)

# Statuses a run is still waiting in. One at a time per user: pressing the
# button twice must not buy two calls over the same CVs.
UNFINISHED_STATUSES: tuple[str, ...] = ("pending",)


def _from_row(row: Any) -> ProfileSuggestionRun:
    return ProfileSuggestionRun(
        id=row.id,
        status=row.status,
        trace_id=row.trace_id,
        proposals=[ProposedSetting.model_validate(item) for item in (row.proposals or [])],
        cv_count=row.cv_count,
        error_code=row.error_code,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresProfileSuggestionRepository(TenantScopedRepository):
    """This user's suggestion runs, and no one else's."""

    def get(self, run_id: uuid.UUID) -> ProfileSuggestionRun | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == run_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def latest(self) -> ProfileSuggestionRun | None:
        """The newest run -- what the profile screen shows."""
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def pending(self) -> ProfileSuggestionRun | None:
        """A run already in flight, if there is one.

        Read by the route before it enqueues, so pressing the button twice
        costs one call. A read-then-write, not a lock: at-least-once delivery
        already requires the handler to be safe to run twice, and the handler
        itself refuses a run that is no longer `pending`.
        """
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(
                table.c.user_id == self._user_id,
                table.c.status.in_(UNFINISHED_STATUSES),
            )
            .order_by(table.c.created_at.desc())
            .limit(1)
        ).first()
        return None if row is None else _from_row(row)

    def create_pending(self, *, trace_id: uuid.UUID) -> ProfileSuggestionRun:
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
        run_id: uuid.UUID,
        proposals: Sequence[ProposedSetting],
        *,
        cv_count: int,
    ) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == run_id, table.c.user_id == self._user_id)
            .values(
                status="done",
                proposals=[p.model_dump(mode="json") for p in proposals],
                cv_count=cv_count,
                error_code=None,
            )
        )

    def mark_failed(self, run_id: uuid.UUID, code: ProfileSuggestionErrorCode) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == run_id, table.c.user_id == self._user_id)
            .values(status="failed", error_code=code)
        )

    def answered_keys(self) -> set[str]:
        """Every proposal key this user has already accepted or rejected.

        The handler subtracts these before storing a new run's proposals, which
        is what makes "no" stick: a later run reading the same CVs folds the
        same suggestion to the same key and it is never shown again. Accepted
        keys are filtered for the same reason -- the setting is on the profile,
        so proposing it again would be asking a question that is answered.
        """
        rows = self._conn.execute(
            select(table.c.proposals).where(table.c.user_id == self._user_id)
        ).all()
        answered: set[str] = set()
        for row in rows:
            for item in row.proposals or []:
                proposal = ProposedSetting.model_validate(item)
                if proposal.state != "open":
                    answered.add(proposal.key)
        return answered

    def set_proposal_state(
        self, run_id: uuid.UUID, key: str, state: SuggestionState
    ) -> ProposedSetting | None:
        """Answer one proposal, by its `setting_key`.

        None if this user has no such run, or no open proposal under that key:
        both read the same way, because a repository scoped to one user cannot
        tell "not yours" from "not there".

        Read-modify-write inside the caller's transaction rather than a JSONB
        path update: the list is a handful of items, and the Pydantic model
        stays the only thing that writes its shape.
        """
        row = self._conn.execute(
            select(table.c.proposals).where(table.c.id == run_id, table.c.user_id == self._user_id)
        ).first()
        if row is None:
            return None
        proposals = [ProposedSetting.model_validate(item) for item in (row.proposals or [])]
        answered: ProposedSetting | None = None
        updated: list[ProposedSetting] = []
        for proposal in proposals:
            if answered is None and proposal.state == "open" and proposal.key == key:
                answered = proposal.model_copy(update={"state": state})
                updated.append(answered)
            else:
                updated.append(proposal)
        if answered is None:
            return None
        self._conn.execute(
            update(table)
            .where(table.c.id == run_id, table.c.user_id == self._user_id)
            .values(proposals=[p.model_dump(mode="json") for p in updated])
        )
        return answered


__all__ = ["PostgresProfileSuggestionRepository"]
