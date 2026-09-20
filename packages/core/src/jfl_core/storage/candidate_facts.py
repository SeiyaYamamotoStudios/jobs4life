"""Candidate facts: what a CV claims, until the user says otherwise -- slice B6.

A CV cannot simply become the corpus. Grounding on CVs makes every later CV
"supported" and switches the over-claim measurement off silently -- the
33-CV analysis found exactly that drift. So a model proposes facts from the CV,
this table holds them, and only what the user **confirms, one at a time**,
reaches the corpus. CLAUDE.md's 2026-09-18 decision, and the reason there is no
global accept-all anywhere in this slice.

Three states and no fourth:

  * `proposed` -- claimed in a CV, not confirmed. Kept, shown, and **never
    grounding**. Scoring may name it ("your CVs claim X; confirm it and this
    moves from 5 to 7"), which is what makes confirming worth the user's time;
  * `confirmed` -- the user said it is true, in their own words where they
    edited it. Only this state carries a `span_id`, and the database enforces
    that rather than trusting a caller;
  * `rejected` -- the user said it is not.

**Confirming is the only path into the corpus, and it goes through markdown.**
`confirm` calls `jfl_core.corpus_source.append_confirmed_fact`, which appends
the user's words to their corpus document and re-parses it, so the result is an
ordinary `provenance='document'` span. Never a direct span insert: see that
module for why, and note that `reject` and a re-confirm with different wording
both call its inverse, so there is never a corpus line the user cannot get rid
of.

No model is on any path in this file.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from jfl_core.corpus_source import append_confirmed_fact, remove_confirmed_fact
from jfl_core.db.tables import candidate_facts as table
from jfl_core.models import CandidateFact, CandidateFactState, FactCounts, ProposedFact, RoleGroup
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.user_id,
    table.c.sent_document_id,
    table.c.role_label,
    table.c.role_key,
    table.c.source_line,
    table.c.fact_text,
    table.c.probe,
    table.c.probe_answer,
    table.c.state,
    table.c.confirmed_text,
    table.c.span_id,
    table.c.fingerprint,
    table.c.created_at,
    table.c.updated_at,
)


def _from_row(row: Any) -> CandidateFact:
    return CandidateFact(
        id=row.id,
        user_id=row.user_id,
        sent_document_id=row.sent_document_id,
        role_label=row.role_label,
        role_key=row.role_key,
        source_line=row.source_line,
        fact_text=row.fact_text,
        probe=row.probe,
        probe_answer=row.probe_answer,
        state=row.state,
        confirmed_text=row.confirmed_text,
        span_id=row.span_id,
        fingerprint=row.fingerprint,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _counter(state: str) -> Any:
    return func.count(case((table.c.state == state, 1)))


class PostgresCandidateFactRepository(TenantScopedRepository):
    """This user's candidate facts, and no one else's."""

    def add_proposed(self, facts: Sequence[ProposedFact]) -> int:
        """Insert what a CV proposed, skipping anything already proposed.

        `ON CONFLICT (user_id, fingerprint) DO NOTHING` rather than a read then
        a write: the owner has thirty-three CVs saying much the same thing, and
        two extractions finishing at once must not mint two rows for one fact.
        Returns how many rows were actually inserted, which is what the
        extraction records as `facts_proposed`.

        A conflict deliberately does **not** update the existing row. A fact the
        user has already confirmed or rejected must not be quietly reset to
        `proposed` because another CV mentioned it again.
        """
        if not facts:
            return 0
        rows = [
            {
                "id": uuid.uuid4(),
                "user_id": self._user_id,
                "sent_document_id": fact.sent_document_id,
                "role_label": fact.role_label,
                "role_key": fact.role_key,
                "source_line": fact.source_line,
                "fact_text": fact.fact_text,
                "probe": fact.probe,
                "state": "proposed",
                "fingerprint": fact.fingerprint,
                "ordinal": fact.ordinal,
            }
            for fact in facts
        ]
        result = self._conn.execute(
            pg_insert(table)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["user_id", "fingerprint"])
            .returning(table.c.id)
        ).all()
        return len(result)

    def list_facts(self, *, state: CandidateFactState | None = None) -> list[CandidateFact]:
        """In CV order within a role, roles in the order they were first seen."""
        query = (
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.role_key, table.c.ordinal, table.c.created_at)
        )
        if state is not None:
            query = query.where(table.c.state == state)
        return [_from_row(row) for row in self._conn.execute(query).all()]

    def roles(self) -> list[RoleGroup]:
        """One entry per role, with its counts, in CV order.

        `role_label` is the label of the role's earliest fact: the same role can
        be written slightly differently across CVs, `role_key` folds those
        together (`jfl_core.ids.role_key`), and one of the spellings has to be
        the one shown. The earliest is the least surprising choice and is
        stable, which matters more than which spelling wins.
        """
        first_ordinal = func.min(table.c.ordinal).label("first_ordinal")
        first_seen = func.min(table.c.created_at).label("first_seen")
        label = func.min(table.c.role_label).label("role_label")
        rows = self._conn.execute(
            select(
                table.c.role_key,
                label,
                _counter("proposed").label("proposed"),
                _counter("confirmed").label("confirmed"),
                _counter("rejected").label("rejected"),
                first_ordinal,
                first_seen,
            )
            .where(table.c.user_id == self._user_id)
            .group_by(table.c.role_key)
            .order_by(first_ordinal, first_seen, table.c.role_key)
        ).all()
        return [
            RoleGroup(
                role_key=row.role_key,
                role_label=row.role_label,
                proposed=row.proposed,
                confirmed=row.confirmed,
                rejected=row.rejected,
            )
            for row in rows
        ]

    def get_fact(self, fact_id: uuid.UUID) -> CandidateFact | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == fact_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def confirm(
        self,
        fact_id: uuid.UUID,
        *,
        text: str | None = None,
        probe_answer: str | None = None,
    ) -> CandidateFact | None:
        """Confirm a fact and put it in the corpus. None if there is no such
        fact for this user.

        `text` is the user's edit, stored verbatim and used instead of the
        model's proposal; leave it out to confirm the proposal as written.
        `probe_answer` is likewise the user's own words -- no model touches
        either on the way in or on the way to the corpus.

        Confirming an already-confirmed fact with different words is an edit:
        the old line comes out of the corpus markdown and the new one goes in,
        so the fact never has two spans and the corpus never keeps a sentence
        the user has replaced.
        """
        existing = self.get_fact(fact_id)
        if existing is None:
            return None

        edited = (text or "").strip() or None
        answer = (probe_answer or "").strip() or existing.probe_answer
        wanted = edited or existing.fact_text

        if existing.state == "confirmed" and existing.span_id is not None:
            if wanted == existing.corpus_text:
                if answer != existing.probe_answer:
                    self._set(fact_id, probe_answer=answer)
                return self.get_fact(fact_id)
            self._release(existing)

        span_id = append_confirmed_fact(
            self._conn, self._user_id, wanted, section=existing.role_label
        )
        self._set(
            fact_id,
            state="confirmed",
            confirmed_text=edited,
            probe_answer=answer,
            span_id=span_id,
        )
        return self.get_fact(fact_id)

    def reject(self, fact_id: uuid.UUID) -> CandidateFact | None:
        """Reject a fact. None if there is no such fact for this user.

        Rejecting one that was confirmed takes its line back out of the corpus
        markdown and retires its span, so "I changed my mind" actually stops the
        tool grounding on it -- see `jfl_core.corpus_source`.
        """
        existing = self.get_fact(fact_id)
        if existing is None:
            return None
        if existing.state == "confirmed" and existing.span_id is not None:
            self._release(existing)
        self._set(fact_id, state="rejected", span_id=None)
        return self.get_fact(fact_id)

    def _release(self, fact: CandidateFact) -> None:
        """Take this fact's line out of the corpus -- unless another confirmed
        fact resolved to the same span.

        Two facts can land on one span only if their confirmed wording is
        identical under the same role, which the markdown treats as one bullet.
        Rare, and silently un-grounding the other one would be exactly the kind
        of quiet wrongness this project measures.
        """
        shared = self._conn.execute(
            select(table.c.id)
            .where(
                table.c.user_id == self._user_id,
                table.c.span_id == fact.span_id,
                table.c.id != fact.id,
                table.c.state == "confirmed",
            )
            .limit(1)
        ).first()
        if shared is None:
            remove_confirmed_fact(
                self._conn, self._user_id, fact.corpus_text, section=fact.role_label
            )

    def counts(self) -> FactCounts:
        row = self._conn.execute(
            select(
                _counter("proposed").label("proposed"),
                _counter("confirmed").label("confirmed"),
                _counter("rejected").label("rejected"),
            ).where(table.c.user_id == self._user_id)
        ).one()
        return FactCounts(proposed=row.proposed, confirmed=row.confirmed, rejected=row.rejected)

    def _set(self, fact_id: uuid.UUID, **values: Any) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == fact_id, table.c.user_id == self._user_id)
            .values(**values, updated_at=func.now())
        )
