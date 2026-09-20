"""Candidate facts read out of a CV, and the one place they become corpus.

PLAN.md B6 as redesigned on 2026-09-18. A CV is stored verbatim in the
sent-document store -- form, never truth -- and a model proposes facts from it.
Those proposals are held here, in a table that is deliberately not `spans`:
nothing in this module is grounding until a person says so.

The rule this module exists to enforce: **a fact becomes corpus only through
`confirm`, only one fact per call, and only with the user's own words.** There
is no method that confirms everything, and a per-role control is built in the
route by calling `confirm` once per fact it was shown -- so "no global
accept-all" is a property of this surface, not of a page that happens not to
offer a button.

`fact_text` is the model's wording and never reaches `spans`; `confirmed_text`
is the user's -- the proposal accepted as written, or their edit of it -- and it
is that which `jfl_core.storage.user_corpus` records verbatim, with no model
anywhere in the path.

A fact carrying a `probe` ("led how many?") cannot be confirmed until the probe
is answered. That is not fussiness: the unstated number is exactly what
`scope_inflation` and `ownership_inflation` turn on, and a CV line that says
"led the platform team" supports a very different claim at six people than at
sixty.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, insert, select, update

from jfl_core.db.tables import candidate_facts as facts_table
from jfl_core.models import CandidateFact, CandidateFactState, FactCounts, RoleGroup
from jfl_core.storage.tenancy import TenantScopedRepository
from jfl_core.storage.user_corpus import PostgresUserCorpusRepository

# The corpus section confirmed CV facts are filed under, one sub-section per
# role, so a span's `section_path` says which job it belongs to.
CORPUS_SECTION_ROOT = "Confirmed from CVs"

# The only punctuation this module adds to anything a user typed: it joins a
# confirmed statement to the answer the probe asked for, so one fact stays one
# span. It contributes no words -- see the module docstring on why no model may.
PROBE_JOIN = " — "


def corpus_section(role_label: str) -> str:
    return f"{CORPUS_SECTION_ROOT} > {role_label}"


class ProbeUnansweredError(ValueError):
    """`confirm` was called on a fact whose probe has no answer.

    Deliberately an error rather than a silent skip: a caller that confirms one
    named fact is acting on a user's click, and quietly doing nothing would
    report success for a fact that did not become corpus.
    """

    def __init__(self, fact_id: uuid.UUID) -> None:
        super().__init__(
            f"candidate fact {fact_id} cannot be confirmed until its probe is answered"
        )


@dataclass(frozen=True, slots=True)
class ProposedFact:
    """One extraction result, before it is stored. Flat and dumb on purpose --
    the extraction step owns what a good proposal looks like; this module owns
    only that a proposal is not yet evidence.
    """

    sent_document_id: uuid.UUID
    role_label: str
    role_key: str
    source_line: str
    fact_text: str
    fingerprint: str
    probe: str | None = None


_COLUMNS = (
    facts_table.c.id,
    facts_table.c.user_id,
    facts_table.c.sent_document_id,
    facts_table.c.role_label,
    facts_table.c.role_key,
    facts_table.c.source_line,
    facts_table.c.fact_text,
    facts_table.c.probe,
    facts_table.c.probe_answer,
    facts_table.c.state,
    facts_table.c.confirmed_text,
    facts_table.c.span_id,
    facts_table.c.fingerprint,
    facts_table.c.created_at,
    facts_table.c.updated_at,
)


def _fact_from_row(row: Any) -> CandidateFact:
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


class PostgresCandidateFactRepository(TenantScopedRepository):
    """One user's candidate facts. Bound to that user at construction; there is
    no per-call override, so no call site can name another tenant.
    """

    def _corpus(self) -> PostgresUserCorpusRepository:
        return PostgresUserCorpusRepository(self._conn, self._user_id)

    # -- writing proposals ----------------------------------------------------

    def add_proposed(self, facts: Sequence[ProposedFact]) -> int:
        """Store newly extracted proposals and return how many were new.

        De-duplicated on `fingerprint` per user: the same fact restated across
        many CVs is one row and one confirmation, not one per document. A
        fingerprint already on the record is left exactly as it is -- including
        a fact the user has already rejected, which a later upload must not
        resurrect as unchecked work.
        """
        added = 0
        for fact in facts:
            existing = self._conn.execute(
                select(facts_table.c.id).where(
                    facts_table.c.user_id == self._user_id,
                    facts_table.c.fingerprint == fact.fingerprint,
                )
            ).first()
            if existing is not None:
                continue
            self._conn.execute(
                insert(facts_table).values(
                    id=uuid.uuid4(),
                    user_id=self._user_id,
                    sent_document_id=fact.sent_document_id,
                    role_label=fact.role_label,
                    role_key=fact.role_key,
                    source_line=fact.source_line,
                    fact_text=fact.fact_text,
                    probe=fact.probe,
                    state="proposed",
                    fingerprint=fact.fingerprint,
                )
            )
            added += 1
        return added

    # -- reading --------------------------------------------------------------

    def list_facts(self, *, state: CandidateFactState | None = None) -> list[CandidateFact]:
        """Every fact, or every fact in one state, in CV order.

        CV order is insertion order (`created_at`, which uses
        `clock_timestamp()` so one upload's rows do not tie). The user is
        reading their own career back and must be able to follow it against the
        document it came from.
        """
        stmt = select(*_COLUMNS).where(facts_table.c.user_id == self._user_id)
        if state is not None:
            stmt = stmt.where(facts_table.c.state == state)
        rows = self._conn.execute(stmt.order_by(facts_table.c.created_at, facts_table.c.id)).all()
        return [_fact_from_row(row) for row in rows]

    def roles(self) -> list[RoleGroup]:
        """One entry per role, in CV order, with its counts per state."""
        first_seen = func.min(facts_table.c.created_at).label("first_seen")
        rows = self._conn.execute(
            select(
                facts_table.c.role_key,
                func.min(facts_table.c.role_label).label("role_label"),
                func.count().filter(facts_table.c.state == "proposed").label("proposed"),
                func.count().filter(facts_table.c.state == "confirmed").label("confirmed"),
                func.count().filter(facts_table.c.state == "rejected").label("rejected"),
                first_seen,
            )
            .where(facts_table.c.user_id == self._user_id)
            .group_by(facts_table.c.role_key)
            .order_by(first_seen)
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
        """None for a fact that does not exist OR belongs to someone else --
        the caller cannot tell those apart, and must not be able to.
        """
        row = self._conn.execute(
            select(*_COLUMNS).where(
                facts_table.c.id == fact_id,
                facts_table.c.user_id == self._user_id,
            )
        ).first()
        return None if row is None else _fact_from_row(row)

    def counts(self) -> FactCounts:
        row = self._conn.execute(
            select(
                func.count().filter(facts_table.c.state == "proposed").label("proposed"),
                func.count().filter(facts_table.c.state == "confirmed").label("confirmed"),
                func.count().filter(facts_table.c.state == "rejected").label("rejected"),
            ).where(facts_table.c.user_id == self._user_id)
        ).one()
        return FactCounts(proposed=row.proposed, confirmed=row.confirmed, rejected=row.rejected)

    # -- state changes --------------------------------------------------------

    def confirm(
        self,
        fact_id: uuid.UUID,
        *,
        text: str | None = None,
        probe_answer: str | None = None,
    ) -> CandidateFact | None:
        """Confirm ONE fact and record it in the corpus, verbatim.

        `text` is the user's edit; blank or absent means "true as written", in
        which case the model's `fact_text` becomes their statement because they
        said so. Either way what is stored is what will be shown back to them,
        untouched.

        `probe_answer` is folded into the same statement, so a fact and the
        number it turned on stay one span rather than two half-facts.

        None if there is no such fact for this user. Raises
        `ProbeUnansweredError` if the fact has a probe and neither this call nor
        the record supplies an answer.
        """
        fact = self.get_fact(fact_id)
        if fact is None:
            return None

        answer = (probe_answer if probe_answer is not None else fact.probe_answer) or ""
        answer = answer.strip()
        if fact.probe and not answer:
            raise ProbeUnansweredError(fact_id)

        confirmed = (text or "").strip() or fact.fact_text
        statement = confirmed if not answer else f"{confirmed}{PROBE_JOIN}{answer}"
        span = self._corpus().record(statement, section=corpus_section(fact.role_label))

        return self._set_state(
            fact_id,
            state="confirmed",
            confirmed_text=confirmed,
            probe_answer=answer or None,
            span_id=span,
        )

    def reject(self, fact_id: uuid.UUID) -> CandidateFact | None:
        """Mark a fact "not true as written". Nothing is deleted: the row stays,
        the page keeps showing it, and it can be brought back.

        If it had already been confirmed, its span is retired -- a statement the
        user has withdrawn must stop grounding claims the moment they withdraw
        it, not at the next re-ingestion.
        """
        fact = self.get_fact(fact_id)
        if fact is None:
            return None
        if fact.span_id is not None:
            self._corpus().retire(fact.span_id)
        return self._set_state(fact_id, state="rejected", span_id=None)

    def restore(self, fact_id: uuid.UUID) -> CandidateFact | None:
        """Bring a rejected fact back to unchecked. Their edit, if they made
        one, is kept -- see `candidate_facts.confirmed_text` in `db.tables` --
        so restoring does not hand the user a blank box where their words were.
        """
        fact = self.get_fact(fact_id)
        if fact is None:
            return None
        if fact.span_id is not None:
            self._corpus().retire(fact.span_id)
        return self._set_state(fact_id, state="proposed", span_id=None)

    def _set_state(
        self,
        fact_id: uuid.UUID,
        *,
        state: CandidateFactState,
        span_id: uuid.UUID | None,
        confirmed_text: str | None = None,
        probe_answer: str | None = None,
    ) -> CandidateFact | None:
        values: dict[str, Any] = {
            "state": state,
            "span_id": span_id,
            "updated_at": dt.datetime.now(dt.UTC),
        }
        if confirmed_text is not None:
            values["confirmed_text"] = confirmed_text
        if probe_answer is not None:
            values["probe_answer"] = probe_answer
        row = self._conn.execute(
            update(facts_table)
            .where(facts_table.c.id == fact_id, facts_table.c.user_id == self._user_id)
            .values(**values)
            .returning(*_COLUMNS)
        ).first()
        return None if row is None else _fact_from_row(row)


__all__ = [
    "CORPUS_SECTION_ROOT",
    "PROBE_JOIN",
    "PostgresCandidateFactRepository",
    "ProbeUnansweredError",
    "ProposedFact",
    "corpus_section",
]
