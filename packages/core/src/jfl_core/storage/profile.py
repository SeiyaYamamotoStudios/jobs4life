"""Profile storage: one append-only JSONB row per save, tenancy-scoped.

`docs/profile-schema.md`, 2026-09-21. The eighteen free-text questions of
PLAN.md B3a are gone, and with them three tables that each had their own
versioning rule. What replaces them is one table and one shape: every save
appends a row, the current profile is the latest one, and a save identical to
the current profile writes nothing -- so re-submitting an untouched form does
not manufacture history. What you believed about yourself in March stays
readable, and undo is free.

`current()` returns an **empty `Profile`**, never None, for a user who has never
saved: "this user has no profile" and "this user has an empty profile" are the
same statement to every caller, and making them ask which is how a `None` check
gets forgotten in one place out of six. An empty section reports "not stated"
and is never defaulted or inferred.

Validation lives in `jfl_core.profile`, not here and not in Postgres. JSONB
carries no CHECK constraint, so the Pydantic model is the only write path and
the only guarantee -- weaker than a CHECK, accepted deliberately (owner,
2026-09-21). Nothing in this module writes `data` it has not validated, and
`current()` validates on the way out too, so a row that somehow stopped being
readable fails loudly here rather than three layers up.

Nothing here logs profile text. Comp floors and deal-breakers are sensitive and
the standing rule applies: never in `runs`, never in a trace, never logged.
"""

from __future__ import annotations

import uuid

from sqlalchemy import insert, select

from jfl_core.db.tables import profiles as table
from jfl_core.profile import (
    SCHEMA_VERSION,
    Capability,
    Profile,
    ProfileVersion,
    propose_capabilities,
    self_assessment_corpus_lines,
)
from jfl_core.storage.candidate_facts import PostgresCandidateFactRepository
from jfl_core.storage.tenancy import TenantScopedRepository
from jfl_core.storage.user_corpus import PostgresUserCorpusRepository

_COLUMNS = (table.c.id, table.c.data, table.c.created_at)


class PostgresProfileRepository(TenantScopedRepository):
    """One user's profile, and no one else's."""

    def current(self) -> Profile:
        """The latest saved profile, or an empty one if there is none.

        Never None and never a guessed default -- see the module docstring.
        """
        row = self._conn.execute(
            select(table.c.data)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc())
            .limit(1)
        ).first()
        return Profile() if row is None else Profile.model_validate(row.data)

    def save(self, profile: Profile) -> Profile:
        """Append a new version, unless it is identical to the current one.

        An identical save is a no-op that returns the current profile: a form
        re-submitted untouched must not appear in history as a decision the
        user made. Comparison is on the validated model, so a reordered JSON
        key is still a no-op and a reordered *list* is still a change -- lists
        here are ranked, and a re-ranking is exactly the kind of change worth
        keeping.
        """
        if profile == self.current():
            return profile
        row = self._conn.execute(
            insert(table)
            .values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                schema_version=SCHEMA_VERSION,
                data=profile.as_json(),
            )
            .returning(table.c.data)
        ).one()
        return Profile.model_validate(row.data)

    def history(self, limit: int = 20) -> list[ProfileVersion]:
        """Saved versions, newest first. `at()` reads one back whole."""
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc())
            .limit(limit)
        ).all()
        return [
            ProfileVersion(
                id=row.id,
                created_at=row.created_at,
                data=Profile.model_validate(row.data),
            )
            for row in rows
        ]

    def at(self, version_id: uuid.UUID) -> Profile | None:
        """One saved version's profile, or None if this user has no such row.

        Another user's version id reads as None rather than as an error: the
        repository is scoped to one user and cannot be asked about another, so
        there is nothing to distinguish "not yours" from "not there".
        """
        row = self._conn.execute(
            select(table.c.data).where(table.c.id == version_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else Profile.model_validate(row.data)


def save_profile(
    profile_repo: PostgresProfileRepository,
    corpus_repo: PostgresUserCorpusRepository,
    profile: Profile,
) -> Profile:
    """Save a profile and bring the corpus into line with its self-assessment.

    The two halves of one action, in one function so no caller performs half of
    it. `self_assessment` holds profile questions 15 and 16 -- where your depth
    is genuine, and the gaps that keep coming up -- which are claims about the
    person rather than preferences, and are therefore also corpus text. The
    profile row is where the screen reads the words back from; the corpus span
    is what the claim gate can cite.

    The corpus write is `PostgresUserCorpusRepository.replace_section`,
    unchanged: the **one** write path from a user's own words into the corpus,
    shared with confirming a CV fact. Never a second one, and no model anywhere
    on it -- the text is stored byte for byte. Replacing rather than appending
    is what makes a re-answer supersede: the older statement's span is retired
    and stops grounding anything.

    The corpus is rewritten even when the profile save was a no-op. That costs
    a re-parse of one small document and it is the honest order: the corpus is
    the store that can drift out of line with the row, so it is the one that
    gets reconciled rather than skipped on a fast path.
    """
    saved = profile_repo.save(profile)
    for section, lines in self_assessment_corpus_lines(saved).items():
        corpus_repo.replace_section(section, lines)
    return saved


def propose_capabilities_from_facts(
    fact_repo: PostgresCandidateFactRepository,
    profile: Profile,
) -> list[Capability]:
    """Capability rows this user's **confirmed** CV facts would seed, grouped by
    role, for the profile screen to offer.

    Proposed, never saved: the rows come back untiered and the user sets the
    depth. Capabilities already on the profile are left out, so offering this
    twice never overwrites a tier someone chose. The grouping rule itself is
    pure and lives in `jfl_core.profile.propose_capabilities`.
    """
    return propose_capabilities(
        fact_repo.list_facts(state="confirmed"), existing=profile.capabilities
    )


__all__ = [
    "PostgresProfileRepository",
    "propose_capabilities_from_facts",
    "save_profile",
]
