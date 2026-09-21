"""The profile, one append-only row per save -- `docs/profile-schema.md`.

**PLACEHOLDER.** The real version of this module is being written in parallel;
this is the minimum `/profile` needs to be built and tested against a live
table. The surface is the one agreed for it -- `current`, `save`, `history`,
`at` -- so replacing this file should not touch the routes.

It sits beside `jfl_core.storage.profile` (the B3a answers/objectives/ruled-out
tables) rather than replacing it, because the scoring path still reads those and
is being moved separately. Nothing writes both.

Append-only, and that is the whole design: every save is a new row, the current
profile is the latest, and "what did I believe in March" costs nothing to
answer. An identical save writes nothing, so re-submitting an untouched section
does not manufacture history -- the same rule `profile_answers` follows.

No model call anywhere in this module, and nothing here logs profile content:
comp floors, deal-breakers and self-assessments are sensitive and the standing
rule is never in `runs`, never in a trace, never logged.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import insert, select

from jfl_core.db.tables import profiles as table
from jfl_core.profile import SCHEMA_VERSION, Profile, ProfileVersion
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.schema_version,
    table.c.data,
    table.c.created_at,
)


def _version_from_row(row: Any) -> ProfileVersion:
    return ProfileVersion(
        id=row.id,
        schema_version=row.schema_version,
        created_at=row.created_at,
        profile=Profile.model_validate(row.data or {}),
    )


def _as_data(profile: Profile) -> dict[str, Any]:
    """JSON-ready, with `Disciplines.not_` written out as `not` -- the design
    doc's own shape, so the stored document reads as the document describes.
    """
    return profile.model_dump(mode="json", by_alias=True)


class PostgresProfileRepository(TenantScopedRepository):
    """One user's profile, and no one else's."""

    def current(self) -> Profile:
        """The latest saved profile, or an empty one when there is none.

        Empty rather than `None` because every section of the page is optional:
        a user who has saved nothing is in exactly the same state as one who has
        saved a blank section, and both read as "not stated".
        """
        version = self.current_version()
        return Profile() if version is None else version.profile

    def current_version(self) -> ProfileVersion | None:
        """The latest version, or None -- what the page needs for "saved when"."""
        row = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc(), table.c.id.desc())
            .limit(1)
        ).first()
        return None if row is None else _version_from_row(row)

    def save(self, profile: Profile) -> Profile:
        """Append a version. An identical save is a no-op.

        Returns the profile as it now stands, which for a no-op is what was
        already there -- so a caller cannot tell the difference, and does not
        need to.
        """
        current = self.current_version()
        if current is not None and current.profile == profile:
            return current.profile
        self._conn.execute(
            insert(table).values(
                id=uuid.uuid4(),
                user_id=self._user_id,
                schema_version=SCHEMA_VERSION,
                data=_as_data(profile),
            )
        )
        return profile

    def history(self, limit: int = 20) -> list[ProfileVersion]:
        """Versions, newest first. Nothing is ever edited or deleted, so this is
        simply the rows.
        """
        rows = self._conn.execute(
            select(*_COLUMNS)
            .where(table.c.user_id == self._user_id)
            .order_by(table.c.created_at.desc(), table.c.id.desc())
            .limit(limit)
        ).all()
        return [_version_from_row(row) for row in rows]

    def at(self, version_id: uuid.UUID) -> Profile | None:
        """One earlier version, or None if it is not this user's."""
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == version_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _version_from_row(row).profile


__all__ = ["PostgresProfileRepository"]
