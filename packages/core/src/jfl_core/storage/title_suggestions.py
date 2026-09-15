"""Suggested title expansions for a saved filter's title includes -- slice C7a.

One row per `(user, phrase_key)`, keyed on the phrase's normalised form
(`jfl_intake.normalise.normalise`, applied by the caller -- this module never
normalises anything itself, matching `job_filters`' own split between what is
stored and what matches). The call runs once per phrase, ever: `create_pending`
uses `ON CONFLICT DO NOTHING` on the unique `(user_id, phrase_key)` rather than a
read-then-write, so two saves racing on the same new phrase mint at most one row
and enqueue at most one model call.

Nothing here ever writes to `job_filters.title_includes`. Turning a suggestion
into a match term is `jfl_web.routes.title_suggestions.accept_title_suggestions`,
a plain filter edit gated by a tickbox -- this repository only stores what the
model proposed and which row a user has dismissed.

No SQL above this layer, and no model call anywhere near it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from jfl_core.db.tables import title_suggestions as table
from jfl_core.models import SuggestedTitle, TitleSuggestion, TitleSuggestionErrorCode
from jfl_core.storage.tenancy import TenantScopedRepository

_COLUMNS = (
    table.c.id,
    table.c.phrase,
    table.c.phrase_key,
    table.c.status,
    table.c.suggestions,
    table.c.error_code,
    table.c.dismissed_at,
    table.c.created_at,
    table.c.updated_at,
)


def _from_row(row: Any) -> TitleSuggestion:
    return TitleSuggestion(
        id=row.id,
        phrase=row.phrase,
        phrase_key=row.phrase_key,
        status=row.status,
        suggestions=[SuggestedTitle.model_validate(item) for item in (row.suggestions or [])],
        error_code=row.error_code,
        dismissed_at=row.dismissed_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class PostgresTitleSuggestionRepository(TenantScopedRepository):
    """This user's title-suggestion rows, and no one else's."""

    def get(self, suggestion_id: uuid.UUID) -> TitleSuggestion | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(table.c.id == suggestion_id, table.c.user_id == self._user_id)
        ).first()
        return None if row is None else _from_row(row)

    def get_by_phrase_key(self, phrase_key: str) -> TitleSuggestion | None:
        row = self._conn.execute(
            select(*_COLUMNS).where(
                table.c.phrase_key == phrase_key, table.c.user_id == self._user_id
            )
        ).first()
        return None if row is None else _from_row(row)

    def create_pending(self, *, phrase: str, phrase_key: str) -> TitleSuggestion | None:
        """A new `pending` row for this phrase, or None if one already exists.

        `ON CONFLICT DO NOTHING` rather than a read then a write -- two saves
        racing on the same new phrase must mint at most one row and enqueue at
        most one call, and the unique constraint is what makes that true under a
        race rather than merely in the common case.
        """
        row = self._conn.execute(
            pg_insert(table)
            .values(id=uuid.uuid4(), user_id=self._user_id, phrase=phrase, phrase_key=phrase_key)
            .on_conflict_do_nothing(index_elements=["user_id", "phrase_key"])
            .returning(*_COLUMNS)
        ).first()
        return None if row is None else _from_row(row)

    def mark_done(self, suggestion_id: uuid.UUID, suggestions: list[SuggestedTitle]) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == suggestion_id, table.c.user_id == self._user_id)
            .values(
                status="done",
                suggestions=[s.model_dump() for s in suggestions],
                error_code=None,
            )
        )

    def mark_failed(self, suggestion_id: uuid.UUID, code: TitleSuggestionErrorCode) -> None:
        self._conn.execute(
            update(table)
            .where(table.c.id == suggestion_id, table.c.user_id == self._user_id)
            .values(status="failed", error_code=code)
        )

    def dismiss(self, suggestion_id: uuid.UUID) -> bool:
        """Hide the row from the panel. The row itself is kept -- see the module
        docstring: the call already ran and paying for it again is not the fix
        for not wanting to see it any more.
        """
        row = self._conn.execute(
            update(table)
            .where(table.c.id == suggestion_id, table.c.user_id == self._user_id)
            .values(dismissed_at=func.now())
            .returning(table.c.id)
        ).first()
        return row is not None
