"""The migrated database accepts exactly the values the table definitions declare.

`packages/core/tests/test_value_lists_agree.py` keeps each `Literal` and its tuple in
step. That would not have caught either real drift this project has had: both times the
Python side was right and the *migration* was stale, so the tuple and the Literal agreed
while Postgres refused the value. This reads the CHECK constraints back out of the
migrated schema and compares them with the tuples, so a model change that forgets its
migration fails here rather than at the first production INSERT.
"""

from __future__ import annotations

import os
import re

import pytest
from jfl_core.db import tables
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.integration

# (table, column, tuple) -- the same sets as test_value_lists_agree.py.
CONSTRAINED_COLUMNS = [
    ("applications", "status", tables._APPLICATION_STATUSES),
    ("applications", "extraction_status", tables._EXTRACTION_STATUSES),
    ("applications", "extraction_error_code", tables._EXTRACTION_ERROR_CODES),
    ("tasks", "status", tables._TASK_STATUSES),
    ("watched_boards", "platform", tables._BOARD_PLATFORMS),
    ("board_checks", "status", tables._BOARD_CHECK_STATUSES),
    ("board_checks", "error_code", tables._BOARD_CHECK_ERROR_CODES),
    ("board_jobs", "workplace", tables._WORKPLACES),
    ("job_filters", "workplaces", tables._WORKPLACES),
    ("board_filter_exceptions", "workplaces", tables._WORKPLACES),
    ("job_filters", "workplace_mode", tables._WORKPLACE_MODES),
    ("job_feed_marks", "kind", tables._BOARD_JOB_EVENT_KINDS),
    ("title_suggestions", "status", tables._TITLE_SUGGESTION_STATUSES),
    ("title_suggestions", "error_code", tables._TITLE_SUGGESTION_ERROR_CODES),
]


def _allowed_values(table: str, column: str) -> set[str]:
    engine = create_engine(os.environ["JFL_DATABASE_URL"])
    with engine.connect() as conn:
        defs = (
            conn.execute(
                text(
                    "select pg_get_constraintdef(c.oid) from pg_constraint c "
                    "join pg_class t on t.oid = c.conrelid "
                    "where c.contype = 'c' and t.relname = :table"
                ),
                {"table": table},
            )
            .scalars()
            .all()
        )
    # Postgres stores `x in ('a','b')` as `(x = ANY (ARRAY['a'::text, 'b'::text]))`,
    # and an array column's subset check `xs <@ array['a','b']::text[]` as
    # `(xs <@ ARRAY['a'::text, 'b'::text])`. Match the column exactly, so `status`
    # does not also pick up `extraction_status`.
    pattern = re.compile(rf"(?<![a-z_]){re.escape(column)} (?:= ANY \(|<@ )ARRAY\[(.*?)\]")
    matches = [m.group(1) for d in defs if (m := pattern.search(d))]
    assert len(matches) == 1, (
        f"expected one value-list CHECK on {table}.{column}, found {len(matches)}"
    )
    return set(re.findall(r"'([^']*)'", matches[0]))


@pytest.mark.parametrize(("table", "column", "values"), CONSTRAINED_COLUMNS)
def test_database_check_accepts_exactly_the_declared_values(
    table: str, column: str, values: tuple[str, ...]
) -> None:
    in_db = _allowed_values(table, column)
    declared = set(values)
    assert in_db == declared, (
        f"{table}.{column}: migration is stale -- only declared: {sorted(declared - in_db)}; "
        f"only in database: {sorted(in_db - declared)}. Write a new migration."
    )
