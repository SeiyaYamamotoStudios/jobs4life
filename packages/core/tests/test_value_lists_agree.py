"""Every closed set of values is written down three times, and the copies must agree.

A status or error code lives in a `Literal` in `jfl_core.models` (what the code may
produce), a tuple in `jfl_core.db.tables` (what the table definition says), and a CHECK
constraint in a migration (what Postgres will actually accept). Those three drifted apart
twice in one week: board platforms, then `duplicate_posting`. Both times the Python side
was widened and the database was not, so the first real use failed at INSERT -- and for
the error code, a check's own failure would have become a second, unrecorded one.

This test compares the Literal and the tuple for every such set. Its integration sibling,
`tests/test_check_constraints_match_models_integration.py`, compares the tuple against the
constraint in the migrated database, which is where both real failures actually were.
"""

from __future__ import annotations

from typing import get_args

import pytest
from jfl_core import models
from jfl_core.db import tables

# (Literal in models, tuple in tables). Add a row whenever a new closed set gets a
# CHECK constraint -- the integration test enumerates the same list.
VALUE_SETS = [
    (models.ApplicationStatus, tables._APPLICATION_STATUSES),
    (models.ExtractionStatus, tables._EXTRACTION_STATUSES),
    (models.ExtractionErrorCode, tables._EXTRACTION_ERROR_CODES),
    (models.TaskStatus, tables._TASK_STATUSES),
    (models.BoardPlatform, tables._BOARD_PLATFORMS),
    (models.BoardCheckStatus, tables._BOARD_CHECK_STATUSES),
    (models.BoardCheckErrorCode, tables._BOARD_CHECK_ERROR_CODES),
    # One set, three constrained columns: board_jobs.workplace, and the
    # workplace arrays on job_filters and board_filter_exceptions.
    (models.Workplace, tables._WORKPLACES),
    (models.TitleSuggestionStatus, tables._TITLE_SUGGESTION_STATUSES),
    (models.TitleSuggestionErrorCode, tables._TITLE_SUGGESTION_ERROR_CODES),
]


@pytest.mark.parametrize(("literal", "values"), VALUE_SETS)
def test_literal_and_table_tuple_hold_the_same_values(
    literal: object, values: tuple[str, ...]
) -> None:
    in_code = set(get_args(literal))
    in_table = set(values)
    assert in_code == in_table, (
        f"only in models: {sorted(in_code - in_table)}; "
        f"only in tables: {sorted(in_table - in_code)}"
    )


def test_no_table_tuple_repeats_a_value() -> None:
    for _, values in VALUE_SETS:
        assert len(values) == len(set(values)), values
