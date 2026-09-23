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
from jfl_core import models, pushback
from jfl_core.cv_document import CvTemplate
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
    (models.WorkplaceMode, tables._WORKPLACE_MODES),
    (models.BoardJobEventKind, tables._BOARD_JOB_EVENT_KINDS),
    (models.TitleSuggestionStatus, tables._TITLE_SUGGESTION_STATUSES),
    (models.CapabilityClusterStatus, tables._CAPABILITY_CLUSTER_STATUSES),
    (models.CapabilityClusterErrorCode, tables._CAPABILITY_CLUSTER_ERROR_CODES),
    (models.ProfileSuggestionStatus, tables._PROFILE_SUGGESTION_STATUSES),
    (models.ProfileSuggestionErrorCode, tables._PROFILE_SUGGESTION_ERROR_CODES),
    (models.TitleSuggestionErrorCode, tables._TITLE_SUGGESTION_ERROR_CODES),
    (models.CandidateFactState, tables._CANDIDATE_FACT_STATES),
    (models.ScoreStatus, tables._SCORE_STATUSES),
    (models.ScoreErrorCode, tables._SCORE_ERROR_CODES),
    (models.AnswerKind, tables._APPLICATION_QUESTION_ANSWER_KINDS),
    (models.AnswerStatus, tables._APPLICATION_QUESTION_ANSWER_STATUSES),
    (models.AnswerErrorCode, tables._APPLICATION_QUESTION_ANSWER_ERROR_CODES),
    (models.DocumentStorageKind, tables._DOCUMENT_STORAGE_KINDS),
    (models.CvExtractionStatus, tables._CV_EXTRACTION_STATUSES),
    (models.CvExtractionErrorCode, tables._CV_EXTRACTION_ERROR_CODES),
    (models.CandidateFactState, tables._CANDIDATE_FACT_STATES),
    (models.PushbackStatus, tables._PUSHBACK_STATUSES),
    (models.PushbackErrorCode, tables._PUSHBACK_ERROR_CODES),
    (models.ClassificationSource, tables._CLASSIFICATION_SOURCES),
    (models.CvDocumentStatus, tables._CV_DOCUMENT_STATUSES),
    (CvTemplate, tables._CV_TEMPLATES),
    # These three live beside the rule they belong to rather than in `models`:
    # `jfl_core.pushback` is what decides what a classification means, so the
    # closed set is declared where it is enforced.
    (pushback.PushbackKind, tables._PUSHBACK_CLASSIFICATIONS),
    (pushback.Disposition, tables._PUSHBACK_DISPOSITIONS),
    (pushback.Axis, tables._SCORE_AXES),
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
