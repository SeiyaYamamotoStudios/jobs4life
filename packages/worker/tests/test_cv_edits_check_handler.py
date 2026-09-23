"""`check_cv_edits`' decisions that need no database: mapping the gate's
sentences back onto the lines it was given, and which failures are permanent.
The handler end to end is in `tests/test_cv_documents_web_integration.py`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from jfl_worker.handlers.cv_edits_check import (
    FAILURE_MARKER,
    _classify,
    _paths,
    verdicts_for_lines,
)
from jfl_worker.registry import PermanentTaskError


@dataclass
class _Sentence:
    text: str
    kind: str
    verdict: str | None
    evidence_note: str = ""


def test_each_line_gets_its_worst_sentence() -> None:
    lines = [
        ("summary.0", "I lead teams. I cut costs by 40%."),
        ("roles.0.bullets.1", "Ran the build farm."),
    ]
    sentences = [
        _Sentence("I lead teams.", "claim", "supported"),
        _Sentence("I cut costs by 40%.", "claim", "unsupported", "No figure recorded."),
        _Sentence("Ran the build farm.", "claim", "review", "Partly."),
    ]
    assert verdicts_for_lines(lines, sentences) == {
        "summary.0": ("I lead teams. I cut costs by 40%.", "unsupported", "No figure recorded."),
        "roles.0.bullets.1": ("Ran the build farm.", "review", "Partly."),
    }


def test_framing_only_line_is_framing_and_unmatched_line_gets_nothing() -> None:
    lines = [("summary.0", "I care about craft."), ("summary.1", "Something else.")]
    sentences = [_Sentence("I care about craft.", "framing", "supported")]
    assert verdicts_for_lines(lines, sentences) == {
        "summary.0": ("I care about craft.", "framing", ""),
    }


def test_the_same_sentence_in_two_lines_lands_in_order() -> None:
    lines = [("a", "Shipped it."), ("b", "Shipped it.")]
    sentences = [
        _Sentence("Shipped it.", "claim", "supported"),
        _Sentence("Shipped it.", "claim", "unsupported"),
    ]
    result = verdicts_for_lines(lines, sentences)
    assert result["a"][1] == "supported"
    assert result["b"][1] == "unsupported"


@pytest.mark.parametrize(
    ("message", "code", "permanent"),
    [
        ("authentication_error: invalid x-api-key", "api_key_rejected", True),
        ("model refused to respond: reasoning_extraction", "model_refused", True),
        ("rate_limited: slow down", "model_error", False),
    ],
)
def test_classify(message: str, code: str, permanent: bool) -> None:
    assert _classify(message) == (code, permanent)


def test_the_failure_marker_is_the_one_the_page_reads() -> None:
    from jfl_web import cvdocs

    assert cvdocs._CHECK_FAILURE_MARKER == FAILURE_MARKER


@pytest.mark.parametrize("payload", [{}, {"paths": "summary.0"}, {"paths": [1]}])
def test_bad_paths_are_permanent(payload: dict[str, object]) -> None:
    with pytest.raises(PermanentTaskError):
        _paths(payload)
