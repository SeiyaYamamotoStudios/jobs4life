"""How the CLI prints claim gate results.

Lives here rather than under `packages/cli/tests/` for the same reason as
test_cli_model_flag.py: there is no such directory. The printers are called
directly -- every real command opens Postgres, and what is under test is only
the rendering of results that are already parsed.
"""

from __future__ import annotations

import pytest
from jfl_cli.main import _print_sentence, _print_summary
from jfl_gate.schema import SentenceResult


def _title() -> SentenceResult:
    return SentenceResult(
        index=1,
        kind="title",
        verdict=None,
        drift_label=None,
        cited_span_ids=[],
        evidence_note="Document title: not checked against the corpus.",
        text="Jane Placeholder -- CV bullets (Staff Engineer -- Example Co)",
    )


def _claim() -> SentenceResult:
    return SentenceResult(
        index=2,
        kind="claim",
        verdict="review",
        drift_label="scope_inflation",
        cited_span_ids=[],
        evidence_note="Team size not stated.",
        text="Led a team of 12.",
    )


def test_a_title_renders_as_not_checked_never_as_a_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_sentence(_title())

    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("[NOT CHECKED]")
    assert "title" in line
    assert "SUPPORTED" not in line
    assert "None" not in line


def test_the_summary_counts_a_title_apart_from_claims_and_framing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_summary([_title(), _claim()])

    out = capsys.readouterr().out
    assert "2 sentences: 1 review, 1 not checked (title)" in out
    assert "supported" not in out
    assert "framing" not in out
