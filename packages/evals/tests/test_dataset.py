"""Unit tests for the golden-set loader (jfl_evals.dataset).

No network, no database -- every test writes its own small JSONL fixture to a
tmp_path rather than reading the real cached dataset, so these stay fast and
independent of what packages/evals/data/fever_slice.jsonl happens to contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jfl_evals.dataset import DatasetError, GoldenItem, load_golden_set


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_GOOD_SUPPORTED = json.dumps(
    {
        "id": "fever-1",
        "claim": "The sky is blue.",
        "expected_verdict": "supported",
        "evidence": ["The sky appears blue due to Rayleigh scattering."],
        "fever_label": "SUPPORTS",
        "source": "fever",
    }
)

_GOOD_NEI = json.dumps(
    {
        "id": "fever-2",
        "claim": "Something unverifiable happened.",
        "expected_verdict": "review",
        "evidence": [],
        "fever_label": "NOT ENOUGH INFO",
        "source": "fever",
    }
)


class TestLoadGoldenSet:
    def test_parses_every_well_formed_line(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        _write(path, [_GOOD_SUPPORTED, _GOOD_NEI])

        items = load_golden_set(path)

        assert len(items) == 2
        assert all(isinstance(item, GoldenItem) for item in items)
        assert items[0].id == "fever-1"
        assert items[0].expected_verdict == "supported"
        assert items[1].evidence == []

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        _write(path, [_GOOD_SUPPORTED, "", "   ", _GOOD_NEI])

        items = load_golden_set(path)

        assert len(items) == 2

    def test_a_malformed_json_line_raises_dataset_error(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        _write(path, [_GOOD_SUPPORTED, "not valid json{{{"])

        with pytest.raises(DatasetError, match="invalid JSON"):
            load_golden_set(path)

    def test_a_line_missing_a_required_field_raises_dataset_error(self, tmp_path: Path) -> None:
        bad = json.dumps({"id": "fever-3", "claim": "Missing fields."})
        path = tmp_path / "golden.jsonl"
        _write(path, [bad])

        with pytest.raises(DatasetError, match="does not match GoldenItem"):
            load_golden_set(path)

    def test_an_unrecognised_expected_verdict_raises_dataset_error(self, tmp_path: Path) -> None:
        bad = json.dumps(
            {
                "id": "fever-4",
                "claim": "Bad verdict.",
                "expected_verdict": "maybe",
                "evidence": [],
                "fever_label": "SUPPORTS",
                "source": "fever",
            }
        )
        path = tmp_path / "golden.jsonl"
        _write(path, [bad])

        with pytest.raises(DatasetError):
            load_golden_set(path)

    def test_error_message_names_the_offending_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        _write(path, [_GOOD_SUPPORTED, "still not json"])

        with pytest.raises(DatasetError, match=r":2:"):
            load_golden_set(path)
