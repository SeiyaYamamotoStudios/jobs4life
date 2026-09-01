"""The tier-1 golden set: FEVER claims mapped onto the claim gate's verdict space.

See DATASET.md for the source, licence, sampling method, and what this dataset
cannot test. The cached file itself (`packages/evals/data/fever_slice.jsonl`) is
produced by `packages/evals/scripts/build_fever_dataset.py`, which is not part of
this package and not run automatically -- the eval reads the cache, never the
network, per CLAUDE.md's golden set discipline: **do not generate synthetic items or
adjust labels to improve a score**. This loader only parses what that script wrote.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

# Same three values as jfl_gate.schema.Verdict. Not imported from there: this
# dataset predates and is independent of any one gate implementation -- coupling
# the golden set's type to the package under test would make a schema change in
# jfl_gate silently reshape the golden set instead of failing a test.
ExpectedVerdict = Literal["supported", "unsupported", "review"]

FeverLabel = Literal["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class GoldenItem(BaseModel):
    """One FEVER claim, its resolved evidence sentences, and the verdict the gate
    is expected to return.

    `evidence` is legitimately empty when `fever_label == "NOT ENOUGH INFO"` --
    FEVER's NEI items have no evidence by construction, and the corresponding
    corpus for that item is an empty corpus, not a missing one. It is never empty
    for SUPPORTS/REFUTES: the build script drops those if FEVER's evidence did not
    resolve to real text, rather than keeping an item with no way to check it.
    """

    id: str
    claim: str
    expected_verdict: ExpectedVerdict
    evidence: list[str]
    fever_label: FeverLabel
    source: Literal["fever"] = "fever"


class DatasetError(ValueError):
    """The cached JSONL is malformed. Raised with the offending line number rather
    than silently skipping the line -- a golden set that silently shrinks when a
    line goes bad hides exactly the kind of corruption that reproducibility exists
    to catch.
    """


def iter_golden_set(path: Path) -> Iterator[GoldenItem]:
    """Stream items from the cached JSONL, one `GoldenItem` per non-blank line."""
    with path.open(encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON: {e}") from e
            try:
                yield GoldenItem.model_validate(data)
            except ValidationError as e:
                raise DatasetError(f"{path}:{lineno}: does not match GoldenItem: {e}") from e


def load_golden_set(path: Path) -> list[GoldenItem]:
    """The whole cached golden set as a list, in file order."""
    return list(iter_golden_set(path))
