#!/usr/bin/env python3
"""One-off script: builds packages/evals/data/fever_slice.jsonl from public FEVER
data on the HuggingFace Hub.

NOT part of the jfl_evals package and NOT a runtime dependency of it: `datasets`
(and the `huggingface_hub` it pulls in) is only needed once, to produce the cached
JSONL that is committed alongside this script and is what the eval and its tests
actually read. Run it with:

    uv run --with datasets python packages/evals/scripts/build_fever_dataset.py

Sampling is seeded (SEED below) but still depends on what HuggingFace serves for
the source dataset at fetch time, so re-running this is "as reproducible as the
upstream mirror" -- not bit-identical by construction. The committed JSONL is the
actual golden set; this script is provenance, not a runtime part of the eval.

See packages/evals/DATASET.md for the source, licence, and sampling method in prose.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

SEED = 20260901
PER_LABEL = 70  # 3 labels -> up to 210 items before evidence-availability filtering
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "fever_slice.jsonl"

# Dzeniks/fever_3way on the HF Hub: a script-free re-publication of FEVER's own
# official 3-way-balanced validation set (paper_dev.jsonl from fever.ai), with each
# claim's gold evidence sentences already resolved to their Wikipedia text (FEVER's
# own release only ships (page, sentence_id) pairs against a separate wiki-pages
# dump). See DATASET.md for why this mirror was used over the original release.
_HF_DATASET = "Dzeniks/fever_3way"
_HF_SPLIT = "validation"

# FEVER label ints -> (display name, the gate's Verdict this maps to). See
# CLAUDE.md's drift taxonomy: "the standard supported / refuted / not-enough-evidence
# split ... public FEVER-style data maps onto it directly."
_LABEL_NAMES = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
_EXPECTED_VERDICT = {0: "supported", 1: "unsupported", 2: "review"}

# FEVER's evidence text comes from a PTB-tokenized Wikipedia dump: brackets and a
# few other tokens are spelled out rather than literal, and punctuation is
# space-separated. Purely cosmetic -- it does not change what the sentence
# asserts -- but literal "-LRB-" in a corpus span would be a strange thing to hand
# the model as a citation, so it is undone before caching.
_BRACKET_TOKENS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-COLON-": ":",
}
_SPACED_PUNCT = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_BEFORE_OPEN = re.compile(r"([(\[])\s+")
# PTB tokenization also splits contractions and possessives off with a leading
# space ("world 's", "does n't"); collapse the common ones.
_SPACED_CLITIC = re.compile(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b")


def _detokenize(text: str) -> str:
    for token, replacement in _BRACKET_TOKENS.items():
        text = text.replace(token, replacement)
    text = _SPACED_PUNCT.sub(r"\1", text)
    text = _SPACED_CLITIC.sub(r"\1", text)
    text = _SPACE_BEFORE_OPEN.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    # Deferred, and type-ignored: `datasets` is not a workspace dependency (see
    # module docstring) -- installed on demand with `uv run --with datasets`.
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(_HF_DATASET, split=_HF_SPLIT)

    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for ex in ds:
        by_label[ex["label"]].append(ex)

    rng = random.Random(SEED)
    items: list[dict[str, Any]] = []
    kept_by_label: dict[int, int] = {0: 0, 1: 0, 2: 0}
    skipped_no_evidence = 0

    for label, pool in by_label.items():
        shuffled = list(pool)
        rng.shuffle(shuffled)
        for ex in shuffled:
            if kept_by_label[label] >= PER_LABEL:
                break
            evidence_raw = [s for s in ex["evidence"].split("\n") if s.strip()]
            if label != 2 and not evidence_raw:
                # SUPPORTS/REFUTES with no resolved evidence sentence: not usable --
                # the brief is explicit that only NEI may legitimately have an empty
                # corpus. Do not invent evidence to keep the item; skip it.
                skipped_no_evidence += 1
                continue
            evidence = [_detokenize(s) for s in evidence_raw]
            items.append(
                {
                    "id": f"fever-{ex['id']}",
                    "claim": _detokenize(ex["claim"]),
                    "expected_verdict": _EXPECTED_VERDICT[label],
                    "evidence": evidence,
                    "fever_label": _LABEL_NAMES[label],
                    "source": "fever",
                }
            )
            kept_by_label[label] += 1

    rng.shuffle(items)  # don't leave the file grouped by label

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for item in items:
            f.write(json.dumps(item, sort_keys=True) + "\n")

    by_name = {_LABEL_NAMES[k]: v for k, v in kept_by_label.items()}
    print(f"wrote {len(items)} items to {OUT_PATH}")
    print(f"kept by label: {by_name}")
    print(f"skipped (SUPPORTS/REFUTES with no resolved evidence): {skipped_no_evidence}")


if __name__ == "__main__":
    main()
