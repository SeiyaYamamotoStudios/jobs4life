"""Build the static demo site from committed fixture results.

Why there is no template engine: every result file is already the complete
input the page needs, and all rendering happens client-side, in vanilla JS,
over the embedded JSON blob (see demo/template/index.html). That leaves the
server side with exactly one substitution to make -- swap a placeholder
token for a JSON literal -- which `str.replace` does as well as any
templating library would. PLAN.md's W4 originally said "Jinja at build
time"; it was amended on 2026-09-05 because a template engine with one
substitution to perform is a dependency earning nothing, and one the build
host would then need too. Writing the HTML from Python f-strings instead
would be worse again: the page stays editable as real HTML this way.

Usage:
    uv run python demo/build_site.py
    uv run python demo/build_site.py --results-dir demo/fixtures/results \\
        --template demo/template/index.html --out demo/site/index.html

Reads every `*.json` file in `--results-dir` (glob, never a hardcoded
filename or count -- more files land there as demo/generate_results.py
finishes each combination), validates each one, embeds the validated set as
a JSON array at the exact token `/*__RESULTS_JSON__*/null` in `--template`,
and writes the result to `--out`. Never touches corpus/ or analysis/, never
imports anything from packages/, and never calls a model -- this script only
reads files already on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_DEFAULT_RESULTS_DIR = Path("demo/fixtures/results")
_DEFAULT_TEMPLATE = Path("demo/template/index.html")
_DEFAULT_OUT = Path("demo/site/index.html")

# The exact token the template carries at its embedding point. build_site.py
# replaces this whole substring (comment plus placeholder value) with the
# serialised results array -- see demo/template/index.html's `const RESULTS`
# line. Kept as one substring, not "comment" + "null" separately, so a
# template edit that reorders or respaces the two can't silently break the
# substitution without also breaking this constant.
_PLACEHOLDER = "/*__RESULTS_JSON__*/null"

_REQUIRED_TOP_LEVEL_KEYS = (
    "candidate",
    "job",
    "draft",
    "requirements",
    "coverage",
    "gate",
    "spans",
    "runs",
    "cost_usd_total",
    "generated_at",
)


class DemoBuildError(RuntimeError):
    """A fixture result file is malformed. Always carries the filename that
    failed, so a bad fixture fails loudly and locatably rather than
    producing a silently broken page.
    """


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_result(data: dict[str, Any], filename: str) -> None:
    """Raise `DemoBuildError` if `data` (one parsed result file) is not safe
    to embed. Checks, in order: every required top-level key is present;
    `gate.sentences` and `coverage` are present and shaped as lists; every
    `cited_span_ids` entry (in `coverage` and in `gate.sentences`) resolves
    against `spans`; every `coverage[].requirement_id` resolves against
    `requirements`. Does not attempt to validate every field of every
    record -- the page's own rendering code is defensive about missing
    optional fields -- only the cross-references a bad fixture could get
    wrong silently.
    """
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            raise DemoBuildError(f"{filename}: missing top-level key {key!r}")

    spans = data["spans"]
    if not isinstance(spans, dict):
        raise DemoBuildError(f"{filename}: 'spans' must be an object keyed by span id")
    known_span_ids = set(spans.keys())

    requirements = data["requirements"]
    if not isinstance(requirements, list):
        raise DemoBuildError(f"{filename}: 'requirements' must be a list")
    known_requirement_ids = set()
    for i, req in enumerate(requirements):
        if not isinstance(req, dict) or "id" not in req:
            raise DemoBuildError(f"{filename}: requirements[{i}] missing 'id'")
        known_requirement_ids.add(req["id"])

    coverage = data["coverage"]
    if not isinstance(coverage, list):
        raise DemoBuildError(f"{filename}: 'coverage' must be a list")
    for i, cov in enumerate(coverage):
        if not isinstance(cov, dict):
            raise DemoBuildError(f"{filename}: coverage[{i}] must be an object")
        req_id = cov.get("requirement_id")
        if req_id not in known_requirement_ids:
            raise DemoBuildError(
                f"{filename}: coverage[{i}].requirement_id {req_id!r} has no matching requirement"
            )
        for span_id in cov.get("cited_span_ids", []):
            if span_id not in known_span_ids:
                raise DemoBuildError(
                    f"{filename}: coverage[{i}].cited_span_ids contains {span_id!r}, "
                    "not present in 'spans'"
                )

    gate = data["gate"]
    if not isinstance(gate, dict) or "sentences" not in gate:
        raise DemoBuildError(f"{filename}: 'gate' must be an object with a 'sentences' list")
    sentences = gate["sentences"]
    if not isinstance(sentences, list):
        raise DemoBuildError(f"{filename}: 'gate.sentences' must be a list")
    for i, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            raise DemoBuildError(f"{filename}: gate.sentences[{i}] must be an object")
        for span_id in sentence.get("cited_span_ids", []):
            if span_id not in known_span_ids:
                raise DemoBuildError(
                    f"{filename}: gate.sentences[{i}].cited_span_ids contains {span_id!r}, "
                    "not present in 'spans'"
                )


# --------------------------------------------------------------------------
# Loading and summarising
# --------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load and validate every `*.json` file in `results_dir`, sorted by
    filename for a deterministic build. Never assumes a count -- called
    fresh at the start of the build and again at the end in `main`, so a
    file that appears mid-build is picked up by the second call.
    """
    results = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        validate_result(data, path.name)
        results.append(data)
    return results


def summarise(data: dict[str, Any]) -> str:
    """One line describing a combination's result, for the build's console
    output. `traced` counts claim-kind sentences with verdict `supported`;
    framing is always reported separately, per CLAUDE.md's rule that framing
    (never checked) must never be folded into the checked counts.
    """
    sentences = data["gate"]["sentences"]
    claims = [s for s in sentences if s.get("kind") != "framing"]
    framing_count = len(sentences) - len(claims)
    supported = sum(1 for s in claims if s.get("verdict") == "supported")
    review = sum(1 for s in claims if s.get("verdict") == "review")
    unsupported = sum(1 for s in claims if s.get("verdict") == "unsupported")
    cost = data.get("cost_usd_total", "0")
    candidate_slug = data["candidate"]["slug"]
    job_slug = data["job"]["slug"]
    return (
        f"{candidate_slug} x {job_slug}: {supported}/{len(claims)} traced, "
        f"{review} review, {unsupported} unsupported, {framing_count} not checked "
        f"(framing) -- ${cost}"
    )


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def _embed(results: list[dict[str, Any]], template_text: str) -> str:
    """Substitute the validated results array into the template at the
    exact `_PLACEHOLDER` token. Escapes `</` to `<\\/` in the serialised
    JSON so a `</script` substring inside any fixture's free text (a job
    ad's raw_text, a generated draft) can never terminate the embedding
    `<script>` tag early -- the escape is valid inside a JS/JSON string
    literal and `</` never occurs anywhere in JSON's structural characters,
    so every occurrence is safely inside a quoted string value.
    """
    if _PLACEHOLDER not in template_text:
        raise DemoBuildError(f"template does not contain the placeholder token {_PLACEHOLDER!r}")
    blob = json.dumps(results, ensure_ascii=False).replace("</", "<\\/")
    return template_text.replace(_PLACEHOLDER, blob)


def build(results_dir: Path, template_path: Path, out_path: Path) -> list[dict[str, Any]]:
    """Load + validate results, embed them into the template, write `out_path`.
    Returns the loaded results so `main` can print the summary without a
    second disk read.
    """
    results = load_results(results_dir)
    template_text = template_path.read_text(encoding="utf-8")
    html = _embed(results, template_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--template", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        results = build(args.results_dir, args.template, args.out)
    except DemoBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print(f"warning: no result files found in {args.results_dir}", file=sys.stderr)

    total_cost = 0.0
    for data in results:
        print(summarise(data))
        total_cost += float(data.get("cost_usd_total", 0))
    print(f"--\n{len(results)} combination(s), ${total_cost:.4f} total, wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
