"""Tests for demo/build_site.py -- the script that reads
demo/fixtures/results/*.json and writes demo/site/index.html, the static
demo page. Lives at tests/test_demo_build_site.py, not demo/tests/, for the
same reason as tests/test_demo_generate.py: pyproject.toml's
`testpaths = ["tests", "packages"]` does not include `demo/`, so a test
under demo/tests/ would silently never run under plain `uv run pytest`.

No real fixtures, no API calls, no Postgres. Every test builds its own tiny
result JSON (and, where needed, its own tiny template) under `tmp_path`, so
none of this depends on demo/fixtures/results/ actually existing or being
in any particular state -- that directory is being filled by a separate,
long-running, real-money process while these tests run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import demo.build_site as build_site

# --------------------------------------------------------------------------
# A minimal but complete valid result, reused and mutated per test.
# --------------------------------------------------------------------------


def _valid_result() -> dict[str, Any]:
    return {
        "candidate": {
            "slug": "jane-doe",
            "source_file": "jane-doe.md",
            "user_id": "11111111-1111-1111-1111-111111111111",
        },
        "job": {
            "slug": "acme-backend-engineer",
            "title": "Backend Engineer",
            "employer": "Acme",
            "location": "Remote",
            "id": "22222222-2222-2222-2222-222222222222",
            "source_file": "acme-backend-engineer.txt",
            "raw_text": "We need a backend engineer.",
        },
        "draft": {
            "id": "33333333-3333-3333-3333-333333333333",
            "kind": "cv_bullets",
            "text": "Jane Doe -- CV bullets.\nLed the platform team.",
            "trace_id": "44444444-4444-4444-4444-444444444444",
        },
        "requirements": [
            {
                "id": "req-1",
                "text": "5+ years of Python",
                "necessity": "essential",
                "ordinal": 0,
            }
        ],
        "coverage": [
            {
                "requirement_id": "req-1",
                "status": "evidenced",
                "evidence_note": "Traces cleanly.",
                "cited_span_ids": ["span-1"],
            }
        ],
        "gate": {
            "sentences": [
                {
                    "index": 0,
                    "text": "Led the platform team.",
                    "kind": "claim",
                    "verdict": "supported",
                    "drift_label": "supported",
                    "evidence_note": "Matches span-1.",
                    "cited_span_ids": ["span-1"],
                    "rule_flags": [],
                },
                {
                    "index": 1,
                    "text": "Technical profile",
                    "kind": "framing",
                    "verdict": "supported",
                    "drift_label": "framing",
                    "evidence_note": "Heading, nothing to check.",
                    "cited_span_ids": [],
                    "rule_flags": [],
                },
            ]
        },
        "spans": {
            "span-1": {
                "text": "Led the platform team at Acme.",
                "section_path": "Acme > Staff Engineer",
                "kind": "bullet",
            }
        },
        "runs": [
            {
                "component": "gate",
                "stage": "baseline",
                "model": "claude-opus-5",
                "cost_usd": "0.01",
                "latency_ms": 1000,
                "tokens_in": 100,
                "tokens_out": 50,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "outcome": "ok",
                "error": None,
            }
        ],
        "cost_usd_total": "0.01",
        "generated_at": "2026-09-04T21:22:06.135450+00:00",
    }


def _write_result(directory: Path, filename: str, data: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


_MINIMAL_TEMPLATE = (
    "<html><body><script>const RESULTS = /*__RESULTS_JSON__*/null;</script></body></html>"
)


def _write_template(path: Path, text: str = _MINIMAL_TEMPLATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# validate_result
# --------------------------------------------------------------------------


def test_valid_result_passes_validation() -> None:
    build_site.validate_result(_valid_result(), "jane-doe__acme.json")


def test_missing_top_level_key_raises_with_filename() -> None:
    data = _valid_result()
    del data["gate"]
    with pytest.raises(build_site.DemoBuildError, match="bad-file.json.*'gate'"):
        build_site.validate_result(data, "bad-file.json")


def test_dangling_gate_cited_span_id_raises_with_filename() -> None:
    data = _valid_result()
    data["gate"]["sentences"][0]["cited_span_ids"] = ["does-not-exist"]
    with pytest.raises(build_site.DemoBuildError, match="bad-file.json.*does-not-exist"):
        build_site.validate_result(data, "bad-file.json")


def test_dangling_coverage_cited_span_id_raises_with_filename() -> None:
    data = _valid_result()
    data["coverage"][0]["cited_span_ids"] = ["nope"]
    with pytest.raises(build_site.DemoBuildError, match="bad-file.json.*nope"):
        build_site.validate_result(data, "bad-file.json")


def test_dangling_requirement_id_raises_with_filename() -> None:
    data = _valid_result()
    data["coverage"][0]["requirement_id"] = "does-not-exist-req"
    with pytest.raises(build_site.DemoBuildError, match="bad-file.json.*does-not-exist-req"):
        build_site.validate_result(data, "bad-file.json")


# --------------------------------------------------------------------------
# build(): end to end against tmp_path
# --------------------------------------------------------------------------


def test_build_fails_loudly_on_malformed_fixture(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    bad = _valid_result()
    bad["coverage"][0]["requirement_id"] = "ghost"
    _write_result(results_dir, "candidate__job.json", bad)
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    with pytest.raises(build_site.DemoBuildError, match="candidate__job.json"):
        build_site.build(results_dir, template_path, out_path)

    assert not out_path.exists()


def test_build_works_with_a_single_result_file(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, "jane-doe__acme.json", _valid_result())
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    results = build_site.build(results_dir, template_path, out_path)

    assert len(results) == 1
    assert out_path.exists()


def test_placeholder_is_replaced_and_json_round_trips(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    data = _valid_result()
    _write_result(results_dir, "jane-doe__acme.json", data)
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    build_site.build(results_dir, template_path, out_path)
    html = out_path.read_text(encoding="utf-8")

    assert "__RESULTS_JSON__" not in html

    match = re.search(r"const RESULTS = (\[.*\]);</script>", html, re.S)
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded == [data]


def test_build_picks_up_files_that_appear_later(tmp_path: Path) -> None:
    """Mirrors the real situation this build runs under: demo/generate_results.py
    keeps writing files to the results directory as build_site.py works, so a
    second `load_results` call must see files that did not exist at the first.
    """
    results_dir = tmp_path / "results"
    _write_result(results_dir, "one.json", _valid_result())

    first = build_site.load_results(results_dir)
    assert len(first) == 1

    second_data = _valid_result()
    second_data["candidate"]["slug"] = "second-candidate"
    _write_result(results_dir, "two.json", second_data)

    second = build_site.load_results(results_dir)
    assert len(second) == 2


def test_output_has_no_external_resource_references(tmp_path: Path) -> None:
    """The file:// guarantee: no http(s) URL anywhere in the built page, so it
    never tries to fetch a stylesheet, font, script, or image over the network.
    """
    results_dir = tmp_path / "results"
    _write_result(results_dir, "jane-doe__acme.json", _valid_result())
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    build_site.build(results_dir, template_path, out_path)
    html = out_path.read_text(encoding="utf-8")

    assert re.search(r"https?://", html) is None


def test_build_raises_if_template_has_no_placeholder(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, "jane-doe__acme.json", _valid_result())
    template_path = _write_template(tmp_path / "template" / "index.html", text="<html></html>")
    out_path = tmp_path / "site" / "index.html"

    with pytest.raises(build_site.DemoBuildError, match="placeholder"):
        build_site.build(results_dir, template_path, out_path)


def test_build_escapes_closing_script_tag_in_embedded_text(tmp_path: Path) -> None:
    """A fixture's free text (job raw_text, draft text) could in principle
    contain a literal `</script>` substring. That must not be able to break
    out of the embedding <script> tag in the built page.
    """
    results_dir = tmp_path / "results"
    data = _valid_result()
    data["job"]["raw_text"] = "Nice try </script><script>alert(1)</script>"
    _write_result(results_dir, "jane-doe__acme.json", data)
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    build_site.build(results_dir, template_path, out_path)
    html = out_path.read_text(encoding="utf-8")

    assert "</script><script>alert" not in html


# --------------------------------------------------------------------------
# summarise()
# --------------------------------------------------------------------------


def test_summarise_counts_framing_separately_from_claims() -> None:
    line = build_site.summarise(_valid_result())
    assert "1/1 traced" in line
    assert "1 not checked (framing)" in line


# --------------------------------------------------------------------------
# main(): argparse wiring + exit codes
# --------------------------------------------------------------------------


def test_main_returns_zero_on_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, "jane-doe__acme.json", _valid_result())
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    rc = build_site.main(
        [
            "--results-dir",
            str(results_dir),
            "--template",
            str(template_path),
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()


def test_main_returns_nonzero_and_reports_filename_on_malformed_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results_dir = tmp_path / "results"
    bad = _valid_result()
    bad["coverage"][0]["requirement_id"] = "ghost"
    _write_result(results_dir, "candidate__job.json", bad)
    template_path = _write_template(tmp_path / "template" / "index.html")
    out_path = tmp_path / "site" / "index.html"

    rc = build_site.main(
        [
            "--results-dir",
            str(results_dir),
            "--template",
            str(template_path),
            "--out",
            str(out_path),
        ]
    )

    assert rc != 0
    captured = capsys.readouterr()
    assert "candidate__job.json" in captured.err
