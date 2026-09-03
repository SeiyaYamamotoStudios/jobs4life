"""Tests for demo/generate_results.py -- the script that runs the real
pipeline over demo/fixtures/ and writes demo/fixtures/results/*.json.

Lives at tests/test_demo_generate.py, not demo/tests/: pyproject.toml's
`testpaths = ["tests", "packages"]` does not include `demo/`, and this repo's
convention for cross-package/script-level tests (test_draft_integration.py,
test_gate_integration.py) is already the root `tests/` directory. Putting
tests under demo/tests/ would silently never run under plain `uv run pytest`.

No real Anthropic client anywhere here -- the root conftest.py blocks one
regardless, and this budget is zero API calls. Instead, `extract_requirements`,
`run_coverage` and `generate_draft` -- the three functions
`demo.generate_results` calls for each combination -- are monkeypatched with
fakes, the same one-level-up pattern `test_draft.py` uses for `check_text`
inside `generate_draft`. Corpus ingestion itself is exercised for real,
against the real demo/fixtures/candidates/*.md files, through an in-memory
repository pair (`_FakeCorpusStore`) conforming to the same
`jfl_core.repositories` Protocols Postgres implements -- so `run_ingestion`
and `jfl_core.ingest.parser.parse_document` actually run, with no faking
needed since neither makes a model call.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from jfl_core.context import RequestContext
from jfl_core.db.tables import LOCAL_USER_ID
from jfl_core.models import (
    Draft,
    GapQuestion,
    Job,
    JobRequirement,
    RequirementCoverage,
    RunRecord,
    Span,
    SpanCandidate,
)
from jfl_generate.schema import ExtractedRequirement, ExtractOutput

import demo.generate_results as generate_results

CANDIDATES_DIR = generate_results.CANDIDATES_DIR
JOB_ADS_DIR = generate_results.JOB_ADS_DIR

MIREILLE = CANDIDATES_DIR / "mireille-fontaine.md"
TOBIAS = CANDIDATES_DIR / "tobias-reyes.md"
LEDGERBRIDGE = JOB_ADS_DIR / "ledgerbridge-senior-backend-engineer.txt"
NIMBUS = JOB_ADS_DIR / "nimbus-engineering-manager.txt"


# --- fakes: real ingestion, fake model calls -----------------------------------


class _FakeCorpusStore:
    """`IngestRepository` + `GroundingRepository` over a plain dict, so real
    `run_ingestion` (no model call) can run against the real fixture files
    without touching Postgres. Mirrors PostgresIngestRepository's upsert
    semantics closely enough for these tests: existence-check, then
    insert-or-update.
    """

    def __init__(self) -> None:
        self._spans: dict[uuid.UUID, Span] = {}

    # IngestRepository
    def upsert_document(
        self,
        user_id: uuid.UUID,
        document_id: uuid.UUID,
        source_uri: str,
        title: str | None,
        content_hash: str,
    ) -> bool:
        return True

    def upsert_span(self, span: Span) -> bool:
        created = span.id not in self._spans
        self._spans[span.id] = span
        return created

    def retire_missing_documents(self, user_id: uuid.UUID, seen: set[uuid.UUID]) -> int:
        return 0

    def retire_missing_spans(self, user_id: uuid.UUID, seen: set[uuid.UUID]) -> int:
        return 0

    # GroundingRepository
    def get_span(self, user_id: uuid.UUID, span_id: uuid.UUID) -> Span | None:
        span = self._spans.get(span_id)
        return span if span is not None and span.user_id == user_id else None

    def search(
        self, user_id: uuid.UUID, embedding: list[float], limit: int = 10
    ) -> list[SpanCandidate]:
        raise NotImplementedError("unused by this pipeline; retrieval is unused in v1")

    def all_spans(self, user_id: uuid.UUID, include_retired: bool = False) -> list[Span]:
        return [s for s in self._spans.values() if s.user_id == user_id]

    def add_adjudicated_span(self, user_id: uuid.UUID, span: Span) -> uuid.UUID:
        self._spans[span.id] = span
        return span.id


class _FakeJobRepository:
    """`JobRepository` over plain dicts -- real enough for
    `_persist_job_from_extraction` (script code, not faked) and the fake
    `run_coverage`/`generate_draft` below to read and write through.
    """

    def __init__(self) -> None:
        self._jobs: dict[uuid.UUID, Job] = {}
        self._requirements: dict[uuid.UUID, list[JobRequirement]] = {}
        self._coverage: list[RequirementCoverage] = []
        self._questions: dict[uuid.UUID, GapQuestion] = {}
        self.drafts: list[Draft] = []

    def upsert_job(self, job: Job) -> bool:
        created = job.id not in self._jobs
        self._jobs[job.id] = job
        return created

    def replace_requirements(
        self, user_id: uuid.UUID, job_id: uuid.UUID, requirements: list[JobRequirement]
    ) -> None:
        self._requirements[job_id] = list(requirements)

    def list_jobs(self, user_id: uuid.UUID) -> list[object]:
        raise NotImplementedError

    def get_job(
        self, user_id: uuid.UUID, job_id: uuid.UUID
    ) -> tuple[Job, list[JobRequirement]] | None:
        job = self._jobs.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        return job, self._requirements.get(job_id, [])

    def record_coverage(self, coverage: RequirementCoverage) -> None:
        self._coverage.append(coverage)

    def latest_coverage(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[RequirementCoverage]:
        req_ids = {r.id for r in self._requirements.get(job_id, [])}
        return [c for c in self._coverage if c.requirement_id in req_ids]

    def upsert_gap_question(self, question: GapQuestion) -> None:
        self._questions[question.id] = question

    def list_open_questions(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[GapQuestion]:
        req_ids = {r.id for r in self._requirements.get(job_id, [])}
        return [
            q
            for q in self._questions.values()
            if q.requirement_id in req_ids and q.status == "open"
        ]

    def get_question(self, user_id: uuid.UUID, question_id: uuid.UUID) -> GapQuestion | None:
        return self._questions.get(question_id)

    def mark_question_answered(
        self,
        user_id: uuid.UUID,
        question_id: uuid.UUID,
        answer_text: str,
        resulting_span_id: uuid.UUID,
    ) -> None:
        raise NotImplementedError

    def record_draft(self, draft: Draft) -> None:
        self.drafts.append(draft)

    def list_drafts(self, user_id: uuid.UUID, job_id: uuid.UUID) -> list[Draft]:
        return [d for d in self.drafts if d.job_id == job_id]


class _FakeRunRepo:
    def __init__(self) -> None:
        self.records: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self.records.append(run)


def _make_repos() -> generate_results.Repos:
    store = _FakeCorpusStore()
    job_repo = _FakeJobRepository()
    return generate_results.Repos(ingest=store, grounding=store, run=_FakeRunRepo(), job=job_repo)


def _run_record(ctx: RequestContext, *, component: str, stage: str, cost: Decimal) -> RunRecord:
    return RunRecord(
        user_id=ctx.user_id,
        trace_id=ctx.trace_id,
        component=component,  # type: ignore[arg-type]
        stage=stage,
        model=ctx.model,
        tokens_in=100,
        tokens_out=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=cost,
        latency_ms=100,
        outcome="ok",
        started_at=datetime.now(UTC),
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extract_cost: Decimal = Decimal("0.01"),
    coverage_cost: Decimal = Decimal("0.15"),
    draft_cost: Decimal = Decimal("0.20"),
    gate_cost: Decimal = Decimal("0.10"),
) -> dict[str, list[object]]:
    """Monkeypatch the three pipeline entry points `generate_results` calls,
    one level up from the model call -- same pattern test_draft.py uses for
    `check_text`. Returns a dict of call logs keyed by stage, for assertions
    on exactly how many times (and for which job/user) each was invoked.
    """
    calls: dict[str, list[object]] = {"extract": [], "coverage": [], "draft": []}

    def fake_extract(ctx: RequestContext, run_repo: object, raw_text: str) -> ExtractOutput:
        calls["extract"].append(raw_text)
        run_repo.record(  # type: ignore[attr-defined]
            _run_record(ctx, component="generate", stage="extract_requirements", cost=extract_cost)
        )
        return ExtractOutput(
            employer="Acme",
            title="Senior Engineer",
            location="Remote",
            requirements=[ExtractedRequirement(text="5+ years of Python", necessity="essential")],
        )

    def fake_run_coverage(
        ctx: RequestContext,
        grounding_repo: object,
        run_repo: object,
        job_repo: _FakeJobRepository,
        job_id: uuid.UUID,
    ) -> list[RequirementCoverage]:
        calls["coverage"].append(job_id)
        coverage_run = _run_record(ctx, component="generate", stage="coverage", cost=coverage_cost)
        run_repo.record(coverage_run)  # type: ignore[attr-defined]
        found = job_repo.get_job(ctx.user_id, job_id)
        assert found is not None
        _job, requirements = found
        rows = []
        for requirement in requirements:
            row = RequirementCoverage(
                user_id=ctx.user_id,
                requirement_id=requirement.id,
                trace_id=ctx.trace_id,
                status="evidenced",
                cited_span_ids=[],
                evidence_note="Traces cleanly.",
            )
            job_repo.record_coverage(row)
            rows.append(row)
        return rows

    def fake_generate_draft(
        ctx: RequestContext,
        job_repo: _FakeJobRepository,
        grounding_repo: object,
        run_repo: object,
        job_id: uuid.UUID,
        kind: str,
    ) -> Draft:
        calls["draft"].append(job_id)
        run_repo.record(_run_record(ctx, component="generate", stage="draft", cost=draft_cost))  # type: ignore[attr-defined]
        run_repo.record(_run_record(ctx, component="gate", stage="baseline", cost=gate_cost))  # type: ignore[attr-defined]
        draft = Draft(
            user_id=ctx.user_id,
            job_id=job_id,
            kind=kind,  # type: ignore[arg-type]
            text="Led the platform team at Acme.",
            gate_result={
                "sentences": [
                    {
                        "index": 1,
                        "kind": "claim",
                        "verdict": "supported",
                        "drift_label": "supported",
                        "cited_span_ids": [],
                        "evidence_note": "Traces cleanly.",
                        "rule_flags": [],
                        "text": "Led the platform team at Acme.",
                    }
                ]
            },
            trace_id=ctx.trace_id,
        )
        job_repo.record_draft(draft)
        return draft

    monkeypatch.setattr(generate_results, "extract_requirements", fake_extract)
    monkeypatch.setattr(generate_results, "run_coverage", fake_run_coverage)
    monkeypatch.setattr(generate_results, "generate_draft", fake_generate_draft)
    return calls


def _base_ctx() -> RequestContext:
    return RequestContext(
        user_id=uuid.uuid4(),  # overwritten per-candidate inside run_demo
        anthropic_api_key="unused",
        database_url="unused-in-tests",
        model="claude-opus-5",
    )


# --- scratch user id -------------------------------------------------------------


def test_scratch_user_id_is_deterministic() -> None:
    a = generate_results.scratch_user_id("mireille-fontaine.md")
    b = generate_results.scratch_user_id("mireille-fontaine.md")
    assert a == b


def test_scratch_user_id_differs_per_candidate() -> None:
    a = generate_results.scratch_user_id("mireille-fontaine.md")
    b = generate_results.scratch_user_id("tobias-reyes.md")
    assert a != b


def test_scratch_user_id_never_equals_local_user_id() -> None:
    for name in ("mireille-fontaine.md", "tobias-reyes.md", "ingrid-solberg.md"):
        assert generate_results.scratch_user_id(name) != LOCAL_USER_ID


def test_scratch_user_id_raises_loudly_if_it_ever_collides_with_local_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a realistic collision (uuid5 practically never lands on one fixed
    UUID) -- this forces the collision by monkeypatching uuid5 itself, to
    prove the guard actually fires rather than being dead code.
    """
    monkeypatch.setattr(generate_results.uuid, "uuid5", lambda *_: LOCAL_USER_ID)
    with pytest.raises(AssertionError, match="LOCAL_USER_ID"):
        generate_results.scratch_user_id("whatever.md")


# --- one combination, JSON round trip --------------------------------------------


def test_output_json_round_trips_and_has_the_required_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fakes(monkeypatch)
    repos = _make_repos()
    combo = generate_results.Combination(MIREILLE, LEDGERBRIDGE)

    report = generate_results.run_demo(repos, _base_ctx(), [combo], tmp_path)

    assert not report.aborted
    assert [o.status for o in report.outcomes] == ["ran"]
    assert calls["extract"] == [LEDGERBRIDGE.read_text(encoding="utf-8")]
    assert len(calls["coverage"]) == 1
    assert len(calls["draft"]) == 1

    result_path = tmp_path / combo.result_filename
    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))

    # candidate/job identity
    assert data["candidate"]["source_file"] == "mireille-fontaine.md"
    assert data["candidate"]["user_id"] == str(generate_results.scratch_user_id(MIREILLE.name))
    assert data["job"]["source_file"] == LEDGERBRIDGE.name
    assert data["job"]["employer"] == "Acme"

    # extracted requirements
    assert len(data["requirements"]) == 1
    assert data["requirements"][0]["text"] == "5+ years of Python"
    assert data["requirements"][0]["necessity"] == "essential"

    # per-requirement coverage: evidence_note present, cited_span_ids a list
    assert len(data["coverage"]) == 1
    coverage_row = data["coverage"][0]
    assert coverage_row["status"] == "evidenced"
    assert coverage_row["evidence_note"] == "Traces cleanly."
    assert coverage_row["cited_span_ids"] == []

    # draft text and gate verdicts
    assert data["draft"]["text"] == "Led the platform team at Acme."
    gate_sentences = data["gate"]["sentences"]
    assert len(gate_sentences) == 1
    sentence = gate_sentences[0]
    for field_name in (
        "index",
        "kind",
        "verdict",
        "drift_label",
        "cited_span_ids",
        "evidence_note",
        "rule_flags",
    ):
        assert field_name in sentence
    assert "reason" not in sentence  # the renamed-away field must never reappear

    # spans: the real corpus was really ingested (no model call involved),
    # and every span id has resolvable text alongside it.
    assert len(data["spans"]) > 0
    for span_id, span_info in data["spans"].items():
        uuid.UUID(span_id)  # well-formed
        assert isinstance(span_info["text"], str) and span_info["text"]

    # runs: real token/cost/latency numbers, one row per model call (extract,
    # coverage, draft, gate baseline).
    assert len(data["runs"]) == 4
    stages = {r["stage"] for r in data["runs"]}
    assert stages == {"extract_requirements", "coverage", "draft", "baseline"}
    for run in data["runs"]:
        assert run["cost_usd"] is not None

    assert Decimal(data["cost_usd_total"]) == Decimal("0.01") + Decimal("0.15") + Decimal(
        "0.20"
    ) + Decimal("0.10")


def test_ingestion_is_isolated_per_candidate_scratch_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The isolation requirement: candidate A's spans must never leak into
    candidate B's scratch user, even though both are ingested through the
    same shared `_FakeCorpusStore` (the real Postgres tables are shared the
    same way, scoped only by `user_id`).
    """
    calls = _install_fakes(monkeypatch)
    repos = _make_repos()
    combos = [
        generate_results.Combination(MIREILLE, LEDGERBRIDGE),
        generate_results.Combination(TOBIAS, LEDGERBRIDGE),
    ]

    generate_results.run_demo(repos, _base_ctx(), combos, tmp_path)

    mireille_uid = generate_results.scratch_user_id(MIREILLE.name)
    tobias_uid = generate_results.scratch_user_id(TOBIAS.name)
    assert mireille_uid != tobias_uid

    mireille_spans = repos.grounding.all_spans(mireille_uid)
    tobias_spans = repos.grounding.all_spans(tobias_uid)
    assert mireille_spans and tobias_spans
    assert {s.id for s in mireille_spans}.isdisjoint({s.id for s in tobias_spans})

    # Extraction happened once for this one shared job ad, reused for both
    # candidates -- not once per candidate.
    assert len(calls["extract"]) == 1
    assert len(calls["coverage"]) == 2
    assert len(calls["draft"]) == 2


# --- --limit -----------------------------------------------------------------


def test_limit_runs_exactly_n_combinations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fakes(monkeypatch)
    repos = _make_repos()
    combos = [
        generate_results.Combination(MIREILLE, LEDGERBRIDGE),
        generate_results.Combination(MIREILLE, NIMBUS),
        generate_results.Combination(TOBIAS, LEDGERBRIDGE),
    ]

    report = generate_results.run_demo(repos, _base_ctx(), combos, tmp_path, limit=1)

    assert len(calls["draft"]) == 1
    ran = [o for o in report.outcomes if o.status == "ran"]
    assert len(ran) == 1
    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    # the other two combinations were left untouched, resumable
    assert len(report.remaining) == 2


# --- skip existing / --force --------------------------------------------------


def test_existing_result_file_is_skipped_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fakes(monkeypatch)
    repos = _make_repos()
    combo = generate_results.Combination(MIREILLE, LEDGERBRIDGE)

    (tmp_path / combo.result_filename).write_text(
        json.dumps({"cost_usd_total": "0.30", "fake": "pre-existing"}), encoding="utf-8"
    )

    report = generate_results.run_demo(repos, _base_ctx(), [combo], tmp_path)

    assert calls["extract"] == []
    assert calls["coverage"] == []
    assert calls["draft"] == []
    assert [o.status for o in report.outcomes] == ["skipped"]
    # the pre-existing file must be untouched
    on_disk = json.loads((tmp_path / combo.result_filename).read_text(encoding="utf-8"))
    assert on_disk == {"cost_usd_total": "0.30", "fake": "pre-existing"}


def test_force_overrides_an_existing_result_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fakes(monkeypatch)
    repos = _make_repos()
    combo = generate_results.Combination(MIREILLE, LEDGERBRIDGE)

    (tmp_path / combo.result_filename).write_text(
        json.dumps({"cost_usd_total": "0.30", "fake": "pre-existing"}), encoding="utf-8"
    )

    report = generate_results.run_demo(repos, _base_ctx(), [combo], tmp_path, force=True)

    assert len(calls["draft"]) == 1
    assert [o.status for o in report.outcomes] == ["ran"]
    on_disk = json.loads((tmp_path / combo.result_filename).read_text(encoding="utf-8"))
    assert on_disk.get("fake") is None
    assert on_disk["draft"]["text"] == "Led the platform team at Acme."


# --- metering / abort ---------------------------------------------------------


def test_metering_aborts_before_starting_a_further_combination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each combination here costs $1.00 (0.1 extract + 0.3 coverage + 0.4
    draft + 0.2 gate). With 3 combinations total, one completed combination
    projects to $3.00 -- above a $2.00 cap -- so the script must stop after
    combination 1 and never touch combination 2 or 3.
    """
    calls = _install_fakes(
        monkeypatch,
        extract_cost=Decimal("0.10"),
        coverage_cost=Decimal("0.30"),
        draft_cost=Decimal("0.40"),
        gate_cost=Decimal("0.20"),
    )
    repos = _make_repos()
    combos = [
        generate_results.Combination(MIREILLE, LEDGERBRIDGE),
        generate_results.Combination(MIREILLE, NIMBUS),
        generate_results.Combination(TOBIAS, LEDGERBRIDGE),
    ]

    report = generate_results.run_demo(
        repos, _base_ctx(), combos, tmp_path, max_spend=Decimal("2.00")
    )

    assert report.aborted
    assert len(calls["draft"]) == 1  # never started combination 2 or 3
    ran = [o for o in report.outcomes if o.status == "ran"]
    assert len(ran) == 1
    assert report.projected_total is not None
    assert report.projected_total > Decimal("2.00")
    # completed combination's file stays on disk after the abort
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(report.remaining) == 2


def test_metering_does_not_abort_when_projection_stays_under_the_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fakes(
        monkeypatch,
        extract_cost=Decimal("0.01"),
        coverage_cost=Decimal("0.05"),
        draft_cost=Decimal("0.05"),
        gate_cost=Decimal("0.02"),
    )
    repos = _make_repos()
    combos = [
        generate_results.Combination(MIREILLE, LEDGERBRIDGE),
        generate_results.Combination(TOBIAS, LEDGERBRIDGE),
    ]

    report = generate_results.run_demo(
        repos, _base_ctx(), combos, tmp_path, max_spend=Decimal("8.00")
    )

    assert not report.aborted
    assert len(calls["draft"]) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert report.remaining == []


def test_resumed_invocation_seeds_the_projection_from_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-existing result file (from a previous invocation) must count
    toward the running mean immediately -- even before any combination runs
    THIS invocation -- so a second `--limit 1` run can abort on its first
    combination if history alone already projects over the cap.
    """
    calls = _install_fakes(monkeypatch)
    combo_done = generate_results.Combination(MIREILLE, LEDGERBRIDGE)
    combo_next = generate_results.Combination(MIREILLE, NIMBUS)
    combo_last = generate_results.Combination(TOBIAS, LEDGERBRIDGE)

    (tmp_path / combo_done.result_filename).write_text(
        json.dumps({"cost_usd_total": "3.00"}), encoding="utf-8"
    )

    repos = _make_repos()
    report = generate_results.run_demo(
        repos,
        _base_ctx(),
        [combo_done, combo_next, combo_last],
        tmp_path,
        max_spend=Decimal("5.00"),
    )

    # combo_done is skipped (no model call), but its $3.00 seeds the mean, so
    # after combo_next's own (cheap) run the projected total for all 3 is
    # still dominated by that $3.00 history and must trip the cap immediately.
    assert report.aborted
    assert len(calls["draft"]) == 1  # only combo_next ran
    assert not (tmp_path / combo_last.result_filename).exists()
