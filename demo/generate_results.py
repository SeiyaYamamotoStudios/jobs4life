"""Generate the demo's committed results: real pipeline output over fictional
fixtures. See demo/fixtures/README.md and PLAN.md's W3.

For each of the 9 (candidate x job ad) combinations under demo/fixtures/, this
runs the real pipeline end to end -- extract -> coverage -> draft (which gates
internally) -- and writes one JSON file per combination to
demo/fixtures/results/. Those files are fiction over real pipeline output and
are committed (see demo/fixtures/README.md's golden-set-vs-fixture distinction).

Isolation (CLAUDE.md's decisions log, "the corpus never leaves the owner's
machine in v1", and the architectural constraint that every table carries
user_id): each fictional candidate is ingested under its OWN deterministic
scratch user id (`scratch_user_id`), NEVER `jfl_core.db.tables.LOCAL_USER_ID`
-- mixing fiction into the real corpus would corrupt the corpus the whole
project measures against. `scratch_user_id` asserts this loudly.

Metering: costs are not estimated up front and trusted -- measured per-call
spend varies too much for that (see the module docstring in the task brief /
PLAN.md's W3). Instead this meters as it goes: a real `RunRepository` (backed
by Postgres, never in-memory -- see CLAUDE.md, "runs under-reports when
scripts use in-memory repos") writes one `runs` row per model call, this
script also captures those rows to compute each combination's actual cost,
and after every completed combination it projects the total spend from the
running mean across all completed combinations (this run's, plus any already
on disk from a previous invocation) and aborts *before starting the next
combination* if that projection exceeds `--max-spend`. Already-written result
files are never touched by an abort.

Usage:
    uv run python demo/generate_results.py --limit 1
    uv run python demo/generate_results.py --max-spend 8.00
    uv run python demo/generate_results.py --force

Requires JFL_DATABASE_URL (Postgres up, `docker compose up -d`) and either
ANTHROPIC_API_KEY or an `ant auth login` profile -- same credential
resolution as everywhere else in this codebase (see RequestContext). This
script makes REAL, BILLED model calls; it is never invoked by the test suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from jfl_core.context import RequestContext
from jfl_core.db.tables import LOCAL_USER_ID
from jfl_core.db.tables import users as users_table
from jfl_core.ids import content_hash
from jfl_core.ids import job_id as derive_job_id
from jfl_core.ids import requirement_id as derive_requirement_id
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.models import (
    Draft,
    DraftKind,
    Job,
    JobRequirement,
    RequirementCoverage,
    RunRecord,
    Span,
)
from jfl_core.repositories import (
    GroundingRepository,
    IngestRepository,
    JobRepository,
    RunRepository,
)
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresIngestRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_gate.gate import GateError
from jfl_generate.draft import generate_draft
from jfl_generate.errors import GenerateError
from jfl_generate.extract import extract_requirements
from jfl_generate.jobs import run_coverage
from jfl_generate.schema import ExtractOutput
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection

# --------------------------------------------------------------------------
# Paths and defaults
# --------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
CANDIDATES_DIR = _FIXTURES_DIR / "candidates"
JOB_ADS_DIR = _FIXTURES_DIR / "job-ads"
RESULTS_DIR = _FIXTURES_DIR / "results"

DEFAULT_MAX_SPEND = Decimal("8.00")
DRAFT_KIND: DraftKind = "cv_bullets"

# Fixed, private namespace for deriving scratch user ids -- deliberately NOT
# jfl_core.ids.NS_ROOT, so a demo scratch id can never collide with a real
# corpus-derived id no matter how that namespace's downstream uses evolve.
_SCRATCH_NAMESPACE = uuid.UUID("d3f0c1a2-8b4e-5f6a-9c1d-2e3f4a5b6c7d")


def scratch_user_id(candidate_filename: str) -> uuid.UUID:
    """Deterministic scratch user id for one fictional candidate file --
    stable across re-runs (so a resumed invocation lands on the same rows),
    and never `LOCAL_USER_ID`. The assertion is not defensive filler: an
    agent once wrote fabricated data into the real corpus, and this is the
    one guard standing between that mistake and this script.
    """
    uid = uuid.uuid5(_SCRATCH_NAMESPACE, candidate_filename)
    if uid == LOCAL_USER_ID:
        raise AssertionError(
            f"scratch user id for {candidate_filename!r} collided with "
            "LOCAL_USER_ID -- refusing to touch the real user's corpus"
        )
    return uid


# --------------------------------------------------------------------------
# Combinations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Combination:
    candidate_path: Path
    job_path: Path

    @property
    def candidate_slug(self) -> str:
        return self.candidate_path.stem

    @property
    def job_slug(self) -> str:
        return self.job_path.stem

    @property
    def result_filename(self) -> str:
        return f"{self.candidate_slug}__{self.job_slug}.json"

    @property
    def label(self) -> str:
        return f"{self.candidate_slug} x {self.job_slug}"


def discover_combinations(candidates_dir: Path, job_ads_dir: Path) -> list[Combination]:
    """All (candidate, job ad) pairs, candidate-major, both sides sorted --
    the fixed iteration order that makes `--limit` and the per-combination
    metering log deterministic across runs.
    """
    candidates = sorted(candidates_dir.glob("*.md"))
    job_ads = sorted(job_ads_dir.glob("*.txt"))
    return [Combination(c, j) for c in candidates for j in job_ads]


# --------------------------------------------------------------------------
# Repositories
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Repos:
    """Every repository the pipeline needs, bundled so `run_demo` takes one
    argument instead of four. Production wires this to Postgres (see `main`);
    tests wire it to in-memory fakes conforming to the same Protocols from
    `jfl_core.repositories` -- `run_demo` itself never imports Postgres.
    """

    ingest: IngestRepository
    grounding: GroundingRepository
    run: RunRepository
    job: JobRepository


class _CapturingRunRepository:
    """Wraps a real `RunRepository` so every `.record()` still writes through
    (CLAUDE.md: exactly one `runs` row per model call, never skipped), while
    also collecting the records locally -- this is how the script gets the
    real token/cost/latency numbers into the result JSON and the running-mean
    projection, without a second read-back query against `runs`.
    """

    def __init__(self, inner: RunRepository) -> None:
        self._inner = inner
        self.records: list[RunRecord] = []

    def record(self, run: RunRecord) -> None:
        self._inner.record(run)
        self.records.append(run)


# --------------------------------------------------------------------------
# Per-combination pipeline
# --------------------------------------------------------------------------


def _ingest_candidate(repos: Repos, ctx: RequestContext, candidate_path: Path) -> None:
    """Ingest exactly ONE candidate's markdown file under `ctx.user_id`.

    Never point this at `demo/fixtures/candidates/` directly: `run_ingestion`
    walks every `*.md` under the directory it is given, so pointing it at the
    shared candidates directory would ingest all three fictional candidates
    into whichever scratch user happens to run first. A fresh temp directory
    holding a copy of just this one file is what keeps each candidate's
    corpus isolated to their own scratch user.
    """
    with tempfile.TemporaryDirectory(prefix="jfl-demo-corpus-") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / candidate_path.name).write_text(
            candidate_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        run_ingestion(ctx, repos.ingest, tmp_path)


def _persist_job_from_extraction(
    job_repo: JobRepository,
    user_id: uuid.UUID,
    raw_text: str,
    extracted: ExtractOutput,
) -> tuple[Job, list[JobRequirement]]:
    """Store a job + its requirements from an ALREADY-extracted `ExtractOutput`.

    Mirrors the storage half of `jfl_generate.jobs.add_job` exactly (same id
    derivation, same field mapping) but deliberately does not call
    `extract_requirements` itself: extraction happens once per job ad and is
    reused across all three candidates (see the module docstring and PLAN.md's
    W3 budget -- 3 extractions total, not 9), so the extraction call and the
    per-user storage of its result have to be decoupled. `add_job` couples
    them, which is correct for the interactive CLI (one job, one user) and
    wrong here.
    """
    jid = derive_job_id(user_id, raw_text)
    job = Job(
        id=jid,
        user_id=user_id,
        source="paste",
        employer=extracted.employer,
        title=extracted.title,
        location=extracted.location,
        raw_text=raw_text,
        content_hash=content_hash(raw_text),
    )
    job_repo.upsert_job(job)

    requirements = [
        JobRequirement(
            id=derive_requirement_id(jid, item.text),
            user_id=user_id,
            job_id=jid,
            ordinal=i,
            text=item.text,
            necessity=item.necessity,
        )
        for i, item in enumerate(extracted.requirements)
    ]
    job_repo.replace_requirements(user_id, jid, requirements)
    return job, requirements


def _build_result_json(
    combo: Combination,
    user_id: uuid.UUID,
    job: Job,
    requirements: list[JobRequirement],
    coverage_rows: list[RequirementCoverage],
    draft: Draft,
    spans: list[Span],
    run_records: list[RunRecord],
) -> dict[str, Any]:
    """Everything the demo page needs to render this combination from this
    file alone. Span ids cited in `coverage` or `gate` resolve against
    `spans`: the full grounding set actually sent to the model for this
    combination (`GroundingRepository.all_spans`, the same call the pipeline
    itself makes) -- not just the cited subset, so a citation added to the
    prompt later without a corresponding code change here still resolves.
    """
    cost_total = sum((r.cost_usd or Decimal(0) for r in run_records), start=Decimal(0))
    return {
        "candidate": {
            "user_id": str(user_id),
            "source_file": combo.candidate_path.name,
            "slug": combo.candidate_slug,
        },
        "job": {
            "id": str(job.id),
            "employer": job.employer,
            "title": job.title,
            "location": job.location,
            "source_file": combo.job_path.name,
            "slug": combo.job_slug,
            "raw_text": job.raw_text,
        },
        "requirements": [
            {
                "id": str(r.id),
                "ordinal": r.ordinal,
                "text": r.text,
                "necessity": r.necessity,
            }
            for r in requirements
        ],
        "coverage": [
            {
                "requirement_id": str(c.requirement_id),
                "status": c.status,
                "cited_span_ids": [str(s) for s in c.cited_span_ids],
                "evidence_note": c.evidence_note,
            }
            for c in coverage_rows
        ],
        "draft": {
            "id": str(draft.id),
            "kind": draft.kind,
            "text": draft.text,
            "trace_id": str(draft.trace_id),
        },
        # The claim gate's own output -- per-sentence index/kind/verdict/
        # drift_label/cited_span_ids/evidence_note/rule_flags. Field is
        # `evidence_note`, never `reason` (see the task brief and commits
        # f87bf78 / f0fd4ec: a `reason` field trips a live-API safety
        # classifier). `draft.gate_result` already carries that name.
        "gate": draft.gate_result,
        "spans": {
            str(s.id): {"text": s.text, "section_path": s.section_path, "kind": s.kind}
            for s in spans
        },
        "runs": [r.model_dump(mode="json") for r in run_records],
        "cost_usd_total": str(cost_total),
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _process_combination(
    repos: Repos,
    base_ctx: RequestContext,
    combo: Combination,
    extracted_by_job: dict[Path, ExtractOutput],
    ingested_users: set[uuid.UUID],
    commit: Callable[[], None],
    captured: list[RunRecord],
) -> dict[str, Any]:
    """Run extract (if not already cached for this job ad) -> coverage ->
    draft (which gates internally) for one combination, and return the JSON
    result. `captured` accumulates every `RunRecord` written along the way,
    by reference, so a caller can still see partial spend if this raises
    partway through.
    """
    user_id = scratch_user_id(combo.candidate_path.name)
    ctx_base = dataclasses.replace(base_ctx, user_id=user_id)

    if user_id not in ingested_users:
        _ingest_candidate(repos, ctx_base, combo.candidate_path)
        ingested_users.add(user_id)
        commit()

    job_ad_text = combo.job_path.read_text(encoding="utf-8")
    if combo.job_path not in extracted_by_job:
        extract_ctx = dataclasses.replace(ctx_base, trace_id=uuid.uuid4())
        capture = _CapturingRunRepository(repos.run)
        extracted_by_job[combo.job_path] = extract_requirements(extract_ctx, capture, job_ad_text)
        captured.extend(capture.records)
        commit()
    extracted = extracted_by_job[combo.job_path]

    job, requirements = _persist_job_from_extraction(repos.job, user_id, job_ad_text, extracted)

    coverage_ctx = dataclasses.replace(ctx_base, trace_id=uuid.uuid4())
    coverage_capture = _CapturingRunRepository(repos.run)
    coverage_rows = run_coverage(coverage_ctx, repos.grounding, coverage_capture, repos.job, job.id)
    captured.extend(coverage_capture.records)
    commit()

    draft_ctx = dataclasses.replace(ctx_base, trace_id=uuid.uuid4())
    draft_capture = _CapturingRunRepository(repos.run)
    draft = generate_draft(draft_ctx, repos.job, repos.grounding, draft_capture, job.id, DRAFT_KIND)
    captured.extend(draft_capture.records)
    commit()

    spans = repos.grounding.all_spans(user_id)
    return _build_result_json(
        combo, user_id, job, requirements, coverage_rows, draft, spans, captured
    )


# --------------------------------------------------------------------------
# Orchestration: metering, --limit, skip/--force
# --------------------------------------------------------------------------

ComboStatus = Literal["ran", "skipped", "errored"]


@dataclass(frozen=True, slots=True)
class ComboOutcome:
    combo: Combination
    status: ComboStatus
    cost_usd: Decimal | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    outcomes: list[ComboOutcome]
    aborted: bool
    projected_total: Decimal | None
    remaining: list[Combination]


def run_demo(
    repos: Repos,
    base_ctx: RequestContext,
    combos: Sequence[Combination],
    results_dir: Path,
    *,
    limit: int | None = None,
    max_spend: Decimal = DEFAULT_MAX_SPEND,
    force: bool = False,
    commit: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = print,
) -> RunReport:
    """Run the pipeline over `combos`, resuming from what is already on disk.

    Skips a combination whose result file already exists unless `force`.
    `limit` caps how many combinations actually get PROCESSED this
    invocation (a combination already satisfied by an existing file does not
    count against it). After every combination that finishes, projects the
    total spend for all of `combos` from the running mean (seeded from any
    pre-existing result files, so a resumed invocation still projects
    honestly) and aborts before starting the next one if that projection
    exceeds `max_spend`. Already-written files are never removed.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    total_combos = len(combos)

    outcomes: list[ComboOutcome] = []
    completed_costs: list[Decimal] = []
    to_process: list[Combination] = []

    for combo in combos:
        path = results_dir / combo.result_filename
        if path.exists() and not force:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                completed_costs.append(Decimal(str(existing["cost_usd_total"])))
            except Exception:  # noqa: BLE001 -- a malformed file just contributes nothing to the mean
                pass
            log(f"skip {combo.label} (already have {path.name})")
            outcomes.append(ComboOutcome(combo, "skipped"))
            continue
        to_process.append(combo)

    if limit is not None:
        to_process = to_process[:limit]

    extracted_by_job: dict[Path, ExtractOutput] = {}
    ingested_users: set[uuid.UUID] = set()

    aborted = False
    projected_total: Decimal | None = None

    for combo in to_process:
        captured: list[RunRecord] = []
        try:
            result = _process_combination(
                repos, base_ctx, combo, extracted_by_job, ingested_users, commit, captured
            )
        except (GenerateError, GateError) as e:
            cost = sum((r.cost_usd or Decimal(0) for r in captured), start=Decimal(0))
            completed_costs.append(cost)
            outcomes.append(ComboOutcome(combo, "errored", cost, str(e)))
            log(f"ERROR {combo.label}: {e} (partial spend ${cost:.2f}; not written, will retry)")
            continue

        cost = Decimal(result["cost_usd_total"])
        (results_dir / combo.result_filename).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        completed_costs.append(cost)
        outcomes.append(ComboOutcome(combo, "ran", cost))

        cumulative = sum(completed_costs, Decimal(0))
        mean = cumulative / len(completed_costs)
        projected_total = mean * total_combos
        log(
            f"{combo.label}: ${cost:.4f}  cumulative ${cumulative:.4f}  "
            f"projected total ${projected_total:.4f}"
        )

        if projected_total > max_spend:
            aborted = True
            remaining = [c for c in combos if not (results_dir / c.result_filename).exists()]
            log(
                f"ABORTING: projected total ${projected_total:.2f} exceeds --max-spend "
                f"${max_spend:.2f}. Completed {len(completed_costs)} combination(s) for "
                f"${cumulative:.2f} so far. {len(remaining)} combination(s) remain: "
                + ", ".join(c.label for c in remaining)
            )
            break

    remaining = [c for c in combos if not (results_dir / c.result_filename).exists()]
    return RunReport(
        outcomes=outcomes, aborted=aborted, projected_total=projected_total, remaining=remaining
    )


# --------------------------------------------------------------------------
# CLI entry point (real Postgres, real API -- never imported by tests)
# --------------------------------------------------------------------------


def _ensure_scratch_users(conn: Connection, combos: Sequence[Combination]) -> None:
    """No `UsersRepository` exists (accounts, domain 10, is unbuilt) -- this is
    the minimal, script-local exception to "no SQL above the repository
    layer": every table has a `user_id` FK to `users`, so a scratch user row
    must exist before ingestion writes a single span. `ON CONFLICT DO NOTHING`
    makes this safe to call on every invocation, including resumed ones.
    """
    for path in sorted({c.candidate_path for c in combos}):
        uid = scratch_user_id(path.name)
        conn.execute(
            pg_insert(users_table)
            .values(
                id=uid,
                email=f"demo+{path.stem}@job4life.invalid",
                display_name=f"Demo fixture: {path.stem}",
            )
            .on_conflict_do_nothing(index_elements=["id"])
        )
    conn.commit()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real job4life pipeline over the demo fixtures and write "
            "one result JSON per candidate x job-ad combination."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N combinations that still need work (default: all).",
    )
    parser.add_argument(
        "--max-spend",
        type=Decimal,
        default=DEFAULT_MAX_SPEND,
        help=(
            "Abort before starting a new combination once the projected total for all "
            "9 exceeds this, in USD (default: 8.00)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run combinations that already have a result file, instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_ctx = RequestContext.from_env()
    combos = discover_combinations(CANDIDATES_DIR, JOB_ADS_DIR)

    engine = create_engine(base_ctx.database_url)
    with engine.connect() as conn:
        _ensure_scratch_users(conn, combos)
        repos = Repos(
            ingest=PostgresIngestRepository(conn),
            grounding=PostgresGroundingRepository(conn),
            run=PostgresRunRepository(conn),
            job=PostgresJobRepository(conn),
        )
        report = run_demo(
            repos,
            base_ctx,
            combos,
            RESULTS_DIR,
            limit=args.limit,
            max_spend=args.max_spend,
            force=args.force,
            commit=conn.commit,
        )

    ran = [o for o in report.outcomes if o.status == "ran"]
    errored = [o for o in report.outcomes if o.status == "errored"]
    total_cost = sum((o.cost_usd or Decimal(0) for o in ran), start=Decimal(0))
    print(f"\n{len(ran)} ran, {len(errored)} errored, ${total_cost:.2f} spent this invocation.")
    if report.remaining:
        remaining_labels = ", ".join(c.label for c in report.remaining)
        print(f"{len(report.remaining)} combination(s) remain: {remaining_labels}")
    return 1 if report.aborted else 0


if __name__ == "__main__":
    sys.exit(main())
