"""jfl CLI. Entry point declared in packages/cli/pyproject.toml as `jfl = jfl_cli.main:app`.

A delivery mechanism, not a domain -- see CLAUDE.md's repo shape. Depends on
every other package; nothing depends on it.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from jfl_core.context import RequestContext
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.models import Draft, DraftKind, Job, JobRequirement, RequirementCoverage

# PostgresIngestRepository backs `ingest` and `answer` -- the gap-answer write-back
# re-ingests the corpus, it never writes a grounding span directly. See CLAUDE.md's
# decisions log, "A gap answer lands in corpus markdown, not the database."
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresIngestRepository,
    PostgresJobRepository,
    PostgresRunRepository,
)
from jfl_gate.gate import GateError, check_text
from jfl_gate.input import read_input
from jfl_gate.schema import GateOutput, SentenceResult
from jfl_generate.draft import generate_draft
from jfl_generate.errors import GenerateError
from jfl_generate.jobs import add_job, answer_question, run_coverage
from sqlalchemy import Connection, create_engine
from sqlalchemy.exc import OperationalError

app = typer.Typer()
job_app = typer.Typer(help="Anchor work on a job: requirements, corpus coverage, gap questions.")
app.add_typer(
    job_app,
    name="job",
    help="Anchor work on a job: requirements, corpus coverage, gap questions.",
)

_VERDICT_COLOR = {
    "supported": typer.colors.GREEN,
    "review": typer.colors.YELLOW,
    "unsupported": typer.colors.RED,
}

# Framing is neither a pass nor a flag -- it is the absence of a check, and it gets
# its own colour so it can never be mistaken for either at a glance.
_FRAMING_COLOR = typer.colors.BLUE

_COVERAGE_COLOR = {
    "evidenced": typer.colors.GREEN,
    "partial": typer.colors.YELLOW,
    "absent": typer.colors.YELLOW,
    "contradicted": typer.colors.RED,
}

# CLI vocabulary ("cv") vs the stored `drafts.kind` CHECK values -- "cv_bullets" is
# what the table and the drift taxonomy call it, "cv" is what a user types.
_DRAFT_KIND_MAP: dict[str, DraftKind] = {"cv": "cv_bullets", "cover_letter": "cover_letter"}


def _redact(url: str) -> str:
    """Never print the password, even in an error the user is about to paste."""
    return re.sub(r"://[^:/@]+:[^@]*@", "://***:***@", url)


@contextmanager
def _transaction(ctx: RequestContext) -> Iterator[Connection]:
    """One transaction, with a readable message when the database is unreachable.

    Without this, a stopped container surfaces as ~17k characters of SQLAlchemy
    and rich traceback, which buries the one fact that matters.
    """
    engine = create_engine(ctx.database_url)
    try:
        with engine.begin() as conn:
            yield conn
    except OperationalError as e:
        typer.secho(f"cannot reach Postgres at {_redact(ctx.database_url)}", fg="red", err=True)
        typer.echo("  start it with: docker compose up -d", err=True)
        raise typer.Exit(code=1) from e


def _read_text_or_file(text: str | None, file: Path | None) -> str:
    """Shared by `check` and `job add`: text as an argument, or --file (PDF included)."""
    if file is not None:
        return read_input(file)
    if text is not None:
        return text
    typer.echo("Provide text as an argument, or pass --file.", err=True)
    raise typer.Exit(code=1)


@app.callback()
def _callback() -> None:
    """job-for-life: the grounding gate and coverage report for AI-generated job
    application text.
    """


@app.command()
def ingest() -> None:
    """Rebuild the corpus index in Postgres from corpus/**/*.md."""
    ctx = RequestContext.from_env()
    with _transaction(ctx) as conn:
        repo = PostgresIngestRepository(conn)
        summary = run_ingestion(ctx, repo)
    typer.echo(f"documents seen: {summary.documents_seen} (retired {summary.documents_retired})")
    typer.echo(
        f"spans created: {summary.spans_created}, updated: {summary.spans_updated}, "
        f"retired: {summary.spans_retired}"
    )


@app.command()
def check(
    text: str | None = typer.Argument(
        None, help="Text to check. Omit this and pass --file instead."
    ),
    file: Path | None = typer.Option(  # noqa: B008 -- typer's documented pattern
        None, "--file", help="Read the text from a file instead. .pdf is extracted."
    ),
) -> None:
    """Run the baseline grounding gate over TEXT (or --file) against the corpus."""
    input_text = _read_text_or_file(text, file)

    ctx = RequestContext.from_env()
    # Catch GateError *inside* the `with` block so the transaction commits even on
    # failure -- check_text already wrote a `runs` row for the failure on this same
    # connection, and letting the exception escape the block would roll that back.
    error: str | None = None
    with _transaction(ctx) as conn:
        grounding_repo = PostgresGroundingRepository(conn)
        run_repo = PostgresRunRepository(conn)
        try:
            result = check_text(ctx, grounding_repo, run_repo, input_text)
        except GateError as e:
            result = None
            error = str(e)

    if error is not None or result is None:
        typer.echo(f"gate failed: {error}", err=True)
        raise typer.Exit(code=1)

    for sentence in result.sentences:
        _print_sentence(sentence)
    _print_summary(result.sentences)


def _print_summary(sentences: list[SentenceResult]) -> None:
    """Framing is tallied on its own, never folded into `supported`.

    The prompt forces every framing sentence to `verdict="supported"`, so counting
    verdicts flat inflates the supported total with sentences nothing ever checked.
    A user reading "40 supported" would take it as forty verified claims.
    """
    checked = [s for s in sentences if s.kind != "framing"]
    framing = len(sentences) - len(checked)
    counts = Counter(s.verdict for s in checked)
    parts = [
        typer.style(f"{counts[v]} {v}", fg=_VERDICT_COLOR[v])
        for v in ("supported", "review", "unsupported")
        if counts[v]
    ]
    if framing:
        parts.append(typer.style(f"{framing} not checked (framing)", fg=_FRAMING_COLOR))
    typer.echo(f"\n{len(sentences)} sentences: " + ", ".join(parts))


def _print_sentence(sentence: SentenceResult) -> None:
    """A framing sentence renders as NOT CHECKED, never as SUPPORTED.

    Framing is the one path through the gate with no check anywhere: the prompt
    forces its verdict to `supported`, it is never compared against the corpus, and
    the deterministic rule tier is forbidden from touching it. Displaying that as
    SUPPORTED asserts a verification that did not happen -- and a claim the model
    misfiles as framing would inherit that false assurance silently. NOT CHECKED is
    what actually occurred.
    """
    is_framing = sentence.kind == "framing"
    label = "NOT CHECKED" if is_framing else sentence.verdict.upper()
    color = _FRAMING_COLOR if is_framing else _VERDICT_COLOR[sentence.verdict]
    typer.secho(
        f"[{label:<11}] {sentence.drift_label:<22} {sentence.text}",
        fg=color,
    )
    if sentence.cited_span_ids:
        cited = ", ".join(str(span_id) for span_id in sentence.cited_span_ids)
        typer.echo(f"    cites: {cited}")
    typer.echo(f"    reason: {sentence.reason}")


@job_app.command("add")
def job_add(
    text: str | None = typer.Argument(None, help="Job ad text. Omit this and pass --file instead."),
    file: Path | None = typer.Option(  # noqa: B008 -- typer's documented pattern
        None, "--file", help="Read the job ad from a file instead. .pdf is extracted."
    ),
) -> None:
    """Extract requirements from a pasted job ad and store them."""
    input_text = _read_text_or_file(text, file)

    ctx = RequestContext.from_env()
    error: str | None = None
    job: Job | None = None
    requirements: list[JobRequirement] = []
    # Same pattern as `check`: catch inside the `with` block so a failed
    # extraction's `runs` row still commits.
    with _transaction(ctx) as conn:
        run_repo = PostgresRunRepository(conn)
        job_repo = PostgresJobRepository(conn)
        try:
            job, requirements = add_job(ctx, run_repo, job_repo, input_text)
        except GenerateError as e:
            error = str(e)

    if error is not None or job is None:
        typer.echo(f"extraction failed: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"job {job.id}")
    typer.echo(f"employer: {job.employer or '(unknown)'}")
    typer.echo(f"title: {job.title or '(unknown)'}")
    typer.echo(f"location: {job.location or '(unknown)'}")
    for i, req in enumerate(requirements, start=1):
        typer.echo(f"  {i}. [{req.necessity}] {req.text}")


@job_app.command("list")
def job_list() -> None:
    """List stored jobs."""
    ctx = RequestContext.from_env()
    with _transaction(ctx) as conn:
        job_repo = PostgresJobRepository(conn)
        summaries = job_repo.list_jobs(ctx.user_id)

    if not summaries:
        typer.echo("no jobs stored yet")
        return
    for s in summaries:
        employer = s.employer or "(unknown employer)"
        title = s.title or "(unknown title)"
        typer.echo(
            f"{s.id}  {employer:<30} {title:<30} {s.requirement_count:>3} reqs  "
            f"{s.created_at:%Y-%m-%d}"
        )


@job_app.command("coverage")
def job_coverage(
    job_id: uuid.UUID = typer.Argument(  # noqa: B008 -- typer's documented pattern
        ..., help="Job id, from `jfl job list`."
    ),
) -> None:
    """Check corpus coverage for a job's requirements and print a verdict per one."""
    ctx = RequestContext.from_env()
    error: str | None = None
    coverage_rows: list[RequirementCoverage] = []
    with _transaction(ctx) as conn:
        grounding_repo = PostgresGroundingRepository(conn)
        run_repo = PostgresRunRepository(conn)
        job_repo = PostgresJobRepository(conn)
        try:
            coverage_rows = run_coverage(ctx, grounding_repo, run_repo, job_repo, job_id)
        except GenerateError as e:
            error = str(e)

        # Read requirement text and open questions on the same connection, whether
        # or not the coverage call itself succeeded.
        found = job_repo.get_job(ctx.user_id, job_id)
        requirements_by_id = {r.id: r for r in found[1]} if found is not None else {}
        open_questions = job_repo.list_open_questions(ctx.user_id, job_id) if found else []

    if error is not None:
        typer.echo(f"coverage failed: {error}", err=True)
        raise typer.Exit(code=1)

    counts: Counter[str] = Counter()
    for row in coverage_rows:
        requirement = requirements_by_id.get(row.requirement_id)
        requirement_text = requirement.text if requirement else "(unknown requirement)"
        color = _COVERAGE_COLOR[row.status]
        typer.secho(f"[{row.status.upper():<12}] {requirement_text}", fg=color)
        if row.cited_span_ids:
            cited = ", ".join(str(span_id) for span_id in row.cited_span_ids)
            typer.echo(f"    cites: {cited}")
        typer.echo(f"    reason: {row.reason}")
        counts[row.status] += 1

    parts = [
        typer.style(f"{counts[s]} {s}", fg=_COVERAGE_COLOR[s])
        for s in ("evidenced", "partial", "absent", "contradicted")
        if counts[s]
    ]
    typer.echo(f"\n{len(coverage_rows)} requirements: " + ", ".join(parts))

    if open_questions:
        typer.echo("\nopen questions:")
        for q in open_questions:
            typer.echo(f"  {q.id}  {q.question}")


@job_app.command("questions")
def job_questions(
    job_id: uuid.UUID = typer.Argument(  # noqa: B008 -- typer's documented pattern
        ..., help="Job id, from `jfl job list`."
    ),
) -> None:
    """List open gap questions for a job, so they can be answered with `jfl answer`."""
    ctx = RequestContext.from_env()
    with _transaction(ctx) as conn:
        job_repo = PostgresJobRepository(conn)
        questions = job_repo.list_open_questions(ctx.user_id, job_id)

    if not questions:
        typer.echo("no open questions")
        return
    for q in questions:
        typer.echo(f"{q.id}  {q.question}")


@app.command()
def answer(
    question_id: uuid.UUID = typer.Argument(  # noqa: B008 -- typer's documented pattern
        ..., help="Gap question id, from `jfl job questions`."
    ),
    answer_text: str = typer.Argument(  # noqa: B008 -- typer's documented pattern
        ...,
        help=(
            "The answer. Stored word-for-word as a corpus fact -- no model rewrites it -- "
            'so write a self-contained statement (e.g. "I led the migration and was '
            'on-call for it"), not a bare "yes".'
        ),
    ),
) -> None:
    """Answer a gap question. No model call: your words become the corpus fact verbatim."""
    ctx = RequestContext.from_env()
    error: str | None = None
    span_id: uuid.UUID | None = None
    # Same pattern as `check`/`job add`: catch inside the `with` block so a
    # failure still commits whatever it needs to.
    with _transaction(ctx) as conn:
        ingest_repo = PostgresIngestRepository(conn)
        job_repo = PostgresJobRepository(conn)
        try:
            span_id = answer_question(ctx, ingest_repo, job_repo, question_id, answer_text)
        except GenerateError as e:
            error = str(e)

    if error is not None or span_id is None:
        typer.echo(f"answer failed: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"stored as span {span_id}")


@app.command()
def draft(
    job_id: uuid.UUID = typer.Argument(  # noqa: B008 -- typer's documented pattern
        ..., help="Job id, from `jfl job list`."
    ),
    kind: str = typer.Option(  # noqa: B008 -- typer's documented pattern
        ..., "--kind", help="cv or cover_letter."
    ),
) -> None:
    """Generate a draft against the job's requirements and latest corpus coverage,
    then run the claim gate on it automatically. Requires `jfl job coverage JOB_ID`
    to have been run first -- this command never runs coverage itself, that would
    be a second, unbudgeted model call.

    A flagged draft is still printed in full: the claim gate informs, it never
    blocks (see CLAUDE.md, "How the claim gate behaves").
    """
    if kind not in _DRAFT_KIND_MAP:
        typer.echo(f"--kind must be one of: {', '.join(_DRAFT_KIND_MAP)}", err=True)
        raise typer.Exit(code=1)
    stored_kind = _DRAFT_KIND_MAP[kind]

    ctx = RequestContext.from_env()
    error: str | None = None
    result: Draft | None = None
    # Same pattern as `check`/`job add`/`job coverage`: catch inside the `with`
    # block so a failed call's `runs` row(s) still commit.
    with _transaction(ctx) as conn:
        grounding_repo = PostgresGroundingRepository(conn)
        run_repo = PostgresRunRepository(conn)
        job_repo = PostgresJobRepository(conn)
        try:
            result = generate_draft(ctx, job_repo, grounding_repo, run_repo, job_id, stored_kind)
        except (GenerateError, GateError) as e:
            error = str(e)

        # Read open questions on the same connection regardless of outcome, same
        # as `job coverage`.
        open_questions = job_repo.list_open_questions(ctx.user_id, job_id)

    if error is not None or result is None:
        typer.echo(f"draft failed: {error}", err=True)
        raise typer.Exit(code=1)

    typer.echo(result.text)

    gate_output = GateOutput.model_validate(result.gate_result)
    typer.echo("")
    for sentence in gate_output.sentences:
        _print_sentence(sentence)
    _print_summary(gate_output.sentences)

    if open_questions:
        typer.echo("\nopen questions:")
        for q in open_questions:
            typer.echo(f"  {q.id}  {q.question}")


if __name__ == "__main__":
    app()
