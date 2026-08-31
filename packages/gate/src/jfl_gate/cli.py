"""jfl CLI. Entry point declared in packages/gate/pyproject.toml as `jfl = jfl_gate.cli:app`."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from jfl_core.context import RequestContext
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresIngestRepository,
    PostgresRunRepository,
)
from sqlalchemy import Connection, create_engine
from sqlalchemy.exc import OperationalError

from jfl_gate.gate import GateError, check_text
from jfl_gate.input import read_input
from jfl_gate.schema import SentenceResult

app = typer.Typer()

_VERDICT_COLOR = {
    "supported": typer.colors.GREEN,
    "review": typer.colors.YELLOW,
    "unsupported": typer.colors.RED,
}


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


@app.callback()
def _callback() -> None:
    """job-for-life: the grounding gate for AI-generated job application text."""


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
    if file is not None:
        input_text = read_input(file)
    elif text is not None:
        input_text = text
    else:
        typer.echo("Provide text as an argument, or pass --file.", err=True)
        raise typer.Exit(code=1)

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
    counts = Counter(s.verdict for s in sentences)
    parts = [
        typer.style(f"{counts[v]} {v}", fg=_VERDICT_COLOR[v])
        for v in ("supported", "review", "unsupported")
        if counts[v]
    ]
    typer.echo(f"\n{len(sentences)} sentences: " + ", ".join(parts))


def _print_sentence(sentence: SentenceResult) -> None:
    color = _VERDICT_COLOR[sentence.verdict]
    typer.secho(
        f"[{sentence.verdict.upper():<11}] {sentence.drift_label:<22} {sentence.text}",
        fg=color,
    )
    if sentence.cited_span_ids:
        cited = ", ".join(str(span_id) for span_id in sentence.cited_span_ids)
        typer.echo(f"    cites: {cited}")
    typer.echo(f"    reason: {sentence.reason}")


if __name__ == "__main__":
    app()
