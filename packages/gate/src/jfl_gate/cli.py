"""jfl CLI. Entry point declared in packages/gate/pyproject.toml as `jfl = jfl_gate.cli:app`."""

from __future__ import annotations

from pathlib import Path

import typer
from jfl_core.context import RequestContext
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresIngestRepository,
    PostgresRunRepository,
)
from sqlalchemy import create_engine

from jfl_gate.gate import GateError, check_text
from jfl_gate.schema import SentenceResult

app = typer.Typer()

_VERDICT_COLOR = {
    "supported": typer.colors.GREEN,
    "review": typer.colors.YELLOW,
    "unsupported": typer.colors.RED,
}


@app.callback()
def _callback() -> None:
    """job-for-life: the grounding gate for AI-generated job application text."""


@app.command()
def ingest() -> None:
    """Rebuild the corpus index in Postgres from corpus/**/*.md."""
    ctx = RequestContext.from_env()
    engine = create_engine(ctx.database_url)
    with engine.begin() as conn:
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
        None, "--file", help="Read the text to check from a file instead of the argument."
    ),
) -> None:
    """Run the baseline grounding gate over TEXT (or --file) against the corpus."""
    if file is not None:
        input_text = file.read_text(encoding="utf-8")
    elif text is not None:
        input_text = text
    else:
        typer.echo("Provide text as an argument, or pass --file.", err=True)
        raise typer.Exit(code=1)

    ctx = RequestContext.from_env()
    engine = create_engine(ctx.database_url)
    # Catch GateError *inside* the `with` block so the transaction commits even on
    # failure -- check_text already wrote a `runs` row for the failure on this same
    # connection, and letting the exception escape the block would roll that back.
    error: str | None = None
    with engine.begin() as conn:
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
