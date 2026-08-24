"""jfl CLI. Entry point declared in packages/gate/pyproject.toml as `jfl = jfl_gate.cli:app`."""

from __future__ import annotations

import typer
from jfl_core.context import RequestContext
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.storage.postgres import PostgresIngestRepository
from sqlalchemy import create_engine

app = typer.Typer()


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


if __name__ == "__main__":
    app()
