"""One end-to-end test: real Anthropic API, real Postgres. Costs money every run --
keep this file to exactly one test. Not marked `integration`, only `e2e`, so it is
never picked up by `pytest -m integration` and needs an explicit `pytest -m e2e`.

Requires `ANTHROPIC_API_KEY`, `docker compose up -d`, and `alembic upgrade head`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from jfl_core.context import RequestContext
from jfl_core.db.tables import users
from jfl_core.ingest.ingest import run_ingestion
from jfl_core.storage.postgres import (
    PostgresGroundingRepository,
    PostgresIngestRepository,
    PostgresRunRepository,
)
from jfl_gate.gate import check_text
from sqlalchemy import create_engine, insert
from sqlalchemy.engine import Connection

pytestmark = pytest.mark.e2e

_CV = """# Jamie Rivera

## Kaluza

- Led the platform team of 4 engineers rebuilding the metering pipeline.
- Reduced billing-run latency from 6 hours to 40 minutes.

## Earlier career

- Reviewed pull requests for the payments team's fraud-scoring service.
"""


def test_obvious_supported_inflated_and_framing_sentences_get_the_right_shape(
    tmp_path: Path,
) -> None:
    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    engine = create_engine(url)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "cv.md").write_text(_CV, encoding="utf-8")

    with engine.connect() as conn:
        tx = conn.begin()
        try:
            _run_case(conn, corpus_dir)
        finally:
            tx.rollback()  # leave the database exactly as this test found it


def _run_case(conn: Connection, corpus_dir: Path) -> None:
    user_id = uuid.uuid4()
    conn.execute(insert(users).values(id=user_id, email=f"{user_id}@test.invalid"))

    ctx = RequestContext(
        user_id=user_id,
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        database_url="unused-in-tests",
    )

    ingest_repo = PostgresIngestRepository(conn)
    summary = run_ingestion(ctx, ingest_repo, corpus_dir)
    assert summary.spans_created > 0

    text = (
        "Led the platform team of 4 engineers rebuilding the metering pipeline. "
        "Owned the payments team's fraud-scoring service end to end. "
        "Wanting to work closer to the infrastructure layer, they moved into platform "
        "engineering."
    )

    grounding_repo = PostgresGroundingRepository(conn)
    run_repo = PostgresRunRepository(conn)
    result = check_text(ctx, grounding_repo, run_repo, text)

    assert len(result.sentences) == 3

    # A claim that traces cleanly to a bullet in the corpus.
    supported = next(s for s in result.sentences if "Led the platform team" in s.text)
    assert supported.verdict == "supported"
    assert supported.cited_span_ids

    # The corpus says "reviewed", not "owned end to end" -- adjacency_substitution,
    # a hard fail.
    inflated = next(s for s in result.sentences if "Owned the payments team" in s.text)
    assert inflated.verdict == "unsupported"

    # Pure motivation framing -- must never be flagged, regardless of the corpus.
    framing = next(s for s in result.sentences if "Wanting to work closer" in s.text)
    assert framing.kind == "framing"
    assert framing.verdict == "supported"
