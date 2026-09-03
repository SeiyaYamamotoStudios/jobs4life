"""Integration test for the gate's write path: a `runs` row lands in Postgres with
the right fields. Needs `docker compose up -d` and `alembic upgrade head`. The
Anthropic API is mocked -- this test is about the repository/schema, not the model.

Follows the transaction-rollback fixture pattern from test_schema_integration.py.
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from jfl_core.context import RequestContext
from jfl_core.db.tables import runs, users
from jfl_core.storage.postgres import PostgresGroundingRepository, PostgresRunRepository
from jfl_gate.gate import check_text
from jfl_gate.pricing import MODEL, compute_cost_usd
from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import Connection

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def engine():
    import os

    url = os.environ.get("JFL_DATABASE_URL", "postgresql+psycopg://jfl:jfl@localhost:5433/jfl")
    return create_engine(url)


@pytest.fixture
def conn(engine):
    with engine.connect() as c:
        tx = c.begin()
        yield c
        tx.rollback()


@pytest.fixture
def user(conn: Connection) -> uuid.UUID:
    uid = uuid.uuid4()
    conn.execute(insert(users).values(id=uid, email=f"{uid}@test.invalid"))
    return uid


def _ctx(user_id: uuid.UUID) -> RequestContext:
    return RequestContext(
        user_id=user_id, anthropic_api_key="test-key", database_url="unused-in-tests"
    )


def _fake_response() -> Message:
    payload = {
        "sentences": [
            {
                # 1-based index into the input sentences; the wire format no longer
                # echoes the text back (see gate._check_alignment).
                "index": 1,
                "kind": "claim",
                "verdict": "supported",
                "drift_label": "supported",
                "cited_span_ids": [],
                "evidence_note": "Matches the corpus.",
            }
        ]
    }
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text=json.dumps(payload))],
        model=MODEL,
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(
            input_tokens=1234,
            output_tokens=56,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=789,
        ),
    )


def test_a_successful_gate_call_writes_one_runs_row_with_correct_fields(
    monkeypatch: pytest.MonkeyPatch, conn: Connection, user: uuid.UUID
) -> None:
    class _FakeStream:
        def __enter__(self) -> _FakeStream:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def get_final_message(self) -> Message:
            return _fake_response()

    class _FakeMessages:
        # Mirrors `client.messages.stream(...)`: the gate streams because its
        # max_tokens is too high for a non-streaming request. It consumes no
        # partial output, so the double only has to yield the final message.
        def stream(self, **kwargs: object) -> _FakeStream:
            return _FakeStream()

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.messages = _FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)

    ctx = _ctx(user)
    grounding_repo = PostgresGroundingRepository(conn)
    run_repo = PostgresRunRepository(conn)

    result = check_text(ctx, grounding_repo, run_repo, "Led the platform team.")
    assert len(result.sentences) == 1

    row = conn.execute(select(runs).where(runs.c.trace_id == ctx.trace_id)).one()
    assert row.user_id == user
    assert row.component == "gate"
    assert row.stage == "baseline"
    assert row.model == MODEL
    assert row.tokens_in == 1234
    assert row.tokens_out == 56
    assert row.cache_read_tokens == 0
    assert row.cache_write_tokens == 789
    # The `runs.cost_usd` column is NUMERIC(12, 6); Postgres rounds on write.
    expected_cost = compute_cost_usd(MODEL, 1234, 56, 0, 789).quantize(Decimal("0.000001"))
    assert row.cost_usd == expected_cost
    assert row.outcome == "ok"
    assert row.error is None
    assert row.latency_ms is not None and row.latency_ms >= 0
    assert row.started_at is not None
