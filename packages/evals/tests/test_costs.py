"""Unit tests for cost/token aggregation across a run's collected RunRecords."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from jfl_core.models import RunRecord
from jfl_evals.costs import summarize_costs

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


def _run(
    outcome: Literal["ok", "error", "refused", "skipped"] = "ok",
    tokens_in: int | None = 100,
    tokens_out: int | None = 20,
    cache_read_tokens: int | None = 0,
    cache_write_tokens: int | None = 0,
    cost_usd: Decimal | None = Decimal("0.001"),
) -> RunRecord:
    return RunRecord(
        user_id=USER,
        trace_id=uuid.uuid4(),
        component="evals",
        stage="baseline",
        model="claude-opus-5",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=cost_usd,
        latency_ms=100,
        outcome=outcome,
        started_at=datetime.now(UTC),
    )


class TestSummarizeCosts:
    def test_empty_list_yields_zero_totals_and_no_mean(self) -> None:
        summary = summarize_costs([])
        assert summary.n_runs == 0
        assert summary.total_cost_usd == Decimal(0)
        assert summary.mean_cost_usd is None

    def test_totals_tokens_and_cost_across_runs(self) -> None:
        runs = [
            _run(tokens_in=1000, tokens_out=100, cost_usd=Decimal("0.02")),
            _run(tokens_in=500, tokens_out=50, cost_usd=Decimal("0.01")),
        ]
        summary = summarize_costs(runs)
        assert summary.tokens_in == 1500
        assert summary.tokens_out == 150
        assert summary.total_cost_usd == Decimal("0.03")
        assert summary.mean_cost_usd == Decimal("0.015")

    def test_counts_ok_vs_error_outcomes(self) -> None:
        runs = [_run(outcome="ok"), _run(outcome="error", cost_usd=None), _run(outcome="ok")]
        summary = summarize_costs(runs)
        assert summary.n_runs == 3
        assert summary.n_ok == 2
        assert summary.n_errors == 1

    def test_none_cost_and_token_fields_are_treated_as_zero(self) -> None:
        """An errored run may have no usage at all -- see jfl_gate.gate.check_text,
        which records tokens_in=None when the API call never returned.
        """
        runs = [_run(tokens_in=None, tokens_out=None, cost_usd=None, outcome="error")]
        summary = summarize_costs(runs)
        assert summary.tokens_in == 0
        assert summary.tokens_out == 0
        assert summary.total_cost_usd == Decimal(0)
