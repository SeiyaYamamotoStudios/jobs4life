"""Totals tokens and cost across the `RunRecord`s an eval run collected.

Separate from `jfl_evals.scoring` deliberately: cost is an operating concern
(what did this run spend), not part of the over-claim/over-flag measurement. The
two must never be combined into one figure -- a cheap run and an accurate run are
different questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from jfl_core.models import RunRecord


@dataclass(frozen=True)
class CostSummary:
    n_runs: int
    n_ok: int
    n_errors: int  # outcome in ("error", "refused") -- still billed if usage came back
    tokens_in: int
    tokens_out: int
    cache_read_tokens: int
    cache_write_tokens: int
    total_cost_usd: Decimal
    mean_cost_usd: Decimal | None  # None when there is nothing to average


def summarize_costs(runs: list[RunRecord]) -> CostSummary:
    n_runs = len(runs)
    n_ok = sum(1 for r in runs if r.outcome == "ok")
    n_errors = n_runs - n_ok

    tokens_in = sum(r.tokens_in or 0 for r in runs)
    tokens_out = sum(r.tokens_out or 0 for r in runs)
    cache_read_tokens = sum(r.cache_read_tokens or 0 for r in runs)
    cache_write_tokens = sum(r.cache_write_tokens or 0 for r in runs)
    total_cost_usd = sum((r.cost_usd or Decimal(0) for r in runs), Decimal(0))
    mean_cost_usd = (total_cost_usd / n_runs) if n_runs else None

    return CostSummary(
        n_runs=n_runs,
        n_ok=n_ok,
        n_errors=n_errors,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_cost_usd=total_cost_usd,
        mean_cost_usd=mean_cost_usd,
    )
