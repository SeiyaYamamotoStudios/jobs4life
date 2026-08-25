"""Unit tests for cost arithmetic -- hand-computed numbers, no API, no database."""

from __future__ import annotations

from decimal import Decimal

from jfl_gate.pricing import compute_cost_usd


def test_input_and_output_tokens_at_list_price() -> None:
    # 1000 input @ $5/MTok + 1000 output @ $25/MTok = $0.005 + $0.025 = $0.03
    cost = compute_cost_usd(
        tokens_in=1000, tokens_out=1000, cache_read_tokens=0, cache_write_tokens=0
    )
    assert cost == Decimal("0.03")


def test_cache_read_bills_at_one_tenth_input_rate() -> None:
    # 1,000,000 cache-read tokens @ $5/MTok * 0.1 = $0.50
    cost = compute_cost_usd(
        tokens_in=0, tokens_out=0, cache_read_tokens=1_000_000, cache_write_tokens=0
    )
    assert cost == Decimal("0.50")


def test_cache_write_bills_at_one_point_two_five_times_input_rate() -> None:
    # 1,000,000 cache-write tokens @ $5/MTok * 1.25 = $6.25
    cost = compute_cost_usd(
        tokens_in=0, tokens_out=0, cache_read_tokens=0, cache_write_tokens=1_000_000
    )
    assert cost == Decimal("6.25")


def test_all_four_components_sum() -> None:
    # in: 200,000 @ $5 = $1.00
    # out: 4,000 @ $25 = $0.10
    # cache_read: 100,000 @ $0.50 = $0.05
    # cache_write: 8,000 @ $6.25 = $0.05
    # total = $1.20
    cost = compute_cost_usd(
        tokens_in=200_000,
        tokens_out=4_000,
        cache_read_tokens=100_000,
        cache_write_tokens=8_000,
    )
    assert cost == Decimal("1.20")


def test_zero_tokens_costs_zero() -> None:
    cost = compute_cost_usd(tokens_in=0, tokens_out=0, cache_read_tokens=0, cache_write_tokens=0)
    assert cost == Decimal("0")


def test_result_is_a_decimal_not_a_float() -> None:
    # Money arithmetic must not go through binary floating point.
    cost = compute_cost_usd(tokens_in=1, tokens_out=1, cache_read_tokens=1, cache_write_tokens=1)
    assert isinstance(cost, Decimal)
