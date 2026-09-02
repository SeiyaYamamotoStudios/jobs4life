"""Unit tests for cost arithmetic -- hand-computed numbers, no API, no database."""

from __future__ import annotations

from decimal import Decimal

import pytest
from jfl_gate.pricing import compute_cost_usd

OPUS = "claude-opus-5"
SONNET = "claude-sonnet-5"


def test_input_and_output_tokens_at_list_price() -> None:
    # 1000 input @ $5/MTok + 1000 output @ $25/MTok = $0.005 + $0.025 = $0.03
    cost = compute_cost_usd(
        OPUS, tokens_in=1000, tokens_out=1000, cache_read_tokens=0, cache_write_tokens=0
    )
    assert cost == Decimal("0.03")


def test_cache_read_bills_at_one_tenth_input_rate() -> None:
    # 1,000,000 cache-read tokens @ $5/MTok * 0.1 = $0.50
    cost = compute_cost_usd(
        OPUS, tokens_in=0, tokens_out=0, cache_read_tokens=1_000_000, cache_write_tokens=0
    )
    assert cost == Decimal("0.50")


def test_cache_write_bills_at_one_point_two_five_times_input_rate() -> None:
    # 1,000,000 cache-write tokens @ $5/MTok * 1.25 = $6.25
    cost = compute_cost_usd(
        OPUS, tokens_in=0, tokens_out=0, cache_read_tokens=0, cache_write_tokens=1_000_000
    )
    assert cost == Decimal("6.25")


def test_all_four_components_sum() -> None:
    # in: 200,000 @ $5 = $1.00
    # out: 4,000 @ $25 = $0.10
    # cache_read: 100,000 @ $0.50 = $0.05
    # cache_write: 8,000 @ $6.25 = $0.05
    # total = $1.20
    cost = compute_cost_usd(
        OPUS,
        tokens_in=200_000,
        tokens_out=4_000,
        cache_read_tokens=100_000,
        cache_write_tokens=8_000,
    )
    assert cost == Decimal("1.20")


def test_zero_tokens_costs_zero() -> None:
    cost = compute_cost_usd(
        OPUS, tokens_in=0, tokens_out=0, cache_read_tokens=0, cache_write_tokens=0
    )
    assert cost == Decimal("0")


def test_result_is_a_decimal_not_a_float() -> None:
    # Money arithmetic must not go through binary floating point.
    cost = compute_cost_usd(
        OPUS, tokens_in=1, tokens_out=1, cache_read_tokens=1, cache_write_tokens=1
    )
    assert isinstance(cost, Decimal)


def test_sonnet_input_and_output_tokens_at_list_price() -> None:
    # 1000 input @ $2/MTok + 1000 output @ $10/MTok = $0.002 + $0.010 = $0.012
    cost = compute_cost_usd(
        SONNET, tokens_in=1000, tokens_out=1000, cache_read_tokens=0, cache_write_tokens=0
    )
    assert cost == Decimal("0.012")


def test_sonnet_cache_read_and_write_scale_off_its_own_input_rate_not_opus() -> None:
    # cache_read: 1,000,000 @ $2/MTok * 0.1 = $0.20
    # cache_write: 1,000,000 @ $2/MTok * 1.25 = $2.50
    read_cost = compute_cost_usd(
        SONNET, tokens_in=0, tokens_out=0, cache_read_tokens=1_000_000, cache_write_tokens=0
    )
    write_cost = compute_cost_usd(
        SONNET, tokens_in=0, tokens_out=0, cache_read_tokens=0, cache_write_tokens=1_000_000
    )
    assert read_cost == Decimal("0.20")
    assert write_cost == Decimal("2.50")


def test_sonnet_is_exactly_forty_percent_of_opus_for_a_pure_input_output_mix() -> None:
    """Opus is $5/$25 per MTok, Sonnet $2/$10 -- both input and output rates are
    exactly 0.4x Opus's, so a mix of pure input and output tokens (no cache) must
    scale by precisely that factor.
    """
    kwargs = dict(tokens_in=123_456, tokens_out=7_890, cache_read_tokens=0, cache_write_tokens=0)
    opus_cost = compute_cost_usd(OPUS, **kwargs)
    sonnet_cost = compute_cost_usd(SONNET, **kwargs)
    assert sonnet_cost == opus_cost * Decimal("0.4")


def test_unknown_model_raises_rather_than_falling_back_to_a_default_rate() -> None:
    with pytest.raises(ValueError, match="claude-haiku-3"):
        compute_cost_usd(
            "claude-haiku-3",
            tokens_in=100,
            tokens_out=100,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
