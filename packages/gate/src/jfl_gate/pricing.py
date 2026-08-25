"""Claude Opus 5 cost arithmetic.

Rates below are from the /claude-api skill's pricing table (cached 2026-06-24). They
go stale -- check https://docs.claude.com/en/docs/about-claude/pricing before trusting
`cost_usd` for anything that matters. Cache reads bill at ~0.1x the input rate, cache
writes at ~1.25x (the 5-minute default TTL; a 1h-TTL write bills higher and is not
modelled here since the gate never requests one).
"""

from __future__ import annotations

from decimal import Decimal

MODEL = "claude-opus-5"

_INPUT_PER_MTOK = Decimal("5.00")
_OUTPUT_PER_MTOK = Decimal("25.00")
_CACHE_READ_MULTIPLIER = Decimal("0.1")
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
_MTOK = Decimal(1_000_000)


def compute_cost_usd(
    tokens_in: int,
    tokens_out: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> Decimal:
    """Dollar cost of one call, given the four token counts off `response.usage`."""
    input_cost = Decimal(tokens_in) * _INPUT_PER_MTOK
    output_cost = Decimal(tokens_out) * _OUTPUT_PER_MTOK
    cache_read_cost = Decimal(cache_read_tokens) * _INPUT_PER_MTOK * _CACHE_READ_MULTIPLIER
    cache_write_cost = Decimal(cache_write_tokens) * _INPUT_PER_MTOK * _CACHE_WRITE_MULTIPLIER
    return (input_cost + output_cost + cache_read_cost + cache_write_cost) / _MTOK
