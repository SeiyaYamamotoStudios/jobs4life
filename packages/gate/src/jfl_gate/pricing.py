"""Per-model cost arithmetic.

Rates below are from the /claude-api skill's pricing table (cached 2026-06-24). They
go stale -- check https://docs.claude.com/en/docs/about-claude/pricing before trusting
`cost_usd` for anything that matters. Cache reads bill at ~0.1x a model's own input
rate unless the model states its own cache-read rate (Opus 5.5 does: $0.20, which is
0.05x its $4 input), cache writes at 1.25x input for every model (the 5-minute
default TTL; a 1h-TTL write bills higher and is not modelled here since nothing here
ever requests one).
"""

from __future__ import annotations

from decimal import Decimal
from typing import NamedTuple

from jfl_core.context import PRODUCT_MODEL

# Default product model id -- `RequestContext.model` (from `JFL_MODEL`) is what call
# sites actually use; this only backstops a context built without one. Read from
# jfl_core.context, the one place the default is defined.
MODEL = PRODUCT_MODEL


class _Rates(NamedTuple):
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    # None means the generic 0.1x-of-input rule below. Set only where the price sheet
    # names a cache-read rate that rule would get wrong.
    cache_read_per_mtok: Decimal | None = None


# Entries verified against the Anthropic pricing table (see the module docstring's
# caveat about staleness).
_RATES: dict[str, _Rates] = {
    # The product model since 2026-09-23. Cache reads are $0.20/MTok -- 0.05x input,
    # not the 0.1x the other models follow, so it is stated rather than derived.
    # Cache writes follow the same 1.25x-of-input rule as every other model: $5.00.
    "claude-opus-5-5": _Rates(
        input_per_mtok=Decimal("4.00"),
        output_per_mtok=Decimal("20.00"),
        cache_read_per_mtok=Decimal("0.20"),
    ),
    # The claim gate's model (jfl_core.context.GATE_MODEL), and still selectable for
    # product calls via `$JFL_MODEL`.
    "claude-opus-5": _Rates(input_per_mtok=Decimal("5.00"), output_per_mtok=Decimal("25.00")),
    "claude-sonnet-5": _Rates(input_per_mtok=Decimal("2.00"), output_per_mtok=Decimal("10.00")),
    # Not a product-model choice (see the module docstring's `MODEL` and
    # CLAUDE.md's 2026-09-05 decision) -- this is the second, cheaper model
    # `jfl_generate.titles.suggest_titles` calls explicitly for slice C7a's
    # title-suggestion call, regardless of which model the user has configured.
    "claude-haiku-4-5": _Rates(input_per_mtok=Decimal("1.00"), output_per_mtok=Decimal("5.00")),
}

_CACHE_READ_MULTIPLIER = Decimal("0.1")
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
_MTOK = Decimal(1_000_000)


def compute_cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> Decimal:
    """Dollar cost of one call, given the model id and the four token counts off
    `response.usage`. An unknown model id raises rather than silently falling back
    to another model's rates -- a wrong cost number is worse than a crash here.
    """
    try:
        rates = _RATES[model]
    except KeyError:
        raise ValueError(
            f"no pricing data for model {model!r} -- known models: {sorted(_RATES)}"
        ) from None

    input_cost = Decimal(tokens_in) * rates.input_per_mtok
    output_cost = Decimal(tokens_out) * rates.output_per_mtok
    cache_read_rate = (
        rates.cache_read_per_mtok
        if rates.cache_read_per_mtok is not None
        else rates.input_per_mtok * _CACHE_READ_MULTIPLIER
    )
    cache_read_cost = Decimal(cache_read_tokens) * cache_read_rate
    cache_write_cost = Decimal(cache_write_tokens) * rates.input_per_mtok * _CACHE_WRITE_MULTIPLIER
    return (input_cost + output_cost + cache_read_cost + cache_write_cost) / _MTOK
