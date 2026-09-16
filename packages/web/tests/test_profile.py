"""Display and form-parsing helpers for profile setup. No database, no network."""

from __future__ import annotations

import pytest
from jfl_web.profile import (
    FormTooLongError,
    InvalidCompFloorError,
    checked_text,
    custom_disciplines,
    parse_comp_floor,
    parse_contract_types,
    parse_disciplines,
    parse_levels,
    selected_values,
)


def test_overlong_text_is_rejected_never_truncated() -> None:
    assert checked_text("a" * 10, 10) == "a" * 10
    with pytest.raises(FormTooLongError):
        checked_text("a" * 11, 10)


def test_parse_levels_keeps_only_real_values_in_canonical_order() -> None:
    assert parse_levels(["above_em", "ic", "made_up"]) == {"selected": ["ic", "above_em"]}


def test_parse_levels_with_nothing_ticked_is_none() -> None:
    assert parse_levels([]) is None
    assert parse_levels(["not-a-level"]) is None


def test_parse_contract_types_keeps_only_real_values_in_canonical_order() -> None:
    assert parse_contract_types(["fixed_term", "permanent"]) == {
        "selected": ["permanent", "fixed_term"]
    }


def test_parse_comp_floor_none_when_amount_blank() -> None:
    assert parse_comp_floor("", "GBP") is None
    assert parse_comp_floor("   ", "") is None


def test_parse_comp_floor_defaults_currency_to_gbp() -> None:
    assert parse_comp_floor("120000", "") == {"amount": "120000", "currency": "GBP"}


def test_parse_comp_floor_uppercases_and_trims_currency() -> None:
    assert parse_comp_floor("95000.50", " usd ") == {"amount": "95000.50", "currency": "USD"}


def test_parse_comp_floor_rejects_non_numeric_amount() -> None:
    with pytest.raises(InvalidCompFloorError):
        parse_comp_floor("lots", "GBP")


def test_parse_comp_floor_rejects_negative_amount() -> None:
    with pytest.raises(InvalidCompFloorError):
        parse_comp_floor("-1", "GBP")


def test_parse_disciplines_orders_defaults_then_customs_then_new() -> None:
    result = parse_disciplines(["frontend", "platform_infra_engineering"], "Developer relations")
    assert result == {"selected": ["platform_infra_engineering", "frontend", "Developer relations"]}


def test_parse_disciplines_keeps_previously_ticked_custom_values() -> None:
    # A custom bucket the user added before is re-ticked as its own checkbox
    # value (see custom_disciplines below) -- it must survive a save that adds
    # no new one.
    result = parse_disciplines(["embedded", "Developer relations"], "")
    assert result == {"selected": ["embedded", "Developer relations"]}


def test_parse_disciplines_blank_custom_adds_nothing() -> None:
    assert parse_disciplines([], "  ") is None
    assert parse_disciplines(["frontend"], "  ") == {"selected": ["frontend"]}


def test_parse_disciplines_does_not_duplicate_an_already_selected_custom() -> None:
    result = parse_disciplines(["Developer relations"], "Developer relations")
    assert result == {"selected": ["Developer relations"]}


def test_selected_values_empty_for_no_structured_answer() -> None:
    assert selected_values(None) == []
    assert selected_values({}) == []


def test_selected_values_reads_the_selected_key() -> None:
    assert selected_values({"selected": ["ic", "em"]}) == ["ic", "em"]


def test_custom_disciplines_excludes_the_eight_defaults() -> None:
    structured = {"selected": ["frontend", "Developer relations", "embedded"]}
    assert custom_disciplines(structured) == ["Developer relations"]


def test_custom_disciplines_empty_for_no_structured_answer() -> None:
    assert custom_disciplines(None) == []
