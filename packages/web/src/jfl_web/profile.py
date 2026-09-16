"""Display and form-parsing for profile setup -- PLAN.md slice B3a.

No SQL and no storage here -- storage is `jfl_core.storage.profile`. This
module only turns submitted form fields into the shapes the repository takes,
and the repository's output into what the page shows. Same split
`jfl_web.jobfilter` draws for the saved filter.

**Text is never truncated, only rejected.** A submission longer than the limit
raises `FormTooLongError`, exactly as `jfl_web.jobfilter.checked_text` does --
a silently shortened answer is not what the user wrote, and this project's
whole claim is measuring distance from what someone actually said.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

from jfl_core.profile_questions import (
    CONTRACT_TYPE_CHOICES,
    DEFAULT_COMP_CURRENCY,
    DEFAULT_DISCIPLINE_CHOICES,
    LEVEL_CHOICES,
)

# Generous: several of these questions (deal-breakers, warning signs, comp
# composition) invite a paragraph, not a phrase. The limit bounds a row, not
# what anyone is allowed to say.
MAX_ANSWER_TEXT = 4000
MAX_RULED_OUT_TEXT = 2000

_LEVEL_VALUES = tuple(v for v, _ in LEVEL_CHOICES)
_CONTRACT_TYPE_VALUES = tuple(v for v, _ in CONTRACT_TYPE_CHOICES)
_DEFAULT_DISCIPLINE_VALUES = tuple(v for v, _ in DEFAULT_DISCIPLINE_CHOICES)


class FormTooLongError(ValueError):
    """A submitted text field exceeds its limit. Rejected, never truncated."""


class InvalidCompFloorError(ValueError):
    """A submitted comp floor amount is not a non-negative number."""


def checked_text(value: str, limit: int = MAX_ANSWER_TEXT) -> str:
    if len(value) > limit:
        raise FormTooLongError(f"That text is longer than {limit} characters.")
    return value


def _parse_choice_set(values: Iterable[str], allowed: tuple[str, ...]) -> list[str]:
    """The submitted checkbox values that are real choices, in canonical
    order. Anything else is ignored rather than stored -- same rule as
    `jfl_web.jobfilter.parse_workplaces`.
    """
    chosen = set(values)
    return [v for v in allowed if v in chosen]


def parse_levels(values: Iterable[str]) -> dict[str, Any] | None:
    selected = _parse_choice_set(values, _LEVEL_VALUES)
    return {"selected": selected} if selected else None


def parse_contract_types(values: Iterable[str]) -> dict[str, Any] | None:
    selected = _parse_choice_set(values, _CONTRACT_TYPE_VALUES)
    return {"selected": selected} if selected else None


def parse_disciplines(ticked: Iterable[str], new_custom: str) -> dict[str, Any] | None:
    """Ticked default buckets, in canonical order, then any previously-added
    custom buckets that are still ticked (alphabetically), then a newly typed
    one. The buckets are "editable per user" per PLAN.md: the page re-renders
    every custom bucket already on the record as its own checkbox (see
    `custom_disciplines`), so unticking one removes it and leaving it ticked
    keeps it -- a save that only resubmitted the defaults would otherwise
    silently drop a user's own addition.
    """
    ticked_values = {v.strip() for v in ticked if v.strip()}
    defaults_selected = [v for v in _DEFAULT_DISCIPLINE_VALUES if v in ticked_values]
    customs_selected = sorted(ticked_values - set(_DEFAULT_DISCIPLINE_VALUES))
    selected = defaults_selected + customs_selected
    extra = new_custom.strip()
    if extra and extra not in selected:
        selected.append(extra)
    return {"selected": selected} if selected else None


def custom_disciplines(structured: dict[str, Any] | None) -> list[str]:
    """Previously added bucket labels outside the eight defaults, so the page
    can re-render each as its own already-ticked checkbox. Without this, a
    save that did not resubmit a custom label as ticked would look identical
    to one that meant to remove it, and the label would quietly vanish.
    """
    return [v for v in selected_values(structured) if v not in _DEFAULT_DISCIPLINE_VALUES]


def parse_comp_floor(amount: str, currency: str) -> dict[str, Any] | None:
    """None if no amount was given -- the structured value is optional even
    when the free-text answer says something. Raises if an amount is given
    but is not a non-negative number.
    """
    amount = amount.strip()
    if not amount:
        return None
    try:
        value = Decimal(amount)
    except InvalidOperation as exc:
        raise InvalidCompFloorError("That amount is not a number.") from exc
    if value < 0:
        raise InvalidCompFloorError("That amount cannot be negative.")
    chosen_currency = currency.strip().upper() or DEFAULT_COMP_CURRENCY
    return {"amount": str(value), "currency": chosen_currency}


def selected_values(structured: dict[str, Any] | None) -> list[str]:
    """The ticked values out of a tickbox question's structured answer, for
    pre-filling a form's checkboxes. Empty for a question with no structured
    answer yet.
    """
    if not structured:
        return []
    values = structured.get("selected", [])
    return list(values) if isinstance(values, list) else []
