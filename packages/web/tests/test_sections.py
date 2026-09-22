"""The section rules, on their own: no database, no browser, no model.

`jfl_web.sections` is deliberately pure, so the three-rule precedence and the
change marker can be pinned down here and the integration tests can spend
themselves on the markup instead.
"""

from __future__ import annotations

import datetime as dt

from jfl_core.storage.ui_sections import SectionState
from jfl_web.sections import counted, joined, marker, resolve

NOW = dt.datetime(2026, 9, 22, 12, 0, tzinfo=dt.UTC)
EARLIER = NOW - dt.timedelta(hours=2)
LATER = NOW + dt.timedelta(hours=2)


def state(
    *, is_open: bool = False, last_opened_at: dt.datetime | None = NOW, key: str = "s"
) -> SectionState:
    return SectionState(
        section_key=key,
        is_open=is_open,
        default_open=True,
        toggles=1,
        against_default=0,
        last_opened_at=last_opened_at,
    )


# --------------------------------------------------------------------------
# Which of the three rules wins
# --------------------------------------------------------------------------


def test_the_default_applies_when_the_user_has_never_touched_it() -> None:
    assert resolve("s", "S", state=None, default_open=True).open is True
    assert resolve("s", "S", state=None, default_open=False).open is False


def test_a_stored_choice_beats_the_default_both_ways() -> None:
    assert resolve("s", "S", state=state(is_open=False), default_open=True).open is False
    assert resolve("s", "S", state=state(is_open=True), default_open=False).open is True


def test_pending_beats_a_stored_choice() -> None:
    """The one exception, and the reason it is one: a pending run is transient
    state that needs watching, and the choice to fold the panel away was made
    about a different situation.
    """
    section = resolve("s", "S", state=state(is_open=False), default_open=False, forced_open=True)
    assert section.open is True


def test_a_forced_section_reports_open_as_its_default() -> None:
    """What goes back on toggle is what this page *would* have shown with no
    stored choice -- forcing included -- or "went against the default" means
    nothing a year from now.
    """
    section = resolve("s", "S", state=None, default_open=False, forced_open=True)
    assert section.default_open is True


# --------------------------------------------------------------------------
# The change marker
# --------------------------------------------------------------------------


def test_no_watermark_means_no_marker() -> None:
    """A section nobody has ever opened or closed announces nothing. Inventing
    a baseline would report a year of old drafts as news on a first visit --
    the same failure a newly watched board's first check avoids.
    """
    assert marker(None, item_times=[LATER]) == ""
    assert marker(state(last_opened_at=None), item_times=[LATER]) == ""


def test_items_newer_than_the_last_look_are_counted() -> None:
    assert marker(state(), item_times=[EARLIER, LATER, LATER]) == "2 new"
    assert marker(state(), item_times=[EARLIER, EARLIER]) == ""


def test_a_section_that_changed_without_a_countable_list_says_updated() -> None:
    assert marker(state(), changed_at=LATER) == "updated"
    assert marker(state(), changed_at=EARLIER) == ""


def test_an_open_section_never_carries_a_marker() -> None:
    """Its contents are on screen; a badge counting them is noise."""
    section = resolve("s", "S", state=state(is_open=True), default_open=False, item_times=[LATER])
    assert section.open is True
    assert section.marker == ""


def test_a_folded_section_carries_the_marker() -> None:
    section = resolve("s", "S", state=state(is_open=False), default_open=True, item_times=[LATER])
    assert section.marker == "1 new"


# --------------------------------------------------------------------------
# The count summary's wording
# --------------------------------------------------------------------------


def test_counted_agrees_with_itself_about_number() -> None:
    assert counted(1, "requirement") == "1 requirement"
    assert counted(3, "requirement") == "3 requirements"
    assert counted(0, "requirement") == "0 requirements"
    assert counted(2, "capability", "capabilities") == "2 capabilities"


def test_joined_drops_the_parts_that_have_nothing_to_say() -> None:
    assert joined("3 requirements", "", "1 essential") == "3 requirements · 1 essential"
    assert joined("", "") == ""
