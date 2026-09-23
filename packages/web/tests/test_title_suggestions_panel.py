"""The /jobs title-suggestions panel, rendered from its view model.

The failure this pins, from the owner's screen: eight identical boxes, one per
saved title phrase, each saying "Add your API key in Settings to get suggested
titles adjacent to ..." -- with a key stored. The phrases predated the key, so
no suggestion row had ever been created, and the panel read "no row" as "no
key". It also offered "Technical Lead Manager -- Already in filter" with
"technical lead" saved, and put Dismiss in a bordered box of its own.

No database, no model call, no network: a dict stands in for the repository
lookup, and the partial is rendered through the app's own Jinja environment.
"""

from __future__ import annotations

import datetime as dt
import pathlib
import re
import uuid
from types import SimpleNamespace

from jfl_core.models import SuggestedTitle, TitleSuggestion
from jfl_intake.normalise import normalise
from jfl_web.templating import _templates
from jfl_web.titlesuggestions import (
    TitleSuggestionPanel,
    split_phrases,
    suggestion_panel,
    visible_suggestions,
)

NOW = dt.datetime(2026, 9, 23, 12, 0, tzinfo=dt.UTC)

OWNERS_PHRASES = (
    "Engineering manager, tech lead, engineering lead, head of engineering, "
    "technical lead, staff engineer, principal engineer, platform lead"
)


def _row(
    phrase: str,
    status: str = "done",
    titles: list[SuggestedTitle] | None = None,
    dismissed: bool = False,
) -> TitleSuggestion:
    return TitleSuggestion(
        id=uuid.uuid4(),
        phrase=phrase,
        phrase_key=normalise(phrase),
        status=status,  # type: ignore[arg-type]
        suggestions=titles or [],
        dismissed_at=NOW if dismissed else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _panel(includes: str, rows: list[TitleSuggestion], *, has_key: bool) -> TitleSuggestionPanel:
    by_key = {r.phrase_key: r for r in rows}

    def lookup(phrase_key: str) -> TitleSuggestion | None:
        return by_key.get(phrase_key)

    return suggestion_panel(split_phrases(includes), lookup, includes, has_key=has_key)


def _render(panel: TitleSuggestionPanel) -> str:
    template = _templates.env.get_template("_title_suggestions.html")
    return template.render(title_suggestions=panel, session=SimpleNamespace(csrf_token="t"))


def _card_count(html: str) -> int:
    return len(re.findall(r'<div class="title-suggestion[ "]', html))


# ----------------------------------------------------------------------
# "no key" is a fact about the credential store, not about missing rows
# ----------------------------------------------------------------------


def test_key_stored_and_phrases_never_suggested_collapse_to_one_line() -> None:
    panel = _panel(OWNERS_PHRASES, [], has_key=True)
    assert len(panel.unsuggested) == 8 and panel.rows == []

    html = _render(panel)
    assert "API key" not in html
    assert _card_count(html) == 1, "one line, not a box per phrase"
    assert "8 title phrases" in html and "have no suggested titles yet" in html
    assert html.count('action="/jobs/filter/titles/suggest"') == 1
    assert "Suggest titles for all 8" in html


def test_one_unsuggested_phrase_is_named() -> None:
    html = _render(_panel("engineering manager", [], has_key=True))
    assert "&ldquo;engineering manager&rdquo; has no suggested titles yet" in html
    assert ">Suggest titles<" in html


def test_no_key_says_so_exactly_once() -> None:
    html = _render(_panel(OWNERS_PHRASES, [], has_key=False))
    assert html.count("Add your API key") == 1
    assert _card_count(html) == 1
    assert "/jobs/filter/titles/suggest" not in html, "nothing to press without a key"


def test_unsuggested_line_sits_beside_live_rows_and_dismissed_ones_are_gone() -> None:
    rows = [
        _row("engineering manager", titles=[SuggestedTitle(title="Head of Platform", gloss="")]),
        _row("tech lead", dismissed=True),
    ]
    panel = _panel("engineering manager, tech lead, staff engineer", rows, has_key=True)
    assert [v.phrase for v in panel.rows] == ["engineering manager"]
    # A dismissed phrase had its suggestions; it is not asked for again.
    assert panel.unsuggested == ["staff engineer"]
    html = _render(panel)
    assert "Head of Platform" in html
    assert "&ldquo;staff engineer&rdquo; has no suggested titles yet" in html
    assert "tech lead" not in html


def test_nothing_renders_with_no_phrases() -> None:
    assert _render(_panel("", [], has_key=True)).strip() == ""


# ----------------------------------------------------------------------
# "already in filter" by the filter's own rule
# ----------------------------------------------------------------------


def test_a_suggestion_the_filter_already_matches_is_not_offered() -> None:
    """Exactly the owner's case: the model said "Already in filter" and the
    panel offered it anyway, because "technical lead manager" is not *equal*
    to any saved key. It does not need to be: the filter matches a title when
    all of an alternative's words are present, so "technical lead" already
    matches every Technical Lead Manager posting."""
    titles = [
        SuggestedTitle(
            title="Technical Lead Manager", gloss="Already in filter; equivalent seniority"
        ),
        SuggestedTitle(title="Senior Engineering Manager", gloss=""),
        SuggestedTitle(title="Lead Engineer", gloss=""),
    ]
    includes = "engineering manager, technical lead"
    assert [s.title for s in visible_suggestions(titles, includes)] == ["Lead Engineer"]

    panel = _panel(includes, [_row("engineering manager", titles=titles)], has_key=True)
    html = _render(panel)
    assert "Technical Lead Manager" not in html
    assert "Senior Engineering Manager" not in html
    assert "Lead Engineer" in html


# ----------------------------------------------------------------------
# one card, one action row
# ----------------------------------------------------------------------


def _only_card(html: str) -> str:
    start = html.index('<div class="title-suggestion"')
    end = html.index("</section>")
    return html[start:end]


def test_dismiss_is_inside_the_card_and_its_form() -> None:
    row = _row("engineering manager", titles=[SuggestedTitle(title="Head of Platform", gloss="")])
    html = _render(_panel("engineering manager", [row], has_key=True))
    card = _only_card(html)
    assert card.count("<form") == 1, "one form per card: Dismiss is not a second box"
    form = card[card.index("<form") : card.index("</form>")]
    assert "Add ticked titles" in form
    assert f'formaction="/jobs/filter/titles/{row.id}/dismiss"' in form
    assert ">Dismiss</button>" in form


def test_dismiss_with_nothing_left_to_offer_is_still_in_the_card() -> None:
    row = _row("engineering manager", titles=[])
    card = _only_card(_render(_panel("engineering manager", [row], has_key=True)))
    assert card.count("<form") == 1
    assert f'action="/jobs/filter/titles/{row.id}/dismiss"' in card
    assert ">Dismiss</button>" in card


def test_the_filter_form_rule_no_longer_boxes_forms_inside_the_panel() -> None:
    """`.job-filter form` put a bordered box around every form in the panel,
    which is where the separate Dismiss box came from."""
    css = (pathlib.Path(__file__).resolve().parents[1] / "src/jfl_web/static/style.css").read_text()
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    assert not re.search(r"\.job-filter form\b", css)
    assert ".job-filter > form {" in css
