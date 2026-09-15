"""Display and wiring for suggested title expansions -- slice C7a.

No SQL and no model call here: storage is `jfl_core.storage.title_suggestions`,
the call is `jfl_generate.titles`. This module only turns their outputs into
what the `/jobs` panel shows, the same split `jfl_web.jobfilter` and
`jfl_web.jobads` draw for their own features.

**A suggestion is never added to the filter by anything here.** `visible`
below is display filtering only -- what the panel offers a tickbox for. The
only write onto `JobFilter.title_includes` is
`jfl_web.routes.title_suggestions.accept_title_suggestions`, gated by the
tickbox the user actually ticked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from jfl_core.models import SuggestedTitle, TitleSuggestion, TitleSuggestionErrorCode
from jfl_intake.filtering import parse_terms
from jfl_intake.normalise import normalise


class _GetByPhraseKey(Protocol):
    """Structural stand-in for
    `PostgresTitleSuggestionRepository.get_by_phrase_key`, named only so
    `suggestion_rows`'s signature reads without importing the storage layer
    into this display module.
    """

    def __call__(self, phrase_key: str) -> TitleSuggestion | None: ...


def split_phrases(text: str) -> list[str]:
    """Comma-separated phrases, trimmed, in order, deduplicated by exact text.

    Deliberately not `jfl_intake.filtering.parse_terms`: this is what the panel
    shows and what a new row's `phrase` column stores, so it keeps the user's
    own wording rather than the normalised word set matching uses.
    """
    seen: list[str] = []
    for part in text.split(","):
        phrase = part.strip()
        if phrase and phrase not in seen:
            seen.append(phrase)
    return seen


def visible_suggestions(
    suggestions: Sequence[SuggestedTitle], title_includes: str
) -> list[SuggestedTitle]:
    """Suggestions not already in the filter -- what gets a tickbox.

    Checked at render time, against the filter as it stands now, rather than
    trusting the sanitisation done when the call was made: the filter can have
    changed since (another suggestion accepted, a manual edit), and this is
    what keeps "not already included" honest against the current state instead
    of a stale one.
    """
    included = set(parse_terms(title_includes))
    visible: list[SuggestedTitle] = []
    for s in suggestions:
        keys = parse_terms(s.title)
        if keys and keys[0] in included:
            continue
        visible.append(s)
    return visible


@dataclass(frozen=True, slots=True)
class TitleSuggestionFailure:
    """What to tell the user. No `fix_url` for most codes -- unlike extraction's
    failures, a rejected or unreadable key is the one case worth pointing at
    Settings; the rest is not actionable beyond "try again".
    """

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[TitleSuggestionErrorCode, TitleSuggestionFailure] = {
    "no_api_key": TitleSuggestionFailure(
        "This needs your own Anthropic API key -- suggesting titles is a model "
        "call, billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": TitleSuggestionFailure(
        "Anthropic rejected the API key stored here.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "model_refused": TitleSuggestionFailure(
        "The model declined to suggest titles for that phrase."
    ),
    "model_error": TitleSuggestionFailure("Suggesting titles failed."),
    "credential_unreadable": TitleSuggestionFailure(
        "Your stored API key could not be unlocked on the server.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
}

# Anything unrecognised -- a code added to the database before this table
# caught up -- still gets a sentence rather than a blank panel.
_UNKNOWN = TitleSuggestionFailure("Suggesting titles failed.")


def title_suggestion_failure(code: TitleSuggestionErrorCode | None) -> TitleSuggestionFailure:
    return _UNKNOWN if code is None else _FAILURES.get(code, _UNKNOWN)


@dataclass(frozen=True, slots=True)
class SuggestionRowView:
    """One phrase's panel state -- what `_title_suggestion_row.html` renders,
    whether reached from the full panel or from the standalone polling route.
    """

    phrase: str
    row: TitleSuggestion | None
    failure: TitleSuggestionFailure | None = None
    visible: list[SuggestedTitle] = field(default_factory=list)


def suggestion_row_view(
    phrase: str, row: TitleSuggestion | None, title_includes: str
) -> SuggestionRowView:
    """One phrase's view, whether or not it has a row.

    `row` is None when nothing was ever enqueued for this phrase -- no API key
    at save time -- and the panel says so rather than showing nothing. A
    dismissed row is treated the same as no row: the caller is expected to have
    already filtered those out (see `jobs.list_jobs`), so a dismissed row
    reaching here would show again, which is why it never does.
    """
    failure = title_suggestion_failure(row.error_code) if row and row.status == "failed" else None
    visible = (
        visible_suggestions(row.suggestions, title_includes) if row and row.status == "done" else []
    )
    return SuggestionRowView(phrase=phrase, row=row, failure=failure, visible=visible)


def suggestion_rows(
    phrases: Sequence[str],
    get_by_phrase_key: _GetByPhraseKey,
    title_includes: str,
) -> list[SuggestionRowView]:
    """One view per current include phrase, in order. A dismissed row is
    treated as though it does not exist -- the panel drops it entirely rather
    than showing it as gone, which is what "Dismiss" means.
    """
    views: list[SuggestionRowView] = []
    for phrase in phrases:
        row = get_by_phrase_key(normalise(phrase))
        if row is not None and row.dismissed_at is not None:
            row = None
        views.append(suggestion_row_view(phrase, row, title_includes))
    return views
