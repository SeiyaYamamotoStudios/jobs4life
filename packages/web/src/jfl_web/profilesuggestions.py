"""Display and merge rules for CV-read profile settings -- the `/profile` panel.

No SQL and no model call here: storage is
`jfl_core.storage.profile_suggestions`, the call is
`jfl_generate.profile_suggestions`. This module turns a run into what the panel
shows and holds the rules that decide what a user's answer does to their
profile.

**A proposal is never applied silently.** Every one is shown with the CV line
or phrase it came from, so a wrong one is obvious rather than plausible, and
the profile changes only when the user presses accept. Same discipline
`jfl_web.capabilityclusters` applies to capabilities and
`jfl_web.routes.candidate_facts` applies to the corpus.

**A setting the user has already stated wins, always.** Three outcomes, and
`decide` is the one place they are worked out:

  * **new** -- the profile says nothing about this. Accepting records it.
  * **agrees** -- the profile already says exactly this. Accepting adds nothing
    and the panel says so; the proposal is simply marked answered.
  * **conflict** -- the profile says something else. **Never applied on an
    ordinary accept.** The panel shows both versions side by side and the
    user resolves it deliberately, by ticking "replace what I have". A
    conflicting accept without that tick is refused.

**A constraint still needs a stance, and a level floor is still the user's own
words.** A CV states where someone worked and what level they were operating
at; it does not state whether London is a must or a nice-to-have, or what they
would now take. So accepting a location or a level asks for must/nice/never
with no default -- the same `MissingStanceError` rule the constraints form
follows -- and the level box starts **empty**, with the CV's observation quoted
beside it rather than typed into it. `level_floor` is a choice; the observation
only informs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from jfl_core.ids import fold
from jfl_core.models import (
    ProfileSuggestionErrorCode,
    ProfileSuggestionKind,
    ProfileSuggestionRun,
    ProposedSetting,
)
from jfl_core.profile import (
    Constraint,
    Disciplines,
    Profile,
    Stance,
    location_value,
    text_value,
)

# A place or a discipline the user has retyped is still a profile value: the
# same ceilings the profile form applies, rejected rather than truncated.
MAX_VALUE = 200
MAX_LEVEL_TEXT = 2000

# The constraint kind each proposal kind writes, where it writes one. Locations
# and levels are constraints; disciplines are their own section.
CONSTRAINT_FOR: dict[ProfileSuggestionKind, str] = {
    "location": "location",
    "level": "level_floor",
}


class SuggestionConflictError(ValueError):
    """An accept that would overwrite something the user has already stated,
    without them saying to.

    Refused rather than resolved: the whole claim of this project is measuring
    distance from what someone actually said, and quietly replacing what they
    said with what a model read off their CV is the failure that would make the
    rest of it meaningless.
    """


class NoPlacesGivenError(ValueError):
    """A location accepted with every place deleted from the box.

    Refused rather than stored as an empty list: a location constraint with no
    places says nothing, and a stance attached to nothing is not a preference.
    """


class MissingLevelTextError(ValueError):
    """A level accepted with an empty box.

    The CV's observation is not typed in for the user on purpose -- see the
    module docstring. "Has been operating at engineering-manager level" is a
    fact about the past; the floor is a decision about the future, and the tool
    does not get to make it for them.
    """


@dataclass(frozen=True, slots=True)
class SettingFailure:
    """What to tell the user. Mirrors `jfl_web.capabilityclusters`'s failures --
    a rejected or unreadable key is the one case worth pointing at Settings;
    the rest is not actionable beyond "try again".
    """

    message: str
    fix_url: str | None = None
    fix_label: str | None = None


_FAILURES: dict[ProfileSuggestionErrorCode, SettingFailure] = {
    "no_api_key": SettingFailure(
        "This needs your own Anthropic API key -- reading your CVs is a model "
        "call, billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": SettingFailure(
        "Anthropic rejected the API key stored here.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "model_refused": SettingFailure("The model declined to read these CVs. Nothing was changed."),
    "model_error": SettingFailure("Reading your CVs failed. Nothing was changed."),
    "credential_unreadable": SettingFailure(
        "Your stored API key could not be unlocked on the server.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
}

# Anything unrecognised -- a code added to the database before this module
# caught up -- still gets a sentence rather than a blank panel.
_UNKNOWN = SettingFailure("Reading your CVs failed. Nothing was changed.")


def setting_failure(code: ProfileSuggestionErrorCode | None) -> SettingFailure:
    return _UNKNOWN if code is None else _FAILURES.get(code, _UNKNOWN)


# The words the panel puts on each kind. Written here rather than in the
# template so that "what the screen offers" is one list, the same rule
# `jfl_web.profile` follows.
KIND_HEADINGS: dict[ProfileSuggestionKind, str] = {
    "discipline": "Something you practise",
    "not_discipline": "Something you do not practise",
    "location": "Where you have worked",
    "level": "The level your CVs describe",
}

KIND_NOTES: dict[ProfileSuggestionKind, str] = {
    "discipline": "Goes on the ranked list of what you practise, at the end.",
    "not_discipline": 'Goes on the "not this" list, which carries the same weight.',
    "location": (
        "Becomes your location constraint, in this order -- first is first "
        "choice. Say whether it is a must, a nice to have or a never; a CV "
        "cannot tell us that."
    ),
    "level": (
        "This is an observation about what you have been doing, not a floor. "
        "The floor is your decision, so type it yourself."
    ),
}

Verdict = Literal["new", "agrees", "conflict"]


@dataclass(frozen=True, slots=True)
class SuggestionView:
    """One proposal as the panel renders it: what it says, the CV's own words
    behind it, and what the profile already says about the same setting.
    """

    key: str
    kind: ProfileSuggestionKind
    heading: str
    note: str
    values: list[str] = field(default_factory=list)
    source_lines: list[str] = field(default_factory=list)
    verdict: Verdict = "new"
    # What the user has already stated, as display text. Empty for `new`.
    current: str = ""

    @property
    def needs_stance(self) -> bool:
        """Location and level become constraints, and a constraint with no
        must/nice/never is refused -- `jfl_web.profile.MissingStanceError`.
        """
        return self.kind in CONSTRAINT_FOR

    @property
    def text(self) -> str:
        """The proposal as one line, for the kinds that carry a single value."""
        return ", ".join(self.values)


@dataclass(frozen=True, slots=True)
class SuggestionsView:
    """One run's panel state -- what `_profile_suggestions.html` renders,
    whether reached inline from `/profile` or from the polling route.
    """

    run: ProfileSuggestionRun | None = None
    suggestions: list[SuggestionView] = field(default_factory=list)
    failure: SettingFailure | None = None
    cost: object = None


def _current_location(profile: Profile) -> list[str]:
    constraint = profile.constraint("location")
    if constraint is None:
        return []
    places = constraint.value.get("places", [])
    return [str(place) for place in places]


def _current_level(profile: Profile) -> str:
    constraint = profile.constraint("level_floor")
    if constraint is None:
        return ""
    return str(constraint.value.get("text", "") or "")


def _folded(values: Sequence[str]) -> list[str]:
    return [fold(value) for value in values]


def decide(profile: Profile, proposal: ProposedSetting) -> tuple[Verdict, str]:
    """(verdict, what the profile already says). The one place rule 3 lives.

    Comparison is folded, so "Engineering Management" and "engineering
    management" agree rather than conflict -- a difference in capitals is not a
    disagreement worth making someone resolve.
    """
    disciplines = profile.disciplines
    if proposal.kind == "discipline":
        value = fold(proposal.values[0]) if proposal.values else ""
        if value in _folded(disciplines.practises):
            return "agrees", ", ".join(disciplines.practises)
        if value in _folded(disciplines.not_practised):
            # The opposite claim, not a missing one: the user has said they do
            # *not* practise this.
            return "conflict", f"not this: {', '.join(disciplines.not_practised)}"
        return "new", ""

    if proposal.kind == "not_discipline":
        value = fold(proposal.values[0]) if proposal.values else ""
        if value in _folded(disciplines.not_practised):
            return "agrees", ", ".join(disciplines.not_practised)
        if value in _folded(disciplines.practises):
            return "conflict", f"you practise: {', '.join(disciplines.practises)}"
        return "new", ""

    if proposal.kind == "location":
        current = _current_location(profile)
        if not current:
            return "new", ""
        if _folded(current) == _folded(proposal.values):
            return "agrees", ", ".join(current)
        return "conflict", ", ".join(current)

    current_level = _current_level(profile)
    if not current_level:
        return "new", ""
    if fold(current_level) == fold(proposal.values[0] if proposal.values else ""):
        return "agrees", current_level
    return "conflict", current_level


def to_view(proposal: ProposedSetting, profile: Profile) -> SuggestionView:
    """One proposal, paired with what the profile already says about it."""
    verdict, current = decide(profile, proposal)
    return SuggestionView(
        key=proposal.key,
        kind=proposal.kind,
        heading=KIND_HEADINGS[proposal.kind],
        note=KIND_NOTES[proposal.kind],
        values=list(proposal.values),
        source_lines=list(proposal.source_lines),
        verdict=verdict,
        current=current,
    )


def suggestions_view(
    run: ProfileSuggestionRun | None,
    profile: Profile,
    *,
    cost: object = None,
) -> SuggestionsView:
    """One run's view, whether or not there is a run.

    A run that failed shows a sentence and no proposals; a finished one shows
    everything still open, each against what the profile currently says.
    """
    if run is None:
        return SuggestionsView(run=None)
    failure = setting_failure(run.error_code) if run.status == "failed" else None
    return SuggestionsView(
        run=run,
        suggestions=[to_view(proposal, profile) for proposal in run.open_proposals],
        failure=failure,
        cost=cost,
    )


def _with_discipline(
    disciplines: Disciplines, value: str, *, practises: bool, replace: bool
) -> Disciplines:
    """The disciplines section with one value added to one of its two lists.

    Resolving a conflict *moves* the word rather than putting it on both lists,
    which would have the profile say someone both practises and does not
    practise the same thing.
    """
    keep_practises = list(disciplines.practises)
    keep_not = list(disciplines.not_practised)
    folded = fold(value)
    if replace:
        keep_practises = [item for item in keep_practises if fold(item) != folded]
        keep_not = [item for item in keep_not if fold(item) != folded]
    target = keep_practises if practises else keep_not
    if folded not in _folded(target):
        target.append(value)
    return Disciplines(practises=keep_practises, not_practised=keep_not)


def _with_constraint(profile: Profile, kind: str, constraint: Constraint) -> list[Constraint]:
    """The constraint list with one kind replaced in place, or appended.

    In place where it existed, so resolving a conflict does not move the row to
    the bottom of a list the user reads in order.
    """
    replaced = list(profile.constraints)
    position = next((i for i, c in enumerate(replaced) if c.kind == kind), None)
    if position is None:
        replaced.append(constraint)
    else:
        replaced[position] = constraint
    return replaced


def applied(
    profile: Profile,
    proposal: ProposedSetting,
    *,
    stance: Stance | None = None,
    places: Sequence[str] | None = None,
    level_text: str = "",
    replace: bool = False,
) -> Profile:
    """The profile with this proposal accepted onto it.

    Raises `SuggestionConflictError` when the profile already says something
    else and `replace` was not ticked -- rule 3, and the reason this function
    is the only way a proposal reaches `profiles.data`.

    `places` and `level_text` are the user's own words from the accept form:
    the places textarea starts prefilled with what the CV said and is theirs to
    edit, the level box starts empty. A proposal that merely agrees with the
    profile returns it unchanged, so accepting one is a no-op that still marks
    the question answered.
    """
    verdict, _current = decide(profile, proposal)
    if verdict == "conflict" and not replace:
        raise SuggestionConflictError(
            "Your profile already says something else here. Tick "
            "“replace what I have” if you want the CV's version instead."
        )
    if verdict == "agrees" and proposal.kind in ("discipline", "not_discipline"):
        return profile

    if proposal.kind in ("discipline", "not_discipline"):
        return profile.model_copy(
            update={
                "disciplines": _with_discipline(
                    profile.disciplines,
                    proposal.values[0],
                    practises=proposal.kind == "discipline",
                    replace=replace,
                )
            }
        )

    # Location and level are constraints, and a constraint with no stance is
    # refused rather than filed under a default -- see `jfl_web.profile`.
    if stance is None:
        from jfl_web.profile import MissingStanceError

        raise MissingStanceError(
            "Say whether this is a must, a nice to have or a never -- a CV does "
            "not state that, and a value without one does not say what it is."
        )

    if proposal.kind == "location":
        chosen = [place for place in (places if places is not None else proposal.values) if place]
        if not chosen:
            raise NoPlacesGivenError("Give at least one place, or reject this suggestion.")
        constraint = Constraint(kind="location", stance=stance, value=location_value(chosen))
        return profile.model_copy(
            update={"constraints": _with_constraint(profile, "location", constraint)}
        )

    text = level_text.strip()
    if not text:
        raise MissingLevelTextError(
            "Type the lowest level you would take. Your CVs say what you have "
            "been doing; what you would take next is your call."
        )
    constraint = Constraint(kind="level_floor", stance=stance, value=text_value(text))
    return profile.model_copy(
        update={"constraints": _with_constraint(profile, "level_floor", constraint)}
    )


__all__ = [
    "CONSTRAINT_FOR",
    "KIND_HEADINGS",
    "KIND_NOTES",
    "MAX_LEVEL_TEXT",
    "MAX_VALUE",
    "MissingLevelTextError",
    "NoPlacesGivenError",
    "SettingFailure",
    "SuggestionConflictError",
    "SuggestionView",
    "SuggestionsView",
    "applied",
    "decide",
    "setting_failure",
    "suggestions_view",
    "to_view",
]
