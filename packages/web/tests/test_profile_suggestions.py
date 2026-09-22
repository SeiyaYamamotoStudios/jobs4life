"""The rules that decide what a CV-read suggestion does to a profile.

No database and no model here -- `jfl_web.profilesuggestions` is pure, and it
holds the three answers this panel exists to get right:

  * a setting the user has already stated **wins**;
  * a proposal that agrees adds nothing;
  * a proposal that conflicts is **never** applied on an ordinary accept.

Plus the two things a CV cannot tell us and the user therefore must: whether a
constraint is a must, a nice to have or a never, and what their level floor
actually is.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import get_args

import pytest
from jfl_core.ids import setting_key
from jfl_core.models import (
    PROFILE_SUGGESTION_KINDS,
    ProfileSuggestionKind,
    ProfileSuggestionRun,
    ProposedSetting,
)
from jfl_core.profile import Constraint, Disciplines, Profile, location_value, text_value
from jfl_web.profile import MissingStanceError
from jfl_web.profilesuggestions import (
    KIND_HEADINGS,
    KIND_NOTES,
    MissingLevelTextError,
    NoPlacesGivenError,
    SuggestionConflictError,
    applied,
    decide,
    setting_failure,
    suggestions_view,
)


def _proposal(kind: str, *values: str, lines: list[str] | None = None) -> ProposedSetting:
    return ProposedSetting(
        kind=kind,  # type: ignore[arg-type]
        key=setting_key(kind, list(values)),
        values=list(values),
        source_lines=lines or [f"a line about {values[0]}"],
    )


def _run(*proposals: ProposedSetting, status: str = "done") -> ProfileSuggestionRun:
    when = dt.datetime(2026, 9, 22, tzinfo=dt.UTC)
    return ProfileSuggestionRun(
        id=uuid.uuid4(),
        status=status,  # type: ignore[arg-type]
        trace_id=uuid.uuid4(),
        proposals=list(proposals),
        cv_count=1,
        created_at=when,
        updated_at=when,
    )


# -- the words the panel puts on each kind ------------------------------------


def test_every_kind_has_a_heading_and_a_note() -> None:
    """The same agreement `jfl_web.profile` keeps for the profile form's own
    value lists: a kind the screen cannot describe is a kind that renders blank.
    """
    kinds = set(get_args(ProfileSuggestionKind))
    assert kinds == set(PROFILE_SUGGESTION_KINDS)
    assert set(KIND_HEADINGS) == kinds
    assert set(KIND_NOTES) == kinds


def test_an_unknown_error_code_still_gets_a_sentence() -> None:
    assert setting_failure(None).message
    assert setting_failure("no_api_key").fix_url == "/settings"


# -- decide -------------------------------------------------------------------


def test_a_discipline_the_profile_says_nothing_about_is_new() -> None:
    assert decide(Profile(), _proposal("discipline", "platform engineering")) == ("new", "")


def test_a_discipline_the_user_already_practises_agrees() -> None:
    profile = Profile(disciplines=Disciplines(practises=["Platform Engineering"]))
    verdict, current = decide(profile, _proposal("discipline", "platform engineering"))
    assert verdict == "agrees"
    assert "Platform Engineering" in current


def test_a_discipline_the_user_has_ruled_out_conflicts() -> None:
    """The opposite claim, not a missing one -- they have said they do *not*
    practise this.
    """
    profile = Profile(disciplines=Disciplines(not_practised=["frontend"]))
    verdict, current = decide(profile, _proposal("discipline", "frontend"))
    assert verdict == "conflict"
    assert "not this" in current


def test_a_not_this_the_user_practises_conflicts() -> None:
    profile = Profile(disciplines=Disciplines(practises=["frontend"]))
    assert decide(profile, _proposal("not_discipline", "frontend"))[0] == "conflict"


def test_a_location_matching_what_the_user_stated_agrees() -> None:
    profile = Profile(
        constraints=[
            Constraint(kind="location", stance="must", value=location_value(["London", "Bristol"]))
        ]
    )
    assert decide(profile, _proposal("location", "London", "Bristol"))[0] == "agrees"


def test_a_location_in_a_different_order_conflicts() -> None:
    """The order is the ranking -- "London then Bristol" and "Bristol then
    London" are different statements, not the same one written twice.
    """
    profile = Profile(
        constraints=[
            Constraint(kind="location", stance="must", value=location_value(["Bristol", "London"]))
        ]
    )
    assert decide(profile, _proposal("location", "London", "Bristol"))[0] == "conflict"


def test_a_level_the_user_has_already_set_conflicts() -> None:
    profile = Profile(
        constraints=[Constraint(kind="level_floor", stance="must", value=text_value("EM or above"))]
    )
    verdict, current = decide(profile, _proposal("level", "has been operating at EM level"))
    assert verdict == "conflict"
    assert current == "EM or above"


# -- applying ------------------------------------------------------------------


def test_accepting_a_discipline_appends_it() -> None:
    updated = applied(Profile(), _proposal("discipline", "platform engineering"))
    assert updated.disciplines.practises == ["platform engineering"]


def test_accepting_a_not_this_goes_on_the_other_list() -> None:
    updated = applied(Profile(), _proposal("not_discipline", "frontend"))
    assert updated.disciplines.not_practised == ["frontend"]
    assert updated.disciplines.practises == []


def test_accepting_something_the_user_already_says_changes_nothing() -> None:
    profile = Profile(disciplines=Disciplines(practises=["Platform Engineering"]))
    assert applied(profile, _proposal("discipline", "platform engineering")) == profile


def test_a_conflicting_accept_is_refused_rather_than_applied() -> None:
    """Rule 3. Quietly replacing what someone said with what a model read off
    their CV is the failure that would make the rest of this meaningless.
    """
    profile = Profile(disciplines=Disciplines(not_practised=["frontend"]))
    with pytest.raises(SuggestionConflictError):
        applied(profile, _proposal("discipline", "frontend"))
    assert profile.disciplines.not_practised == ["frontend"]


def test_a_conflict_the_user_chose_to_resolve_moves_the_word() -> None:
    """Resolving moves it rather than putting it on both lists, which would
    have the profile say someone both practises and does not practise one
    thing.
    """
    profile = Profile(disciplines=Disciplines(not_practised=["frontend"]))
    updated = applied(profile, _proposal("discipline", "frontend"), replace=True)
    assert updated.disciplines.practises == ["frontend"]
    assert updated.disciplines.not_practised == []


def test_a_location_needs_a_stance() -> None:
    """A CV says where someone worked. It does not say whether London is a line
    or a preference, and that difference is the only reason to record it.
    """
    with pytest.raises(MissingStanceError):
        applied(Profile(), _proposal("location", "London"))


def test_accepting_a_location_records_the_order_the_user_left() -> None:
    updated = applied(
        Profile(),
        _proposal("location", "London", "Bristol"),
        stance="nice",
        places=["Bristol", "London", "remote"],
    )
    constraint = updated.constraint("location")
    assert constraint is not None
    assert constraint.stance == "nice"
    assert constraint.value["places"] == ["Bristol", "London", "remote"]


def test_a_location_with_every_place_deleted_is_refused() -> None:
    with pytest.raises(NoPlacesGivenError):
        applied(Profile(), _proposal("location", "London"), stance="must", places=[])


def test_replacing_a_location_keeps_its_position_in_the_list() -> None:
    profile = Profile(
        constraints=[
            Constraint(kind="location", stance="must", value=location_value(["Leeds"])),
            Constraint(kind="workplace", stance="must", value=text_value("remote")),
        ]
    )
    updated = applied(
        profile,
        _proposal("location", "London"),
        stance="must",
        places=["London"],
        replace=True,
    )
    assert [c.kind for c in updated.constraints] == ["location", "workplace"]
    assert updated.constraint("location").value["places"] == ["London"]  # type: ignore[union-attr]


def test_a_level_is_never_written_from_the_observation() -> None:
    """`level_floor` is a choice the user makes, not a fact about their past.
    The box starts empty and an empty box is refused rather than filled in with
    what the CV happened to describe.
    """
    with pytest.raises(MissingLevelTextError):
        applied(
            Profile(),
            _proposal("level", "has been operating at engineering-manager level"),
            stance="must",
            level_text="   ",
        )


def test_accepting_a_level_records_the_users_own_words() -> None:
    updated = applied(
        Profile(),
        _proposal("level", "has been operating at engineering-manager level"),
        stance="must",
        level_text="Engineering manager or above",
    )
    constraint = updated.constraint("level_floor")
    assert constraint is not None
    assert constraint.value["text"] == "Engineering manager or above"
    assert constraint.stance == "must"


def test_a_level_needs_a_stance_too() -> None:
    with pytest.raises(MissingStanceError):
        applied(Profile(), _proposal("level", "EM level"), level_text="EM or above")


# -- the view ------------------------------------------------------------------


def test_no_run_renders_no_claim_about_one() -> None:
    view = suggestions_view(None, Profile())
    assert view.run is None and view.suggestions == []


def test_a_finished_run_shows_each_proposal_against_what_the_profile_says() -> None:
    profile = Profile(disciplines=Disciplines(practises=["frontend"]))
    run = _run(
        _proposal("discipline", "platform engineering"),
        _proposal("not_discipline", "frontend"),
    )
    view = suggestions_view(run, profile, cost=None)
    assert [s.verdict for s in view.suggestions] == ["new", "conflict"]
    assert all(s.source_lines for s in view.suggestions), "a proposal with no CV words behind it"


def test_an_answered_proposal_is_not_offered_again_on_the_screen() -> None:
    answered = _proposal("discipline", "platform engineering").model_copy(
        update={"state": "rejected"}
    )
    view = suggestions_view(_run(answered), Profile())
    assert view.suggestions == []


def test_a_failed_run_shows_a_sentence_and_no_proposals() -> None:
    run = _run(status="failed")
    run = run.model_copy(update={"error_code": "no_api_key"})
    view = suggestions_view(run, Profile())
    assert view.failure is not None
    assert view.failure.fix_url == "/settings"
    assert view.suggestions == []


def test_a_location_and_a_level_are_the_two_that_need_a_stance() -> None:
    run = _run(
        _proposal("discipline", "platform engineering"),
        _proposal("location", "London"),
        _proposal("level", "EM level"),
    )
    view = suggestions_view(run, Profile())
    assert [s.needs_stance for s in view.suggestions] == [False, True, True]
