"""CV header settings and interests on the profile: validated, not trusted, and
never on the path to the corpus. Pure; fixtures are fictional."""

from __future__ import annotations

import pytest
from jfl_core.profile import (
    CORPUS_SECTIONS,
    SCHEMA_VERSION,
    CvHeaderLink,
    CvHeaderSettings,
    Profile,
    self_assessment_corpus_lines,
)
from pydantic import ValidationError


def test_schema_version_was_bumped_and_old_rows_still_read() -> None:
    assert SCHEMA_VERSION == 2
    old_row = {
        "constraints": [],
        "capabilities": [],
        "disciplines": {"practises": [], "not": []},
        "objectives": [],
        "self_assessment": {"depth_genuine": "", "recurring_gaps": ""},
    }
    profile = Profile.model_validate(old_row)
    assert profile.cv_header == CvHeaderSettings()
    assert profile.interests == []
    assert profile.is_empty


def test_a_full_header_round_trips() -> None:
    header = CvHeaderSettings(
        name="Robin Example",
        tagline="Engineering Manager | Platform",
        phone="+44 7700 900000",
        email="robin@example.test",
        location="Bristol, UK",
        links=[CvHeaderLink(label="github.com/robin", url="github.com/robin")],
    )
    profile = Profile(cv_header=header, interests=["  Sea   swimming ", "", "Chess"])
    again = Profile.model_validate(profile.as_json())
    assert again == profile
    assert again.cv_header.links[0].url == "https://github.com/robin"
    assert again.interests == ["Sea swimming", "Chess"]
    assert again.cv_header.contact == ["+44 7700 900000", "robin@example.test", "Bristol, UK"]


@pytest.mark.parametrize("email", ["not-an-email", "a@b", "two@@example.test", "a b@example.test"])
def test_a_bad_email_is_refused(email: str) -> None:
    with pytest.raises(ValidationError):
        CvHeaderSettings(email=email)


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "data:text/html,hi", "ftp://example.test", "https://localhost", "   "],
)
def test_a_link_that_is_not_a_web_address_is_refused(url: str) -> None:
    with pytest.raises(ValidationError):
        CvHeaderLink(label="Mine", url=url)


def test_a_phone_with_letters_is_refused() -> None:
    with pytest.raises(ValidationError):
        CvHeaderSettings(phone="call me maybe")


def test_too_many_links_or_interests_are_refused() -> None:
    link = CvHeaderLink(label="x", url="https://example.test")
    with pytest.raises(ValidationError):
        CvHeaderSettings(links=[link] * 6)
    with pytest.raises(ValidationError):
        Profile(interests=[f"interest {n}" for n in range(13)])


def test_header_and_interests_never_reach_the_corpus_lines() -> None:
    """The one mapping `save_profile` writes to the corpus from names only the
    self-assessment. Settings on the CV header are not claims."""
    profile = Profile(
        cv_header=CvHeaderSettings(name="Robin Example", email="robin@example.test"),
        interests=["Sea swimming"],
    )
    lines = self_assessment_corpus_lines(profile)
    assert set(lines) == set(CORPUS_SECTIONS.values())
    assert all(not texts for texts in lines.values())
