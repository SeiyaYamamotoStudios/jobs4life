"""The CV header and interests, read from the profile's settings."""

from __future__ import annotations

from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvLink
from jfl_core.cv_header import header_from_profile, header_name
from jfl_core.profile import CvHeaderLink, CvHeaderSettings, Profile


def _profile(**settings: object) -> Profile:
    interests = settings.pop("interests", [])
    return Profile(cv_header=CvHeaderSettings(**settings), interests=interests)  # type: ignore[arg-type]


class TestHeaderName:
    def test_the_profile_name_wins(self) -> None:
        assert header_name(_profile(name="Morgan F."), "Account Name") == "Morgan F."

    def test_the_account_name_when_the_profile_names_no_one(self) -> None:
        assert header_name(Profile(), "Morgan Fictional") == "Morgan Fictional"

    def test_empty_when_neither_names_anyone(self) -> None:
        # jfl_generate.cv_document then falls back to the corpus title.
        assert header_name(Profile(), "  ") == ""


def _doc() -> CvDocument:
    return CvDocument(
        header=CvHeader(name="From Generation"),
        summary=[CvLine(text="Led a platform team.", verdict="supported")],
    )


class TestHeaderFromProfile:
    def test_every_setting_flows_into_the_document(self) -> None:
        profile = _profile(
            name="Morgan Fictional",
            tagline="Engineering Manager | Platform",
            phone="07700 900000",
            email="morgan@example.org",
            location="Bristol, UK",
            links=[CvHeaderLink(label="github.com/morgan", url="https://github.com/morgan")],
            interests=["Fell running", "Choral singing"],
        )
        out = header_from_profile(profile, _doc())
        assert out.header == CvHeader(
            name="Morgan Fictional",
            tagline="Engineering Manager | Platform",
            contact=["07700 900000", "morgan@example.org", "Bristol, UK"],
            links=[CvLink(label="github.com/morgan", url="https://github.com/morgan")],
        )
        assert out.interests == ["Fell running", "Choral singing"]

    def test_a_blank_profile_name_keeps_the_documents_name(self) -> None:
        out = header_from_profile(_profile(phone="07700 900000"), _doc())
        assert out.header.name == "From Generation"
        assert out.header.contact == ["07700 900000"]

    def test_the_claims_and_their_verdicts_are_untouched(self) -> None:
        doc = _doc()
        out = header_from_profile(_profile(name="X", interests=["Chess"]), doc)
        assert out.summary == doc.summary
        assert out.generated_lines() == doc.generated_lines()
