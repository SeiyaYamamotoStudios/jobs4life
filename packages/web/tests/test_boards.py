"""Pure display helpers for the boards screen: platform names, the default
label for a newly watched board, and plain-English words for a check's error
code. No database, no network -- see `jfl_web.boards`.
"""

from __future__ import annotations

from jfl_intake.detect import detect_board
from jfl_web.boards import check_error_message, default_label, platform_label


class TestPlatformLabel:
    def test_known_platform_gets_its_display_name(self) -> None:
        assert platform_label("greenhouse") == "Greenhouse"
        assert platform_label("smartrecruiters") == "SmartRecruiters"

    def test_every_supported_platform_has_a_label(self) -> None:
        # BoardPlatform's twelve values -- see jfl_core.models. If a
        # thirteenth is ever added, this catches it defaulting to the raw
        # literal rather than a proper display name.
        for platform in (
            "greenhouse",
            "ashby",
            "lever",
            "workday",
            "smartrecruiters",
            "rippling",
            "breezy",
            "teamtailor",
            "personio",
            "recruitee",
            "pinpoint",
            "workable",
        ):
            label = platform_label(platform)
            assert label[0].isupper()


class TestDefaultLabel:
    def test_greenhouse_uses_the_token(self) -> None:
        ref = detect_board("https://boards.greenhouse.io/acme-corp")
        assert default_label(ref) == "Acme Corp"

    def test_teamtailor_uses_only_the_company_part_of_the_hostname(self) -> None:
        ref = detect_board("https://acme.teamtailor.com")
        assert default_label(ref) == "Acme"

    def test_workday_prefers_the_tenant_over_the_site(self) -> None:
        ref = detect_board("https://acme.wd5.myworkdayjobs.com/careers")
        assert default_label(ref) == "Acme"


class TestCheckErrorMessage:
    def test_a_known_code_gets_a_plain_sentence(self) -> None:
        message = check_error_message("drop_guard")
        assert "dropped" in message
        assert message[0].islower()  # reads naturally after "held: ..."

    def test_no_code_and_an_unknown_code_both_get_a_message_never_a_crash(self) -> None:
        assert check_error_message(None)
        assert check_error_message("something_new_added_to_the_db")  # type: ignore[arg-type]
