"""The paste box's two pure decisions: what to call the row before anything has
read the ad, and what to do with the optional link.

Both run in the request path, so both are deliberately dumb -- the expensive,
accurate answer comes back from the worker half a minute later.
"""

from __future__ import annotations

import pytest
from jfl_web.jobads import (
    FALLBACK_TITLE,
    MAX_TITLE_CHARS,
    extraction_failure,
    normalise_url,
    provisional_title,
)


class TestProvisionalTitle:
    def test_the_first_line_becomes_the_title(self) -> None:
        assert provisional_title("Senior Platform Engineer\n\nAcme Corp") == (
            "Senior Platform Engineer"
        )

    def test_leading_blank_lines_are_skipped(self) -> None:
        assert provisional_title("\n\n   \nStaff Engineer\nAcme") == "Staff Engineer"

    def test_markdown_and_bullet_decoration_is_stripped(self) -> None:
        assert provisional_title("## Staff Engineer ##") == "Staff Engineer"
        assert provisional_title("* Staff Engineer") == "Staff Engineer"
        assert provisional_title("--- \nStaff Engineer") == "Staff Engineer"

    def test_a_line_of_pure_punctuation_is_not_a_title(self) -> None:
        """A rule you find out you need the first time someone pastes a PDF."""
        assert provisional_title("=====\n-----\nStaff Engineer") == "Staff Engineer"

    def test_internal_whitespace_is_collapsed(self) -> None:
        assert provisional_title("Staff    Engineer\t\tII") == "Staff Engineer II"

    def test_a_long_first_line_is_cut_at_a_word_boundary(self) -> None:
        title = provisional_title("Senior " * 40)
        assert len(title) <= MAX_TITLE_CHARS
        assert title.endswith("…")
        assert not title.endswith(" …")

    def test_an_unbroken_run_is_still_cut(self) -> None:
        title = provisional_title("x" * 500)
        assert len(title) == MAX_TITLE_CHARS
        assert title.endswith("…")

    def test_text_with_no_words_at_all_falls_back(self) -> None:
        assert provisional_title("\n\n---\n") == FALLBACK_TITLE
        assert provisional_title("") == FALLBACK_TITLE


class TestNormaliseUrl:
    def test_blank_is_none_rather_than_an_error(self) -> None:
        assert normalise_url("") is None
        assert normalise_url("   ") is None

    @pytest.mark.parametrize("value", ["https://acme.example/jobs/1", "http://acme.example/jobs/1"])
    def test_http_and_https_are_kept(self, value: str) -> None:
        assert normalise_url(f"  {value} ") == value

    @pytest.mark.parametrize(
        "value", ["acme.example/jobs/1", "javascript:alert(1)", "ftp://acme.example"]
    )
    def test_anything_that_is_not_a_web_link_is_rejected(self, value: str) -> None:
        """The URL is rendered as an `href`. Rejecting `javascript:` is the
        whole of the defence needed, because nothing here ever fetches it.
        """
        with pytest.raises(ValueError, match="http"):
            normalise_url(value)


class TestExtractionFailure:
    def test_a_missing_key_sends_the_user_to_settings(self) -> None:
        failure = extraction_failure("no_api_key")
        assert failure.fix_url == "/settings"
        assert failure.fix_label

    def test_a_rejected_key_sends_the_user_to_settings_too(self) -> None:
        assert extraction_failure("api_key_rejected").fix_url == "/settings"

    def test_a_model_error_offers_no_settings_link_because_settings_would_not_help(
        self,
    ) -> None:
        assert extraction_failure("model_error").fix_url is None

    def test_an_absent_code_still_produces_a_sentence(self) -> None:
        """A blank panel is worse than a vague one."""
        assert extraction_failure(None).message
