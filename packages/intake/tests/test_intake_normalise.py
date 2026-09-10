"""The fingerprint normalises exactly what its docstring says, and nothing more."""

from __future__ import annotations

from jfl_intake.normalise import clean_text, fingerprint, normalise


def test_case_punctuation_and_whitespace_do_not_distinguish_a_role() -> None:
    assert fingerprint("Sr. Engineer (Remote)", "London, UK") == fingerprint(
        "  sr   engineer remote ", "london uk"
    )


def test_punctuation_becomes_a_space_rather_than_fusing_words() -> None:
    assert normalise("Engineer,Platform") == "engineer platform"
    assert normalise("Front-end") == "front end"


def test_unicode_is_folded_compatibly() -> None:
    assert normalise("Ｅｎｇｉｎｅｅｒ") == "engineer"  # full-width
    assert normalise("STRASSE") == normalise("straße")  # casefold, not lower


def test_no_synonyms_or_stemming_are_applied() -> None:
    """A guess about sameness would write a false "reposted" into the history."""
    assert fingerprint("Sr Engineer", "London") != fingerprint("Senior Engineer", "London")
    assert fingerprint("Engineer", "London") != fingerprint("Engineers", "London")


def test_a_missing_location_is_the_empty_string() -> None:
    assert fingerprint("Engineer", None) == fingerprint("Engineer", "   ") == "engineer|"


def test_the_separator_cannot_occur_inside_either_half() -> None:
    assert fingerprint("a|b", "c") == "a b|c"
    assert fingerprint("a", "b|c") == "a|b c"


def test_clean_text_keeps_what_the_owner_reads() -> None:
    assert clean_text("  Staff Engineer, Platform  ") == "Staff Engineer, Platform"
    assert clean_text("   ") is None
    assert clean_text(None) is None
    assert clean_text({"name": "London"}) is None  # a shape error, not a title
