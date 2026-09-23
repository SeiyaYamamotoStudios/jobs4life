"""The generated CV's check in plain words, the export warning, the text
export and the file name. Pure; fixtures are fictional."""

from __future__ import annotations

import pytest
from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvLink, CvRole, CvSkill
from jfl_web.cvdocs import (
    check_failure,
    cv_check,
    cv_filename,
    cv_plain_text,
    export_warning,
    line_key,
    line_meaning,
)


def _doc(*bullets: CvLine) -> CvDocument:
    return CvDocument(
        header=CvHeader(
            name="Robin Example",
            tagline="Engineering Manager",
            contact=["robin@example.test"],
            links=[CvLink(label="github.com/robin", url="https://github.com/robin")],
        ),
        summary=[CvLine(text="A summary.", verdict="framing")],
        skills=[CvSkill(label="Platform", text=CvLine(text="Builds.", verdict="supported"))],
        roles=[
            CvRole(
                title="EM",
                employer="Fictional Freight",
                dates="2021 – Present",
                bullets=list(bullets),
            )
        ],
        education=[CvLine(text="BSc Computing", origin="fact")],
        interests=["Chess"],
    )


def test_framing_and_unchecked_lines_are_never_marked_supported() -> None:
    assert line_key(CvLine(text="x", verdict="framing")) == "not_checked"
    assert line_key(CvLine(text="x", verdict=None)) == "not_checked"
    assert line_key(CvLine(text="x", verdict="supported")) == "supported"


def test_an_edited_line_says_it_has_not_been_checked_since() -> None:
    assert line_meaning(CvLine(text="x", origin="user")) == "Not checked since you edited it."


def test_the_warning_counts_lines_the_facts_do_not_back() -> None:
    doc = _doc(
        CvLine(text="One.", verdict="unsupported"),
        CvLine(text="Two.", verdict="review"),
        CvLine(text="Three.", verdict="supported"),
    )
    assert export_warning(doc) == "2 lines aren't backed by your confirmed facts."
    one = _doc(CvLine(text="One.", verdict="unsupported"))
    assert export_warning(one) == "1 line isn't backed by your confirmed facts."


def test_the_warning_names_unchecked_edits_too_and_is_empty_when_all_is_well() -> None:
    doc = _doc(CvLine(text="Mine.", origin="user"))
    assert export_warning(doc) == "1 line you edited hasn't been checked."
    assert export_warning(_doc(CvLine(text="Fine.", verdict="supported"))) == ""


def test_the_check_groups_worst_first_and_keeps_edits_apart() -> None:
    check = cv_check(
        _doc(
            CvLine(text="Review.", verdict="review"),
            CvLine(text="Bad.", verdict="unsupported"),
            CvLine(text="Mine.", origin="user"),
        )
    )
    assert [v.line.text for v in check.flagged] == ["Bad.", "Review."]
    assert [v.line.text for v in check.edited_unchecked] == ["Mine."]
    assert [v.line.text for v in check.unchecked] == ["A summary."]
    assert check.flagged[0].action is not None
    assert check.flagged[0].where == "EM, Fictional Freight"


def test_plain_text_is_the_words_only() -> None:
    doc = _doc(CvLine(text="Cut costs.", verdict="unsupported", note="SECRET NOTE"))
    text = cv_plain_text(doc)
    for words in ("Robin Example", "A summary.", "Builds.", "Cut costs.", "BSc Computing", "Chess"):
        assert words in text
    assert "SECRET NOTE" not in text
    assert "Not supported" not in text


@pytest.mark.parametrize(
    ("name", "employer", "expected"),
    [
        ("Robin Example", "Acme Ltd", "Robin_Example_CV_Acme_Ltd.pdf"),
        ("Zoë O'Brien", 'Café "Quotes"/Slash', "Zoe_O_Brien_CV_Cafe_Quotes_Slash.pdf"),
        ("", None, "CV.pdf"),
    ],
)
def test_file_names_are_sanitised(name: str, employer: str | None, expected: str) -> None:
    assert cv_filename(name, employer, "pdf") == expected


def test_check_failure_reads_the_code_and_nothing_else() -> None:
    failure = check_failure("PermanentTaskError: cv edit check failed permanently: no_api_key")
    assert failure.fix_url == "/settings"
    assert "tried" not in check_failure("something else sk-ant-secret").message
    assert "sk-ant" not in check_failure("something else sk-ant-secret").message
