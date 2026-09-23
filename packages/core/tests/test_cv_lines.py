"""Editing a generated CV: wording is editable, facts are not, and an edited
line loses its verdict. Pure -- no database, no model. Fixtures are fictional."""

from __future__ import annotations

import pytest
from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvRole, CvSkill
from jfl_core.cv_lines import (
    InvalidEditError,
    ProtectedFieldError,
    apply_edits,
    editable_fields,
    line_at,
    line_paths,
    unchecked_edits,
    with_verdicts,
    worst_verdict,
)


def _doc() -> CvDocument:
    return CvDocument(
        header=CvHeader(name="Robin Example", contact=["robin@example.test"]),
        summary=[CvLine(text="Engineering leader for platform teams.", verdict="framing")],
        skills=[
            CvSkill(
                label="Platform",
                text=CvLine(text="Ran the build farm for four teams.", verdict="supported"),
            )
        ],
        roles=[
            CvRole(
                title="Engineering Manager",
                employer="Fictional Freight Ltd",
                dates="Jan 2021 – Present",
                descriptor="Logistics software.",
                bullets=[
                    CvLine(text="Led a team of eight engineers.", verdict="supported"),
                    CvLine(text="Cut costs by 40%.", verdict="unsupported", note="No figure."),
                ],
            )
        ],
        education=[CvLine(text="BSc Computing, Example University", origin="fact")],
        interests=["Sea swimming"],
    )


def test_paths_cover_every_checkable_line_in_reading_order() -> None:
    doc = _doc()
    assert [p for p, _ in line_paths(doc)] == [
        "summary.0",
        "skills.0.text",
        "roles.0.bullets.0",
        "roles.0.bullets.1",
    ]
    assert [line for _, line in line_paths(doc)] == doc.generated_lines()


def test_facts_are_never_offered_as_fields() -> None:
    paths = {f.path for f in editable_fields(_doc())}
    for fact in ("roles.0.title", "roles.0.employer", "roles.0.dates", "roles.0.location"):
        assert fact not in paths
    assert not any(p.startswith(("education.", "header", "interests.")) for p in paths)
    assert {"roles.0.descriptor", "skills.0.label", "skills_heading"} <= paths


def test_an_edited_line_becomes_the_users_and_loses_its_verdict() -> None:
    result = apply_edits(_doc(), {"roles.0.bullets.1": "Cut hosting costs."})
    line = result.doc.roles[0].bullets[1]
    assert line.text == "Cut hosting costs."
    assert line.origin == "user"
    assert line.verdict is None and line.note == ""
    assert result.edited == 1
    assert unchecked_edits(result.doc) == ["roles.0.bullets.1"]


def test_an_unchanged_line_keeps_its_origin_and_verdict() -> None:
    doc = _doc()
    result = apply_edits(doc, {"roles.0.bullets.0": "  Led a team of eight\r\nengineers. "})
    assert not result.changed
    assert result.doc == doc


@pytest.mark.parametrize(
    "field",
    [
        "roles.0.title",
        "roles.0.employer",
        "roles.0.dates",
        "roles.0.location",
        "education.0",
        "header.name",
        "interests",
        "summary.9",
        "template",
    ],
)
def test_a_fact_or_unknown_field_is_refused(field: str) -> None:
    with pytest.raises(ProtectedFieldError):
        apply_edits(_doc(), {"roles.0.bullets.0": "Changed.", field: "Chief Executive"})


def test_emptying_a_line_cuts_it_and_a_skill_goes_with_its_line() -> None:
    result = apply_edits(_doc(), {"roles.0.bullets.1": "", "skills.0.text": " "})
    assert [b.text for b in result.doc.roles[0].bullets] == ["Led a team of eight engineers."]
    assert result.doc.skills == []
    assert result.removed == 2


def test_emptied_heading_or_label_is_an_error() -> None:
    with pytest.raises(InvalidEditError):
        apply_edits(_doc(), {"skills_heading": ""})
    with pytest.raises(InvalidEditError):
        apply_edits(_doc(), {"skills.0.label": ""})


def test_headings_descriptors_and_labels_are_editable() -> None:
    result = apply_edits(
        _doc(),
        {
            "skills_heading": "Skills",
            "skills.0.label": "Delivery",
            "roles.0.descriptor": "Freight software.",
        },
    )
    assert result.doc.skills_heading == "Skills"
    assert result.doc.skills[0].label == "Delivery"
    assert result.doc.roles[0].descriptor == "Freight software."
    # A label change does not touch the line's verdict.
    assert result.doc.skills[0].text.verdict == "supported"


def test_over_long_value_is_refused_not_truncated() -> None:
    with pytest.raises(InvalidEditError):
        apply_edits(_doc(), {"summary.0": "x" * 5000})


def test_verdicts_land_only_on_lines_whose_text_still_matches() -> None:
    edited = apply_edits(
        _doc(), {"roles.0.bullets.1": "Cut hosting costs.", "summary.0": "New summary."}
    ).doc
    checked = with_verdicts(
        edited,
        {
            "roles.0.bullets.1": ("Cut hosting costs.", "review", "Partly."),
            "summary.0": ("An older wording.", "supported", ""),
        },
    )
    assert checked.roles[0].bullets[1].verdict == "review"
    assert checked.roles[0].bullets[1].note == "Partly."
    assert checked.roles[0].bullets[1].origin == "user"
    assert line_at(checked, "summary.0") is not None
    assert checked.summary[0].verdict is None


def test_worst_verdict() -> None:
    assert worst_verdict(["supported", "unsupported", "review"]) == "unsupported"
    assert worst_verdict(["framing", "supported"]) == "supported"
    assert worst_verdict(["framing"]) == "framing"
    assert worst_verdict([]) is None
