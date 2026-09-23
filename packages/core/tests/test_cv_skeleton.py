"""The CV skeleton: roles, dates and education read out of the corpus with no
model anywhere. Every fixture here is fictional.

Spans are produced by the real parser (`parse_document`), so these tests read
exactly the `section_path` shapes ingestion writes -- both the owner-authored
markdown shape and the hosted, CV-onboarded one (`jfl_core.corpus_source`).
"""

from __future__ import annotations

import uuid

import pytest
from jfl_core.corpus_source import SOURCE_URI as HOSTED_URI
from jfl_core.corpus_source import append_line
from jfl_core.cv_document import CvDocument, CvHeader, CvLine, CvRole, CvSkill
from jfl_core.cv_skeleton import build_skeleton, name_from_title, split_trailing_dates
from jfl_core.ingest.parser import parse_document
from jfl_core.models import Span

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")

OWNER_CORPUS = """\
# Morgan Fictional — Career Record

## Northwind Traders
Northwind sells tea wholesale across the UK.

### Head of Engineering, Nov 2021 – Present
- Location: Bristol, UK
- Led a platform team of eight engineers.

#### Numbers
- Cut deploy time from 40 to 12 minutes.

### Senior Engineer, Mar 2018 – Oct 2021
- Built the pricing service.

### Interim Lead
- Covered the lead role during a hiring gap.

## Contoso (2012–2018)

### Engineer | Leeds, 2012 – 2018
- Wrote the order pipeline in Java.

## Education
- BSc Computer Science, University of Nowhere, 2008

### MSc Data Science, 2011

## Independent / Non-Employment Activity

### Open-source maintainer, 2019 – Present
- Maintains a small scheduling library.

## Things stated as NOT true
- Morgan has never managed a budget.
- Morgan has not worked in Kubernetes in production.
"""


def _spans(uri: str, markdown: str) -> list[Span]:
    return parse_document(uri, markdown, USER).spans


def _hosted(*facts: tuple[str, str]) -> list[Span]:
    """A hosted corpus built through the real write path's markdown function."""
    content = ""
    for section, line in facts:
        content = append_line(content, section, line)
    return _spans(HOSTED_URI, content)


class TestOwnerShape:
    def setup_method(self) -> None:
        self.skeleton = build_skeleton(_spans("file:corpus/record.md", OWNER_CORPUS))

    def test_roles_are_read_verbatim_most_recent_first(self) -> None:
        got = [(r.employer, r.title, r.dates) for r in self.skeleton.roles]
        assert got == [
            ("Northwind Traders", "Head of Engineering", "Nov 2021 – Present"),
            ("Northwind Traders", "Senior Engineer", "Mar 2018 – Oct 2021"),
            ("Contoso", "Engineer", "2012 – 2018"),
            # Undated, beside dated siblings: still a role, text kept, sorted last.
            ("Northwind Traders", "Interim Lead", ""),
        ]

    def test_facts_include_sub_sections_and_leave_out_the_location_line(self) -> None:
        head = self.skeleton.roles[0]
        assert head.location == "Bristol, UK"
        assert head.facts == (
            "Led a platform team of eight engineers.",
            "Cut deploy time from 40 to 12 minutes.",
        )
        assert len(head.fact_span_ids) == 2

    def test_a_location_written_on_the_heading_is_split_off(self) -> None:
        contoso = self.skeleton.roles[2]
        assert (contoso.title, contoso.location) == ("Engineer", "Leeds")

    def test_education_lines_are_verbatim_and_in_corpus_order(self) -> None:
        assert self.skeleton.education == (
            "BSc Computer Science, University of Nowhere, 2008",
            "MSc Data Science, 2011",
        )

    def test_the_boundaries_section_is_never_a_role_or_an_education_line(self) -> None:
        assert self.skeleton.boundaries == (
            "Morgan has never managed a budget.",
            "Morgan has not worked in Kubernetes in production.",
        )
        rendered = [f for r in self.skeleton.roles for f in r.facts]
        rendered += list(self.skeleton.education)
        assert not any("never managed" in line for line in rendered)

    def test_independent_activity_is_not_a_job(self) -> None:
        assert all("Open-source" not in r.title for r in self.skeleton.roles)

    def test_the_document_title_is_kept_for_the_header_fallback(self) -> None:
        assert self.skeleton.corpus_title == "Morgan Fictional — Career Record"
        assert name_from_title(self.skeleton.corpus_title) == "Morgan Fictional"


class TestHostedShape:
    def test_role_labels_split_into_employer_title_and_dates(self) -> None:
        skeleton = build_skeleton(
            _hosted(
                ("Acme Ltd -- Engineering Manager, 2021-2024", "Ran hiring for the team."),
                ("Globex — Staff Engineer, Jan 2016 – Jun 2021", "Wrote the billing system."),
                ("Senior Developer at Initech, 2012-2015", "Maintained the TPS service."),
            )
        )
        got = [(r.employer, r.title, r.dates, r.facts) for r in skeleton.roles]
        assert got == [
            ("Acme Ltd", "Engineering Manager", "2021-2024", ("Ran hiring for the team.",)),
            ("Globex", "Staff Engineer", "Jan 2016 – Jun 2021", ("Wrote the billing system.",)),
            ("Initech", "Senior Developer", "2012-2015", ("Maintained the TPS service.",)),
        ]

    def test_a_label_with_no_readable_dates_keeps_its_text_and_sorts_last(self) -> None:
        skeleton = build_skeleton(
            _hosted(
                ("Umbrella Corp -- Platform Lead, sometime recently", "Ran the platform."),
                ("Acme Ltd -- Engineering Manager, 2021-2024", "Ran hiring."),
            )
        )
        last = skeleton.roles[-1]
        assert (last.employer, last.title, last.dates) == (
            "Umbrella Corp",
            "Platform Lead, sometime recently",
            "",
        )
        assert skeleton.roles[0].employer == "Acme Ltd"

    def test_profile_and_unfiled_sections_are_not_roles(self) -> None:
        skeleton = build_skeleton(
            _hosted(
                ("Depth and exposure", "Deep in Python, working in Go."),
                ("Recurring gaps", "No regulated-industry work."),
                ("Confirmed Facts", "Speaks at meetups."),
                ("Acme Ltd -- Engineering Manager, 2021-2024", "Ran hiring."),
            )
        )
        assert [r.employer for r in skeleton.roles] == ["Acme Ltd"]
        assert skeleton.education == ()

    def test_education_under_the_cvs_own_heading(self) -> None:
        skeleton = build_skeleton(
            _hosted(
                ("Education", "BA History, University of Elsewhere, 2005"),
                ("Acme Ltd -- Engineering Manager, 2021-2024", "Ran hiring."),
            )
        )
        assert skeleton.education == ("BA History, University of Elsewhere, 2005",)

    def test_the_hosted_title_is_not_a_name(self) -> None:
        skeleton = build_skeleton(_hosted(("Acme Ltd -- EM, 2021-2024", "Ran hiring.")))
        assert skeleton.corpus_title is None


def test_missing_education_is_empty_never_guessed() -> None:
    skeleton = build_skeleton(
        _spans("file:c.md", "# Corpus\n\n## Acme\n\n### Engineer, 2019 – 2020\n- Wrote code.\n")
    )
    assert skeleton.education == ()
    assert [(r.employer, r.title) for r in skeleton.roles] == [("Acme", "Engineer")]


def test_an_employer_named_like_a_section_keyword_stays_an_employer() -> None:
    corpus = (
        "# C\n\n## Pearson Education\n\n### Developer, 2010 – 2012\n- Built the LMS.\n\n"
        "## Acme Projects Ltd\n\n### Analyst, 2008 – 2010\n- Wrote reports.\n"
    )
    skeleton = build_skeleton(_spans("file:c.md", corpus))
    assert [r.employer for r in skeleton.roles] == ["Pearson Education", "Acme Projects Ltd"]
    assert skeleton.education == ()


def test_a_role_in_two_documents_is_listed_once_with_both_documents_facts() -> None:
    first = _spans("file:a.md", "# A\n\n## Acme\n\n### Engineer, 2019 – 2020\n- Wrote code.\n")
    second = _spans("file:b.md", "# B\n\n## Acme\n\n### Engineer, 2019 – 2020\n- Fixed bugs.\n")
    skeleton = build_skeleton(first + second)
    assert len(skeleton.roles) == 1
    assert set(skeleton.roles[0].facts) == {"Wrote code.", "Fixed bugs."}


def test_an_empty_corpus_is_an_empty_skeleton() -> None:
    assert build_skeleton([]) == build_skeleton([])
    assert build_skeleton([]).roles == ()


@pytest.mark.parametrize(
    ("heading", "before", "dates"),
    [
        ("Title, Nov 2024 – Present", "Title", "Nov 2024 – Present"),
        ("Title (2019-2021)", "Title", "2019-2021"),
        ("Title, 06/2019 to 03/2020", "Title", "06/2019 to 03/2020"),
        ("Title — Sept 2019 - now", "Title", "Sept 2019 - now"),
        ("Title, 2020", "Title", "2020"),
    ],
)
def test_trailing_dates_are_returned_exactly_as_written(
    heading: str, before: str, dates: str
) -> None:
    got_before, got_dates, key = split_trailing_dates(heading)
    assert (got_before, got_dates) == (before, dates)
    assert key is not None


@pytest.mark.parametrize("heading", ["Head of 2030 Programme", "Engineer", "Engineer, 13/2020"])
def test_a_heading_without_readable_trailing_dates_has_none(heading: str) -> None:
    assert split_trailing_dates(heading) == (heading, "", None)


def test_an_open_end_sorts_above_a_closed_one() -> None:
    assert split_trailing_dates("A, 2010 – Present")[2] > split_trailing_dates("B, 2019 – 2024")[2]  # type: ignore[operator]


def test_generated_lines_are_summary_then_skills_then_bullets_by_reference() -> None:
    summary, skill, bullet = CvLine(text="s"), CvLine(text="k"), CvLine(text="b")
    document = CvDocument(
        header=CvHeader(name="Morgan"),
        summary=[summary],
        skills=[CvSkill(label="L", text=skill)],
        roles=[CvRole(title="T", employer="E", dates="2020", bullets=[bullet])],
        education=[CvLine(text="BSc", origin="fact")],
    )
    lines = document.generated_lines()
    assert [line.text for line in lines] == ["s", "k", "b"]
    lines[0].verdict = "review"
    assert document.summary[0].verdict == "review"


def test_a_note_beneath_a_qualification_is_never_an_education_line() -> None:
    """The owner's record keeps caveats under a qualification -- that a 2011
    certificate "should not be represented as current or applied AI expertise".
    That is written for the tool, never for a reader, and it was once printed
    onto a CV verbatim. Only the qualification itself is a line."""
    corpus = (
        "# Test Person — Verification Record\n\n"
        "## Education\n\n"
        "### PGCert (Distinction), Intelligent Systems — Example University, 2011\n\n"
        "Named topics: neural networks, fuzzy logic. This is coursework from 2011 and "
        "should not be represented as current expertise on its own.\n\n"
        "- A bullet of further caveats about the same certificate.\n\n"
        "### BSc (Hons), Computing — Example Polytechnic, 2006\n"
    )
    skeleton = build_skeleton(parse_document("corpus/record.md", corpus, USER).spans)
    assert skeleton.education == (
        "PGCert (Distinction), Intelligent Systems — Example University, 2011",
        "BSc (Hons), Computing — Example Polytechnic, 2006",
    )
    assert not any("should not be represented" in line for line in skeleton.education)
