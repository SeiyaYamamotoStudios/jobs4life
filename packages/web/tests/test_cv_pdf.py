"""The CV renderer: both templates, real text, nothing from the claim gate.

The fixture is a fictional person. No database, no network -- and the renderer
fetches nothing, which the socket guard in the root conftest would catch.
"""

from __future__ import annotations

import io
import re
from typing import Any

import pytest
from jfl_core.cv_document import (
    CvDocument,
    CvHeader,
    CvLine,
    CvLink,
    CvRole,
    CvSkill,
)
from jfl_web.cv_pdf import render_cv_html, render_cv_pdf
from pypdf import PdfReader

VERDICT_NOTE = "GATE-NOTE corpus is silent on the headcount of this team"


def _gen(text: str, verdict: str | None = "supported") -> CvLine:
    return CvLine(text=text, origin="generated", verdict=verdict, note=VERDICT_NOTE)  # type: ignore[arg-type]


def fictional_cv(template: str = "modern", *, roles: int = 3) -> CvDocument:
    """Marguerite Okonkwo-Lindqvist is not a real person."""
    employers = [
        (
            "Head of Platform Engineering",
            "Fernwhistle Logistics",
            "Leeds, UK",
            "Mar 2022 – Present",
            "Freight-matching marketplace for regional hauliers",
        ),
        (
            "Engineering Manager, Payments",
            "Brackenfold Bank",
            "Remote",
            "Jan 2018 – Feb 2022",
            "Challenger bank serving small businesses",
        ),
        (
            "Senior Software Engineer",
            "Quillmere Analytics",
            "Sheffield, UK",
            "Jun 2013 – Dec 2017",
            "Retail footfall analytics",
        ),
    ]
    role_list = []
    for i in range(roles):
        title, employer, location, dates, descriptor = employers[i % len(employers)]
        role_list.append(
            CvRole(
                title=title if i < len(employers) else f"{title} ({i})",
                employer=employer,
                location=location,
                dates=dates,
                descriptor=descriptor,
                bullets=[
                    _gen(
                        f"Led a team of seven through a migration to event-driven "
                        f"dispatch, cutting nightly batch failures to zero (role {i})."
                    ),
                    _gen(
                        f"Introduced blameless incident review and a weekly "
                        f"reliability forum adopted by three sister teams (role {i}).",
                        "review",
                    ),
                    _gen(
                        f"Hired and developed four engineers, two since promoted to "
                        f"senior (role {i}).",
                        "unsupported",
                    ),
                    _gen(
                        f"Chose to prioritise observability before new features (role {i}).",
                        "framing",
                    ),
                ],
            )
        )
    doc = CvDocument(
        template=template,  # type: ignore[arg-type]
        header=CvHeader(
            name="Marguerite Okonkwo-Lindqvist",
            tagline="Engineering Manager | Platform & Reliability | Applied AI",
            contact=["07700 900123", "m.okonkwo@example.com", "Leeds, UK"],
            links=[
                CvLink(
                    label="linkedin.com/in/example-mol", url="https://linkedin.com/in/example-mol"
                )
            ],
        ),
        summary=[
            _gen(
                "Engineering manager with fifteen years building logistics and "
                "payments platforms, most recently leading platform engineering."
            ),
            _gen("Known for turning fragile batch systems into observable services.", "review"),
        ],
        skills=[
            CvSkill(
                label="Platform & Distributed Systems",
                text=_gen("Event-driven architecture, Kafka, Postgres at scale."),
            ),
            CvSkill(
                label="Leadership",
                text=_gen("Hiring, coaching, and running teams of up to twelve."),
            ),
        ],
        roles=role_list,
        education=[
            CvLine(text="BSc Computer Science, University of Examplefield, 2012", origin="fact"),
            CvLine(text="Applied Machine Learning certificate, 2024", origin="fact"),
        ],
        interests=["Fell running and allotment gardening."],
    )
    if template == "classic":
        doc.skills_heading = "Skills & Expertise"
    return doc


def _pdf_text(pdf: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(pdf))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return re.sub(r"\s+", " ", text), len(reader.pages)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


@pytest.mark.parametrize("template", ["classic", "modern"])
def test_pdf_carries_every_line_of_the_document(template: str) -> None:
    doc = fictional_cv(template)
    pdf = render_cv_pdf(doc)
    assert pdf.startswith(b"%PDF")
    text, _ = _pdf_text(pdf)

    assert doc.header.name in text
    for role in doc.roles:
        assert role.title in text
        assert role.dates in text
        assert role.employer in text
        for bullet in role.bullets:
            assert _norm(bullet.text) in text
    for line in doc.summary + doc.education:
        assert _norm(line.text) in text
    for skill in doc.skills:
        assert skill.label in text


@pytest.mark.parametrize("template", ["classic", "modern"])
def test_nothing_from_the_claim_gate_reaches_the_pdf(template: str) -> None:
    pdf = render_cv_pdf(fictional_cv(template))
    text, _ = _pdf_text(pdf)
    assert "GATE-NOTE" not in text
    for word in ("supported", "unsupported", "framing", "Check this", "Not checked"):
        assert word.lower() not in text.lower()
    assert b"GATE-NOTE" not in pdf


@pytest.mark.parametrize("template", ["classic", "modern"])
def test_headings_are_the_documents_own_strings(template: str) -> None:
    doc = fictional_cv(template)
    doc.experience_heading = "Where I Have Worked"
    doc.education_heading = "Study"
    html = render_cv_html(doc)
    for heading in (doc.skills_heading, "Where I Have Worked", "Study", doc.interests_heading):
        assert f"<h2>{heading.replace('&', '&amp;')}</h2>" in html
    assert "Professional Experience" not in html
    text, _ = _pdf_text(render_cv_pdf(doc))
    # modern uppercases by CSS; the PDF text layer carries the transformed form.
    heading = "Where I Have Worked"
    assert (heading.upper() if template == "modern" else heading) in text


def test_pdf_metadata_title_names_the_person() -> None:
    reader = PdfReader(io.BytesIO(render_cv_pdf(fictional_cv())))
    assert reader.metadata is not None
    assert reader.metadata.title == "Marguerite Okonkwo-Lindqvist — CV"


def test_fonts_are_embedded() -> None:
    reader = PdfReader(io.BytesIO(render_cv_pdf(fictional_cv("classic"))))
    embedded = False
    for page in reader.pages:
        resources: Any = page["/Resources"]
        fonts: Any = resources["/Font"]
        for ref in fonts.values():
            font = ref.get_object()
            descendants = font.get("/DescendantFonts")
            descriptor = (
                descendants[0].get_object()["/FontDescriptor"]
                if descendants
                else font.get("/FontDescriptor")
            )
            if descriptor is None:
                continue
            d = descriptor.get_object()
            assert any(k in d for k in ("/FontFile", "/FontFile2", "/FontFile3")), d
            embedded = True
    assert embedded


@pytest.mark.parametrize("template", ["classic", "modern"])
def test_a_long_document_runs_to_several_pages(template: str) -> None:
    doc = fictional_cv(template, roles=12)
    text, pages = _pdf_text(render_cv_pdf(doc))
    assert pages > 1
    assert doc.roles[-1].title in text


@pytest.mark.parametrize("template", ["classic", "modern"])
def test_html_preview_carries_the_same_text(template: str) -> None:
    doc = fictional_cv(template)
    html = render_cv_html(doc)
    assert "GATE-NOTE" not in html
    assert "<title>Marguerite Okonkwo-Lindqvist — CV</title>" in html
    assert f'class="cv cv-{template}"' in html
    for role in doc.roles:
        assert role.title in html
        assert role.dates in html
        for bullet in role.bullets:
            assert bullet.text in html
    for line in doc.summary + doc.education:
        assert line.text in html
    assert 'href="https://linkedin.com/in/example-mol"' in html


def test_text_is_escaped_not_interpreted() -> None:
    doc = fictional_cv()
    doc.summary = [CvLine(text="Shipped <script>alert(1)</script> & more")]
    html = render_cv_html(doc)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_an_unclickable_link_scheme_becomes_plain_text() -> None:
    doc = fictional_cv()
    doc.header.links = [CvLink(label="my site", url="javascript:alert(1)")]
    html = render_cv_html(doc)
    assert "javascript:" not in html
    assert "my site" in html


def test_rendering_does_not_mutate_the_callers_document() -> None:
    doc = fictional_cv()
    render_cv_html(doc)
    assert doc.roles[0].bullets[0].note == VERDICT_NOTE
    assert doc.roles[0].bullets[0].verdict == "supported"
