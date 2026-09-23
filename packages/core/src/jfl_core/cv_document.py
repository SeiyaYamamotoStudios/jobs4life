"""The complete CV as one document: header, summary, skills, roles, education.

Shared by three parts of the system -- generation (`jfl_generate.cv_document`),
PDF rendering and the edit/export screens -- so this module is the contract
between them and nothing else. Pure pydantic; no framework, no model, no
storage.

The rules the model carries, which every consumer relies on:

  * **Titles, employers, dates and education lines are never generated.** They
    are copied verbatim from confirmed facts (the user's corpus). A date the
    model wrote is a date that can drift, and a title it wrote is a claim to
    have held it.
  * **Header contact details and interests are profile settings, not claims**,
    and are never checked by the claim gate.
  * **Every `origin="generated"` line is checked by the claim gate.** Framing
    is shown as not checked, never as supported. A flagged line is still
    rendered -- the claim gate informs, it never blocks.
  * A line the user edits becomes `origin="user"`; it is re-checked if they
    ask, and the export renders exactly what they approved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CvTemplate = Literal["classic", "modern"]
LineOrigin = Literal["fact", "generated", "user"]  # where a line of text came from


class CvLink(BaseModel):
    label: str  # "linkedin.com/in/example"
    url: str  # "https://linkedin.com/in/example"


class CvHeader(BaseModel):
    name: str
    tagline: str = ""  # "Engineering Manager | Platform | Applied AI"
    contact: list[str] = Field(default_factory=list)  # ["07700 900000", "a@b.com", "Bristol, UK"]
    links: list[CvLink] = Field(default_factory=list)


class CvLine(BaseModel):
    """One sentence-bearing line. `origin` says who wrote it. `verdict` is the claim
    gate's verdict for a generated line (None until checked, and always None for a
    `fact` line, which is copied verbatim from a confirmed fact)."""

    text: str
    origin: LineOrigin = "generated"
    verdict: Literal["supported", "review", "unsupported", "framing"] | None = None
    note: str = ""  # the gate's evidence note, verbatim


class CvSkill(BaseModel):
    label: str  # "Platform & Distributed Systems"
    text: CvLine


class CvRole(BaseModel):
    title: str  # verbatim from facts
    employer: str  # verbatim from facts
    location: str = ""
    dates: str  # verbatim, e.g. "Nov 2024 – Present"
    descriptor: str = ""  # one line on what the company does
    bullets: list[CvLine] = Field(default_factory=list)


class CvDocument(BaseModel):
    template: CvTemplate = "modern"
    header: CvHeader
    summary: list[CvLine] = Field(default_factory=list)  # paragraphs
    skills_heading: str = "What I Bring"  # classic: "Skills & Expertise"
    skills: list[CvSkill] = Field(default_factory=list)
    experience_heading: str = "Professional Experience"
    roles: list[CvRole] = Field(default_factory=list)
    education_heading: str = "Education & Credentials"
    education: list[CvLine] = Field(default_factory=list)  # origin="fact", verbatim
    interests_heading: str = "Personal Interests"
    interests: list[str] = Field(default_factory=list)  # from the profile, the user's words

    def generated_lines(self) -> list[CvLine]:
        """Summary paragraphs, then skill texts, then every role's bullets, in
        document order -- the lines a claim-gate pass covers.

        Returned by reference, so a caller can set `verdict` and `note` on them in
        place. Filtered by nothing: a line the user has since edited
        (`origin="user"`) is still one of these positions, and whether to
        re-check it is the caller's decision, not this method's.
        """
        lines: list[CvLine] = list(self.summary)
        lines.extend(skill.text for skill in self.skills)
        for role in self.roles:
            lines.extend(role.bullets)
        return lines
