"""PLACEHOLDER (cvedit branch) -- the shared CV document model.

The generation branch owns the real module at this path; at merge its version
wins and this file is dropped. The public surface below is copied exactly from
the shared interface note so the editing/export screens can be built against it.

Rules every consumer relies on:
- Titles, employers, dates and education lines are never generated -- copied
  verbatim from confirmed facts.
- Header contact details and interests are profile settings, not claims, and
  are never checked by the claim gate.
- Every `origin="generated"` line is checked by the claim gate. Framing is shown
  as not checked, never as supported. A flagged line is still rendered.
- A line the user edits becomes `origin="user"`; it is re-checked if they ask,
  and the export renders exactly what they approved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CvTemplate = Literal["classic", "modern"]
LineOrigin = Literal["fact", "generated", "user"]


class CvLink(BaseModel):
    label: str
    url: str


class CvHeader(BaseModel):
    name: str
    tagline: str = ""
    contact: list[str] = Field(default_factory=list)
    links: list[CvLink] = Field(default_factory=list)


class CvLine(BaseModel):
    """One sentence-bearing line. `origin` says who wrote it. `verdict` is the
    claim gate's verdict for a generated line (None until checked, and always
    None for a `fact` line)."""

    text: str
    origin: LineOrigin = "generated"
    verdict: Literal["supported", "review", "unsupported", "framing"] | None = None
    note: str = ""


class CvSkill(BaseModel):
    label: str
    text: CvLine


class CvRole(BaseModel):
    title: str
    employer: str
    location: str = ""
    dates: str
    descriptor: str = ""
    bullets: list[CvLine] = Field(default_factory=list)


class CvDocument(BaseModel):
    template: CvTemplate = "modern"
    header: CvHeader
    summary: list[CvLine] = Field(default_factory=list)
    skills_heading: str = "What I Bring"
    skills: list[CvSkill] = Field(default_factory=list)
    experience_heading: str = "Professional Experience"
    roles: list[CvRole] = Field(default_factory=list)
    education_heading: str = "Education & Credentials"
    education: list[CvLine] = Field(default_factory=list)
    interests_heading: str = "Personal Interests"
    interests: list[str] = Field(default_factory=list)

    def generated_lines(self) -> list[CvLine]:
        """Summary + skill texts + bullets, in order."""
        lines = list(self.summary)
        lines.extend(skill.text for skill in self.skills)
        for role in self.roles:
            lines.extend(role.bullets)
        return lines
