"""The shared CV document model.

PLACEHOLDER: the generation slice owns the real module; this copy carries the
agreed public surface only, so the PDF renderer can be built against it.
"""

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
    """One sentence-bearing line. `origin` says who wrote it. `verdict` is the claim
    gate's verdict for a generated line (None until checked, and always None for a
    `fact` line, which is copied verbatim from a confirmed fact)."""

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
        lines = list(self.summary)
        lines.extend(skill.text for skill in self.skills)
        for role in self.roles:
            lines.extend(role.bullets)
        return lines
