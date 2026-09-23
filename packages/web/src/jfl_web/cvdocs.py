"""Display helpers for the generated CV document: its check in plain words,
the export warning, the plain-text export and the download file name.

Pure -- no storage, no network, no model -- the same split `jfl_web.drafts`
draws. The four marks are `jfl_web.drafts`' own (`VERDICT_WORDS`): a line is
Supported, Check this, Not supported or Not checked, and nothing else. Framing
and any line never checked read as "Not checked", never as supported. A line the
user rewrote has had its verdict cleared (`jfl_core.cv_lines`) and says so:
"not checked since you edited it".

**The claim gate informs, it never blocks.** Nothing here decides whether a CV
may be downloaded; `export_warning` only says, plainly, how many lines in the
version being downloaded are not backed by the user's confirmed facts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from jfl_core.cv_document import CvDocument, CvLine
from jfl_core.cv_lines import line_paths

from jfl_web.drafts import (
    VERDICT_MEANINGS,
    VERDICT_WORDS,
    GenerationFailure,
    NextAction,
    VerdictKey,
    cost_range,
    headline,
    next_action,
)

TEMPLATE_LABELS: dict[str, str] = {"classic": "Classic", "modern": "Modern"}

# What a stored version's status reads as in the version list. Unknown values
# (the generation branch may add its own) are shown as they are.
STATUS_WORDS: dict[str, str] = {
    "generated": "written",
    "edited": "your edits",
    "checked": "your edits, checked",
    "template": "template changed",
    "header": "header updated from your profile",
}

# "Check my edits" is one claim-gate call over the edited lines. Its cost is
# mostly re-reading the user's confirmed facts, so it is much the same for one
# line as for ten. The low end is CLAUDE.md's measured ~$0.02 for a marginal
# check against an already-cached set of facts; the high end is an ESTIMATE
# (not measured) for a cold read of a 30-50k-token set. Replace with a measured
# figure once `runs` holds a few of these.
CHECK_EDITS_COST: tuple[Decimal, Decimal] = (Decimal("0.02"), Decimal("0.25"))


def check_edits_cost() -> str:
    return cost_range(*CHECK_EDITS_COST)


# `jfl_worker.handlers.cv_edits_check.FAILURE_MARKER`, and the closed set of
# codes that follow it. Parsed, never shown verbatim.
_CHECK_FAILURE_MARKER = "cv edit check failed permanently: "
_CHECK_FAILURES: dict[str, GenerationFailure] = {
    "no_api_key": GenerationFailure(
        "Checking needs your own Anthropic API key -- it is billed to you rather than to this app.",
        fix_url="/settings",
        fix_label="Add an API key",
    ),
    "api_key_rejected": GenerationFailure(
        "Anthropic rejected the API key stored here. Replace it and try again.",
        fix_url="/settings",
        fix_label="Replace the key",
    ),
    "credential_unreadable": GenerationFailure(
        "Your stored API key could not be unlocked on the server. Setting it again will fix it.",
        fix_url="/settings",
        fix_label="Set the key again",
    ),
    "model_refused": GenerationFailure(
        "The model declined to respond. Trying again is worth a go."
    ),
}


def check_failure(last_error: str | None) -> GenerationFailure:
    """A failed "Check my edits" task, in plain words."""
    if last_error and _CHECK_FAILURE_MARKER in last_error:
        code = last_error.rsplit(_CHECK_FAILURE_MARKER, 1)[-1].strip()
        if code in _CHECK_FAILURES:
            return _CHECK_FAILURES[code]
    return GenerationFailure("The last check of your edits failed. Trying again is worth a go.")


def line_key(line: CvLine) -> VerdictKey:
    """Which of the four marks a line gets. Only a real verdict earns one of
    the three checked marks; framing and "never checked" are Not checked."""
    if line.verdict == "supported":
        return "supported"
    if line.verdict == "review":
        return "review"
    if line.verdict == "unsupported":
        return "unsupported"
    return "not_checked"


def line_meaning(line: CvLine) -> str:
    if line.verdict is None and line.origin == "user":
        return "Not checked since you edited it."
    if line.verdict is None:
        return "Not checked yet."
    return VERDICT_MEANINGS[line_key(line)]


@dataclass(frozen=True, slots=True)
class LineView:
    path: str
    where: str
    line: CvLine
    key: VerdictKey
    label: str
    style: str
    meaning: str
    action: NextAction | None
    edited: bool


def _where(doc: CvDocument, path: str) -> str:
    parts = path.split(".")
    if parts[0] == "summary":
        return "Summary"
    if parts[0] == "skills":
        return doc.skills[int(parts[1])].label
    role = doc.roles[int(parts[1])]
    return f"{role.title}, {role.employer}"


def line_view(doc: CvDocument, path: str, line: CvLine) -> LineView:
    key = line_key(line)
    flagged = key in ("review", "unsupported")
    action = next_action({"kind": "claim", "verdict": key}) if flagged else None
    return LineView(
        path=path,
        where=_where(doc, path),
        line=line,
        key=key,
        label=VERDICT_WORDS[key],
        style="verdict-" + key.replace("_", "-"),
        meaning=line_meaning(line),
        action=action,
        edited=line.origin == "user",
    )


@dataclass(frozen=True, slots=True)
class CvCheck:
    """One version's lines, counted and grouped for the page."""

    supported: int
    review: int
    unsupported: int
    headline: str
    flagged: tuple[LineView, ...]
    backed: tuple[LineView, ...]
    unchecked: tuple[LineView, ...]
    edited_unchecked: tuple[LineView, ...]

    @property
    def claims(self) -> int:
        return self.supported + self.review + self.unsupported


def cv_check(doc: CvDocument) -> CvCheck:
    views = [line_view(doc, path, line) for path, line in line_paths(doc)]
    by = {k: [v for v in views if v.key == k] for k in ("supported", "review", "unsupported")}
    unchecked = [v for v in views if v.key == "not_checked"]
    return CvCheck(
        supported=len(by["supported"]),
        review=len(by["review"]),
        unsupported=len(by["unsupported"]),
        headline=headline(len(by["supported"]), len(by["review"]), len(by["unsupported"])),
        # Worst first, each in reading order.
        flagged=tuple(by["unsupported"] + by["review"]),
        backed=tuple(by["supported"]),
        unchecked=tuple(v for v in unchecked if not (v.edited and v.line.verdict is None)),
        edited_unchecked=tuple(v for v in unchecked if v.edited and v.line.verdict is None),
    )


def _lines(n: int) -> str:
    return "1 line" if n == 1 else f"{n} lines"


def export_warning(doc: CvDocument) -> str:
    """What the download says about the version being downloaded, or "".

    "2 lines aren't backed by your confirmed facts." -- plus, where it is true,
    how many lines the user rewrote and has not had checked. It never stops the
    download: what is sent is the user's call.
    """
    check = cv_check(doc)
    flagged = len(check.flagged)
    edited = len(check.edited_unchecked)
    parts: list[str] = []
    if flagged:
        verb = "isn't" if flagged == 1 else "aren't"
        parts.append(f"{_lines(flagged)} {verb} backed by your confirmed facts")
    if edited:
        verb = "hasn't" if edited == 1 else "haven't"
        parts.append(f"{_lines(edited)} you edited {verb} been checked")
    if not parts:
        return ""
    sentence = " and ".join(parts)
    return sentence[:1].upper() + sentence[1:] + "."


def cv_plain_text(doc: CvDocument) -> str:
    """The CV as plain text -- exactly the words of the version, no marks, no
    notes. What "Download as text" serves and what "Copy" copies."""
    out: list[str] = [doc.header.name]
    if doc.header.tagline:
        out.append(doc.header.tagline)
    contact = [*doc.header.contact, *(f"{link.label} ({link.url})" for link in doc.header.links)]
    if contact:
        out.append(" | ".join(contact))
    if doc.summary:
        out.append("")
        out.extend(line.text for line in doc.summary)
    if doc.skills:
        out += ["", doc.skills_heading.upper()]
        out.extend(f"{skill.label}: {skill.text.text}" for skill in doc.skills)
    if doc.roles:
        out += ["", doc.experience_heading.upper()]
        for role in doc.roles:
            out.append("")
            out.append(f"{role.title} -- {role.employer}")
            out.append(" | ".join(x for x in (role.location, role.dates) if x))
            if role.descriptor:
                out.append(role.descriptor)
            out.extend(f"- {line.text}" for line in role.bullets)
    if doc.education:
        out += ["", doc.education_heading.upper()]
        out.extend(line.text for line in doc.education)
    if doc.interests:
        out += ["", doc.interests_heading.upper(), ", ".join(doc.interests)]
    return "\n".join(out).strip() + "\n"


def _filename_part(value: str) -> str:
    ascii_only = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_")[:60]


def cv_filename(name: str, employer: str | None, extension: str) -> str:
    """`Jane_Doe_CV_Acme.pdf` -- ASCII letters, digits and underscores only, so
    it is safe in a header and on every file system."""
    parts = [p for p in (_filename_part(name), "CV", _filename_part(employer or "")) if p]
    return "_".join(parts) + f".{extension}"


__all__ = [
    "CHECK_EDITS_COST",
    "STATUS_WORDS",
    "TEMPLATE_LABELS",
    "CvCheck",
    "LineView",
    "check_edits_cost",
    "check_failure",
    "cv_check",
    "cv_filename",
    "cv_plain_text",
    "export_warning",
    "line_key",
    "line_meaning",
    "line_view",
]
