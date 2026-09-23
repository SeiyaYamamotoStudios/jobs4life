"""Addressing the lines of a `CvDocument`, and editing the ones a user may edit.

Shared by the web screens (review, edit, export) and the worker's "check my
edits" handler, which is why it is in `jfl_core` and pure: no storage, no
network, no model.

**A path names one field**, e.g. `summary.0`, `skills.2.text`,
`roles.1.bullets.3`, `roles.1.descriptor`, `skills_heading`. The edit form
posts each editable field under its path, and the check task names the lines
it checked by path.

**Two kinds of field, and the line between them is the point.**

* *Editable* -- every line the model wrote (summary, skill texts, bullets), the
  descriptors, the skill labels and the section headings. Wording.
* *Facts* -- titles, employers, locations, dates and education lines, copied
  verbatim from what the user confirmed, plus the header and interests, which
  come from the profile. **Never editable here.** A date typed into a CV
  screen is a date that drifts from the record; changing a fact is done by
  changing the fact. `apply_edits` refuses a submission that names any field
  outside the editable set -- a crafted POST changing a title is an error,
  and nothing is stored.

An edited line becomes `origin="user"` with its verdict cleared: the check it
had was of different words. A line whose submitted text is unchanged keeps its
origin and its verdict untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from jfl_core.cv_document import CvDocument, CvLine, CvRole, CvSkill

# One line of a CV, or one heading, never needs more than this. A bound on a
# row, not on what anyone may say -- rejected, never truncated.
MAX_FIELD_CHARS = 1500

HEADING_FIELDS: tuple[str, ...] = (
    "skills_heading",
    "experience_heading",
    "education_heading",
    "interests_heading",
)

FieldKind = Literal["line", "text", "heading"]


@dataclass(frozen=True, slots=True)
class EditableField:
    path: str
    kind: FieldKind  # "line" carries a verdict; "text" and "heading" do not
    value: str
    line: CvLine | None = None


class ProtectedFieldError(ValueError):
    """A submission tried to change a fact (title, employer, dates, education,
    the header) or named a field that does not exist. Nothing is stored."""


class InvalidEditError(ValueError):
    """An editable field came back in a shape that cannot be stored -- too long,
    or a heading or skill label emptied."""


def line_paths(doc: CvDocument) -> list[tuple[str, CvLine]]:
    """Every checkable line (the generated or user-written ones), in reading
    order -- summary, skills, then each role's bullets. The same order as
    `CvDocument.generated_lines`, with each line's path."""
    out: list[tuple[str, CvLine]] = [(f"summary.{i}", line) for i, line in enumerate(doc.summary)]
    out.extend((f"skills.{i}.text", skill.text) for i, skill in enumerate(doc.skills))
    for r, role in enumerate(doc.roles):
        out.extend((f"roles.{r}.bullets.{b}", line) for b, line in enumerate(role.bullets))
    return out


def editable_fields(doc: CvDocument) -> list[EditableField]:
    """Everything the edit form offers, in reading order."""
    fields: list[EditableField] = [
        EditableField(f"summary.{i}", "line", line.text, line) for i, line in enumerate(doc.summary)
    ]
    fields.append(EditableField("skills_heading", "heading", doc.skills_heading))
    for i, skill in enumerate(doc.skills):
        fields.append(EditableField(f"skills.{i}.label", "text", skill.label))
        fields.append(EditableField(f"skills.{i}.text", "line", skill.text.text, skill.text))
    fields.append(EditableField("experience_heading", "heading", doc.experience_heading))
    for r, role in enumerate(doc.roles):
        fields.append(EditableField(f"roles.{r}.descriptor", "text", role.descriptor))
        for b, line in enumerate(role.bullets):
            fields.append(EditableField(f"roles.{r}.bullets.{b}", "line", line.text, line))
    fields.append(EditableField("education_heading", "heading", doc.education_heading))
    fields.append(EditableField("interests_heading", "heading", doc.interests_heading))
    return fields


def line_at(doc: CvDocument, path: str) -> CvLine | None:
    return dict(line_paths(doc)).get(path)


def unchecked_edits(doc: CvDocument) -> list[str]:
    """Paths of lines the user wrote that have not been checked since."""
    return [
        path for path, line in line_paths(doc) if line.origin == "user" and line.verdict is None
    ]


def _clean(value: str) -> str:
    """One line of a CV holds no line breaks; whitespace is tidied, nothing
    else is changed."""
    return " ".join(value.split())


def _edited(line: CvLine, submitted: str | None) -> CvLine | None:
    """The line after an edit: unchanged (same object), rewritten by the user
    (origin "user", verdict cleared), or None -- emptied, so removed."""
    if submitted is None:
        return line
    text = _clean(submitted)
    if text == _clean(line.text):
        return line
    if not text:
        return None
    return CvLine(text=text, origin="user", verdict=None, note="")


@dataclass(frozen=True, slots=True)
class EditResult:
    doc: CvDocument
    edited: int
    removed: int

    @property
    def changed(self) -> bool:
        return bool(self.edited or self.removed)


def apply_edits(doc: CvDocument, submitted: Mapping[str, str]) -> EditResult:
    """A new document with the submitted wording applied. Pure.

    `submitted` maps path -> text. A path left out keeps its text. An emptied
    line is removed (cutting is one of the two things a flagged line's next
    step suggests). Raises `ProtectedFieldError` if any key is not an editable
    path of `doc` -- including every fact field -- and `InvalidEditError` for
    an over-long value or an emptied heading or skill label.
    """
    allowed = {field.path for field in editable_fields(doc)}
    for key, value in submitted.items():
        if key not in allowed:
            raise ProtectedFieldError(key)
        if len(value) > MAX_FIELD_CHARS:
            raise InvalidEditError(f"One line is longer than {MAX_FIELD_CHARS} characters.")

    edited = removed = 0

    def edit_line(path: str, line: CvLine) -> CvLine | None:
        nonlocal edited, removed
        after = _edited(line, submitted.get(path))
        if after is None:
            removed += 1
        elif after is not line:
            edited += 1
        return after

    def edit_text(path: str, current: str, *, required: str | None = None) -> str:
        nonlocal edited
        if path not in submitted:
            return current
        text = _clean(submitted[path])
        if not text and required:
            raise InvalidEditError(required)
        if text == _clean(current):
            return current
        edited += 1
        return text

    summary = [
        kept
        for i, line in enumerate(doc.summary)
        if (kept := edit_line(f"summary.{i}", line)) is not None
    ]
    skills: list[CvSkill] = []
    for i, skill in enumerate(doc.skills):
        text = edit_line(f"skills.{i}.text", skill.text)
        if text is None:
            continue  # a skill with its line cut goes, label and all
        label = edit_text(f"skills.{i}.label", skill.label, required="A skill needs a label.")
        skills.append(CvSkill(label=label, text=text))
    roles: list[CvRole] = []
    for r, role in enumerate(doc.roles):
        bullets = [
            kept
            for b, line in enumerate(role.bullets)
            if (kept := edit_line(f"roles.{r}.bullets.{b}", line)) is not None
        ]
        descriptor = edit_text(f"roles.{r}.descriptor", role.descriptor)
        # Title, employer, location and dates are carried over untouched.
        roles.append(role.model_copy(update={"bullets": bullets, "descriptor": descriptor}))
    headings = {
        name: edit_text(name, getattr(doc, name), required="A section heading can't be empty.")
        for name in HEADING_FIELDS
    }
    new_doc = doc.model_copy(
        update={"summary": summary, "skills": skills, "roles": roles, **headings}
    )
    return EditResult(doc=new_doc, edited=edited, removed=removed)


# Worst first. `framing` is never compared with the facts and so ranks with
# "not checked", below every real verdict.
_RANK = {"unsupported": 3, "review": 2, "supported": 1, "framing": 0}


def with_verdicts(
    doc: CvDocument,
    verdicts: Mapping[str, tuple[str, str, str]],
) -> CvDocument:
    """A copy of `doc` with verdicts written onto the lines at the given paths.

    `verdicts` maps path -> (expected text, verdict, note). A line whose text
    no longer matches what was checked is left alone: the user has edited it
    again since, and a verdict on different words would be a lie.
    """

    def apply(path: str, line: CvLine) -> CvLine:
        found = verdicts.get(path)
        if found is None:
            return line
        text, verdict, note = found
        if line.text != text or verdict not in _RANK:
            return line
        return line.model_copy(update={"verdict": verdict, "note": note})

    return doc.model_copy(
        update={
            "summary": [apply(f"summary.{i}", line) for i, line in enumerate(doc.summary)],
            "skills": [
                skill.model_copy(update={"text": apply(f"skills.{i}.text", skill.text)})
                for i, skill in enumerate(doc.skills)
            ],
            "roles": [
                role.model_copy(
                    update={
                        "bullets": [
                            apply(f"roles.{r}.bullets.{b}", line)
                            for b, line in enumerate(role.bullets)
                        ]
                    }
                )
                for r, role in enumerate(doc.roles)
            ],
        }
    )


def worst_verdict(verdicts: list[str]) -> str | None:
    """One line's verdict from its sentences' verdicts: the worst of them."""
    known = [v for v in verdicts if v in _RANK]
    return max(known, key=lambda v: _RANK[v]) if known else None


__all__ = [
    "HEADING_FIELDS",
    "MAX_FIELD_CHARS",
    "EditResult",
    "EditableField",
    "InvalidEditError",
    "ProtectedFieldError",
    "apply_edits",
    "editable_fields",
    "line_at",
    "line_paths",
    "unchecked_edits",
    "with_verdicts",
    "worst_verdict",
]
