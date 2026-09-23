"""The fixed half of a CV, read deterministically out of the corpus: which roles,
in what order, with what titles, employers and dates -- and the education lines.

**No model is anywhere in this module.** A title, an employer or a date on a CV
is a claim to have held that job at that time; if a model wrote it, it could
drift, and the claim gate would then be checking a line the tool itself made
up. So the skeleton is copied out of confirmed facts, verbatim, and the one
model call that writes a CV (`jfl_generate.cv_document`) is handed it by index
and never asked to restate any of it.

Two corpus shapes, both read from `Span.section_path` (the heading breadcrumb
`jfl_core.ingest.parser` builds, " > "-joined, the lone-h1 document title left
out):

  * **Owner-authored markdown**: `## Employer` with `### Title, Mon YYYY – Mon
    YYYY` beneath. A heading whose text ends in a date range and sits under a
    parent heading is a role; the parent is the employer. An undated heading
    beside dated siblings is a role too -- its dates are simply not known.
  * **CV-onboarded (hosted) corpus**: confirmed facts filed under their role
    label (`jfl_core.corpus_source`), one top-level section per role, written
    the way the CV-facts prompt asks: "Acme Ltd -- Engineering Manager,
    2021-2024". The label is split into employer and title at its first
    separator, and its trailing dates read off the end.

**Dates are parsed only to sort.** What is stored is the text as written
("Nov 2024 – Present"), never a normalised form. A role whose dates cannot be
parsed keeps its text and sorts last -- a date is never invented to place it.

**A section the corpus lacks is empty**, never guessed: no education recorded
means no education lines. And a section of **boundaries** ("things stated as
NOT true") is never rendered anywhere on a CV; its lines are returned
separately so the generation call can be told, explicitly, what not to claim.

Sections that are corpus but not CV structure -- the profile's self-assessment
("Depth and exposure", "Recurring gaps"), unfiled confirmed facts, independent
or non-employment activity -- are not roles, even where a heading beneath them
carries dates. Their text still reaches the generation call through the whole
corpus; it just never becomes a job on the CV.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from jfl_core.corpus_source import TITLE as HOSTED_TITLE
from jfl_core.ids import normalise
from jfl_core.models import Span

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_WORD = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_ONE_DATE = rf"(?:{_MONTH_WORD}\s+|\d{{1,2}}/)?(?:19|20)\d{{2}}"
_OPEN_END = r"(?:present|current|now|today|ongoing|date)"
_RANGE_SEPARATOR = r"\s*(?:–|—|--|-|to|until)\s*"
# The date range at the END of a heading -- optionally in brackets, optionally
# set off by a comma, bar or dash. Anchored at the end so a year inside a title
# ("Head of 2030 Programme") is never read as the role's dates.
_TRAILING_DATES = re.compile(
    rf"(?:^|(?<=[\s,(|·—–-]))\(?(?P<dates>(?P<start>{_ONE_DATE})"
    rf"(?:{_RANGE_SEPARATOR}(?P<end>{_ONE_DATE}|{_OPEN_END}))?)\)?\s*$",
    re.IGNORECASE,
)
_PARSE_ONE = re.compile(
    rf"^(?:(?P<month>{_MONTH_WORD})\s+|(?P<num>\d{{1,2}})/)?(?P<year>(?:19|20)\d{{2}})$",
    re.IGNORECASE,
)
_OPEN = re.compile(rf"^{_OPEN_END}$", re.IGNORECASE)

# Between a role label's employer and title, in the order tried. " at " reads
# the other way round ("Engineering Manager at Acme").
_LABEL_SEPARATORS = (" — ", " – ", " -- ", " - ", " | ", " at ", ", ")
# Between a title and a location written on the same heading.
_LOCATION_SEPARATORS = (" | ", " · ")

_BOUNDARY_SECTION = re.compile(
    r"\bnot true\b|^boundaries\b|\bboundaries to hold\b|\bnever claim|\bdo not claim"
    r"|\bstated as not\b",
    re.IGNORECASE,
)
# Anchored at the start, so an employer called "Pearson Education" stays an
# employer.
_EDUCATION_SECTION = re.compile(
    r"^(?:education|qualifications?|certifications?|credentials|training)\b",
    re.IGNORECASE,
)
# A top-level hosted label that names a qualification rather than a job --
# CV-facts files education under the CV's own heading, but a CV that wrote each
# degree as its own entry produces labels like "University of X -- BSc Physics,
# 2001-2004". Degree words only: "University" or "College" is as often an
# employer.
_EDUCATION_LABEL = re.compile(
    r"\b(?:b\.?sc|m\.?sc|ph\.?d|mba|b\.?eng|m\.?eng|degree|diploma|certificat\w*"
    r"|a-levels?|gcses?)\b",
    re.IGNORECASE,
)
# Whole-heading matches, so "Acme Projects Ltd" is still an employer.
_NOT_A_ROLE_SECTION = re.compile(
    r"^(?:independent\b.*|.*non-employment.*|depth and exposure|recurring gaps|confirmed facts"
    r"|summary|profile|(?:key |core )?skills.*|(?:personal )?interests|referees|references"
    r"|contact.*|(?:side |personal )?projects)$",
    re.IGNORECASE,
)
_GENERIC_PARENT = re.compile(
    r"^(?:(?:professional\s+|work\s+)?experience|employment(?:\s+history)?|career(?:\s+history)?"
    r"|work\s+history|roles|positions)$",
    re.IGNORECASE,
)
_LOCATION_LINE = re.compile(r"^location\s*:\s*(?P<where>\S.*)$", re.IGNORECASE)

_PATH_SEPARATOR = " > "


@dataclass(frozen=True, slots=True)
class SkeletonRole:
    """One role, every field copied from the corpus as written."""

    title: str
    employer: str
    dates: str  # "" when the corpus states none that can be read
    location: str = ""
    facts: tuple[str, ...] = ()  # the role's confirmed facts, verbatim, in corpus order
    fact_span_ids: tuple[uuid.UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class CvSkeleton:
    roles: tuple[SkeletonRole, ...] = ()  # most recent first; undated last
    education: tuple[str, ...] = ()  # verbatim, in corpus order
    boundaries: tuple[str, ...] = ()  # never rendered; a constraint for the model
    corpus_title: str | None = None  # the first corpus document's own title, if any


@dataclass
class _RoleDraft:
    path: tuple[str, ...]
    title: str
    employer: str
    dates: str
    sort_key: tuple[int, int, int, int] | None
    order: int
    location: str = ""
    facts: list[str] = field(default_factory=list)
    fact_span_ids: list[uuid.UUID] = field(default_factory=list)


def _date_value(text: str, *, is_end: bool) -> tuple[int, int] | None:
    if _OPEN.match(text.strip()):
        return (9999, 12) if is_end else None
    match = _PARSE_ONE.match(text.strip())
    if match is None:
        return None
    year = int(match.group("year"))
    if match.group("month"):
        month = _MONTHS[match.group("month").lower()[:3]]
    elif match.group("num"):
        month = int(match.group("num"))
        if not 1 <= month <= 12:
            return None
    else:
        # A bare year sorts as the end of that year when it ends a role and as
        # its start when it starts one -- ordering only, never shown.
        month = 12 if is_end else 1
    return (year, month)


def split_trailing_dates(text: str) -> tuple[str, str, tuple[int, int, int, int] | None]:
    """(text before the dates, the dates as written, a sort key) -- or
    (text, "", None) when the heading ends in no readable date.

    The sort key is (end year, end month, start year, start month); an open
    end ("Present") sorts above every closed one. Only for ordering: the dates
    string returned is the heading's own characters, untouched.
    """
    match = _TRAILING_DATES.search(text)
    if match is None:
        return text, "", None
    start = _date_value(match.group("start"), is_end=False)
    if start is None:
        return text, "", None
    raw_end = match.group("end")
    end = _date_value(raw_end, is_end=True) if raw_end else start
    if end is None:
        return text, "", None
    before = text[: match.start()].rstrip(" ,|·—–-(\t")
    return before, match.group("dates"), (end[0], end[1], start[0], start[1])


def _split_label(label: str) -> tuple[str, str]:
    """(employer, title) from a hosted role label, already stripped of dates."""
    for separator in _LABEL_SEPARATORS:
        if separator in label:
            left, right = label.split(separator, 1)
            left, right = left.strip(), right.strip()
            if not left or not right:
                continue
            if separator == " at ":
                return right, left
            return left, right
    return "", label.strip()


def _split_location(title: str) -> tuple[str, str]:
    for separator in _LOCATION_SEPARATORS:
        if separator in title:
            head, rest = title.split(separator, 1)
            return head.strip(), rest.strip()
    return title, ""


def _components(span: Span) -> tuple[str, ...]:
    return tuple(span.section_path.split(_PATH_SEPARATOR)) if span.section_path else ()


def _is_title_heading(span: Span) -> bool:
    return span.kind == "heading" and not span.section_path


def _by_document(spans: Sequence[Span]) -> list[list[Span]]:
    documents: dict[uuid.UUID, list[Span]] = {}
    for span in spans:
        if span.document_id is None:
            continue  # an adjudicated span has no place in any document's structure
        documents.setdefault(span.document_id, []).append(span)
    return [sorted(group, key=lambda s: s.ordinal or 0) for group in documents.values()]


def _read_document(
    spans: Sequence[Span], order_start: int
) -> tuple[list[_RoleDraft], list[str], list[str], str | None]:
    title: str | None = None
    headings: list[tuple[str, ...]] = []
    for span in spans:
        if _is_title_heading(span):
            title = title or span.text
        elif span.kind == "heading":
            headings.append(_components(span))

    def special(path: tuple[str, ...], pattern: re.Pattern[str]) -> int | None:
        return next((i for i, part in enumerate(path) if pattern.search(part)), None)

    # Which heading paths are roles.
    dated_children: dict[tuple[str, ...], int] = {}
    for path in headings:
        if len(path) >= 2 and split_trailing_dates(path[-1])[2] is not None:
            dated_children[path[:-1]] = dated_children.get(path[:-1], 0) + 1
    has_content: set[tuple[str, ...]] = {
        _components(s) for s in spans if s.kind != "heading" and s.section_path
    }

    roles: dict[tuple[str, ...], _RoleDraft] = {}
    education_paths: set[tuple[str, ...]] = set()
    order = order_start
    for path in headings:
        if (
            special(path, _BOUNDARY_SECTION) is not None
            or special(path, _EDUCATION_SECTION) is not None
            or special(path, _NOT_A_ROLE_SECTION) is not None
        ):
            continue
        if any(path[:i] in roles for i in range(1, len(path))):
            continue  # a sub-heading inside a role already found
        if dated_children.get(path, 0):
            continue  # an employer heading with dated roles beneath it, even if itself dated
        before, dates, key = split_trailing_dates(path[-1])
        if len(path) == 1:
            employer, title_text = _split_label(before)
            if key is None and not employer:
                continue  # no dates and no employer/title separator: not a role label
            if key is None and path not in has_content:
                continue
            if _EDUCATION_LABEL.search(before):
                education_paths.add(path)
                continue
        else:
            parent = path[-2]
            if key is None and (dated_children.get(path[:-1], 0) == 0 or path not in has_content):
                continue
            # An employer heading may carry the span of years there ("Contoso
            # (2012-2018)"); the role line shows its own dates, so the
            # employer is the heading's words before them.
            employer, title_text = split_trailing_dates(parent)[0], before
            if _GENERIC_PARENT.match(parent):
                employer, title_text = _split_label(before)
        title_text, location = _split_location(title_text)
        roles[path] = _RoleDraft(
            path=path,
            title=title_text,
            employer=employer,
            dates=dates,
            sort_key=key,
            order=order,
            location=location,
        )
        order += 1

    education: list[str] = []
    boundaries: list[str] = []
    for span in spans:
        if _is_title_heading(span):
            continue
        path = _components(span)
        if special(path, _BOUNDARY_SECTION) is not None:
            if span.kind != "heading":
                boundaries.append(span.text)
            continue
        education_at = special(path, _EDUCATION_SECTION)
        if education_at is not None:
            # Three shapes, and only two of them are CV lines:
            #   - the section's own heading: structure, never a line;
            #   - a sub-heading directly under it ("### BSc ..., 2006"), or a
            #     bullet sitting directly in the section with no sub-heading
            #     (a flat list): the qualification itself -- a line;
            #   - prose or bullets *beneath* a sub-heading: a NOTE about that
            #     qualification, written for the tool, never for a reader. The
            #     owner's record says, of a 2011 PGCert, that it "should not be
            #     represented as current or applied AI expertise" -- a caveat
            #     that was printed onto a CV verbatim before this rule. Notes
            #     stay out of the document; the model still reads them in the
            #     corpus, which is where a caveat belongs.
            depth = len(path) - (education_at + 1)
            is_qualification = (span.kind == "heading" and depth == 1) or (
                span.kind != "heading" and depth == 0
            )
            if is_qualification:
                education.append(span.text)
            continue
        if path[:1] in education_paths:
            education.append(span.text)
            continue
        if span.kind == "heading":
            continue
        owner = next((roles[path[:i]] for i in range(len(path), 0, -1) if path[:i] in roles), None)
        if owner is None:
            continue
        location_line = _LOCATION_LINE.match(span.text.strip())
        if location_line is not None and not owner.location:
            owner.location = location_line.group("where").strip()
            continue
        owner.facts.append(span.text)
        owner.fact_span_ids.append(span.id)

    return list(roles.values()), education, boundaries, title


def build_skeleton(spans: Sequence[Span]) -> CvSkeleton:
    """The CV's fixed structure from this user's live corpus spans, both shapes.

    Roles that appear in more than one corpus document (same employer, title
    and dates, compared normalised) are one role with their facts merged, so a
    corpus split across files does not list a job twice.
    """
    merged: dict[tuple[str, str, str], _RoleDraft] = {}
    education: list[str] = []
    boundaries: list[str] = []
    corpus_title: str | None = None
    seen_roles = 0
    for document in _by_document(spans):
        roles, doc_education, doc_boundaries, title = _read_document(document, seen_roles)
        seen_roles += len(roles)
        if corpus_title is None and title and title != HOSTED_TITLE:
            corpus_title = title
        for role in roles:
            key = (normalise(role.employer), normalise(role.title), normalise(role.dates))
            existing = merged.get(key)
            if existing is None:
                merged[key] = role
                continue
            for text, span_id in zip(role.facts, role.fact_span_ids, strict=True):
                if normalise(text) not in {normalise(f) for f in existing.facts}:
                    existing.facts.append(text)
                    existing.fact_span_ids.append(span_id)
            existing.location = existing.location or role.location
        education.extend(doc_education)
        boundaries.extend(doc_boundaries)

    dated = sorted(
        (r for r in merged.values() if r.sort_key is not None),
        key=lambda r: (r.sort_key, -r.order),
        reverse=True,
    )
    undated = sorted((r for r in merged.values() if r.sort_key is None), key=lambda r: r.order)
    return CvSkeleton(
        roles=tuple(
            SkeletonRole(
                title=r.title,
                employer=r.employer,
                dates=r.dates,
                location=r.location,
                facts=tuple(r.facts),
                fact_span_ids=tuple(r.fact_span_ids),
            )
            for r in (*dated, *undated)
        ),
        education=tuple(_dedupe(education)),
        boundaries=tuple(_dedupe(boundaries)),
        corpus_title=corpus_title,
    )


def _dedupe(lines: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        key = normalise(line)
        if key and key not in seen:
            seen.add(key)
            kept.append(line)
    return kept


_NAME_SEPARATORS = (" — ", " – ", " -- ", " - ", " | ", ": ")


def name_from_title(title: str | None) -> str:
    """The name part of a corpus document's own title ("Jane Doe — Career
    record" -> "Jane Doe"), or "" -- the header's last fallback, after the
    profile and the account's display name. Only ever the title's own words.
    """
    if not title:
        return ""
    for separator in _NAME_SEPARATORS:
        if separator in title:
            return title.split(separator, 1)[0].strip()
    return title.strip()
