"""Pure markdown parser: content string in, spans out.

No filesystem access here -- a web upload path will feed this the same way a
local file does, so touching disk would make this untestable in one place and
wrong in the other. `source.py` is the only module allowed to read files.

Citation units are headings, bullets and paragraphs, split on blank lines and
line-type changes (a bullet marker always starts a new span, even directly
below another bullet or mid-paragraph). Position is expressed as a heading
breadcrumb (`section_path`) rather than an ordinal, per ids.py: reordering or
inserting content elsewhere must not change an unrelated span's identity.

All heading levels contribute to `section_path` EXCEPT a document title -- an
h1 that is the only one in the file. Including it would put the title in every
span id in the document, so reformatting your own name at the top of a CV would
change every id beneath it and orphan every reference. A file using h1s
structurally (several of them) keeps them in the breadcrumb, since there they
carry position rather than identity.

There is otherwise no special-casing that excludes a heading from
the path. Title extraction (the first h1's text) is a separate, orthogonal
read of the same heading.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from jfl_core.ids import content_hash, document_id, normalise, span_id
from jfl_core.models import Sentence, Span

_H1 = re.compile(r"^#[^#].*$", re.MULTILINE)
# A line that is nothing but a bold run is a heading in practice -- people
# write "**Numbers**" over a list rather than "#### Numbers". Treating it as a
# paragraph makes it a citable span saying only "Numbers", and strips that
# context off every bullet beneath it. Level 4 nests it under real h1-h3.
_BOLD_LABEL = re.compile(r"^\*\*(?P<text>[^*]+)\*\*$")
_BOLD_LABEL_LEVEL = 4
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# Case-insensitive; matched against the word (including its own trailing dot)
# immediately preceding a candidate sentence boundary.
_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "ltd.",
    "inc.",
    "co.",
    "corp.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "jr.",
    "sr.",
    "st.",
}
_SINGLE_INITIAL = re.compile(r"^[A-Z]\.$")
# Two or more short dotted segments written as one token: run-together initials
# ("R.R.", "J.R.R."), dotted acronyms ("U.S.", "U.K."), and degree abbreviations
# ("Ph.D.", "M.Sc.", "B.Sc."). Before this existed, "George R.R. Martin." split
# into "...George R.R." and a fragment "Martin." -- observed on FEVER item
# fever-13515, where the eval then scored the item as a harness error. That was
# first recorded as the model re-splitting its input; it was this splitter.
#
# Segments are capped at three letters so an ordinary word followed by an
# abbreviation cannot match. The cost is the same one the Ltd./Inc. entries
# above already accept: a sentence that genuinely ends on such a token ("...in
# the U.S. The team grew.") stays merged with the next one. That is the cheaper
# direction -- a merged unit is still checked, whole; a split one sends a name
# fragment to the claim gate as a claim of its own.
_DOTTED_ABBREVIATION = re.compile(r"^(?:[A-Za-z]{1,3}\.){2,}$")
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?=\s|$)")
_TRAILING_WORD = re.compile(r"(\S+)$")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    source_uri: str
    document_id: uuid.UUID
    title: str | None
    content_hash: str
    spans: list[Span]


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of each sentence within `text`.

    A regex-based splitter, not a real sentence-boundary algorithm. It covers
    the abbreviation ("e.g.", "Ltd.", ...) and single-initial ("J. Smith")
    cases common in CVs by refusing to split there, and likewise run-together
    dotted tokens ("R.R.", "U.S.", "Ph.D."). Known gaps, left
    unhandled on purpose: decimal numbers ("v2.0 shipped."), quoted sentences,
    and abbreviations not in the fixed list above. A real corpus is short
    enough that misplaced sentence boundaries cost little -- they only affect
    how tightly a citation can be scoped, never whether grounding works.
    """
    stripped_len = len(text.rstrip())
    if stripped_len == 0:
        return []

    boundaries: list[int] = []
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        word_match = _TRAILING_WORD.search(text[:end])
        word = word_match.group(1) if word_match else ""
        if (
            word.lower() in _ABBREVIATIONS
            or _SINGLE_INITIAL.match(word)
            or _DOTTED_ABBREVIATION.match(word)
        ):
            continue
        rest = text[end:].lstrip()
        if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'([“‘"):
            continue
        boundaries.append(end)

    spans: list[tuple[int, int]] = []
    start = 0
    for end in boundaries:
        seg_start, seg_end = start, end
        while seg_start < seg_end and text[seg_start].isspace():
            seg_start += 1
        while seg_end > seg_start and text[seg_end - 1].isspace():
            seg_end -= 1
        if seg_end > seg_start:
            spans.append((seg_start, seg_end))
        start = end
    seg_start = start
    while seg_start < stripped_len and text[seg_start].isspace():
        seg_start += 1
    if seg_start < stripped_len:
        spans.append((seg_start, stripped_len))
    return spans


def _iter_lines(content: str) -> Iterator[tuple[int, int, str]]:
    """Yield (start_offset, end_offset, text) per line; end excludes the newline."""
    pos = 0
    for raw_line in content.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        yield pos, pos + len(line), line
        pos += len(raw_line)


def parse_document(source_uri: str, content: str, user_id: uuid.UUID) -> ParsedDocument:
    doc_id = document_id(user_id, source_uri)
    title: str | None = None
    # A lone h1 is the document title, not a section. Several h1s are structure.
    title_is_structural = len(_H1.findall(content)) != 1
    heading_stack: list[tuple[int, str]] = []
    occurrences: Counter[tuple[str, str]] = Counter()
    spans: list[Span] = []
    ordinal = 0

    # The one open bullet or paragraph block, if any. A block never crosses a
    # heading or blank line, and a new bullet marker always closes whatever is
    # open -- nested bullets are siblings for identity purposes, not children.
    block_kind: str | None = None
    block_start = 0
    block_end = 0

    def section_path() -> str:
        return " > ".join(t for _, t in heading_stack)

    def make_span(kind: str, text: str, char_start: int, char_end: int) -> Span:
        nonlocal ordinal
        path = section_path()
        key = (path, normalise(text))
        occurrence = occurrences[key]
        occurrences[key] += 1
        sid = span_id(user_id, source_uri, path, text, occurrence)
        sentences = [
            Sentence(idx=i, start_offset=s, end_offset=e)
            for i, (s, e) in enumerate(split_sentences(text))
        ]
        result = Span(
            id=sid,
            user_id=user_id,
            document_id=doc_id,
            provenance="document",
            kind=kind,  # type: ignore[arg-type]
            section_path=path,
            ordinal=ordinal,
            text=text,
            content_hash=content_hash(text),
            char_start=char_start,
            char_end=char_end,
            sentences=sentences,
        )
        ordinal += 1
        return result

    def close_block() -> None:
        nonlocal block_kind
        if block_kind is None:
            return
        text = content[block_start:block_end]
        if text:
            spans.append(make_span(block_kind, text, block_start, block_end))
        block_kind = None

    for line_start, _line_end, line in _iter_lines(content):
        if not line.strip():
            close_block()
            continue

        heading_match = _HEADING.match(line)
        bold_match = None if heading_match else _BOLD_LABEL.match(line.strip())
        if heading_match or bold_match:
            close_block()
            if heading_match:
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).rstrip()
            else:
                assert bold_match is not None
                level = _BOLD_LABEL_LEVEL
                heading_text = bold_match.group("text").strip().rstrip(":.")
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            is_title = level == 1 and title is None and not title_is_structural
            if not is_title:
                heading_stack.append((level, heading_text))
            if title is None and level == 1:
                title = heading_text
            h_start = line_start + line.index(heading_text)
            h_end = h_start + len(heading_text)

            spans.append(make_span("heading", heading_text, h_start, h_end))
            continue

        bullet_match = _BULLET_PREFIX.match(line)
        if bullet_match:
            close_block()
            block_kind = "bullet"
            block_start = line_start + bullet_match.end()
            block_end = line_start + len(line.rstrip())
            continue

        # Plain content line: starts a paragraph, or continues whatever block is open.
        if block_kind is None:
            block_kind = "paragraph"
            block_start = line_start + (len(line) - len(line.lstrip()))
        block_end = line_start + len(line.rstrip())

    close_block()

    return ParsedDocument(
        source_uri=source_uri,
        document_id=doc_id,
        title=title,
        content_hash=content_hash(content),
        spans=spans,
    )
