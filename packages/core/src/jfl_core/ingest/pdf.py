"""PDF in, plain text out -- one definition, used by everything that reads a PDF.

This started as `jfl_gate.input._from_pdf`, tuned against 32 real CVs. It moved
here when CV upload needed the same thing, because the alternative was two
readers: the claim gate would see one rendering of a PDF and the confirmation
screen another, and the whole point of the confirmation screen is that the line
quoted back is the line the tool actually read. One reader, one answer.

`pypdf` is the library. Pure Python, BSD-3, no system binary (no poppler, no
`pdftotext` subprocess, nothing to install on the VPS), actively released, and
already a dependency here. The alternatives were weighed: PyMuPDF extracts
better and is AGPL, which is not available to a hosted product; pdfminer.six is
pure Python and permissive but slower and effectively in maintenance; pdfplumber
sits on pdfminer and buys layout features this does not use.

**What cannot be read honestly is refused, not guessed at.** A scan or a photo
of a CV has no text layer, and `extract_text()` answers it with nothing, or with
a stray page number from a footer. Returning that as "your CV" would put three
characters of garbage into the confirmation screen and call them the author's
own words. `PdfHasNoTextLayer` is raised instead -- there is no OCR here and
this module is not the place to add one silently.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from pypdf import PasswordType, PdfReader
from pypdf.errors import PyPdfError

# Runs of spaces inside a line come from PDF column layout, not from the author.
_RUNS = re.compile(r"[ \t]{2,}")

# A line starting with one of these is a bullet: bare glyphs a PDF bullet font
# extracts as (•, ▪), plain-text-style markers (-, *), or a numbered "1." /
# "1)". Exported because jfl_gate.gate needs the identical pattern to split
# whatever this module hands it back into blocks -- one definition, so the two
# modules can never quietly drift apart on what counts as a bullet.
BULLET_START = re.compile(r"^(?:[•▪\-*]|\d+[.)])\s+")

# A line ending here is a finished clause, never a mid-sentence wrap. Only
# .!? count, matching jfl_core.ingest.parser.split_sentences's own definition
# of a sentence boundary -- a semicolon or colon most often continues a list
# within one bullet ("1:1s that leave engineers...; direct feedback that...")
# and must not be read as the bullet ending.
_TERMINAL = re.compile(r"[.!?]\s*$")

# CVs put dates at the end of a title line by convention: "Engineering Manager
# Nov 2024 -- Present", "Postgraduate Certificate ... 2011". A line ending this
# way is a title in its own right. Without this check, a long bullet that
# happens to end without a period (common where the last item in a list has no
# final full stop) would read as "still wrapping" by length alone and swallow
# the next credential's title straight into it.
_TRAILING_DATE = re.compile(r"(?:(?:19|20)\d{2}|present|current)\s*$", re.IGNORECASE)

# A line's length relative to the longest line this document produced. Prose
# that wraps because it hit the page margin sits close to that maximum; a
# heading, a title-and-dates line, or a company-and-location line ends
# wherever its content ends -- almost always well short of it. Relative
# rather than a fixed character count so it adapts to the page's own margins
# and font size instead of being tuned to one document.
_WRAP_THRESHOLD_RATIO = 0.75

# A line consisting of nothing but a bare year (or "present"/"current") --
# same tokens as _TRAILING_DATE, anchored at both ends instead of just the
# end. Nobody titles a CV entry "2020": a line that is *only* this token is
# never a heading in its own right, so it is always the stranded tail of a
# title-and-dates line whose date range itself got wrapped ("... Feb 2011 --
# Mar" / "2020"), a case actually observed on real CVs. See _joins_forward.
_BARE_DATE_LINE = re.compile(r"^(?:(?:19|20)\d{2}|present|current)$", re.IGNORECASE)

# A hyphen immediately after a letter or digit, at the very end of a line --
# no space before it. This is what a PDF extractor produces both for a
# genuinely hyphenated compound that happens to fall at the page margin
# ("cross-\nborder", "on-\ncall") and, in principle, for an old-style
# soft/typesetting hyphen inserted purely to break a word across the line
# ("col-\nlaboration"). See _join_wrapped for which way this project resolves
# that ambiguity and why.
_WORD_HYPHEN_EOL = re.compile(r"[A-Za-z0-9]-$")

# Letters and digits only: what a person would count as text on the page.
_TEXTUAL = re.compile(r"[^\W_]", re.UNICODE)

# The floor below which a PDF is called a scan rather than a document.
#
# The rule has to separate "no text layer at all" from "a real, short CV", and
# those are nowhere near each other: the shortest plausible one-page CV runs to
# a couple of thousand letters and digits, while an image-only page yields zero
# -- or, where the producer stamped a footer or a page number into a text layer
# over the image, a couple of dozen. 200 sits in the empty middle of that gap,
# so it needs no tuning and is not a threshold anyone has to defend per file.
#
# Counted across the whole document rather than per page on purpose. A PDF that
# is half typed and half scanned still has its typed half read, and the review
# screen is where the author sees the missing half and pastes it in -- which is
# a better outcome than refusing a document we can partly read.
MIN_TEXT_CHARS = 200


class PdfUnreadable(Exception):
    """The bytes are not a PDF this library can open, or they are encrypted.

    The message is written for a person: it is rendered straight onto the
    upload screen, and it never carries library text, a path or a byte count.
    """


class PdfHasNoTextLayer(PdfUnreadable):
    """A scan or a photo. There is text on the page to a human eye and none in
    the file, and no amount of parsing will change that.
    """


@dataclass(frozen=True, slots=True)
class PdfText:
    """What one PDF held: its text, how many pages it came off, and how much of
    it a person would count as text -- letters and digits, not punctuation and
    not whitespace. `textual_chars` is what the scan check is made of.
    """

    text: str
    pages: int
    textual_chars: int


def extract_pdf_text(source: Path | IO[bytes] | bytes) -> str:
    """`read_pdf`'s text alone, with the scan check off.

    For `jfl check <file>`, which is handed one document the author has chosen
    to check right now and shows its result immediately: an empty result is
    visible in the same breath, so a floor there would only refuse short
    documents that are perfectly real. The check belongs to the path where
    silence gets stored and later quoted back as somebody's own words, which is
    CV upload -- see `read_pdf`.
    """
    return read_pdf(source, require_text_layer=False).text


def read_pdf(source: Path | IO[bytes] | bytes, *, require_text_layer: bool = True) -> PdfText:
    """Extract text, then rebuild the block structure the PDF layout destroyed.

    `extract_text()` returns one physical line per line of type on the page --
    headings, bullets, job-title-and-dates lines, and paragraph text all come
    back as bare newline-separated lines with no signal for which are real
    mid-sentence wraps and which are independent claims that merely sit next
    to each other. Left alone, every one of those looks like a wrap and the
    gate is handed one fused blob per section instead of one unit per claim.

    Each pair of adjacent lines is either joined with a single space (a
    genuine wrap) or separated by a blank line (a real break); no bare
    newline survives. `jfl_gate.gate` never needs to see a PDF line break
    again -- a blank line is unambiguously "new block", exactly as it is in
    corpus markdown.

    Raises `PdfHasNoTextLayer` for a scan and `PdfUnreadable` for anything
    else that will not open. Neither is retryable and both say so in words a
    person can act on. `require_text_layer=False` drops only the first of
    those, and the one caller that does it says why.
    """
    pages = _page_texts(source)
    blocks = _rebuild(pages)
    textual = len(_TEXTUAL.findall(blocks))
    if require_text_layer and textual < MIN_TEXT_CHARS:
        raise PdfHasNoTextLayer(
            "looks like a scan or a photo rather than a text PDF -- there is no text "
            "in the file to read. This step does not read images. Export or print the "
            "CV to PDF from the document it came from, or paste the text in instead."
        )
    return PdfText(text=blocks, pages=len(pages), textual_chars=textual)


def _page_texts(source: Path | IO[bytes] | bytes) -> list[str]:
    stream: Path | IO[bytes] = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            # An empty user password is the common "restricted printing" case
            # and decrypts silently; a real password does not, and is refused.
            try:
                opened = reader.decrypt("")
            except Exception:
                opened = PasswordType.NOT_DECRYPTED
            if opened == PasswordType.NOT_DECRYPTED:
                raise PdfUnreadable(
                    "is password-protected, so its text cannot be read. Save an "
                    "unprotected copy, or paste the text in instead."
                )
        return [page.extract_text() or "" for page in reader.pages]
    except PdfUnreadable:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, TypeError, RecursionError):
        # Deliberately broad and deliberately wordless about the cause: pypdf
        # raises a wide family on a damaged file, and the exception text is
        # library detail that means nothing to the person who chose the file.
        raise PdfUnreadable(
            "could not be opened as a PDF. If it opens in a reader, try saving "
            "it again from there, or paste the text in instead."
        ) from None


def _rebuild(pages: list[str]) -> str:
    lines: list[str] = []
    for i, page_text in enumerate(pages):
        if i > 0:
            lines.append("")  # a page break is always a real break
        lines.extend(_RUNS.sub(" ", line.strip()) for line in page_text.split("\n"))

    max_len = max((len(line) for line in lines if line), default=0)
    threshold = max_len * _WRAP_THRESHOLD_RATIO

    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(_join_wrapped(current))
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if BULLET_START.match(line):
            flush()
            current.append(line)
            continue
        if current and _joins_forward(current[-1], line, threshold):
            current.append(line)
        else:
            flush()
            current.append(line)
    flush()

    return "\n\n".join(blocks).strip()


def _joins_forward(prev_line: str, next_line: str, threshold: float) -> bool:
    """Should `next_line` be pulled onto the end of `prev_line` as one wrapped
    line, rather than starting a new block?
    """
    if _TERMINAL.search(prev_line):
        return False
    if BULLET_START.match(next_line):
        return False
    # A "|"-delimited line (a subtitle, a contact-details line) is header
    # metadata, not prose -- it can be long enough to clear the wrap
    # threshold without ever being a mid-sentence wrap.
    if "|" in prev_line:
        return False
    # A bare "2020" (or "Present"/"Current") is never a title in its own
    # right, so it always belongs to whatever came before it -- join it
    # unconditionally rather than let the checks below reject it. This
    # matters because it would otherwise fail two different ways: the
    # trailing-date check below reads it as a title boundary (that is
    # defect 2 from NEXT.md), and even with that check removed the length
    # check would still reject it on real data -- a four-character line
    # essentially never clears the wrap-length threshold on its own. Real
    # CVs never title an entry with a bare year, so joining unconditionally
    # has no observed downside; see packages/gate/tests/test_input.py.
    if _BARE_DATE_LINE.match(next_line):
        return True
    if _TRAILING_DATE.search(next_line):
        return False
    return len(prev_line) >= threshold


def _join_wrapped(lines: list[str]) -> str:
    """Join one block's physical lines into the text `extract_pdf_text` returns.

    Ordinarily a single space -- an honest word-wrap. But a line ending in a
    hyphen directly after a letter or digit, no space before it, is the wrap
    point of either a genuinely hyphenated compound ("cross-" / "border") or
    an old-style soft hyphen inserted purely to break a word across the line
    ("col-" / "laboration"). Joining either with a bare space, as a plain
    `" ".join` does, produces a stray space around the hyphen -- confirmed on
    real CVs as "trade- offs", "cross- border", "AI- assisted", never a
    soft-hyphen break.

    Chosen rule: always keep the hyphen, only remove the stray space, giving
    "trade-offs" / "cross-border". This is a deliberate judgement call, not a
    detected distinction -- nothing in the extracted text tells a compound's
    hyphen apart from a soft one; pypdf does not preserve a separate
    soft-hyphen codepoint, and every hyphen-at-line-end actually observed
    across 32 real CVs turned out to be a genuine compound. Per this
    project's rule against inventing categories without evidence (see
    CLAUDE.md's drift taxonomy), there is no basis yet for the alternative
    (drop the hyphen). Failure mode: a document from a source that really
    does soft-hyphenate ("col-laboration" for "collaboration") would keep an
    unwanted hyphen. That corrupts a word's spelling but never merges or
    splits a claim, so it is the cheaper of the two failure directions --
    the same trade-off `_joins_forward` above makes for defect 2.

    Since this is now what the confirmation screen quotes back, the author
    sees any such word and can fix it before a single fact is read out of it.
    """
    if not lines:
        return ""
    joined: list[str] = [lines[0]]
    for line in lines[1:]:
        if _WORD_HYPHEN_EOL.search(joined[-1]):
            joined[-1] = joined[-1] + line
        else:
            joined.append(line)
    return " ".join(joined)


__all__ = [
    "BULLET_START",
    "MIN_TEXT_CHARS",
    "PdfHasNoTextLayer",
    "PdfText",
    "PdfUnreadable",
    "extract_pdf_text",
    "read_pdf",
]
