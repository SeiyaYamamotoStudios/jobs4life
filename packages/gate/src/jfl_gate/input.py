"""Reading the text to be checked.

PDF is here because it is what the real inputs are: application CVs arrive as
PDFs, and requiring `pdftotext` first put a shell pipeline between the user and
the tool.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

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


def read_input(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _from_pdf(path)
    return path.read_text(encoding="utf-8")


def _from_pdf(path: Path) -> str:
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
    """
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]

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
    """Join one block's physical lines into the text `read_input` returns.

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
