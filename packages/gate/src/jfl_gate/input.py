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
            blocks.append(" ".join(current))
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
    if _TRAILING_DATE.search(next_line):
        return False
    # A "|"-delimited line (a subtitle, a contact-details line) is header
    # metadata, not prose -- it can be long enough to clear the wrap
    # threshold without ever being a mid-sentence wrap.
    if "|" in prev_line:
        return False
    return len(prev_line) >= threshold
