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
# A hard-wrapped line ends mid-sentence; a blank line is a real break.
_SOFT_WRAP = re.compile(r"(?<![.!?:;])\n(?!\n)")


def read_input(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _from_pdf(path)
    return path.read_text(encoding="utf-8")


def _from_pdf(path: Path) -> str:
    """Extract text, then undo the line breaks the PDF layout introduced.

    Without rejoining, every wrapped line looks like a sentence boundary to the
    splitter and the gate is handed fragments instead of claims.
    """
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    # Rejoin first: collapsing runs beforehand leaves a doubled space wherever a
    # line ended in one and the newline became another.
    text = _SOFT_WRAP.sub(" ", text)
    text = _RUNS.sub(" ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
