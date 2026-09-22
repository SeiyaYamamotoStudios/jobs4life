"""Reading the text to be checked.

PDF is here because it is what the real inputs are: application CVs arrive as
PDFs, and requiring `pdftotext` first put a shell pipeline between the user and
the tool.

The extraction itself moved to `jfl_core.ingest.pdf` when CV upload needed the
same thing. Two readers would have meant the claim gate seeing one rendering of
a PDF and the confirmation screen another, which is unaffordable in a tool whose
claim is that the line it quotes back is the line it read. What is left here is
the dispatch on file suffix, which is the CLI's business and not core's.
"""

from __future__ import annotations

from pathlib import Path

from jfl_core.ingest.pdf import (
    BULLET_START,
    PdfHasNoTextLayer,
    PdfUnreadable,
    extract_pdf_text,
)

__all__ = ["BULLET_START", "PdfHasNoTextLayer", "PdfUnreadable", "read_input"]


def read_input(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8")
