"""`jfl_core.ingest.pdf`: what comes out of a PDF, and what is refused.

Every PDF here is built in-process with reportlab, never committed as a binary
fixture and never downloaded -- the same rule `packages/gate/tests/test_input.py`
follows, and for the same reasons: a binary in the repo is a thing nobody can
read in a diff, and a download is a network call in a test run that forbids one.

No model call and no network. The block-rebuilding rules themselves are pinned
in the gate's own test file, which predates this module and still exercises it
through `read_input`; what is new here is the scan refusal and the byte-stream
entry point the web upload uses.
"""

from __future__ import annotations

import io

import pytest
from jfl_core.ingest.pdf import (
    MIN_TEXT_CHARS,
    PdfHasNoTextLayer,
    PdfUnreadable,
    extract_pdf_text,
    read_pdf,
)

reportlab = pytest.importorskip("reportlab", reason="reportlab builds the test PDFs")


def _pdf(*pages: list[str]) -> bytes:
    """A PDF with real, selectable text: one page per argument."""
    from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for lines in pages:
        y = 800
        for line in lines:
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


def _image_only_pdf(pages: int = 1) -> bytes:
    """A PDF with marks on the page and no text layer at all -- what a scanner
    or a phone camera produces. Drawn as filled rectangles rather than an
    embedded photograph so the test needs no image file, but the property that
    matters is identical: `extract_text()` has nothing to return.
    """
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for _ in range(pages):
        y = 780
        for _ in range(40):
            c.rect(72, y, 400, 6, fill=1, stroke=0)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


# A CV's worth of real sentences: comfortably past MIN_TEXT_CHARS, so these
# tests are about extraction and not about the floor.
CV_LINES = [
    "Jane Doe",
    "Engineering Manager | London, UK",
    "Acme Ltd, Engineering Manager Nov 2021 - Present",
    "- Led a platform team of eight engineers through a migration off a",
    "  monolith, with the on-call rota and the hiring budget.",
    "- Owned the pricing service end to end, including its error budget.",
    "- Delivered a rewrite of the settlement pipeline in twelve weeks.",
    "Northwind, Senior Engineer Feb 2017 - Oct 2021",
    "- Wrote the scheduler that replaced a nightly batch job.",
    "- Reviewed the cross-border payments design and its failure modes.",
]


# -- reading -------------------------------------------------------------------


def test_text_comes_out_of_a_pdf_given_as_bytes() -> None:
    result = read_pdf(_pdf(CV_LINES))

    assert "Jane Doe" in result.text
    assert "Owned the pricing service end to end" in result.text
    assert result.pages == 1
    assert result.textual_chars >= MIN_TEXT_CHARS


def test_a_file_object_and_bytes_read_the_same() -> None:
    raw = _pdf(CV_LINES)
    assert read_pdf(io.BytesIO(raw)).text == read_pdf(raw).text


def test_a_path_reads_the_same_as_bytes(tmp_path: object) -> None:
    import pathlib

    assert isinstance(tmp_path, pathlib.Path)
    raw = _pdf(CV_LINES)
    path = tmp_path / "cv.pdf"
    path.write_bytes(raw)
    assert read_pdf(path).text == read_pdf(raw).text


def test_every_page_is_counted() -> None:
    assert read_pdf(_pdf(CV_LINES, CV_LINES, CV_LINES)).pages == 3


def test_a_page_break_is_always_a_block_break() -> None:
    text = read_pdf(_pdf(["Led the team."], CV_LINES)).text
    assert text.startswith("Led the team.")
    assert "\n\n" in text


# -- refusing what cannot be read honestly -------------------------------------


def test_a_scan_is_refused_rather_than_read_as_a_few_characters() -> None:
    """The whole reason PDF was refused before: three characters of garbage
    quoted back as somebody's own words. Detected, not produced.
    """
    with pytest.raises(PdfHasNoTextLayer) as caught:
        read_pdf(_image_only_pdf(pages=3))

    message = str(caught.value)
    assert "scan" in message
    assert "paste" in message


def test_the_scan_message_does_not_promise_ocr() -> None:
    with pytest.raises(PdfHasNoTextLayer) as caught:
        read_pdf(_image_only_pdf())
    assert "does not read images" in str(caught.value)


def test_a_document_with_only_a_stray_line_of_text_is_still_a_scan() -> None:
    """A scanner that stamps a page number into a text layer over the image
    yields a handful of characters. That is not a CV.
    """
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 780
    for _ in range(40):
        c.rect(72, y, 400, 6, fill=1, stroke=0)
        y -= 14
    c.drawString(300, 40, "Page 1 of 3")
    c.save()

    with pytest.raises(PdfHasNoTextLayer):
        read_pdf(buf.getvalue())


def test_bytes_that_are_not_a_pdf_are_refused_without_library_detail() -> None:
    with pytest.raises(PdfUnreadable) as caught:
        read_pdf(b"%PDF-1.4 not really a pdf at all")

    message = str(caught.value)
    assert "could not be opened as a PDF" in message
    # The message is rendered onto a page: no exception text, no stack, no path.
    assert "Traceback" not in message and "pypdf" not in message


def test_a_scan_is_a_kind_of_unreadable_so_one_handler_catches_both() -> None:
    assert issubclass(PdfHasNoTextLayer, PdfUnreadable)


# -- the gate's entry point keeps its own behaviour -----------------------------


def test_extract_pdf_text_does_not_apply_the_scan_floor() -> None:
    """`jfl check <file>` is handed one document and shows its result at once,
    so a floor there would only refuse short documents that are real. The floor
    belongs where silence gets stored and quoted back later.
    """
    assert extract_pdf_text(_pdf(["Led the team."])) == "Led the team."
    assert extract_pdf_text(_image_only_pdf()) == ""
