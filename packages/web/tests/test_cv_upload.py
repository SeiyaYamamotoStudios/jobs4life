"""`jfl_web.corpus`: what the upload form takes, and what it says when it does not.

No database, no model call, no network. The PDFs are built in-process with
reportlab, never committed and never fetched.
"""

from __future__ import annotations

import io

import pytest
from jfl_core.cv_limits import MAX_CV_READ_CHARS, for_reading
from jfl_web.corpus import (
    MAX_CV_CHARS,
    MAX_FILE_BYTES,
    MAX_FILES_PER_UPLOAD,
    MAX_UPLOAD_BYTES,
    UploadRejected,
    check_length,
    check_suffix,
    check_upload_size,
    clean_filename,
    read_note,
    read_upload,
)

pytest.importorskip("reportlab", reason="reportlab builds the test PDFs")

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


def make_pdf(lines: list[str] = CV_LINES, *, pages: int = 1) -> bytes:
    from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for _ in range(pages):
        y = 800
        for line in lines:
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


def make_scan() -> bytes:
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 780
    for _ in range(40):
        c.rect(72, y, 400, 6, fill=1, stroke=0)
        y -= 14
    c.save()
    return buf.getvalue()


# -- which formats are taken ---------------------------------------------------


@pytest.mark.parametrize("name", ["cv.pdf", "CV.PDF", "cv.md", "cv.txt", "a.markdown", "b.text"])
def test_the_formats_that_work_are_taken(name: str) -> None:
    check_suffix(name)


@pytest.mark.parametrize("name", ["cv.docx", "cv.doc", "cv.odt", "cv.rtf", "cv.pages"])
def test_a_word_processor_file_is_refused_by_name(name: str) -> None:
    """Still refused -- and the message names what does work rather than
    leaving the user to guess.
    """
    with pytest.raises(UploadRejected) as caught:
        check_suffix(name)

    message = str(caught.value)
    assert "PDF" in message and ".md" in message and ".txt" in message
    assert "paste" in message


def test_an_unknown_extension_is_refused_the_same_way() -> None:
    with pytest.raises(UploadRejected):
        check_suffix("cv.pages.zip")


# -- reading a PDF -------------------------------------------------------------


def test_a_pdf_comes_back_as_text_marked_as_extracted() -> None:
    item = read_upload("cv.pdf", make_pdf())

    assert item.extracted is True
    assert "Jane Doe" in item.text
    assert item.note is not None and "cv.pdf" in item.note


def test_the_note_counts_the_pages_it_read() -> None:
    assert "1 page of" in (read_upload("cv.pdf", make_pdf()).note or "")
    assert "3 pages of" in (read_upload("cv.pdf", make_pdf(pages=3)).note or "")


def test_a_scan_is_refused_with_a_message_naming_the_file() -> None:
    with pytest.raises(UploadRejected) as caught:
        read_upload("scanned-cv.pdf", make_scan())

    message = str(caught.value)
    assert message.startswith("scanned-cv.pdf")
    assert "scan" in message
    assert "does not read images" in message


def test_a_damaged_pdf_is_refused_without_library_detail() -> None:
    with pytest.raises(UploadRejected) as caught:
        read_upload("cv.pdf", b"%PDF-1.4 truncated")
    assert "could not be opened as a PDF" in str(caught.value)


# -- plain text is untouched ---------------------------------------------------


def test_a_markdown_upload_is_the_bytes_that_were_sent() -> None:
    body = "# Jane Doe\n\n- Led a team of 8.\n"
    item = read_upload("cv.md", body.encode("utf-8"))

    assert item.text == body
    assert item.extracted is False
    assert item.note is None


def test_text_that_is_not_utf8_is_refused() -> None:
    with pytest.raises(UploadRejected) as caught:
        read_upload("cv.txt", b"\xff\xfe not utf-8")
    assert "not UTF-8" in str(caught.value)


# -- the ceilings, and what each one protects ----------------------------------


def test_the_storage_ceiling_is_far_above_the_read_ceiling() -> None:
    """They were one number and could not be raised independently. The point of
    splitting them is that storage is cheap and a model call is not.
    """
    assert MAX_CV_CHARS > MAX_CV_READ_CHARS
    assert MAX_CV_CHARS == 500_000
    assert MAX_CV_READ_CHARS == 100_000


def test_a_cv_past_the_storage_ceiling_is_refused() -> None:
    with pytest.raises(UploadRejected) as caught:
        check_length("huge.md", "x" * (MAX_CV_CHARS + 1))
    assert "past the" in str(caught.value)


def test_a_cv_between_the_two_ceilings_is_taken_and_said_to_be_read_in_part() -> None:
    body = "x" * (MAX_CV_READ_CHARS + 5_000)
    assert check_length("long.md", body) == body

    note = read_note(len(body))
    assert note is not None
    assert "Stored in full" in note
    assert f"{MAX_CV_READ_CHARS:,}" in note


def test_a_cv_under_the_read_ceiling_gets_no_note() -> None:
    assert read_note(MAX_CV_READ_CHARS) is None


def test_only_the_read_ceiling_reaches_the_model() -> None:
    sent, truncated = for_reading("line\n" * 60_000)
    assert truncated is True
    assert len(sent) <= MAX_CV_READ_CHARS
    # Cut on a line boundary, so the model is never handed half a sentence and
    # asked to quote it back as somebody's own words.
    assert not sent.endswith("lin")


def test_a_short_cv_reaches_the_model_whole() -> None:
    body = "Led a team of eight."
    assert for_reading(body) == (body, False)


def test_an_empty_file_is_refused() -> None:
    with pytest.raises(UploadRejected):
        check_length("cv.md", "   \n ")


def test_one_oversized_file_is_refused_before_it_is_parsed() -> None:
    raw = b"x" * (MAX_FILE_BYTES + 1)
    with pytest.raises(UploadRejected) as caught:
        check_upload_size("big.pdf", raw, len(raw))
    assert "one file" in str(caught.value)


def test_too_many_bytes_in_one_go_is_refused() -> None:
    with pytest.raises(UploadRejected) as caught:
        check_upload_size("cv.pdf", b"x", MAX_UPLOAD_BYTES + 1)
    assert "in one go" in str(caught.value)


def test_the_request_ceiling_sits_below_cloudflares() -> None:
    """A limit the app cannot explain is a limit the user cannot act on: past
    Cloudflare's 100 MB body ceiling they would get its error page instead of
    our sentence.
    """
    assert MAX_UPLOAD_BYTES < 100 * 1000 * 1000


def test_the_file_count_covers_a_real_cv_collection() -> None:
    """The owner's case is thirty-three CVs in one upload."""
    assert MAX_FILES_PER_UPLOAD >= 33 * 2


# -- filenames come back from the review form, so they are input ----------------


def test_a_filename_is_trimmed_flattened_and_capped() -> None:
    assert clean_filename("  my   cv.pdf \n") == "my cv.pdf"
    assert clean_filename("../../etc/passwd") == "..-..-etc-passwd"
    assert len(clean_filename("a" * 500)) == 200


def test_an_empty_filename_still_names_something() -> None:
    assert clean_filename("   ") == "CV"
