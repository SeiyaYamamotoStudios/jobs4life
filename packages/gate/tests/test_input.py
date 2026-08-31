"""Reading the text under check. No API, no database."""

from __future__ import annotations

from pathlib import Path

from jfl_gate.input import read_input


def test_plain_text_is_returned_unchanged(tmp_path: Path) -> None:
    p = tmp_path / "draft.md"
    p.write_text("Led the team.\nShipped it.\n", encoding="utf-8")
    assert read_input(p) == "Led the team.\nShipped it.\n"


def test_suffix_match_is_case_insensitive(tmp_path: Path) -> None:
    p = tmp_path / "CV.PDF"
    p.write_bytes(_one_page_pdf("Led the team."))
    assert "Led the team." in read_input(p)


class TestPdf:
    """PDF layout introduces line breaks the author never wrote.

    Left in place, every wrapped line looks like a sentence boundary and the gate
    is handed fragments instead of claims.
    """

    def test_soft_wrapped_lines_are_rejoined(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("I led the platform", "team at Visa."))
        assert "I led the platform team at Visa." in read_input(p)

    def test_a_real_sentence_end_is_kept_apart(self, tmp_path: Path) -> None:
        """A blank line, not a bare newline: a bare newline is what caused the
        original bug (parser.py's block model, and jfl_gate.gate's, both read
        a bare newline as "still the same block"). A blank line is the one
        signal that reliably reads as "new block" downstream.
        """
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Led the team.", "Shipped it."))
        assert "Led the team.\n\nShipped it." in read_input(p)

    def test_rejoining_leaves_single_spaces(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("I led the ", "team at Visa."))
        assert "  " not in read_input(p)

    def test_column_padding_is_collapsed(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Led    the    team."))
        assert "Led the team." in read_input(p)


class TestPdfBlockStructure:
    """Real CVs are headings, title-and-dates lines, and bullets sitting directly
    on top of each other with no blank line in the raw extraction -- see
    `_from_pdf`'s docstring. These are the cases that used to fuse an entire
    section (e.g. EDUCATION) into one 400+ character span with one verdict.
    """

    def test_a_heading_is_kept_apart_from_the_bullet_beneath_it(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("EDUCATION & CREDENTIALS", "- Completed a course"))
        blocks = _blocks(read_input(p))
        assert "EDUCATION & CREDENTIALS" in blocks
        assert "- Completed a course" in blocks

    def test_a_job_title_and_dates_line_is_kept_apart_from_the_bullet_beneath_it(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(
            _one_page_pdf(
                "Senior System Developer Aug 2005 - Feb 2011",
                "- Led development of a real-time event streaming system",
            )
        )
        blocks = _blocks(read_input(p))
        assert "Senior System Developer Aug 2005 - Feb 2011" in blocks
        assert "- Led development of a real-time event streaming system" in blocks

    def test_consecutive_bullets_never_merge(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("- First point", "- Second point", "- Third point"))
        blocks = _blocks(read_input(p))
        assert blocks == ["- First point", "- Second point", "- Third point"]

    def test_a_long_bullet_lacking_a_final_period_does_not_swallow_the_next_title(
        self, tmp_path: Path
    ) -> None:
        """The one case line length alone can't settle: a bullet ending near the
        page margin, with no final period because the author dropped one (common
        for the last item in a comma list), directly followed by the next
        credential's title. By length alone this reads as "still wrapping".
        Resolved by the trailing-date signal instead: a line ending in a year is
        never a valid continuation of what came before it.
        """
        p = tmp_path / "cv.pdf"
        p.write_bytes(
            _one_page_pdf(
                "- Advanced neural network architectures, model evaluation, and failure modes",
                "Postgraduate Certificate (Distinction) 2011",
            )
        )
        blocks = _blocks(read_input(p))
        assert "Postgraduate Certificate (Distinction) 2011" in blocks


def _blocks(text: str) -> list[str]:
    """Split `read_input`'s output on the blank lines that mark a real block
    break -- the same separator `jfl_gate.gate.split_blocks` reads.
    """
    return [block for block in text.split("\n\n") if block]


def _one_page_pdf(*lines: str) -> bytes:
    """Minimal single-page PDF. Avoids committing a binary fixture."""
    import io

    from pypdf import PdfWriter
    from reportlab.pdfgen import canvas  # type: ignore[import-untyped]

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 14
    c.save()
    buf.seek(0)
    writer = PdfWriter(clone_from=buf)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
