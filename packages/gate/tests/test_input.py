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
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Led the team.", "Shipped it."))
        assert "Led the team.\nShipped it." in read_input(p)

    def test_rejoining_leaves_single_spaces(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("I led the ", "team at Visa."))
        assert "  " not in read_input(p)

    def test_column_padding_is_collapsed(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Led    the    team."))
        assert "Led the team." in read_input(p)


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
