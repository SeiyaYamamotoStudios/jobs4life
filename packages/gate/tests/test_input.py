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

        This is also the record of NEXT.md's "known to be wrong" acceptance for
        defect 2: over-splitting (a title wrongly kept apart) is far cheaper than
        fusing (two unrelated claims merged into one verdict), so the trailing-date
        signal stays even though it is a heuristic that can occasionally misfire --
        see TestBareDateFragment below for the one real-CV misfire this project has
        actually found, and why fixing it narrowly does not touch this case.
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


class TestHyphenRejoin:
    """A word broken across a line break by a hyphen used to come back with a
    stray space around the hyphen -- "cross-\\nborder" as "cross- border".
    Confirmed on real CVs: "cross- border", "trade- offs", "on- call",
    "AI- assisted" -- every one a genuinely hyphenated compound that merely
    happened to fall at the page margin, never an old-style soft hyphen
    inserted purely to break a word. See `_join_wrapped`'s docstring for the
    chosen rule (always keep the hyphen, only remove the stray space) and its
    failure mode.
    """

    def test_a_compound_word_broken_at_the_hyphen_rejoins_with_no_space(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Delivered a large cross-", "border payments programme."))
        text = read_input(p)
        assert "cross-border" in text
        assert "cross - border" not in text
        assert "cross- border" not in text
        assert "cross -border" not in text

    def test_a_hyphen_that_would_be_a_soft_break_also_keeps_the_hyphen(
        self, tmp_path: Path
    ) -> None:
        """The other direction of the same judgement call: nothing in the
        extracted text distinguishes an old-style soft hyphen ("colla-\\nboration"
        for "collaboration") from a real compound's hyphen landing at the same
        spot, so this project's rule -- keep the hyphen, fix only the spacing --
        applies here too, deliberately. The result ("colla-boration") is a wrong
        spelling, not a merged or split claim, which is why `_join_wrapped`
        accepts it as the cheaper failure mode rather than guessing.
        """
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Improved cross-team colla-", "boration on delivery."))
        text = read_input(p)
        assert "colla-boration" in text
        assert "colla- boration" not in text

    def test_a_hyphen_with_a_space_before_it_is_not_a_wrap_point(self, tmp_path: Path) -> None:
        """A suspended/shared hyphen inside one physical line ("performance- and
        reliability-critical", meaning "performance-critical and
        reliability-critical") is the author's own typography, not a PDF line
        wrap -- `_join_wrapped` only ever looks at the join *between* physical
        lines, so a hyphen with a space already before it inside one line must
        pass through untouched.
        """
        p = tmp_path / "cv.pdf"
        p.write_bytes(_one_page_pdf("Operated a performance- and reliability-critical system."))
        text = read_input(p)
        assert "performance- and reliability-critical" in text


class TestBareDateFragment:
    """A wrapped title-and-dates line whose own date range gets split across
    two physical lines by the PDF layout -- "Feb 2011 - Mar" / "2020" -- used
    to read the stranded "2020" as a title boundary in its own right (the same
    trailing-date signal that correctly separates real titles in
    TestPdfBlockStructure). Confirmed on one real CV: the fragment landed as
    its own useless one-line block, upstream of the company/location line that
    should follow the full title. See `_joins_forward`'s `_BARE_DATE_LINE`
    check.
    """

    def test_a_stranded_bare_year_rejoins_onto_its_title(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(
            _one_page_pdf(
                "Founder and Software Engineer Feb 2011 - Mar",
                "2020",
                "Yamamoto Studios | Bristol, UK",
            )
        )
        blocks = _blocks(read_input(p))
        assert "Founder and Software Engineer Feb 2011 - Mar 2020" in blocks
        # The rejoin must not go on to swallow the next, unrelated line --
        # that would be the fusion this project treats as the worse failure.
        assert "Yamamoto Studios | Bristol, UK" in blocks

    def test_a_stranded_present_or_current_also_rejoins(self, tmp_path: Path) -> None:
        p = tmp_path / "cv.pdf"
        p.write_bytes(
            _one_page_pdf(
                "Engineering Manager Nov 2024 -",
                "Present",
                "giffgaff | London, UK",
            )
        )
        blocks = _blocks(read_input(p))
        assert "Engineering Manager Nov 2024 - Present" in blocks
        assert "giffgaff | London, UK" in blocks


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
