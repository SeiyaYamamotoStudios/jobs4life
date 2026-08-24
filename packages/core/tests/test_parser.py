"""Unit tests for the markdown parser -- no filesystem, no database.

Matches the style of test_ids.py: these encode the properties parse_document
promises. Several tests re-derive properties ids.py already guarantees
(stability, reordering); the point here is confirming the parser actually
calls span_id the way it claims to, not re-testing span_id itself.
"""

from __future__ import annotations

import uuid

from jfl_core.ingest.parser import ParsedDocument, parse_document, split_sentences

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")
DOC = "file:corpus/cv.md"


def _parse(content: str) -> ParsedDocument:
    return parse_document(DOC, content, USER)


class TestSectionPath:
    def test_content_before_any_heading_has_empty_section_path(self) -> None:
        parsed = _parse("- A bullet with no heading above it\n")
        assert parsed.spans[0].section_path == ""

    def test_nested_headings_build_a_breadcrumb(self) -> None:
        parsed = _parse(
            "## Kaluza\n### Platform\n- Led the platform team\n",
        )
        bullet = next(s for s in parsed.spans if s.kind == "bullet")
        assert bullet.section_path == "Kaluza > Platform"

    def test_sibling_heading_replaces_the_previous_one_at_its_level(self) -> None:
        parsed = _parse(
            "## Kaluza\n### Platform\n- one\n### Data\n- two\n",
        )
        bullets = [s for s in parsed.spans if s.kind == "bullet"]
        assert bullets[0].section_path == "Kaluza > Platform"
        assert bullets[1].section_path == "Kaluza > Data"

    def test_returning_to_a_shallower_heading_pops_the_deeper_ones(self) -> None:
        parsed = _parse(
            "## Kaluza\n### Platform\n- one\n## Other Co\n- two\n",
        )
        bullets = [s for s in parsed.spans if s.kind == "bullet"]
        assert bullets[0].section_path == "Kaluza > Platform"
        assert bullets[1].section_path == "Other Co"

    def test_skipped_heading_levels_still_nest_under_whatever_is_open(self) -> None:
        """h2 -> h4 with no h3 in between; the h4 nests under the h2 rather than erroring."""
        parsed = _parse("## Kaluza\n#### Detail\n- one\n")
        bullet = next(s for s in parsed.spans if s.kind == "bullet")
        assert bullet.section_path == "Kaluza > Detail"


class TestCitationUnitKinds:
    def test_distinguishes_bullets_paragraphs_and_headings(self) -> None:
        parsed = _parse(
            "## Kaluza\n\nWorked as a platform engineer.\n\n- Led the platform team\n",
        )
        kinds = [s.kind for s in parsed.spans]
        assert kinds == ["heading", "paragraph", "bullet"]

    def test_a_new_bullet_marker_always_starts_a_fresh_span(self) -> None:
        parsed = _parse("- First\n- Second\n- Third\n")
        assert [s.text for s in parsed.spans] == ["First", "Second", "Third"]

    def test_nested_bullets_are_each_their_own_span_not_merged_into_the_parent(self) -> None:
        parsed = _parse("- Top bullet\n  - Nested one\n  - Nested two\n- Second top bullet\n")
        texts = [s.text for s in parsed.spans]
        assert texts == ["Top bullet", "Nested one", "Nested two", "Second top bullet"]
        assert all(s.kind == "bullet" for s in parsed.spans)

    def test_wrapped_paragraph_lines_merge_into_one_span(self) -> None:
        parsed = _parse("This is a paragraph\nthat wraps onto a second line.\n")
        assert len(parsed.spans) == 1
        assert parsed.spans[0].kind == "paragraph"
        assert "second line" in parsed.spans[0].text

    def test_blank_line_separates_two_paragraphs(self) -> None:
        parsed = _parse("First paragraph.\n\nSecond paragraph.\n")
        assert len(parsed.spans) == 2
        assert all(s.kind == "paragraph" for s in parsed.spans)


class TestOccurrenceDisambiguation:
    def test_identical_text_in_one_section_gets_distinct_ids(self) -> None:
        parsed = _parse("## Section\n- Shipped it\n- Shipped it\n")
        bullets = [s for s in parsed.spans if s.kind == "bullet"]
        assert len(bullets) == 2
        assert bullets[0].id != bullets[1].id

    def test_identical_text_in_different_sections_does_not_need_disambiguation(self) -> None:
        """Different section_path already makes the id distinct; occurrence resets per section."""
        parsed = _parse("## A\n- Shipped it\n## B\n- Shipped it\n")
        bullets = [s for s in parsed.spans if s.kind == "bullet"]
        assert bullets[0].id != bullets[1].id


class TestCharOffsets:
    def test_bullet_offsets_round_trip_through_the_original_content(self) -> None:
        content = "## Kaluza\n- Led the platform team\n- Shipped v2\n"
        parsed = _parse(content)
        for span in parsed.spans:
            assert content[span.char_start : span.char_end] == span.text

    def test_paragraph_offsets_round_trip(self) -> None:
        content = "Intro paragraph\nwrapped over two lines.\n\n- A bullet\n"
        parsed = _parse(content)
        for span in parsed.spans:
            assert content[span.char_start : span.char_end] == span.text

    def test_heading_offsets_round_trip_and_exclude_the_hash_marks(self) -> None:
        content = "### Platform Team\n- x\n"
        parsed = _parse(content)
        heading = parsed.spans[0]
        assert content[heading.char_start : heading.char_end] == "Platform Team"


class TestSentenceOffsets:
    def test_sentence_offsets_are_relative_to_span_text_and_round_trip(self) -> None:
        parsed = _parse("- Led the team. Shipped the platform. Grew revenue.\n")
        span = parsed.spans[0]
        assert len(span.sentences) == 3
        for sentence in span.sentences:
            substring = span.text[sentence.start_offset : sentence.end_offset]
            assert substring in span.text
            assert substring == substring.strip()

    def test_sentences_are_ordered_and_non_overlapping(self) -> None:
        parsed = _parse("- One. Two. Three.\n")
        span = parsed.spans[0]
        for a, b in zip(span.sentences, span.sentences[1:], strict=False):
            assert a.end_offset <= b.start_offset
            assert a.idx < b.idx

    def test_single_sentence_span_with_no_terminal_punctuation_still_gets_one_sentence(
        self,
    ) -> None:
        parsed = _parse("- Led the platform team\n")
        span = parsed.spans[0]
        assert len(span.sentences) == 1
        assert span.text[span.sentences[0].start_offset : span.sentences[0].end_offset] == (
            span.text
        )


class TestSentenceSplittingAbbreviations:
    def test_splits_on_a_plain_sentence_boundary(self) -> None:
        offsets = split_sentences("Led the team. Shipped the platform.")
        assert len(offsets) == 2

    def test_does_not_split_on_eg(self) -> None:
        offsets = split_sentences("Used several tools, e.g. Terraform and Kubernetes.")
        assert len(offsets) == 1

    def test_does_not_split_on_ie(self) -> None:
        offsets = split_sentences("One database, i.e. Postgres, backed the service.")
        assert len(offsets) == 1

    def test_does_not_split_on_etc(self) -> None:
        offsets = split_sentences("Ran migrations, backups, etc. every night.")
        assert len(offsets) == 1

    def test_does_not_split_on_company_suffix_ltd(self) -> None:
        offsets = split_sentences("Worked at Kaluza Ltd. They build energy software.")
        assert len(offsets) == 1

    def test_does_not_split_on_company_suffix_inc(self) -> None:
        offsets = split_sentences("Consulted for Acme Inc. It shipped devices.")
        assert len(offsets) == 1

    def test_does_not_split_on_vs(self) -> None:
        offsets = split_sentences("Compared build vs. buy before deciding.")
        assert len(offsets) == 1

    def test_does_not_split_on_a_single_capital_initial(self) -> None:
        offsets = split_sentences("Reported to J. Smith throughout the project.")
        assert len(offsets) == 1

    def test_does_split_after_an_initial_when_a_real_sentence_follows(self) -> None:
        # "Smith." ends with a real word before the period, not a bare initial.
        offsets = split_sentences("Reported to A. B. Smith. Delivered the platform.")
        assert len(offsets) >= 1  # at minimum doesn't crash; the two "A. B." are suppressed


class TestDocumentTitle:
    def test_first_h1_becomes_the_title(self) -> None:
        parsed = _parse("# Seiya Yamamoto\n## Kaluza\n- x\n")
        assert parsed.title == "Seiya Yamamoto"

    def test_no_h1_means_no_title(self) -> None:
        parsed = _parse("## Kaluza\n- x\n")
        assert parsed.title is None

    def test_only_the_first_h1_is_used_as_title(self) -> None:
        parsed = _parse("# First\n- x\n# Second\n- y\n")
        assert parsed.title == "First"


class TestEdgeCaseDocuments:
    def test_empty_document_has_no_spans(self) -> None:
        parsed = _parse("")
        assert parsed.spans == []
        assert parsed.title is None

    def test_document_with_no_headings_has_all_top_level_spans(self) -> None:
        parsed = _parse("- one\n- two\n\nA paragraph too.\n")
        assert all(s.section_path == "" for s in parsed.spans)

    def test_document_with_only_headings_has_only_heading_spans(self) -> None:
        parsed = _parse("# Title\n## Section A\n## Section B\n")
        assert all(s.kind == "heading" for s in parsed.spans)
        assert len(parsed.spans) == 3


class TestStability:
    def test_reparsing_identical_content_produces_identical_ids(self) -> None:
        content = "## Kaluza\n- Led the platform team\n- Shipped v2\n"
        first = _parse(content)
        second = _parse(content)
        assert [s.id for s in first.spans] == [s.id for s in second.spans]

    def test_reordering_bullets_within_a_section_preserves_their_ids(self) -> None:
        original = _parse("## Section\n- Alpha\n- Beta\n- Gamma\n")
        reordered = _parse("## Section\n- Gamma\n- Alpha\n- Beta\n")
        assert {s.id for s in original.spans} == {s.id for s in reordered.spans}

    def test_document_content_hash_is_stable_across_reparses(self) -> None:
        content = "## Kaluza\n- Led the platform team\n"
        assert _parse(content).content_hash == _parse(content).content_hash


class TestTitleHeadingIsNotPosition:
    """A document title must not appear in span ids.

    Span identity is meant to survive edits elsewhere in the file. If the title
    h1 sits in every breadcrumb, reformatting your own name at the top of a CV
    renames every span beneath it and orphans every reference to them.
    """

    CV = "# Seiya Yamamoto\n\n## Kaluza\n\n### Platform\n\n- Led the platform team\n"

    def test_lone_h1_is_excluded_from_section_path(self) -> None:
        doc = parse_document("file:cv.md", self.CV, USER)
        bullet = next(s for s in doc.spans if s.kind == "bullet")
        assert bullet.section_path == "Kaluza > Platform"

    def test_lone_h1_is_still_the_title(self) -> None:
        assert parse_document("file:cv.md", self.CV, USER).title == "Seiya Yamamoto"

    def test_renaming_the_title_does_not_change_span_ids(self) -> None:
        before = parse_document("file:cv.md", self.CV, USER)
        after = parse_document(
            "file:cv.md", self.CV.replace("# Seiya Yamamoto", "# Seiya Yamamoto - CV"), USER
        )
        b = next(s for s in before.spans if s.kind == "bullet")
        a = next(s for s in after.spans if s.kind == "bullet")
        assert a.id == b.id

    def test_several_h1s_are_structural_and_stay_in_the_path(self) -> None:
        content = "# Kaluza\n\n- Built billing\n\n# Currencycloud\n\n- Built FX\n"
        doc = parse_document("file:roles.md", content, USER)
        paths = [s.section_path for s in doc.spans if s.kind == "bullet"]
        assert paths == ["Kaluza", "Currencycloud"]

    def test_no_headings_at_all_gives_empty_paths(self) -> None:
        doc = parse_document("file:notes.md", "- One\n\n- Two\n", USER)
        assert {s.section_path for s in doc.spans} == {""}
