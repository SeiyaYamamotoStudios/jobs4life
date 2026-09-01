"""Unit tests for `jfl_core.ingest.gap_answers` -- no filesystem outside a
tmp_path, no database, no model. Matches the style of test_parser.py: these
encode the properties the module promises, cross-checked against
`parse_document` itself rather than against a hand-derived expectation, so a
drift between this module's id formula and the parser's would show up here.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from jfl_core.ingest.gap_answers import (
    FILENAME,
    SECTION_PATH,
    SOURCE_URI,
    append_gap_answer,
    gap_answer_span_id,
)
from jfl_core.ingest.parser import parse_document
from jfl_core.models import Span

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")


def _file(corpus_dir: Path) -> Path:
    return corpus_dir / FILENAME


def _spans_for(corpus_dir: Path) -> list[Span]:
    content = _file(corpus_dir).read_text(encoding="utf-8")
    return parse_document(SOURCE_URI, content, USER).spans


class TestFileCreation:
    def test_creates_the_file_with_heading_skeleton_when_absent(self, tmp_path: Path) -> None:
        assert not _file(tmp_path).exists()

        append_gap_answer(tmp_path, "I led the Q3 database migration.")

        content = _file(tmp_path).read_text(encoding="utf-8")
        assert content.startswith("# Answered Questions\n")
        assert "## Gap Answers\n" in content
        assert "- I led the Q3 database migration.\n" in content

    def test_a_lone_h1_leaves_the_title_out_of_section_path_and_the_h2_stable(
        self, tmp_path: Path
    ) -> None:
        """The heading structure this module writes: a single h1 (excluded from
        section_path by the parser's lone-h1 heuristic) plus one h2, so every
        answer's section_path is the stable, non-empty "Gap Answers".
        """
        append_gap_answer(tmp_path, "I led the Q3 database migration.")

        spans = _spans_for(tmp_path)
        title_heading = next(s for s in spans if s.text == "Answered Questions")
        assert title_heading.section_path == ""  # excluded: it is the document title

        bullet = next(s for s in spans if s.kind == "bullet")
        assert bullet.section_path == SECTION_PATH == "Gap Answers"

    def test_appending_to_an_existing_file_does_not_disturb_earlier_bullets(
        self, tmp_path: Path
    ) -> None:
        append_gap_answer(tmp_path, "First answer.")
        append_gap_answer(tmp_path, "Second answer.")

        spans = _spans_for(tmp_path)
        bullets = [s for s in spans if s.kind == "bullet"]
        assert [b.text for b in bullets] == ["First answer.", "Second answer."]


class TestSpanIdMatchesTheParser:
    def test_computed_id_matches_what_parse_document_actually_assigns(self, tmp_path: Path) -> None:
        answer = "I led the Q3 database migration and was on-call for the cutover."
        append_gap_answer(tmp_path, answer)

        computed = gap_answer_span_id(USER, answer)
        bullet = next(s for s in _spans_for(tmp_path) if s.kind == "bullet")
        assert bullet.id == computed

    def test_a_second_distinct_answer_gets_a_different_id_that_still_matches(
        self, tmp_path: Path
    ) -> None:
        first = "First distinct answer."
        second = "Second distinct answer."
        append_gap_answer(tmp_path, first)
        append_gap_answer(tmp_path, second)

        bullets = [s for s in _spans_for(tmp_path) if s.kind == "bullet"]
        assert bullets[0].id == gap_answer_span_id(USER, first)
        assert bullets[1].id == gap_answer_span_id(USER, second)
        assert bullets[0].id != bullets[1].id


class TestVerbatimPreservation:
    def test_leading_and_trailing_whitespace_is_trimmed_not_stored(self, tmp_path: Path) -> None:
        append_gap_answer(tmp_path, "   I have AWS certification, obtained 2024.   ")

        bullet = next(s for s in _spans_for(tmp_path) if s.kind == "bullet")
        assert bullet.text == "I have AWS certification, obtained 2024."

    def test_embedded_newlines_are_collapsed_to_a_single_bullet(self, tmp_path: Path) -> None:
        # A literal newline in the answer (e.g. from a shell $'...' argument)
        # must not split into a blank-line-separated paragraph -- that would
        # close the bullet block and start a second span, splitting one
        # answer into two.
        append_gap_answer(tmp_path, "First line.\nSecond line.")

        bullets = [s for s in _spans_for(tmp_path) if s.kind == "bullet"]
        assert len(bullets) == 1
        assert bullets[0].text == "First line. Second line."

    def test_a_multi_sentence_answer_is_stored_as_one_span_with_several_sentences(
        self, tmp_path: Path
    ) -> None:
        answer = "I owned the migration end to end. I was on-call for the cutover."
        append_gap_answer(tmp_path, answer)

        bullet = next(s for s in _spans_for(tmp_path) if s.kind == "bullet")
        assert bullet.text == answer
        assert len(bullet.sentences) == 2

    def test_markdown_characters_in_the_answer_survive_unmangled(self, tmp_path: Path) -> None:
        answer = "Used *bold-sounding* tools and _italic_ framing, but honestly."
        append_gap_answer(tmp_path, answer)

        bullet = next(s for s in _spans_for(tmp_path) if s.kind == "bullet")
        assert bullet.text == answer

    def test_a_leading_dash_typed_by_the_user_is_preserved_as_content(self, tmp_path: Path) -> None:
        # The user's own leading "-" must not be mistaken for a second bullet
        # marker: it is part of their sentence, not markdown structure.
        answer = "- Already led a smaller migration before this one."
        append_gap_answer(tmp_path, answer)

        bullet = next(s for s in _spans_for(tmp_path) if s.kind == "bullet")
        assert bullet.text == answer
        assert gap_answer_span_id(USER, answer) == bullet.id


class TestIdempotency:
    def test_answering_with_the_same_text_twice_does_not_duplicate_the_line(
        self, tmp_path: Path
    ) -> None:
        answer = "I led the Q3 database migration and was on-call for the cutover."
        first = append_gap_answer(tmp_path, answer)
        second = append_gap_answer(tmp_path, answer)

        assert first is True
        assert second is False
        content = _file(tmp_path).read_text(encoding="utf-8")
        assert content.count("I led the Q3 database migration") == 1

    def test_repeat_answer_resolves_to_the_same_single_span(self, tmp_path: Path) -> None:
        answer = "I led the Q3 database migration and was on-call for the cutover."
        append_gap_answer(tmp_path, answer)
        append_gap_answer(tmp_path, answer)

        bullets = [s for s in _spans_for(tmp_path) if s.kind == "bullet"]
        assert len(bullets) == 1
        assert bullets[0].id == gap_answer_span_id(USER, answer)

    def test_idempotent_across_incidental_whitespace_differences(self, tmp_path: Path) -> None:
        append_gap_answer(tmp_path, "I have AWS certification, obtained 2024.")
        second = append_gap_answer(tmp_path, "  I have AWS   certification, obtained 2024.  ")

        # Collapsed-whitespace variants of the same answer normalise to the
        # same key, so this must not mint a second, differently-worded bullet.
        assert second is False
        bullets = [s for s in _spans_for(tmp_path) if s.kind == "bullet"]
        assert len(bullets) == 1
