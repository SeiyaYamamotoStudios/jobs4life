"""The markdown half of the corpus write-back, without a database.

What is worth pinning here is the round trip: a line appended by
`append_line` must parse back out as exactly one bullet span, under a
`section_path` that names the role, with the user's words untouched. That is
the whole claim of the write-back path -- go through markdown and the span is
an ordinary document span -- and it is testable in one place with no Postgres.

The database half (`append_confirmed_fact`, `remove_confirmed_fact`) is covered
in `tests/test_confirmed_facts_grounding_integration.py`.
"""

from __future__ import annotations

import uuid

from jfl_core.corpus_source import (
    _HEADER,
    DEFAULT_SECTION,
    SOURCE_URI,
    append_line,
    remove_line,
    set_section,
)
from jfl_core.ingest.parser import parse_document

USER = uuid.UUID("0425d123-ed29-5a6a-a06d-d00267574046")

ROLE = "Acme Ltd -- Engineering Manager, 2021-2024"


def bullets(content: str) -> list[tuple[str, str]]:
    """(section_path, text) for every bullet span the parser finds."""
    parsed = parse_document(SOURCE_URI, content, USER)
    return [(s.section_path or "", s.text) for s in parsed.spans if s.kind == "bullet"]


class TestAppendLine:
    def test_a_new_section_is_created_and_the_bullet_parses_back_out(self) -> None:
        content = append_line(_HEADER, ROLE, "Led a team of eight engineers.")
        assert bullets(content) == [(ROLE, "Led a team of eight engineers.")]

    def test_the_title_stays_out_of_the_section_path(self) -> None:
        """A lone h1 is the document title. If it were in the breadcrumb,
        renaming the document would re-mint every span id beneath it.
        """
        content = append_line(_HEADER, ROLE, "Ran the on-call rota.")
        parsed = parse_document(SOURCE_URI, content, USER)
        assert parsed.title is not None
        assert all(parsed.title not in (s.section_path or "") for s in parsed.spans)

    def test_a_second_fact_joins_the_same_section(self) -> None:
        content = append_line(_HEADER, ROLE, "First fact.")
        content = append_line(content, ROLE, "Second fact.")
        assert bullets(content) == [(ROLE, "First fact."), (ROLE, "Second fact.")]

    def test_two_roles_get_two_sections(self) -> None:
        content = append_line(_HEADER, ROLE, "Acme fact.")
        content = append_line(content, "Northwind -- Staff Engineer, 2018-2021", "Northwind fact.")
        assert bullets(content) == [
            (ROLE, "Acme fact."),
            ("Northwind -- Staff Engineer, 2018-2021", "Northwind fact."),
        ]

    def test_appending_the_same_fact_twice_changes_nothing(self) -> None:
        once = append_line(_HEADER, ROLE, "Led a team of eight.")
        twice = append_line(once, ROLE, "Led  a team of eight.")
        assert once == twice
        assert len(bullets(twice)) == 1

    def test_the_same_words_under_a_different_role_are_a_separate_bullet(self) -> None:
        content = append_line(_HEADER, ROLE, "Led a team of six.")
        content = append_line(content, "Northwind -- EM", "Led a team of six.")
        assert len(bullets(content)) == 2

    def test_the_users_words_are_stored_untouched(self) -> None:
        words = "Owned the FX pricing platform's on-call, *including* weekends."
        content = append_line(_HEADER, ROLE, words)
        assert bullets(content) == [(ROLE, words)]

    def test_a_default_section_is_used_when_there_is_no_role(self) -> None:
        content = append_line(_HEADER, DEFAULT_SECTION, "MSc, 2010.")
        assert bullets(content) == [(DEFAULT_SECTION, "MSc, 2010.")]


class TestRemoveLine:
    def test_a_removed_fact_stops_parsing_out(self) -> None:
        content = append_line(_HEADER, ROLE, "First fact.")
        content = append_line(content, ROLE, "Second fact.")
        content = remove_line(content, ROLE, "First fact.")
        assert bullets(content) == [(ROLE, "Second fact.")]

    def test_removing_something_absent_changes_nothing(self) -> None:
        content = append_line(_HEADER, ROLE, "Only fact.")
        assert remove_line(content, ROLE, "Never said this.") == content

    def test_it_only_removes_from_the_named_section(self) -> None:
        content = append_line(_HEADER, ROLE, "Led a team of six.")
        content = append_line(content, "Northwind -- EM", "Led a team of six.")
        content = remove_line(content, ROLE, "Led a team of six.")
        assert bullets(content) == [("Northwind -- EM", "Led a team of six.")]

    def test_the_heading_survives_an_emptied_section(self) -> None:
        """A role the user has cleared out must not silently stop existing in a
        document they can read.
        """
        content = append_line(_HEADER, ROLE, "Only fact.")
        content = remove_line(content, ROLE, "Only fact.")
        assert f"## {ROLE}" in content
        assert bullets(content) == []

    def test_append_after_remove_restores_the_same_span_id(self) -> None:
        """Span identity is content plus section, never insertion order -- so a
        fact taken out and put back resolves to the row it had before.
        """
        first = append_line(_HEADER, ROLE, "Ran the migration.")
        before = parse_document(SOURCE_URI, first, USER)
        removed = remove_line(first, ROLE, "Ran the migration.")
        again = append_line(removed, ROLE, "Ran the migration.")
        after = parse_document(SOURCE_URI, again, USER)
        wanted = [s.id for s in before.spans if s.kind == "bullet"]
        assert wanted == [s.id for s in after.spans if s.kind == "bullet"]


class TestSetSection:
    """`append_line`'s bulk form: the section ends up holding exactly what was
    asked for. This is what a re-answered profile question needs -- the newer
    words replace the older ones rather than joining them.
    """

    def test_a_section_ends_up_holding_exactly_what_was_given(self) -> None:
        content = set_section(_HEADER, "Recurring gaps", ["Terraform.", "Kafka."])
        assert bullets(content) == [("Recurring gaps", "Terraform."), ("Recurring gaps", "Kafka.")]

    def test_a_replacement_supersedes_rather_than_joins(self) -> None:
        content = set_section(_HEADER, "Depth and exposure", ["Deep in the JVM."])
        content = set_section(content, "Depth and exposure", ["Actually, deep in data."])
        assert bullets(content) == [("Depth and exposure", "Actually, deep in data.")]

    def test_an_empty_replacement_clears_the_section_but_keeps_its_heading(self) -> None:
        content = set_section(_HEADER, "Depth and exposure", ["Deep in the JVM."])
        content = set_section(content, "Depth and exposure", [])
        assert bullets(content) == []
        assert "## Depth and exposure" in content

    def test_other_sections_are_left_alone(self) -> None:
        content = append_line(_HEADER, ROLE, "Ran the rota.")
        content = set_section(content, "Recurring gaps", ["Terraform."])
        content = set_section(content, "Recurring gaps", ["Kafka."])
        assert bullets(content) == [(ROLE, "Ran the rota."), ("Recurring gaps", "Kafka.")]

    def test_clearing_a_section_that_was_never_there_changes_nothing(self) -> None:
        """Nothing is minted for a question the user never answered."""
        assert set_section("", "Recurring gaps", []) == ""
        assert set_section(_HEADER, "Recurring gaps", []) == _HEADER

    def test_lines_that_normalise_alike_collapse_to_one(self) -> None:
        """Two bullets in one section that normalise alike would parse to one
        span with two occurrences, making the second's id depend on insertion
        order. `append_line` refuses the duplicate; so does this.
        """
        content = set_section(_HEADER, "Recurring gaps", ["Terraform.", "Terraform."])
        assert len(bullets(content)) == 1
