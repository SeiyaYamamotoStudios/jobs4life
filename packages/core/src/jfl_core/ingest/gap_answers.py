"""Write-back for `jfl answer`: the user's own words land in corpus markdown,
never straight in the database.

See CLAUDE.md's decisions log, "A gap answer lands in corpus markdown, not the
database" -- `jfl answer` used to write an `adjudicated` span directly, which
made "the database is a rebuildable index over the corpus markdown" false the
moment anyone answered a question, and left the fact nowhere the author could
read, edit, or delete it. This module only ever appends a plain bullet under a
fixed heading; `jfl_core.ingest.ingest.run_ingestion` is what turns that line
into a `provenance='document'` span, the same as any other corpus fact.

No model call anywhere in this module -- the caller's text goes into the file
byte-for-byte (aside from collapsing embedded newlines and trimming outer
whitespace, so one answer is always exactly one bullet line).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from jfl_core.ids import normalise, span_id

FILENAME = "answered-questions.md"

# A single, structural h1 -- the parser's lone-h1 heuristic treats it as the
# document title and excludes it from `section_path` (see parser.py). The h2
# below it is what actually appears in `section_path`, so every answer gets a
# stable, non-empty one: "Gap Answers". Verified empirically by running
# `parse_document` over a file built from this skeleton -- see this
# workstream's report for the observed spans.
_HEADER = "# Answered Questions\n\n## Gap Answers\n\n"

SECTION_PATH = "Gap Answers"
# Matches what `walk_corpus` derives for this file regardless of the real
# corpus directory name (it always uses the literal "file:corpus/" prefix --
# see source.py), so this constant and a live ingestion pass always agree.
SOURCE_URI = f"file:corpus/{FILENAME}"


def _path(corpus_dir: Path) -> Path:
    return corpus_dir / FILENAME


def _line_text(answer_text: str) -> str:
    """Trim outer whitespace and collapse embedded newlines, so a shell
    multi-line argument still lands as exactly one bullet -- a bullet that
    spans a blank line would close and reopen as two spans, splitting one
    answer into two unrelated-looking facts.

    Nothing else is touched: internal spacing, punctuation, and markdown
    characters the user typed (including a leading "-" of their own) survive
    as written. See `packages/core/tests/test_gap_answers.py` for the
    parser round-trip that confirms this.
    """
    return " ".join(answer_text.strip().splitlines())


def _existing_bullets(content: str) -> list[str]:
    return [line[2:] for line in content.splitlines() if line.startswith("- ")]


def append_gap_answer(corpus_dir: Path, answer_text: str) -> bool:
    """Append `answer_text` as a bullet under corpus/answered-questions.md's
    "## Gap Answers" heading, creating the file with its heading skeleton if
    absent.

    Idempotent by content, not by caller: appending a line that is already
    present (compared via `jfl_core.ids.normalise`, the same comparison the
    parser uses to key occurrences) is a no-op. That is what keeps
    `gap_answer_span_id` below correct without re-deriving the parser's
    occurrence-counting logic -- as long as this function is the only writer,
    no two bullets in the file ever normalise to the same text, so every
    answer's occurrence is always 0.

    Returns True if a new line was appended, False if an equivalent line was
    already there.
    """
    line = _line_text(answer_text)
    path = _path(corpus_dir)
    content = path.read_text(encoding="utf-8") if path.exists() else _HEADER

    key = normalise(line)
    if any(normalise(existing) == key for existing in _existing_bullets(content)):
        return False

    if not content.endswith("\n"):
        content += "\n"
    content += f"- {line}\n"

    corpus_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def gap_answer_span_id(user_id: uuid.UUID, answer_text: str) -> uuid.UUID:
    """The id `parse_document` will assign the bullet `append_gap_answer`
    writes for this text, computed the same way the parser computes it
    (`ids.span_id` over user, source_uri, section_path, text, occurrence).
    Occurrence is always 0 -- see `append_gap_answer`'s docstring for why
    that is guaranteed, not assumed.
    """
    return span_id(user_id, SOURCE_URI, SECTION_PATH, _line_text(answer_text), occurrence=0)
