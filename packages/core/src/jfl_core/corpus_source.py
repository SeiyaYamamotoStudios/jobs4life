"""The hosted corpus document: markdown as the source of truth, with no file.

**This module is the one write path by which a hosted user's own words become
corpus.** Two flows need it and neither gets its own: confirming a fact read out
of a CV (`jfl_core.storage.candidate_facts`), and answering profile questions 15
and 16 -- where your depth is genuine, and the gaps that keep coming up
(`jfl_core.storage.user_corpus`, a tenancy-scoped wrapper over the functions
here, which is what the web layer is handed). Both are the user stating
something true about themselves; both land in one document, in one shape, with
one id scheme. Two mechanisms for one kind of fact is how one sentence ends up
with two span ids and the claim gate reads it as two independent pieces of
evidence -- CLAUDE.md, "Never both paths for one fact".

`jfl_core.ingest.gap_answers` established the rule this module extends. A fact
the user states about themselves lands in **corpus markdown**, and ingestion is
what turns that markdown into a `provenance='document'` span -- never a direct
span insert. The reasons, from CLAUDE.md's 2026-09-01 decision:

  * "markdown is the source of truth; the database is a rebuildable index over
    it" stays true, instead of quietly becoming false the moment anyone
    confirms anything;
  * the user can read, edit and delete the exact line recorded about them. In a
    truthfulness tool, "I cannot find or fix the fact you stored about me" is
    disqualifying;
  * there is exactly **one** write path per fact. Two paths means two span ids
    for one sentence and a gate that sees it twice.

What is new here is that a hosted user has no `corpus/` directory. `jfl answer`
appends to a file; this cannot. So the markdown itself is stored, in
`documents.text`, on a row whose `storage_kind` is `hosted` -- see the
2026-09-18 migration. The document is parsed by the same `parse_document` and
written by the same `PostgresIngestRepository` as any corpus file, so the spans
it produces are ordinary document spans and nothing downstream can tell the
difference. That is the point.

**No model is anywhere on this path.** The text handed in is the user's own
words, byte for byte apart from trimming the ends and collapsing embedded
newlines so that one fact is always exactly one bullet. Having a model tidy a
confirmed fact into neater prose is the ratchet in miniature: the user is then
held to wording they did not choose, by a tool whose whole claim is that it
measures distance from what they actually said.

## How a user reads, edits or deletes what is stored

The markdown lives in one row: `documents` where `user_id` is theirs and
`source_uri` is `hosted:confirmed-facts.md`, with the whole document in `text`.
`document_markdown` below returns it verbatim, which is what a "show me my
corpus" screen renders and what an export writes out.

Editing and deleting are the same round trip this module performs on the way
in -- change the markdown, re-parse, re-upsert -- so `remove_confirmed_fact`
takes the line out and the span it produced is **retired**, which is what takes
it out of grounding. Retired rather than deleted, per `spans.retired_at`: an
old citation still resolves to a real row, it just no longer grounds anything.
The retire is scoped to this one document (`retire_document_spans`), because
the per-user sweep `run_ingestion` uses would retire every other document's
spans along with it.

`storage_kind='hosted'` is load-bearing for the same reason: `run_ingestion`'s
sweep over `corpus/` is silent about a document that has no file, and silence is
not absence -- see `PostgresIngestRepository.retire_missing_documents`. A
hosted-user document stored under any other `storage_kind` would be retired,
with every fact in it, by one `jfl ingest`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.engine import Connection

from jfl_core.ids import document_id, normalise
from jfl_core.ingest.parser import parse_document
from jfl_core.models import Span
from jfl_core.storage.postgres import PostgresIngestRepository

FILENAME = "confirmed-facts.md"
# `hosted:` rather than `file:corpus/...`: the prefix says where the markdown
# actually is, and nothing on disk will ever answer to this URI.
SOURCE_URI = f"hosted:{FILENAME}"

# Names both of what the document holds -- facts confirmed off a CV and the
# statements a user wrote on the profile page -- without naming either source,
# since the gate reads this heading as the corpus's top line.
TITLE = "Facts you have confirmed"
# Sole h1, so `parse_document` reads it as the document title and leaves it out
# of every `section_path` beneath it -- see parser.py's docstring. That is what
# keeps a span's identity free of the title's wording.
_HEADER = f"# {TITLE}\n"

# Where a statement with no section of its own goes. A fact confirmed against a
# role is filed under the role's own label instead, and a profile answer under
# its question's section (`jfl_core.profile.CORPUS_SECTIONS`), so
# `section_path` carries the employer and title into the gate's view of the
# corpus -- "Led a team of six" is a very different claim under one role than
# under another, and the breadcrumb is what the gate sees.
DEFAULT_SECTION = "Confirmed Facts"


def _line_text(text: str) -> str:
    """One fact, one bullet. Outer whitespace trimmed and embedded newlines
    collapsed -- a bullet spanning a blank line would close and reopen as two
    spans, splitting one confirmed fact into two unrelated-looking ones.
    Nothing else is touched: the user's punctuation, casing and markdown
    characters survive as written.
    """
    return " ".join(text.strip().split("\n"))


def _is_heading(line: str, level: int) -> bool:
    stripped = line.strip()
    return stripped.startswith("#" * level + " ") and not stripped.startswith("#" * (level + 1))


def _heading_text(line: str) -> str:
    return line.strip().lstrip("#").strip()


def append_line(content: str, section: str, line: str) -> str:
    """The pure half: markdown in, markdown out.

    Appends `- {line}` at the end of the `## {section}` block, creating that
    block at the end of the document if it is not there. Idempotent by content
    within the section, compared with `jfl_core.ids.normalise` -- the same
    comparison the parser uses to key occurrences, which is what guarantees no
    two bullets in one section ever normalise alike and therefore that every
    bullet's occurrence is 0.

    Separated from the database work below so the markdown rules can be tested
    without Postgres, and so that a future edit-and-delete screen has one
    obvious place to grow.
    """
    if not content.strip():
        content = _HEADER
    lines = content.splitlines()

    start: int | None = None
    for index, text in enumerate(lines):
        if _is_heading(text, 2) and _heading_text(text) == section:
            start = index
            break

    if start is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.extend(["", f"## {section}", "", f"- {line}"])
        return "\n".join(lines) + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _is_heading(lines[index], 1) or _is_heading(lines[index], 2):
            end = index
            break

    key = normalise(line)
    for text in lines[start + 1 : end]:
        if text.startswith("- ") and normalise(text[2:]) == key:
            return "\n".join(lines) + "\n"

    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, f"- {line}")
    return "\n".join(lines) + "\n"


def remove_line(content: str, section: str, line: str) -> str:
    """`append_line`'s inverse: the markdown with that bullet taken out of that
    section. Unchanged if it is not there.

    The heading is left behind even when it empties, so a role the user has
    cleared out does not silently stop existing in a document they can read.
    """
    key = normalise(line)
    lines = content.splitlines()
    inside = False
    kept: list[str] = []
    removed = False
    for text in lines:
        if _is_heading(text, 1) or _is_heading(text, 2):
            inside = _is_heading(text, 2) and _heading_text(text) == section
        elif inside and not removed and text.startswith("- ") and normalise(text[2:]) == key:
            removed = True
            continue
        kept.append(text)
    if not removed:
        return content
    return "\n".join(kept) + "\n"


def _strip_section(content: str, section: str) -> str:
    """Every bullet taken out of one section, the section's heading left where
    it is. `remove_line`'s bulk form, and the first half of `set_section`.
    """
    inside = False
    kept: list[str] = []
    for text in content.splitlines():
        if _is_heading(text, 1) or _is_heading(text, 2):
            inside = _is_heading(text, 2) and _heading_text(text) == section
        elif inside and text.startswith("- "):
            continue
        kept.append(text)
    return "\n".join(kept) + "\n" if kept else content


def set_section(content: str, section: str, lines: Sequence[str]) -> str:
    """The markdown with this section holding exactly `lines`, in that order.

    `append_line`'s bulk form, and what a re-answered profile question needs:
    the user's newer words replace their older ones rather than joining them,
    and an emptied box empties the section. Lines that normalise alike collapse
    to one, the same way `append_line` refuses a duplicate -- two bullets in one
    section that normalise alike would parse to one span with two occurrences
    and make the second one's id depend on insertion order.

    An empty `lines` against a section that is not there is a no-op, so clearing
    a question the user never answered does not mint a document for them.
    """
    wanted: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = normalise(line)
        if key and key not in seen:
            seen.add(key)
            wanted.append(line)

    if not content.strip():
        if not wanted:
            return content
        content = _HEADER

    stripped = _strip_section(content, section)
    for line in wanted:
        stripped = append_line(stripped, section, line)
    return stripped


def document_markdown(conn: Connection, user_id: uuid.UUID) -> str | None:
    """This user's corpus markdown, verbatim, or None if they have none yet."""
    repo = PostgresIngestRepository(conn)
    return repo.document_text(user_id, document_id(user_id, SOURCE_URI))


def section_name(section: str | None) -> str:
    """The `## ` heading a statement filed under `section` actually lands under,
    which is also the `section_path` its span carries.

    Public because a caller that wants to find a statement's span has to be able
    to name its section without re-deriving the rule -- see
    `jfl_core.storage.candidate_facts.corpus_section`.

    One line, no markdown marks of its own: a heading that wrapped or that
    started with "#" would change the document's structure rather than name a
    section in it.
    """
    return " ".join((section or "").split()).lstrip("#").strip() or DEFAULT_SECTION


def _rewrite(conn: Connection, user_id: uuid.UUID, content: str) -> list[Span]:
    """Store the markdown and bring its spans in line with it.

    Every span the document now has is upserted; every span it used to have and
    no longer does is retired -- scoped to this document, never the per-user
    sweep `run_ingestion` uses, which would retire the rest of the corpus.
    """
    repo = PostgresIngestRepository(conn)
    parsed = parse_document(SOURCE_URI, content, user_id)
    repo.upsert_document(
        user_id,
        parsed.document_id,
        parsed.source_uri,
        parsed.title,
        parsed.content_hash,
        storage_kind="hosted",
        text=content,
    )
    for span in parsed.spans:
        repo.upsert_span(span)
    repo.retire_document_spans(user_id, parsed.document_id, {s.id for s in parsed.spans})
    return parsed.spans


def append_confirmed_fact(
    conn: Connection,
    user_id: uuid.UUID,
    text: str,
    *,
    section: str | None = None,
) -> uuid.UUID:
    """Record one confirmed fact and return the corpus span it became.

    Appends the user's words to their corpus markdown, stores the markdown,
    re-parses the whole document and brings its spans in line. Calling it twice
    with the same text and section is a no-op that returns the same span id.
    """
    heading = section_name(section)
    line = _line_text(text)
    if not line:
        raise ValueError("a confirmed fact cannot be empty")

    repo = PostgresIngestRepository(conn)
    doc_id = document_id(user_id, SOURCE_URI)
    content = append_line(repo.document_text(user_id, doc_id) or _HEADER, heading, line)
    return _find_span(_rewrite(conn, user_id, content), heading, line).id


def remove_confirmed_fact(
    conn: Connection,
    user_id: uuid.UUID,
    text: str,
    *,
    section: str | None = None,
) -> bool:
    """Take one confirmed fact back out of the corpus. Returns whether it was
    there to remove.

    The line goes from the markdown and the span it produced is retired, so
    nothing grounds on it any more. This is what makes "I can delete the fact
    you recorded about me" true rather than aspirational.
    """
    heading = section_name(section)
    line = _line_text(text)
    repo = PostgresIngestRepository(conn)
    doc_id = document_id(user_id, SOURCE_URI)
    content = repo.document_text(user_id, doc_id)
    if content is None:
        return False
    updated = remove_line(content, heading, line)
    if updated == content:
        return False
    _rewrite(conn, user_id, updated)
    return True


def replace_section(
    conn: Connection,
    user_id: uuid.UUID,
    texts: Sequence[str],
    *,
    section: str | None = None,
) -> list[uuid.UUID]:
    """Make `texts` exactly what one section holds, and return their span ids in
    the order given.

    `append_confirmed_fact`'s bulk form, for a statement the user can re-answer
    rather than add to: profile questions 15 and 16. Whatever else was live in
    the section comes out of the markdown and its span is retired, so a
    superseded statement about the user stops grounding claims instead of
    sitting beside its replacement. An empty `texts` clears the section -- an
    answer the user deleted must not go on being cited at them.

    Blank entries are dropped rather than recorded: silence is not a corpus
    fact.
    """
    heading = section_name(section)
    lines = [line for line in (_line_text(text) for text in texts) if line]

    repo = PostgresIngestRepository(conn)
    doc_id = document_id(user_id, SOURCE_URI)
    content = repo.document_text(user_id, doc_id)
    if content is None:
        if not lines:
            return []
        content = _HEADER

    updated = set_section(content, heading, lines)
    spans = _rewrite(conn, user_id, updated)
    return [_find_span(spans, heading, line).id for line in lines]


def _find_span(spans: list[Span], section: str, line: str) -> Span:
    """The span the appended bullet produced.

    Located in the parse output rather than recomputed with `ids.span_id`:
    occurrence counting is the parser's business, and a heading whose text
    happens to match a bullet's would make a recomputed occurrence wrong in a
    way nothing would notice until a citation resolved to the wrong sentence.
    """
    key = normalise(line)
    for span in spans:
        if span.kind == "bullet" and span.section_path == section and normalise(span.text) == key:
            return span
    raise RuntimeError(  # pragma: no cover -- append_line just put it there
        "the appended fact did not parse back out of the corpus markdown"
    )
