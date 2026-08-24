"""Corpus filesystem walker. The only ingestion module that touches disk."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def walk_corpus(corpus_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (source_uri, content) for every `*.md` file under `corpus_dir`.

    `source_uri` always uses the literal "file:corpus/" prefix regardless of
    `corpus_dir`'s actual name, so tests can point this at a fixture
    directory and still get realistic ids. It becomes part of the span id
    (see ids.py), so it must stay stable across machines -- no absolute path.
    """
    for path in sorted(corpus_dir.rglob("*.md")):
        relative = path.relative_to(corpus_dir)
        source_uri = f"file:corpus/{relative.as_posix()}"
        content = path.read_bytes().decode("utf-8")
        yield source_uri, content
