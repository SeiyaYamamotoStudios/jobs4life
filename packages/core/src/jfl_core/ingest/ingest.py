"""Ingestion orchestration: walk the corpus, parse, write through the repository.

No SQL here -- everything storage-shaped goes through `IngestRepository`. This
module only owns control flow: which documents and spans were seen this pass,
so the repository knows what to retire.

Embeddings are untouched deliberately: retrieval is unused in v1 (see
CLAUDE.md's decisions log), so `span_embeddings` stays empty.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from jfl_core.context import RequestContext
from jfl_core.ingest.parser import parse_document
from jfl_core.ingest.source import walk_corpus
from jfl_core.repositories import IngestRepository

DEFAULT_CORPUS_DIR = Path("corpus")


@dataclass(frozen=True, slots=True)
class IngestSummary:
    documents_seen: int
    documents_retired: int
    spans_created: int
    spans_updated: int
    spans_retired: int


def run_ingestion(
    ctx: RequestContext,
    repo: IngestRepository,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> IngestSummary:
    seen_document_ids: set[uuid.UUID] = set()
    seen_span_ids: set[uuid.UUID] = set()
    documents_seen = 0
    spans_created = 0
    spans_updated = 0

    for source_uri, content in walk_corpus(corpus_dir):
        parsed = parse_document(source_uri, content, ctx.user_id)
        seen_document_ids.add(parsed.document_id)
        repo.upsert_document(
            ctx.user_id, parsed.document_id, parsed.source_uri, parsed.title, parsed.content_hash
        )
        documents_seen += 1

        for span in parsed.spans:
            seen_span_ids.add(span.id)
            if repo.upsert_span(span):
                spans_created += 1
            else:
                spans_updated += 1

    spans_retired = repo.retire_missing_spans(ctx.user_id, seen_span_ids)
    documents_retired = repo.retire_missing_documents(ctx.user_id, seen_document_ids)

    return IngestSummary(
        documents_seen=documents_seen,
        documents_retired=documents_retired,
        spans_created=spans_created,
        spans_updated=spans_updated,
        spans_retired=spans_retired,
    )
