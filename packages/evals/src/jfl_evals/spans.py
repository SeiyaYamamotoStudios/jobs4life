"""Turns one golden item's evidence sentences into the small in-memory corpus the
gate grounds against for that item -- never the whole golden set at once. Each
`check_text` call in the eval sees only the evidence for the claim under test, the
same way a real run sees only one user's corpus.
"""

from __future__ import annotations

import uuid

from jfl_core.ids import content_hash, document_id, span_id
from jfl_core.models import Span

from jfl_evals.dataset import GoldenItem

# Provenance choice: "document", not "adjudicated".
#
# tables.py's CheckConstraint ("document_id_iff_document") ties the two together:
# provenance="document" requires a document_id, provenance="adjudicated" requires
# document_id to be NULL. Nothing here is ever persisted, so the constraint itself
# never fires, but the in-memory Span objects should still mean what the model says
# they mean.
#
# "adjudicated" is not a free second option -- it has a specific meaning in this
# system: a user's own verbatim answer to a gap question, written back with no
# model anywhere in that path (CLAUDE.md, "A gap answer is stored verbatim").
# Reusing it for FEVER evidence sentences -- text nobody here wrote or vouched for
# -- would misrepresent what these spans are. "document" is the honest match: text
# presented as though extracted from a source, exactly what a real ingested corpus
# span is. So each item gets a synthetic `document_id`, minted the same
# deterministic way real ingestion mints one (`jfl_core.ids.document_id`), from a
# source_uri that can never collide with a real document because no real source
# is ever named "fever:...".


def build_spans(item: GoldenItem, user_id: uuid.UUID) -> list[Span]:
    """The corpus for one golden item: one Span per evidence sentence, or an empty
    list for a NOT ENOUGH INFO item with no evidence at all.
    """
    if not item.evidence:
        return []

    source_uri = f"fever:{item.id}"
    section_path = f"FEVER evidence [{item.id}]"
    doc_id = document_id(user_id, source_uri)

    spans: list[Span] = []
    occurrence_by_hash: dict[str, int] = {}
    for ordinal, text in enumerate(item.evidence):
        h = content_hash(text)
        occurrence = occurrence_by_hash.get(h, 0)
        occurrence_by_hash[h] = occurrence + 1
        spans.append(
            Span(
                id=span_id(user_id, source_uri, section_path, text, occurrence),
                user_id=user_id,
                document_id=doc_id,
                provenance="document",
                kind="paragraph",
                section_path=section_path,
                ordinal=ordinal,
                text=text,
                content_hash=h,
            )
        )
    return spans
