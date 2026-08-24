"""Deterministic identifiers.

A span's id must survive re-ingestion. The obvious derivation -- hash(document,
ordinal, text) -- does not: inserting a bullet at the top of a file renumbers
every ordinal below it and orphans every id in the golden set.

So position is expressed as the *section heading path*, which is stable under
reordering within a section, and the ordinal is used only to disambiguate spans
whose normalised text is genuinely identical within the same section.

    span_id = uuid5(NS_SPAN, user_id | doc_path | section_path | text_hash | occurrence)

Consequences, stated plainly:
  * reordering bullets within a section  -> id unchanged
  * inserting or deleting other bullets  -> id unchanged
  * moving a bullet to another section   -> new id (it is arguably a new claim)
  * editing the text at all              -> new id; the old span is retired, not
    deleted, so anything pointing at it still resolves
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid

NS_ROOT = uuid.UUID("6f5c2f7e-1c5a-5f9e-9b1e-6a3d0c8f4a21")
NS_DOCUMENT = uuid.uuid5(NS_ROOT, "document")
NS_SPAN = uuid.uuid5(NS_ROOT, "span")
NS_SENTENCE = uuid.uuid5(NS_ROOT, "sentence")

_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Canonical form used for hashing and identity.

    Strips the markdown list marker and collapses whitespace, so re-wrapping a
    bullet or switching `-` to `*` does not mint a new id. Deliberately does NOT
    lowercase or strip punctuation: a change in casing inside a claim can change
    its meaning, and this is a truthfulness tool.
    """
    text = unicodedata.normalize("NFC", text)
    text = _LIST_MARKER.sub("", text)
    return _WS.sub(" ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def document_id(user_id: str, path: str) -> uuid.UUID:
    return uuid.uuid5(NS_DOCUMENT, f"{user_id}|{path}")


def span_id(
    user_id: str, doc_path: str, section_path: str, text: str, occurrence: int = 0
) -> uuid.UUID:
    key = f"{user_id}|{doc_path}|{section_path}|{content_hash(text)}|{occurrence}"
    return uuid.uuid5(NS_SPAN, key)


def adjudicated_span_id(user_id: str, claim_text: str, review_item_id: uuid.UUID) -> uuid.UUID:
    """Adjudicated spans have no file, so identity comes from the adjudication."""
    return uuid.uuid5(NS_SPAN, f"{user_id}|adjudicated|{review_item_id}|{content_hash(claim_text)}")


def sentence_id(span: uuid.UUID, idx: int) -> uuid.UUID:
    return uuid.uuid5(NS_SENTENCE, f"{span}|{idx}")
