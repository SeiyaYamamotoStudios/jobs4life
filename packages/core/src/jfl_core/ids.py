"""Deterministic identifiers.

A span's id must survive re-ingestion. The obvious derivation -- hash(document,
ordinal, text) -- does not: inserting a bullet at the top of a file renumbers
every ordinal below it and orphans every id in the golden set.

So position is expressed as the *section heading path*, which is stable under
reordering within a section, and the ordinal is used only to disambiguate spans
whose normalised text is genuinely identical within the same section.

    span_id = uuid5(NS_SPAN, user_id | source_uri | section_path | text_hash | occurrence)

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
NS_JOB = uuid.uuid5(NS_ROOT, "job")
NS_REQUIREMENT = uuid.uuid5(NS_ROOT, "requirement")
NS_GAP_QUESTION = uuid.uuid5(NS_ROOT, "gap_question")

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


def document_id(user_id: uuid.UUID, source_uri: str) -> uuid.UUID:
    return uuid.uuid5(NS_DOCUMENT, f"{user_id}|{source_uri}")


def span_id(
    user_id: uuid.UUID, source_uri: str, section_path: str, text: str, occurrence: int = 0
) -> uuid.UUID:
    key = f"{user_id}|{source_uri}|{section_path}|{content_hash(text)}|{occurrence}"
    return uuid.uuid5(NS_SPAN, key)


def adjudicated_span_id(
    user_id: uuid.UUID, claim_text: str, review_item_id: uuid.UUID
) -> uuid.UUID:
    """Adjudicated spans have no file, so identity comes from the adjudication."""
    return uuid.uuid5(NS_SPAN, f"{user_id}|adjudicated|{review_item_id}|{content_hash(claim_text)}")


def sentence_id(span: uuid.UUID, idx: int) -> uuid.UUID:
    return uuid.uuid5(NS_SENTENCE, f"{span}|{idx}")


def job_id(user_id: uuid.UUID, raw_text: str) -> uuid.UUID:
    """Deterministic on the pasted text itself, so re-pasting the same ad resolves to
    the same job row instead of minting a duplicate (see `jobs`' unique constraint).
    """
    return uuid.uuid5(NS_JOB, f"{user_id}|{content_hash(raw_text)}")


def requirement_id(job: uuid.UUID, text: str) -> uuid.UUID:
    """Scoped to the job, not to position: re-extracting after an edited ad keeps the
    ids of requirements whose wording did not change.
    """
    return uuid.uuid5(NS_REQUIREMENT, f"{job}|{content_hash(text)}")


def gap_question_id(requirement: uuid.UUID) -> uuid.UUID:
    """Derived from the requirement ALONE, not the question text -- re-running
    coverage must refresh one stable question per requirement, never accumulate
    near-duplicates of it.
    """
    return uuid.uuid5(NS_GAP_QUESTION, str(requirement))


def adjudicated_span_id_from_answer(
    user_id: uuid.UUID, answer_text: str, gap_question: uuid.UUID
) -> uuid.UUID:
    """Adjudicated span minted from a gap answer. Distinct namespace key
    ("gap_answer") from `adjudicated_span_id`'s ("adjudicated") so the two
    write-back paths -- review-item adjudication and gap-question answers --
    can never collide even given the same user and text.
    """
    return uuid.uuid5(NS_SPAN, f"{user_id}|gap_answer|{gap_question}|{content_hash(answer_text)}")
